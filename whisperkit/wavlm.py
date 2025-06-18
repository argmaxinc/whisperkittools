#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from argmaxtools import _sdpa
from argmaxtools import utils as argmaxtools_utils
from argmaxtools.nn import Attention, AttentionType, LayerNorm, FFN

from transformers.activations import ACT2FN
from transformers.models.wavlm.configuration_wavlm import WavLMConfig
from transformers.models.wavlm import modeling_wavlm as hf_wavlm

logger = argmaxtools_utils.get_logger(__name__)

SDPA_IMPL = _sdpa.Cat


class WavLMFeatureEncoderLayer(nn.Module):
    def __init__(self, config, do_norm, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )

        self.activation = ACT2FN[config.feat_extract_activation]

        if do_norm:
            self.layer_norm = nn.GroupNorm(
                num_groups=self.out_conv_dim,
                num_channels=self.out_conv_dim,
                affine=True)

    def forward(self, hidden_states):
        hidden_states = F.conv2d(hidden_states,
                                 weight=self.conv.weight[:, :, None, :],
                                 bias=self.conv.bias,
                                 stride=self.conv.stride)

        if hasattr(self, "layer_norm"):
            hidden_states = self.layer_norm(hidden_states)
        return self.activation(hidden_states)


class WavLMFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.feat_extract_norm == "group"
        self.conv_layers = nn.ModuleList([
            WavLMFeatureEncoderLayer(config, layer_id=i, do_norm=i == 0)
            for i in range(config.num_feat_extract_layers)
        ])

    def forward(self, inputs):
        hidden_states = inputs

        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


class WavLMFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size,)

        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)

    def forward(self, hidden_states):
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = F.conv2d(norm_hidden_states,
                                 weight=self.projection.weight[:, :, None, None],
                                 bias=self.projection.bias)

        return hidden_states, self.pos_conv_embed(hidden_states)


class WavLMPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        weight_norm = nn.utils.parametrizations.weight_norm
        self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.unpad_from_right = 1 if config.num_conv_pos_embeddings % 2 == 0 else 0
        assert config.feat_extract_activation == "gelu"
        self.activation = F.gelu

    def forward(self, hidden_states):
        hidden_states = F.conv2d(hidden_states,
                                 weight=self.conv.weight[:, :, None, :],
                                 bias=self.conv.bias,
                                 padding=(0, self.conv.padding[0]),
                                 groups=self.conv.groups)
        if self.unpad_from_right > 0:
            hidden_states = hidden_states[..., :-self.unpad_from_right]
        return self.activation(hidden_states)


class WavLMAttention(Attention):
    def __init__(self, embed_dim, num_heads, num_buckets=320, max_distance=800):
        super().__init__(embed_dim=embed_dim,
                         n_heads=num_heads,
                         attention_type=AttentionType.SelfAttention)

        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sdpa_implementation = SDPA_IMPL

        self.gru_rel_pos_const = nn.Parameter(torch.ones(1, 1, self.num_heads, 1))
        self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)

    @classmethod
    def from_transformers(cls, source: hf_wavlm.WavLMAttention):
        return cls(embed_dim=source.embed_dim,
                   num_heads=source.num_heads,
                   num_buckets=source.num_buckets,
                   max_distance=source.max_distance)

    def forward(self, hidden_states, attention_mask, position_bias):
        batch_size = hidden_states.shape[0]
        bhcx = (batch_size, self.n_heads, self.per_head_dim, -1)

        gate_a, gate_b = F.conv2d(
            hidden_states.view(*bhcx).transpose(1, 2),
            weight=self.gru_rel_pos_linear.weight[:, :, None, None],
            bias=self.gru_rel_pos_linear.bias
        ).split(4, dim=1)

        gate_a = F.sigmoid(gate_a.sum(dim=1, keepdims=True))
        gate_b = F.sigmoid(gate_b.sum(dim=1, keepdims=True))
        gate_output = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0

        if len(position_bias.shape) == 3:
            position_bias = position_bias.unsqueeze(0)
        elif len(position_bias.shape) != 4:
            raise ValueError(
                "position_bias must have shape (batch_size, num_heads, seq_len, seq_len)"
            )

        qk_mask = gate_output.view(batch_size, self.num_heads, -1, 1) * position_bias

        key_padding_mask = (1. - attention_mask) * -1e4 if attention_mask is not None else None

        return super().forward(input_embeds=hidden_states,
                               key_padding_mask=key_padding_mask,
                               qk_mask=qk_mask)


class WavLMEncoderLayer(nn.Module):
    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.attention = WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
        )
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        assert config.intermediate_size % config.hidden_size == 0
        self.feed_forward = FFN(embed_dim=config.hidden_size,
                                expansion_factor=config.intermediate_size//config.hidden_size,
                                activation_fn=F.gelu)
        self.final_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.attention._register_load_state_dict_pre_hook(
            linear_to_conv2d_map_attention_wavlm)

        self.feed_forward._register_load_state_dict_pre_hook(
            linear_to_conv2d_map_ffn_wavlm)

    def forward(self,
                hidden_states,
                attention_mask=None,
                position_bias=None,):
        residual = hidden_states
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )[0]
        hidden_states = self.layer_norm(residual + hidden_states)
        return self.final_layer_norm(hidden_states + self.feed_forward(hidden_states))


class WavLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([
            WavLMEncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states, positional_embeddings, attention_mask=None):
        assert hasattr(self, "relative_position_bias"), "relative_position_bias must be precomputed"

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, None, None, :]
        hidden_states = hidden_states + positional_embeddings

        intermediate_outputs = []
        hidden_states = self.layer_norm(hidden_states)

        for _, layer in enumerate(self.layers):
            intermediate_outputs.append(hidden_states)
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=self.relative_position_bias,
            )

        intermediate_outputs.append(hidden_states)
        return hidden_states, intermediate_outputs


class WavLMPreprocessor(nn.Module):
    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.config = config
        self.feature_extractor = WavLMFeatureEncoder(config)
        self.feature_projection = WavLMFeatureProjection(config)
        self._register_load_state_dict_pre_hook(move_weights_to_preprocessor)

    def forward(self, waveforms):
        if len(waveforms.shape) == 2:
            waveforms = waveforms[:, None, None, :]

        extract_features = self.feature_extractor(waveforms)
        return self.feature_projection(extract_features)


class WavLM(nn.Module):
    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.config = config
        self.preprocessor = WavLMPreprocessor(config)
        self.encoder = WavLMEncoder(config)

        assert not config.add_adapter
        self._register_load_state_dict_pre_hook(move_weights_to_preprocessor)

    def forward(self, input_values, attention_mask=None):
        hidden_states, positional_embeddings = self.preprocessor(waveforms=input_values)

        return self.encoder(
            hidden_states,
            positional_embeddings,
            attention_mask=attention_mask,
        )


class WavLMWithoutPreprocessor(nn.Module):
    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.config = config
        self.encoder = WavLMEncoder(config)
        assert not config.add_adapter

    def forward(self, hidden_states, positional_embeddings, attention_mask=None):
        return self.encoder(
            hidden_states,
            positional_embeddings,
            attention_mask=attention_mask,
        )


class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = F.relu

    def forward(self, hidden_states):
        weight = self.kernel.weight.view(
            self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)[:, :, None, :]
        hidden_states = F.conv2d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)
        return self.activation(hidden_states)


class WavLMForXVector(nn.Module):
    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.config = config
        self.wavlm = WavLM(config)

        if config.use_weighted_layer_sum:
            n = config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.ones(n) / n)

        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])
        self.tdnn = nn.Sequential(*[
            TDNNLayer(config, i) for i in range(len(config.tdnn_dim))
        ])
        self.feature_extractor = nn.Linear(
            config.tdnn_dim[-1] * 2,
            config.xvector_output_dim
        )

    def xvector_head(self, hidden_states, intermediate_outputs, attention_mask):
        if self.config.use_weighted_layer_sum:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                hidden_states = sum([
                    w * h for w, h in
                    zip(F.softmax(self.layer_weights), intermediate_outputs)
                ])

        hidden_states = F.conv2d(hidden_states,
                                 weight=self.projector.weight[:, :, None, None],
                                 bias=self.projector.bias)

        # Note: This hidden state is frequently >99% sparse
        hidden_states = self.tdnn(hidden_states).transpose(1, 3)

        # TDNN contracts the sequence length due to temporal kernels without padding
        # Adjust the mask to match the contracted sequence length
        seq_len_diff = attention_mask.shape[1] - hidden_states.shape[1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            assert seq_len_diff % 2 == 0

        # TODO(atila): seq_len_diff due to self.tdnn padding behavior impacts
        # left and right sides of the sequence equally. However, reference
        # implementation assumes it only affects the right side. Test whether
        # [seq_len_diff // 2:-seq_len_diff // 2] would work equally well if not better
        # attention_mask = attention_mask[:, seq_len_diff // 2:-seq_len_diff // 2, None, None]
        attention_mask = attention_mask[:, :-seq_len_diff, None, None]

        # Mask sequence for active frames prior to temporal statistics pooling
        masked_hidden_states = hidden_states * attention_mask

        # Compensate for masked out frames in the stats aggregation
        num_nonzero = attention_mask.sum(dim=1, keepdims=True).clamp(min=1, max=hidden_states.shape[1])
        mean = (masked_hidden_states / num_nonzero).sum(dim=1, keepdims=True)
        std = (((masked_hidden_states - mean) / (num_nonzero - 1).clamp(min=1, max=hidden_states.shape[1]
                                                                        ).sqrt()
                ).square().sum(dim=1, keepdims=True)).sqrt()
        pooled_stats = torch.cat([mean.transpose(1, 3), std.transpose(1, 3)], dim=1)

        return F.conv2d(pooled_stats,
                        weight=self.feature_extractor.weight[:, :, None, None],
                        bias=self.feature_extractor.bias).squeeze(-1).squeeze(-1)

    def forward(self, input_values, attention_mask=None):
        hidden_states, intermediate_outputs = self.wavlm(input_values,
                                                         attention_mask=attention_mask)

        return self.xvector_head(hidden_states, intermediate_outputs, attention_mask)


class WavLMForXVectorWithoutPreprocessor(WavLMForXVector):
    def __init__(self, config: WavLMConfig):
        super().__init__(config)
        self.wavlm = WavLMWithoutPreprocessor(config)

    def forward(self, hidden_states, positional_embeddings, attention_mask=None):
        hidden_states, intermediate_outputs = self.wavlm(hidden_states,
                                                         positional_embeddings,
                                                         attention_mask=attention_mask)

        return self.xvector_head(hidden_states, intermediate_outputs, attention_mask)


class WavLMForSequenceClassification(nn.Module):
    def __init__(self, config: WavLMConfig):
        super().__init__()
        self.config = config
        self.wavlm = WavLM(config)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

    def classifier_head(self, hidden_states, attention_mask):
        hidden_states = F.conv2d(hidden_states,
                                 self.projector.weight[:, :, None, None],
                                 self.projector.bias)

        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            masked_hidden_states = hidden_states * attention_mask
            pooled_output = masked_hidden_states.sum(dim=-1, keepdims=True) \
                / masked_hidden_states.sum(dim=-1, keepdims=True)

        return F.conv2d(pooled_output,
                        self.classifier.weight[:, :, None, None],
                        self.classifier.bias)

    def forward(self, input_values, attention_mask=None):
        hidden_states = self.wavlm(
            input_values,
            attention_mask=attention_mask,
        )[0]

        return self.classifier_head(hidden_states, attention_mask)


class WavLMForSequenceClassificationWithoutPreprocessor(WavLMForSequenceClassification):
    def __init__(self, config: WavLMConfig):
        super().__init__(config)
        self.wavlm = WavLMWithoutPreprocessor(config)

    def forward(self, hidden_states, positional_embeddings, attention_mask=None):
        hidden_states = self.wavlm(hidden_states,
                                   positional_embeddings,
                                   attention_mask=attention_mask)[0]

        return self.classifier_head(hidden_states, attention_mask)


def move_weights_to_preprocessor(state_dict, prefix, local_metadata, strict,
                                 missing_keys, unexpected_keys, error_msgs):

    keys_to_pop = []
    state_dict_update = dict()
    for k in state_dict:
        if "preprocessor" not in k:
            if "pos_conv_embed" in k:
                new_k = k.replace("wavlm.encoder", "wavlm.preprocessor.feature_projection")
                if new_k not in state_dict:
                    keys_to_pop.append(k)
                    state_dict_update[new_k] = state_dict[k]
                    state_dict_update[k.replace("wavlm.encoder", "feature_projection")] = state_dict[k]

            if any(prefix in k for prefix in ["feature_extractor", "feature_projection"]):
                new_k = k.replace("wavlm", "wavlm.preprocessor")
                if new_k not in state_dict:
                    keys_to_pop.append(k)
                    state_dict_update[new_k] = state_dict[k]
                    state_dict_update[k.replace("wavlm.", "")] = state_dict[k]

    for k in keys_to_pop:
        state_dict.pop(k)

    state_dict.update(state_dict_update)

    return state_dict


def linear_to_conv2d_map_attention_wavlm(state_dict, prefix, local_metadata, strict,
                                         missing_keys, unexpected_keys, error_msgs):
    if prefix + "rel_attn_embed.weight" in state_dict:
        state_dict.pop(prefix + "rel_attn_embed.weight")

    if prefix + "gru_rel_pos_const" in state_dict:
        state_dict[prefix + "gru_rel_pos_const"] = state_dict[prefix + "gru_rel_pos_const"].transpose(1, 2)

    return argmaxtools_utils.linear_to_conv2d_map_attention(state_dict, prefix, local_metadata, strict,
                                                            missing_keys, unexpected_keys, error_msgs)


def linear_to_conv2d_map_ffn_wavlm(state_dict, prefix, local_metadata, strict,
                                   missing_keys, unexpected_keys, error_msgs):
    common_name_mappings = {
        "fc1": ["intermediate_dense"],
        "fc2": ["output_dense"],
    }
    return argmaxtools_utils.linear_to_conv2d_map_base(common_name_mappings,
                                                       state_dict, prefix, local_metadata,
                                                       strict, missing_keys, unexpected_keys, error_msgs)
