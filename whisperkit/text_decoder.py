#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#
from copy import deepcopy
from itertools import product
import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from argmaxtools import _sdpa
from argmaxtools import utils as argmaxtools_utils
from argmaxtools.nn import Attention, AttentionType, LayerNorm
from beartype import beartype
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.whisper.configuration_whisper import WhisperConfig

import whisperkit.tensor_typing as tt

logger = argmaxtools_utils.get_logger(__name__)

# Scaled Dot-Product Attention (SDPA) implementation to use for WhisperTextDecoder
SDPA_IMPL = _sdpa.Cat


class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()

        # Decoder Self-Attention
        self.self_attn = Attention(
            embed_dim=config.d_model,
            n_heads=config.decoder_attention_heads,
            attention_type=AttentionType.KVCachedSelfAttention,
        )
        self.self_attn.sdpa_implementation = SDPA_IMPL
        self.self_attn._register_load_state_dict_pre_hook(
            argmaxtools_utils.linear_to_conv2d_map_attention
        )
        self.self_attn_layer_norm = LayerNorm(config.d_model)

        # Encoder Decoder Self-Attention
        self.encoder_attn = Attention(
            embed_dim=config.d_model,
            n_heads=config.decoder_attention_heads,
            attention_type=AttentionType.EncoderDecoderCrossAttention,
        )

        self.encoder_attn.sdpa_implementation = SDPA_IMPL
        self.encoder_attn._register_load_state_dict_pre_hook(
            argmaxtools_utils.linear_to_conv2d_map_attention
        )
        self.encoder_attn_layer_norm = LayerNorm(config.d_model)

        # Feedforward Network
        self.fc1 = nn.Conv2d(config.d_model, config.decoder_ffn_dim, 1)
        self.act_fn = ACT2FN[config.activation_function]
        self.fc2 = nn.Conv2d(config.decoder_ffn_dim, config.d_model, 1)
        self.final_layer_norm = LayerNorm(config.d_model)
        self._register_load_state_dict_pre_hook(
            argmaxtools_utils.linear_to_conv2d_map_ffn
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        kv_cache_update_mask: torch.Tensor,
        encoder_output_embeds: torch.Tensor,
        decoder_key_padding_mask: Optional[torch.Tensor] = None,
        encoder_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Shapes:
            Inputs:
                input_embeds:               (batch_size, embed_dim, 1, q_seq_len=1)
                key_cache:                  (batch_size, embed_dim, 1, kv_seq_len)
                value_cache:                (batch_size, embed_dim, 1, kv_seq_len)
                kv_cache_update_mask:       (batch_size, kv_seq_len)
                encoder_output_embeds:      (batch_size, embed_dim, 1, kv_seq_len)
                decoder_key_padding_mask:   (batch_size, decoder_kv_seq_len)
                encoder_key_padding_mask:   (batch_size, encoder_kv_seq_len)


            Outputs:
                outputs:                (batch_size, embed_dim, 1, q_seq_len=1)
                current_key:            (batch_size, embed_dim, 1, q_seq_len=1)
                current_value:          (batch_size, embed_dim, 1, q_seq_len=1)
        """
        residual = input_embeds
        hidden_states = self.self_attn_layer_norm(input_embeds)

        # Self Attention
        hidden_states, key_cache_update, value_cache_update = self.self_attn(
            hidden_states,
            decoder_key_padding_mask,
            key_cache,
            value_cache,
            kv_cache_update_mask,
        )
        hidden_states = residual + hidden_states

        # Encoder Cross-Attention
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states,
            key_padding_mask=encoder_key_padding_mask,
            encoder_output_embeds=encoder_output_embeds,
        )[0]
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.act_fn(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states

        return hidden_states, key_cache_update, value_cache_update


@beartype
class WhisperTextDecoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.d = config.d_model
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_target_positions, config.d_model)

        self.layers = nn.ModuleList(
            [WhisperDecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.layer_norm = LayerNorm(config.d_model)
        self._alignment_heads = None

    def configure_for_token_timestamps(self, generation_config: GenerationConfig) -> None:
        """ Setup forward pass to return attention weights from alignment heads as output
        """
        self._alignment_heads = generation_config.alignment_heads

        def save_w(module, input, output):
            assert isinstance(module, Attention), type(module)
            assert len(output) > 1, len(output)
            setattr(module, "current_w", output[-1])

        for i in range(len(self.layers)):
            self.layers[i].encoder_attn._return_w = True
            self.layers[i].encoder_attn.register_forward_hook(save_w)

    def compute_alignment_heads_attention_weights(self):
        assert self._alignment_heads is not None, \
            "Call configure_for_token_timestamps() to enable this feature"

        alignment_weights = []
        for layer, head in self._alignment_heads:
            logger.debug(f"Gathering token-level alignment weights for layer={layer}, head={head}")

            w = self.layers[layer].encoder_attn.current_w
            _, h, q, _ = w.shape
            assert q == 1 and h == self.config.decoder_attention_heads
            alignment_weights.append(w[:, head:head+1, 0, :])

        return torch.cat(alignment_weights, dim=1).mean(dim=1)

    def forward(
        self,
        input_ids: tt.WhisperTextDecoderInputIdsType,
        cache_length: tt.WhisperTextDecoderCacheLengthType,
        key_cache: tt.WhisperTextDecoderKVCacheType,
        value_cache: tt.WhisperTextDecoderKVCacheType,
        kv_cache_update_mask: tt.WhisperTextDecoderAttentionMaskType,
        encoder_output_embeds: tt.WhisperTextDecoderEncoderOutputEmbedsType,
        decoder_key_padding_mask: Optional[
            tt.WhisperTextDecoderAttentionMaskType
        ] = None,
    ) -> tt.WhisperTextDecoderOutputType:
        """
        Shapes:
            Inputs:
                input_ids:                  (batch, )
                cache_length:               (batch, )
                key_cache:                  (batch, embed_dim*n_layers, 1, decoder_kv_seq_len)
                value_cache:                (batch, embed_dim*n_layers, 1, decoder_kv_seq_len)
                kv_cache_update_mask:       (batch, kv_seq_len)
                encoder_output_embeds:      (batch, embed_dim, 1, encoder_kv_seq_len)
                decoder_key_padding_mask:   (batch, decoder_kv_seq_len)

            Outputs:
                logits:                     (batch, decoder_q_seq_len, vocab_size)
                current_key:                (batch, embed_dim*n_layers, 1, decoder_q_seq_len)
                current_value:              (batch, embed_dim*n_layers, 1, decoder_q_seq_len)
        """
        # Vocabulary and positional embeddings
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(
            cache_length
        )
        hidden_states = hidden_states[:, :, None, None]

        # Ready per-layer key/value caches
        layer_key_caches = key_cache.split(self.d, dim=1)
        layer_value_caches = value_cache.split(self.d, dim=1)

        key_cache_updates = []
        value_cache_updates = []
        for idx, layer in enumerate(self.layers):
            hidden_states, key_cache_update, value_cache_update = layer(
                hidden_states,
                layer_key_caches[idx],
                layer_value_caches[idx],
                kv_cache_update_mask,
                encoder_output_embeds,
                decoder_key_padding_mask,
            )

            # Update key-value caches with current query token(s)
            key_cache_updates.append(key_cache_update)
            value_cache_updates.append(value_cache_update)

        hidden_states = self.layer_norm(hidden_states)

        # Project reusing input embeddings
        logits = F.linear(
            hidden_states.squeeze(2).transpose(1, 2), self.embed_tokens.weight
        )

        outputs = (
            logits,
            torch.cat(key_cache_updates, dim=1),
            torch.cat(value_cache_updates, dim=1),
        )

        # If configured for token-level timestamps, prepare return average attention
        # weights from the alignment heads
        if self._alignment_heads is not None:
            outputs += (self.compute_alignment_heads_attention_weights(),)

        return outputs


# TODO(atila): Extend support to `.en` Whisper versions
class WhisperTextDecoderContextPrefill(nn.Module):
    def __init__(
        self,
        whisper_decoder: WhisperTextDecoder,
        encoder_output_embeds: tt.WhisperTextDecoderEncoderOutputEmbedsType = None,
    ) -> None:
        """
        Enumerates all TextDecoder context prefixes that define a valid task, precomputes the KV
        cache values and initializes an `nn.Embedding` for look-up during inference.

        whisper_decoder:        A `WhisperTextDecoder` instance used to pre-compute key-value caches
                                for each task spec
        encoder_output_embeds:  Optional input used to anchor the generated caches on a specific
                                AudioEncoder context (used for output correctness testing)
        """
        super().__init__()

        self.embed_dim = whisper_decoder.config.d_model
        self.num_layers = whisper_decoder.config.decoder_layers
        self.whisper_version = whisper_decoder.config._name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.whisper_version)
        self.inverse_vocab = dict(
            zip(self.tokenizer.vocab.values(), self.tokenizer.vocab.keys())
        )

        self.lang_id_offset = self.tokenizer.vocab["<|en|>"]  # For OpenAI models: 50259

        if self.whisper_version.endswith(".en"):
            self.num_langs = 1
            self.tasks = ["<|transcribe|>"]
        else:
            # Note: 99 for all except for large-v3 which has 100 (added <|yue|>)
            self.num_langs = (
                self.tokenizer.vocab["<|translate|>"] - self.tokenizer.vocab["<|en|>"]
            )
            self.tasks = ["<|transcribe|>", "<|translate|>"]

        logger.info(f"Context prefill detected {self.num_langs} languages")

        self.prefill_seq_len = 3
        valid_task_specs = list(
            product(
                # This first token is implied in all task specs
                # 1st token: ["<|startoftranscript|>"],
                # 2nd token: Language specifier
                [
                    self.inverse_vocab[self.lang_id_offset + lang_idx]
                    for lang_idx in range(self.num_langs)
                ],
                # 3rd token: Task specifier
                self.tasks,
                # 4th token: First user-specified token to decode post-prefill
                # ["<|notimestamps|>", "<|0.00|>"]
            )
        )

        logger.info(
            f"{len(valid_task_specs)} task spec decoder context prefixes identified."
        )
        self.valid_task_specs = valid_task_specs

        self.cache_shape = (
            1,
            self.embed_dim * self.num_layers,
            1,
            self.prefill_seq_len,
        )

        # Look-up table: keys=valid task specs, values=flattened key/value caches
        self.key_cache_lut = nn.Embedding(
            len(self.valid_task_specs),
            self.embed_dim * self.num_layers * self.prefill_seq_len,
        )

        self.value_cache_lut = nn.Embedding(
            len(self.valid_task_specs),
            self.embed_dim * self.num_layers * self.prefill_seq_len,
        )

        self._fill_lut(whisper_decoder, encoder_output_embeds)

    def _fill_lut(
        self,
        whisper_decoder: WhisperTextDecoder,
        encoder_output_embeds: tt.WhisperTextDecoderEncoderOutputEmbedsType,
    ) -> None:
        """Given a `WhisperTextDecoder` instance, pre-compute the autoregressive decoder loop
        using forced decoder context prefix tokens and save results to nn.Embedding look-up tables
        """
        str2int = self.tokenizer.vocab
        dev = os.getenv("TEST_DEV", None) or argmaxtools_utils.get_fastest_device()

        # Note: The cache technically shouldn't be pre-computable because, even though the (forced)
        # decoder `input_ids` are known ahead of time, the cache is also a function of the runtime-
        # determined `encoder_output_embeds`. However, during training, decoder key-value embeddings
        # are uncorrelated with `encoder_output_embeds` up until the actual transcribed text tokens
        # tokens due to causal masking. Hence, we can estimate the prefill with a random batch.
        batch = encoder_output_embeds.shape[0]

        def _get_empty_context_inputs():
            """Build and return inputs for the first decoding step with empty caches.
            Should be finalized with <|startoftranscript|> query tokens downstream
            """
            cache_shape = (
                batch,
                self.embed_dim * self.num_layers,
                1,
                self.prefill_seq_len,
            )

            empty_ctx_inputs = dict(
                [
                    ("input_ids", None),  # inserted by _update_cache_related_inputs()
                    ("cache_length", torch.zeros(batch, dtype=torch.int32)),
                    ("key_cache", torch.zeros(*cache_shape)),
                    ("value_cache", torch.zeros(*cache_shape)),
                    (
                        "kv_cache_update_mask",
                        torch.cat(
                            [
                                torch.ones(batch, 1),
                                torch.zeros(batch, self.prefill_seq_len - 1),
                            ],
                            dim=1,
                        ),
                    ),
                    ("encoder_output_embeds", encoder_output_embeds),
                    # Mask out all positions except first
                    (
                        "decoder_key_padding_mask",
                        torch.cat(
                            [
                                torch.zeros(batch, 1),
                                torch.ones(batch, self.prefill_seq_len - 1),
                            ],
                            dim=1,
                        )
                        * -1e4,
                    ),
                ]
            )
            return {
                k: v.to(dev) if v is not None else v
                for k, v in empty_ctx_inputs.items()
            }

        def _update_cache_related_inputs(for_query, current_inputs):
            """Run the decoder on current inputs and update kv caches"""
            cache_len = current_inputs["cache_length"][0]
            first_token = "<|startoftranscript|>"
            if cache_len.eq(0) and for_query != first_token:
                raise ValueError(
                    f"{first_token} is the only legal query token for cache_length=0"
                )

            current_inputs["input_ids"] = torch.tensor(
                [str2int[for_query]] * batch, device=dev
            )

            # Compute cache updates (disregard logits as the token sequence is forced)
            _, key_cache_updates, value_cache_updates = whisper_decoder(
                **current_inputs
            )[:3]

            # Update kv caches
            current_inputs["key_cache"][
                ..., cache_len: cache_len + 1
            ] = key_cache_updates
            current_inputs["value_cache"][
                ..., cache_len: cache_len + 1
            ] = value_cache_updates

            # Advance the context and cache cursors
            current_inputs["cache_length"] = current_inputs["cache_length"] + 1
            current_inputs["kv_cache_update_mask"] = torch.roll(
                current_inputs["kv_cache_update_mask"], shifts=1, dims=1
            )
            current_inputs["decoder_key_padding_mask"] = torch.roll(
                current_inputs["decoder_key_padding_mask"], shifts=1, dims=1
            )
            current_inputs["decoder_key_padding_mask"][:, 0] = 0.0

            # Remove input_ids (next call will reset it)
            current_inputs.pop("input_ids")

            # Don't touch encoder_output_embeds and return updated cache-related inputs
            return current_inputs

        # Precompute first token cache as this is shared by all task specs
        whisper_decoder = whisper_decoder.to(dev)
        inputs_post_first_token = _update_cache_related_inputs(
            for_query="<|startoftranscript|>",
            current_inputs=_get_empty_context_inputs(),
        )

        for task_spec_idx, valid_task_spec in tqdm(enumerate(self.valid_task_specs)):
            # Reuse inputs post first token across tasks
            current_inputs = deepcopy(inputs_post_first_token)

            for current_query_token in valid_task_spec:
                current_inputs = _update_cache_related_inputs(
                    for_query=current_query_token,
                    current_inputs=current_inputs,
                )

                logger.debug(
                    f"Updated kv caches to length={current_inputs['cache_length']} "
                    f"with current query={current_query_token}"
                )

            # Take the batch mean estimate (over random encoder outputs) and flatten for look-up
            key_cache_prefill = current_inputs["key_cache"].mean(0).flatten()
            value_cache_prefill = current_inputs["value_cache"].mean(0).flatten()
            self.key_cache_lut.weight.data[task_spec_idx] = key_cache_prefill
            self.value_cache_lut.weight.data[task_spec_idx] = value_cache_prefill

        logger.info("Key-value cache look-up tables were filled")

    def task_and_language_to_task_idx(
        self, task: Union[int, torch.Tensor], language: Union[int, torch.Tensor]
    ) -> Union[int, torch.Tensor]:
        """
        Args:
            task:      0->transcribe, 1->translate
            language:  Original token index from vocab, e.g. <|en|>=50259
        """
        return 2 * (language - self.lang_id_offset) + task

    def task_idx_to_task_and_language(
        self, task_idx: Union[int, torch.Tensor]
    ) -> Tuple[Union[int, torch.Tensor]]:
        """
        Args:
            task_idx: Refers to the flattened index representing selected task & language
        """
        return task_idx % 2, self.lang_id_offset + task_idx // 2

    def forward(
        self, task: torch.Tensor, language: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """Look-up key-value cache prefill values for a given task spec.

        Args:
            task:      0->transcribe, 1->translate
            language:  Original token index from vocab, e.g. <|en|>=50259

        Return shapes:
            key_cache_prefill:    (1, embed_dim*num_layers, 1, self.prefill_seq_len)
            value_cache_prefill:  (1, embed_dim*num_layers, 1, self.prefill_seq_len)
        """
        # Arithmetic to mirror the task ordinality used during _fill_lut()
        task_idx = self.task_and_language_to_task_idx(task, language)

        key_cache_prefill = self.key_cache_lut(task_idx).view(*self.cache_shape)
        value_cache_prefill = self.value_cache_lut(task_idx).view(*self.cache_shape)

        return key_cache_prefill, value_cache_prefill
