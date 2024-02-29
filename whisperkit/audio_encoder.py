#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from argmaxtools import _sdpa
from argmaxtools.compress.sparse_outlier import DecomposedModule
from argmaxtools.nn import Attention, AttentionType, LayerNorm
from argmaxtools.utils import (linear_to_conv2d_map_attention,
                               linear_to_conv2d_map_ffn)
from transformers.activations import ACT2FN
from transformers.models.whisper.configuration_whisper import WhisperConfig

import whisperkit.tensor_typing as tt

# Scaled Dot-Product Attention (SDPA) implementation to use for WhisperAudioEncoder
SDPA_IMPL = _sdpa.Cat


class WhisperAudioEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()

        # Encoder Self-Attention
        self.self_attn = Attention(
            embed_dim=config.d_model,
            n_heads=config.encoder_attention_heads,
            attention_type=AttentionType.SelfAttention,
        )
        self.self_attn._register_load_state_dict_pre_hook(linear_to_conv2d_map_attention)

        # Configure the SDPA implementation
        self.self_attn.sdpa_implementation = SDPA_IMPL

        self.self_attn_layer_norm = LayerNorm(config.d_model)

        # Feedforward Network
        self.fc1 = nn.Conv2d(config.d_model, config.decoder_ffn_dim,  1)
        self.act_fn = ACT2FN[config.activation_function]
        self.fc2 = nn.Conv2d(config.decoder_ffn_dim, config.d_model, 1)
        self._register_load_state_dict_pre_hook(linear_to_conv2d_map_ffn)
        self.final_layer_norm = LayerNorm(config.d_model)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            Inputs:
                input_embeds:   (batch_size, embed_dim, 1, encoder_seq_len)
            Outputs:
                outputs:        (batch_size, embed_dim, 1, encoder_seq_len)
        """
        # Self-attention
        residual = input_embeds
        hidden_states = self.self_attn_layer_norm(input_embeds)
        hidden_states, = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.act_fn(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperAudioEncoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(
            config.num_mel_bins,
            config.d_model,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = nn.ModuleList([
            WhisperAudioEncoderLayer(config) for _ in range(config.encoder_layers)
        ])
        self.layer_norm = LayerNorm(config.d_model)

    def forward(self,
                melspectrogram_features: tt.WhisperAudioEncoderInputType
                ) -> tt.WhisperAudioEncoderOutputType:
        """
        Shapes:
            Inputs:
                melspectrogram_features:    (batch_size, num_mel_bins, 1, encoder_seq_len * 2)

            Outputs:
                outputs:                    (batch_size, embed_dim, 1, encoder_seq_len)
        """
        hidden_states = self.pre_transformer_proj(melspectrogram_features)

        output_seq_len = hidden_states.shape[3]
        pos_embeds = self.embed_positions.weight.transpose(0, 1)[None, :, None, :output_seq_len]

        hidden_states = hidden_states + pos_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.layer_norm(hidden_states)

    def pre_transformer_proj(self, melspectrogram_features: torch.Tensor) -> torch.Tensor:
        # TODO(atiorh): Fix the leaky abstraction caused here due to direct weight access
        if isinstance(self.conv1, DecomposedModule):
            hidden_states = F.gelu(F.conv2d(
                melspectrogram_features,
                self.conv1.outlier_module.weight.data[:, :, None, :],
                torch.zeros_like(self.conv1.inlier_module.bias.data),
                padding=(0, 1),
            ) + F.conv2d(
                melspectrogram_features,
                self.conv1.inlier_module.weight.data[:, :, None, :],
                self.conv1.inlier_module.bias.data,
                padding=(0, 1),
            ))
        else:
            hidden_states = F.gelu(F.conv2d(
                melspectrogram_features,
                self.conv1.weight.data[:, :, None, :],
                self.conv1.bias.data,
                padding=(0, 1),
            ))

        if isinstance(self.conv2, DecomposedModule):
            return F.gelu(F.conv2d(
                hidden_states,
                self.conv2.outlier_module.weight.data[:, :, None, :],
                torch.zeros_like(self.conv2.inlier_module.bias.data),
                padding=(0, 1),
                stride=2
            ) + F.conv2d(
                hidden_states,
                self.conv2.inlier_module.weight.data[:, :, None, :],
                self.conv2.inlier_module.bias.data,
                padding=(0, 1),
                stride=2
            ))
        else:
            return F.gelu(F.conv2d(
                hidden_states,
                self.conv2.weight.data[:, :, None, :],
                self.conv2.bias.data,
                padding=(0, 1),
                stride=2
            ))


MEL_FILTERS_URL = \
    "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"


# Reference: https://github.com/openai/whisper/blob/main/whisper/audio.py
class WhisperMelSpectrogram(nn.Module):
    def __init__(self, n_mels=80, n_fft=400, hop_length=160):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("hann_window", torch.hann_window(n_fft))
        self.register_buffer("mel_filters", self.get_mel_filters(n_mels))

    @staticmethod
    def get_mel_filters(n_mels):
        from io import BytesIO

        import numpy as np
        import requests
        mel_filters_dict = np.load(BytesIO(requests.get(MEL_FILTERS_URL)._content))

        key = f"mel_{n_mels}"
        if key not in mel_filters_dict:
            raise KeyError(f"{key} is not available. Options: {list(mel_filters_dict)}")

        return torch.from_numpy(mel_filters_dict[key])

    def forward(self,
                audio: tt.WhisperMelSpectrogramInputType
                ) -> tt.WhisperMelSpectrogramOutputType:
        stft = torch.stft(
            audio,
            self.n_fft,
            self.hop_length,
            window=self.hann_window,
            return_complex=True,
        )

        # complex-to-real with abs() ** 2 as squared L2
        magnitudes = stft.abs() ** 2
        magnitudes = magnitudes[..., :-1]
        mel_spec = self.mel_filters @ magnitudes
        mel_spec = mel_spec + 1e-10
        log_spec = mel_spec.log10()

        # Range normalization
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        # Fake batch dimension and reshaping to match WhisperAudioEncoder's expected input shape
        return log_spec[None, :, None, :]
