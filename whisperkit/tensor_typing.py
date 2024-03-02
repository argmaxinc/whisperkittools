#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

from beartype.typing import Tuple, Union
from jaxtyping import Float, Int
from torch import Tensor

# audio_encoder.WhisperMelSpectrogram type hints
# Default Whisper input is at 16kHz for 30s: 480000
WhisperMelSpectrogramInputType = Float[Tensor, "num_audio_samples"]
WhisperMelSpectrogramOutputType = Float[Tensor, "1 num_mel_bins 1 2*encoder_seq_len"]

# audio_encoder.WhisperAudioEncoder type hints
WhisperAudioEncoderInputType = Float[Tensor, "batch num_mel_bins 1 2*encoder_seq_len"]
WhisperAudioEncoderOutputType = Float[Tensor, "batch embed_dim 1 encoder_seq_len"]

# text_decoder.WhisperTextDecoder type hints
# Inputs
WhisperTextDecoderInputIdsType = Int[Tensor, "batch"]
WhisperTextDecoderCacheLengthType = Int[Tensor, "batch"]
WhisperTextDecoderKVCacheType = Float[Tensor, "batch embed_dim_x_n_layers 1 decoder_kv_seq_len"]
WhisperTextDecoderAttentionMaskType = Float[Tensor, "batch decoder_kv_seq_len"]
WhisperTextDecoderEncoderOutputEmbedsType = Float[Tensor, "batch embed_dim 1 encoder_kv_seq_len"]

# Outputs
WhisperTextDecoderLogitsType = Float[Tensor, "batch decoder_q_seq_len vocab_size"]
WhisperTextDecoderKVCacheUpdateType = \
    Float[Tensor, "batch embed_dim_x_n_layers 1 decoder_q_seq_len"]
WhisperTextDecoderAlignmentHeadsWeightsType = \
    Float[Tensor, "batch  decoder_kv_seq_len"]

WhisperTextDecoderOutputType = Tuple[
    WhisperTextDecoderLogitsType,
    WhisperTextDecoderKVCacheUpdateType,
    WhisperTextDecoderKVCacheUpdateType
]

WhisperTextDecoderOutputType = Union[
    WhisperTextDecoderOutputType,
    Tuple[
        WhisperTextDecoderLogitsType,
        WhisperTextDecoderKVCacheUpdateType,
        WhisperTextDecoderKVCacheUpdateType,
        WhisperTextDecoderAlignmentHeadsWeightsType,
    ]
]
