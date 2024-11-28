#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

from collections import OrderedDict
from enum import Enum
from typing import List

import coremltools as ct
import torch
from argmaxtools.test_utils import (AppleSiliconContextMixin,
                                    InferenceContextSpec)
from transformers.models.whisper import modeling_whisper


class BenchmarkContext(AppleSiliconContextMixin, InferenceContextSpec):
    """ Context specifier to reproduce an inference context
    """
    def __init__(self, code_commit_hash, model_commit_hash):
        self.code_commit_hash = code_commit_hash
        self.model_commit_hash = model_commit_hash

    def code_spec(self):
        return {"code_commit_hash": self.code_commit_hash}

    def model_spec(self):
        return {"model_commit_hash": self.model_commit_hash}


# Overrides sequence length for the decoder
TEST_DEC_KV_SEQ_LEN = None


# TextDecoder utils
def _prepare_test_inputs_for_decoder(embed_dim,
                                     vocab_size,
                                     n_heads,
                                     n_layers,
                                     batch_size,
                                     enc_seq_len,
                                     dec_kv_seq_len,
                                     active_dec_kv_seq_len):
    """
    Prepare random test inputs with:
    - Key-value cache
    - Fixed length encoder context utilization
    - Randomized length decoder context utilization (via padding masks)
    """
    dec_q_seq_len = 1
    shared_query_tokens = torch.randint(0, vocab_size, size=(batch_size, dec_q_seq_len))

    shared_decoder_kv_cache = [(
        # key cache
        torch.randn(
            batch_size,
            n_heads,
            dec_kv_seq_len,
            embed_dim // n_heads,
        ),
        # value cache
        torch.randn(
            batch_size,
            n_heads,
            dec_kv_seq_len,
            embed_dim // n_heads,
        ),
    ) for _ in range(n_layers)]

    shared_encoder_hidden_states = torch.randn(
        batch_size,
        enc_seq_len,
        embed_dim,
    )

    # Hugging Face transformers implementation inputs
    huggingface_inputs = OrderedDict([
        ("input_ids", shared_query_tokens),
        ("attention_mask", torch.ones(size=(batch_size, active_dec_kv_seq_len))),
        ("encoder_hidden_states", shared_encoder_hidden_states),
        ("past_key_values", [
            [cache[:, :, :active_dec_kv_seq_len - 1, :] for cache in layer_decoder_kv_cache]
            for layer_decoder_kv_cache in shared_decoder_kv_cache
        ]),
    ])

    # Argmax implementation inputs
    bcs = (batch_size, -1, embed_dim)
    argmax_inputs = OrderedDict([
        ("input_ids", shared_query_tokens.squeeze(1)),
        ("cache_length", torch.tensor([active_dec_kv_seq_len - 1] * batch_size)),
        ("key_cache", torch.cat([
                layer_kv_cache[0].transpose(1, 2).reshape(*bcs).transpose(1, 2).unsqueeze(2)
                for layer_kv_cache in shared_decoder_kv_cache
            ],
            dim=1,
        )),
        ("value_cache", torch.cat([
                layer_kv_cache[1].transpose(1, 2).reshape(*bcs).transpose(1, 2).unsqueeze(2)
                for layer_kv_cache in shared_decoder_kv_cache
            ],
            dim=1,
        )),
        # This a one-hot encoding of the current token index to decode, hot value inserted below
        ("kv_cache_update_mask", torch.zeros((batch_size, dec_kv_seq_len))),
        ("encoder_output_embeds", shared_encoder_hidden_states.transpose(1, 2).unsqueeze(2)),
        ("decoder_key_padding_mask", torch.cat([
                torch.zeros((batch_size, active_dec_kv_seq_len)),
                torch.ones((batch_size, dec_kv_seq_len - active_dec_kv_seq_len)) * -1e4
            ],
            dim=1,
        )),
    ])

    argmax_inputs["kv_cache_update_mask"][:, active_dec_kv_seq_len - 1] = 1.

    return argmax_inputs, huggingface_inputs


def _prepare_test_inputs_for_decoder_from_cfg(batch_size: int,
                                              cfg: modeling_whisper.WhisperConfig):
    return _prepare_test_inputs_for_decoder(
        embed_dim=cfg.d_model,
        vocab_size=cfg.vocab_size,
        n_heads=cfg.decoder_attention_heads,
        n_layers=cfg.decoder_layers,
        batch_size=batch_size,
        enc_seq_len=cfg.max_source_positions,
        dec_kv_seq_len=TEST_DEC_KV_SEQ_LEN or cfg.max_target_positions,
        active_dec_kv_seq_len=(TEST_DEC_KV_SEQ_LEN or cfg.max_target_positions) - 1,
    )


def _get_context_prefill_from(hf_whisper_decoder: modeling_whisper.WhisperDecoder,
                              prefill_decoder_ids: List[int],
                              encoder_output_embeds: torch.Tensor):
    """
    Helper function to run the HuggingFace WhisperDecoder implementation in prefill mode using the
    I/O format of Argmax WhisperDecoder implementation
    """
    batch_size = encoder_output_embeds.shape[0]
    bcs = (batch_size, -1, hf_whisper_decoder.config.d_model)
    dev = hf_whisper_decoder.device

    caches = hf_whisper_decoder(
        input_ids=torch.tensor(prefill_decoder_ids, device=dev).unsqueeze(0).expand(batch_size, -1),
        encoder_hidden_states=encoder_output_embeds.squeeze(2).transpose(1, 2).to(dev),
        use_cache=True,
    ).past_key_values

    key_cache = torch.cat([
        caches[layer_idx][0].transpose(1, 2).reshape(*bcs).transpose(1, 2).unsqueeze(2)
        for layer_idx in range(len(caches))
    ], dim=1).mean(0, keepdims=True).flatten()

    value_cache = torch.cat([
        caches[layer_idx][1].transpose(1, 2).reshape(*bcs).transpose(1, 2).unsqueeze(2)
        for layer_idx in range(len(caches))
    ], dim=1).mean(0, keepdims=True).flatten()

    return key_cache, value_cache


def set_metadata_for_whisper_decoder(coreml_model: ct.models.MLModel, whisper_version: str) -> None:
    """
    Add metadata to Core ML models for traceability
    """
    coreml_model.short_description = "Whisper Automatic Speech Recognition Model: Text Decoder"
    model_card_ref = \
        f"Please refer to the Model Card available at huggingface.co/{whisper_version}"
    coreml_model.author = model_card_ref
    coreml_model.license = model_card_ref
    coreml_model.version = whisper_version

    from argmaxtools._version import __version__ as argmaxtools_version

    from whisperkit._version import __version__ as whisperkit_version
    coreml_model.whisperkit_version = whisperkit_version
    coreml_model.argmaxtools_version = argmaxtools_version
    # coreml_model.coremltools_version = ct.version.__version__  # already written by Apple

    # Set the input descriptions
    coreml_model.input_description["input_ids"] = "Token id for current decoding step"
    coreml_model.input_description["cache_length"] = \
        "Numbers of tokens decoded so far in the current context"
    coreml_model.input_description["key_cache"] = \
        "Per-layer key projection outputs cached from previous decoding steps concat across dim=1"
    coreml_model.input_description["value_cache"] = \
        "Per-layer value projection outputs cached from previous decoding steps concat across dim=1"
    coreml_model.input_description["kv_cache_update_mask"] = \
        "One-hot encoded vector that marks the current token index in the sequence"
    coreml_model.input_description["encoder_output_embeds"] = "Output tensor from AudioEncoder"
    coreml_model.input_description["decoder_key_padding_mask"] = \
        "Mask that disables attention on invalid (e.g. padded) decoder tokens"

    # Set the output descriptions
    coreml_model.output_description["logits"] = "Logits over the vocabulary for next token"
    coreml_model.output_description["key_cache_updates"] = \
        "Slice update for key_cache computed during the latest forward pass"
    coreml_model.output_description["value_cache_updates"] = \
        "Slice update for value_cache computed during the latest forward pass"


# AudioEncoder utils
def _prepare_test_inputs_for_encoder_from_cfg(batch_size: int,
                                              cfg: modeling_whisper.WhisperConfig):
    return _prepare_test_inputs_for_encoder(
        embed_dim=cfg.d_model,
        n_mels=cfg.num_mel_bins,
        batch_size=batch_size,
        melspectrogram_seq_len=cfg.max_source_positions * 2,
    )


def _prepare_test_inputs_for_encoder(embed_dim,
                                     n_mels,
                                     batch_size,
                                     melspectrogram_seq_len,
                                     **kwargs):
    melspectrogram_features = torch.randn(
        batch_size,
        n_mels,
        1,
        melspectrogram_seq_len,
    )

    argmax_inputs = dict(melspectrogram_features=melspectrogram_features)
    huggingface_transformers_inputs = dict(input_features=melspectrogram_features.squeeze(2))

    return argmax_inputs, huggingface_transformers_inputs
