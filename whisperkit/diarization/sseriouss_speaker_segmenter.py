# Source:
# https://github.com/pyannote/pyannote-audio/blob/main/pyannote/audio/models/segmentation/SSeRiouSS.py

#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.
#

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.wavlm.configuration_wavlm import WavLMConfig
from pyannote.core.utils.generators import pairwise
import pyannote.audio
from pyannote.audio.models.segmentation import SSeRiouSS as PyannoteSSeRiouSS
from pyannote.audio.utils.powerset import Powerset

from argmaxtools.utils import get_logger

from whisperkit import wavlm

torch.set_grad_enabled(False)
logger = get_logger(__name__)

TEST_SAMPLE_RATE = 16_000
TEST_MODEL_NATIVE_WINDOW_SIZE = 20  # 20s

# Sliding window constants
SLIDING_WINDOW_STRIDE = 5 * TEST_SAMPLE_RATE  # 5s
SLIDING_WINDOW_SIZE = TEST_MODEL_NATIVE_WINDOW_SIZE * TEST_SAMPLE_RATE   # 20s

# Training time constants/assumptions that can not be overriden at inference time
MAX_SPEAKERS_PER_CHUNK = 3  # at most 3 interlopers in a single model window
MAX_SPEAKERS_PER_FRAME = 2  # at most 2 speakers overlapping at any given time

TEST_SEQ_LEN = 999  # 20s of audio at 16kHz yields 999 seqlen after the Convolutional preprocessor


class SSeRiouSS(nn.Module):
    """Self-Supervised Representation for Speaker Segmentation

    wav2vec > LSTM > Feed forward > Classifier
    """

    WAV2VEC_DEFAULT_VERSION = "microsoft/wavlm-base-plus-sv"
    WAVLM_DEFAULT_CONFIG = WavLMConfig.from_pretrained(WAV2VEC_DEFAULT_VERSION)
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 4,
        "bidirectional": True,
        "dropout": 0.0,
        "batch_first": True,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(self, sliding_window_stride: int | None = None):
        super().__init__()
        # only supports TorchAudio compatible local models

        self.wavlm = wavlm.WavLM(self.WAVLM_DEFAULT_CONFIG)
        self.sliding_window_stride = sliding_window_stride or SLIDING_WINDOW_STRIDE

        wavlm_dim = self.WAVLM_DEFAULT_CONFIG.hidden_size
        wavlm_num_layers = self.WAVLM_DEFAULT_CONFIG.num_hidden_layers
        self.wavlm_layer_weights = nn.Parameter(
            data=torch.ones(wavlm_num_layers)
        )

        self.lstm = nn.LSTM(wavlm_dim, **self.LSTM_DEFAULTS)
        lstm_out_features: int = self.LSTM_DEFAULTS["hidden_size"] * (
            2 if self.LSTM_DEFAULTS["bidirectional"] else 1
        )

        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.LINEAR_DEFAULTS["hidden_size"]]
                    * self.LINEAR_DEFAULTS["num_layers"]
                )
            ]
        )

        # Set the powerset
        self.powerset = Powerset(
            MAX_SPEAKERS_PER_CHUNK,
            MAX_SPEAKERS_PER_FRAME
        )
        self.dimension = self.powerset.num_powerset_classes

        self.classifier = nn.Linear(self.LINEAR_DEFAULTS["hidden_size"], self.dimension)
        self.activation = nn.LogSoftmax(dim=-1)

        # Post-processing
        self.postproc = self.powerset.to_multilabel

    @classmethod
    def from_pretrained(
        self,
        local_checkpoint_path: str,
        sliding_window_stride: int | None = None
    ) -> "SSeRiouSS":
        pyannote_segmentation_model = pyannote.audio.Model.from_pretrained(local_checkpoint_path)
        assert isinstance(pyannote_segmentation_model, PyannoteSSeRiouSS)

        argmax_model = SSeRiouSS(sliding_window_stride=sliding_window_stride or 5)

        # Load state dicts
        argmax_model.wavlm.load_state_dict(
            wavlm_torchaudio2hf_dict_adjustment(
                pyannote_segmentation_model.wav2vec.state_dict()
            )
        )

        argmax_model.wavlm_layer_weights.data = pyannote_segmentation_model.wav2vec_weights.data

        argmax_model.lstm.load_state_dict(
            pyannote_segmentation_model.lstm.state_dict()
        )

        argmax_model.linear.load_state_dict(
            pyannote_segmentation_model.linear.state_dict()
        )

        argmax_model.classifier.load_state_dict(
            pyannote_segmentation_model.classifier.state_dict()
        )

        # Precompute relative positional embeddings (This now becomes the maximum sequence length)
        argmax_model.wavlm.encoder.register_buffer(
            "relative_position_bias",
            pyannote_segmentation_model.wav2vec.encoder.transformer.layers[0].attention.compute_bias(
                TEST_SEQ_LEN, TEST_SEQ_LEN
            )
        )

        return argmax_model

    def get_powerset_probs(self, waveform_sliding_window: torch.Tensor, attention_mask=None):
        _, intermediate_outputs = self.wavlm(
            waveform_sliding_window,
            attention_mask=attention_mask
        )

        hidden_states = sum([
            w * h for w, h in
            zip(F.softmax(self.wavlm_layer_weights, dim=0), intermediate_outputs[1:])
        ])

        # LSTM accepts 3D tensors, so squeeze the 3rd dim
        hidden_states = hidden_states.squeeze(2).transpose(1, 2)
        outputs, _ = self.lstm(hidden_states)

        for linear in self.linear:
            outputs = F.leaky_relu(linear(outputs))

        powerset_probs = self.activation(self.classifier(outputs))

        return powerset_probs

    def forward(self, waveform: torch.Tensor, attention_mask=None):
        # Form sliding window from long waveform
        waveform_sliding_window = waveform.unfold(
            0, SLIDING_WINDOW_SIZE, self.sliding_window_stride)
        waveform_sliding_window = waveform_sliding_window.unsqueeze(1).unsqueeze(1)

        powerset_probs = self.get_powerset_probs(
            waveform_sliding_window,
            attention_mask=attention_mask
        )

        # Extract signals from raw model outputs
        speaker_probs = self.postproc(powerset_probs, soft=True)
        speaker_ids = self.postproc(powerset_probs)
        voice_activity = speaker_probs.max(dim=-1).values

        # Aggregate results across sliding windows
        num_windows = voice_activity.shape[0]
        output_num_frames = voice_activity.shape[1]  # e.g. 999 frames for 20s ~ 20.02 ms/frame
        frame_duration = TEST_MODEL_NATIVE_WINDOW_SIZE / output_num_frames

        # VAD
        sliding_window_stride_num_frames = (self.sliding_window_stride / 16_000) / frame_duration

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            total_output_frames = int(
                (num_windows - 1) * sliding_window_stride_num_frames + output_num_frames
            )

            _aggregated_voice_activity = torch.zeros(total_output_frames)
            aggregated_denom = torch.zeros(total_output_frames)
            logger.debug(
                f"{total_output_frames=} {num_windows=} "
                f"{output_num_frames=} {frame_duration=} {sliding_window_stride_num_frames=}")

            start = 0
            for i in range(num_windows):
                start = int(i * sliding_window_stride_num_frames)
                end = start + output_num_frames
                logger.debug(f"{start=} {end=}")
                _aggregated_voice_activity[start:end] += voice_activity[i]
                aggregated_denom[start:end] += 1.

            aggregated_voice_activity = _aggregated_voice_activity / aggregated_denom

            # Number of frames each speaker is active in each window
            # (used downstream for skipping speaker embedding extraction per speaker per window)
            speaker_activity = speaker_ids.sum(1)

            # Overlapped speaker activity map for masking during embedding extraction
            overlapped_speaker_activity = (speaker_ids.sum(2) > 1.).float()

            # squeeze the waveform_sliding_window to (batch, channel, sample)
            waveform_sliding_window = waveform_sliding_window.squeeze(1)

            return speaker_probs, speaker_ids, speaker_activity, overlapped_speaker_activity, \
                aggregated_voice_activity, waveform_sliding_window


def wavlm_torchaudio2hf_dict_adjustment(state_dict):

    # Split qkv proj (in_proj) and rename:
    # *attention.in_proj_{weigth/bias}  -> *attention.{q/k/v}_proj.{weigth/bias}
    keys_to_pop = []
    state_dict_update = {}
    for k in state_dict:
        if "in_proj" in k:
            keys_to_pop.append(k)
            for name, weight in zip(["q", "k", "v"], torch.chunk(state_dict[k], 3)):
                state_dict_update[k.replace("in_proj", f"{name}_proj")] = (
                    weight if "weight" in k else weight
                )

    [state_dict.pop(k) for k in keys_to_pop]
    state_dict.update(state_dict_update)

    state_dict = {
        k.replace("_proj_weight", "_proj.weight"): v for k, v in state_dict.items()
    }
    state_dict = {
        k.replace("_proj_bias", "_proj.bias"): v for k, v in state_dict.items()
    }

    # Rename feature projection:
    state_dict = {
        k.replace(
            "encoder.feature_projection",
            "preprocessor.feature_projection"
        ): v for k, v in state_dict.items()
    }

    # Rename encoder:
    state_dict = {
        k.replace("encoder.transformer", "encoder"): v for k, v in state_dict.items()
    }

    # Rename attention:
    state_dict = {
        k.replace(".attention.attention.", ".attention."): v for k, v in state_dict.items()
    }

    # Rename feature extractor:
    state_dict = {
        k.replace("feature_extractor.", "preprocessor.feature_extractor."): v for k, v in state_dict.items()
    }

    # Rename pos_conv_embed:
    state_dict = {
        k.replace(
            "encoder.pos_conv_embed.",
            "preprocessor.feature_projection.pos_conv_embed."
        ): v for k, v in state_dict.items()
    }

    return state_dict
