# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.signal import get_window

from whisperkit.audio_encoder import WhisperMelSpectrogram as ReferenceWhisperMelSpectrogram
from whisperkit import tensor_typing as tt


def pad_center(data, *, size: int, axis: int = -1, **kwargs):
    """ Same function as `librosa.util.pad_center`
    """
    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(
            f"Target size ({size:d}) must be at least input size ({n:d})"
        )

    return np.pad(data, lengths, **kwargs)


class DecomposedSTFT(nn.Module):
    def __init__(self, filter_length=1024, hop_length=512, win_length=None, window='hann'):
        """ Decomposition of `torch.stft` using non-complex dtypes

        Reference: https://github.com/pseeth/torch-stft
        """
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length

        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        self.cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:self.cutoff, :]),
                                   np.imag(fourier_basis[:self.cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])

        assert(filter_length >= self.win_length)
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()
        forward_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())

    def forward(self, audio: tt.WhisperMelSpectrogramInputType) -> tt.WhisperMelSpectrogramOutputType:
        """ Apply STFT on input (audio) data
        """
        audio = audio.unsqueeze(0).unsqueeze(1)
        padded_audio = F.pad(
            audio,
            (self.pad_amount, self.pad_amount, 0, 0),
            mode='reflect'  # Padding consistent with librosa
        )

        forward_transform = F.conv1d(
            padded_audio,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        real_part = forward_transform[:, :self.cutoff, :]
        imag_part = forward_transform[:, self.cutoff:, :]
        return (real_part**2 + imag_part**2)


class WhisperMelSpectrogram(nn.Module):
    def __init__(self, n_mels=80, n_fft=400, hop_length=160):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer(
            "mel_filters",
            ReferenceWhisperMelSpectrogram.get_mel_filters(n_mels)
        )

        self.stft = DecomposedSTFT(
            filter_length=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.n_fft,
            window='hann'
        )

    def forward(self, audio: tt.WhisperMelSpectrogramInputType) -> tt.WhisperMelSpectrogramOutputType:
        
        transformed = self.stft(audio)
        magnitudes = transformed[..., :-1]
        mel_spec = self.mel_filters @ magnitudes
        mel_spec = mel_spec + 1e-10
        log_spec = mel_spec.log10()

        # Range normalization
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec


class WhisperDecoderPostProc(nn.Module):
    """ Logits post-processing
    """
    def forward(self, logits):
        TOKEN_TIMESTAMP_BEGIN = 50363
        TOKEN_NO_SPEECH = 50361
        
        # logprobs = F.log_softmax(logits, dim=0)
        logprobs = torch.log(F.softmax(logits, dim=0))
        timestamp_logprob = torch.logsumexp(logprobs[TOKEN_TIMESTAMP_BEGIN:], dim=0)
        max_text_token_logprob = torch.max(logprobs[:TOKEN_TIMESTAMP_BEGIN])

        # TODO(keith): Avoid slice assingment + overwrite
        logprobs[0] = timestamp_logprob
        logprobs[1] = max_text_token_logprob
        logprobs[2] = logprobs[TOKEN_NO_SPEECH]
        return logprobs[:3]


class EnergyVAD(nn.Module):
    def forward(self, input_frames, energy_threshold):
        """ Compute RMSE for audio waveform
        """
        square = torch.square(input_frames)
        mean = torch.mean(square, dim=1)
        return torch.sqrt(mean) - energy_threshold[0]
