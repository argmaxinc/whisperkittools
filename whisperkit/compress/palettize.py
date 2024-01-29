#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#
import torch
import torch.nn.functional as F

from transformers import WhisperForConditionalGeneration
from typing import Dict, Tuple

from argmaxtools.compress.palettize import Palettizer
from argmaxtools.test_utils import compute_psnr
from whisperkit import (
    audio_encoder,
    text_decoder,
    test_utils as whisperkit_test_utils
)


torch.set_grad_enabled(False)

TEST_BATCH_SIZE = 8


class WhisperTextDecoderPalettizer(Palettizer):
    """ Palettizer for WhisperTextDecoder
    """
    default_dtype: torch.dtype = torch.float16

    def init_model_and_test_data(self,
                                 model_version: str,
                                 ) -> Tuple[
                                    text_decoder.WhisperTextDecoder,
                                    Dict[str, torch.Tensor]
                                ]:
        """ Initialize WhisperTextDecoder and a test mini-batch of random data
        """
        hf_model = WhisperForConditionalGeneration.from_pretrained(
            model_version,
            torch_dtype=self.default_dtype,
        ).model.decoder

        argmax_model = text_decoder.WhisperTextDecoder(
            hf_model.config).to(self.default_dtype).eval()
        argmax_model.load_state_dict(hf_model.state_dict())

        test_data = whisperkit_test_utils._prepare_test_inputs_for_decoder_from_cfg(
            TEST_BATCH_SIZE, hf_model.config
        )[0]

        return argmax_model, test_data

    def divergence_fn(self, reference: torch.Tensor, proxy: torch.Tensor) -> float:
        """ WhisperTextDecoder emits logits over a token vocabulary. The function
        used to quantify output change is KL divergence (lower the better)
        """
        div = F.kl_div(
            F.log_softmax(
                proxy.squeeze(1).cpu().to(torch.float64),
                dim=1,
                dtype=torch.float64,
            ),
            target=F.log_softmax(
                reference.squeeze(1).cpu().to(torch.float64),
                dim=1,
                dtype=torch.float64
            ),
            log_target=True,
            reduction="batchmean").item()

        if div < 0:
            raise ValueError(f"KL divergence is negative: {div}")
        return div

    def plot_specs(self, f, ax):
        ax.set_yscale("log")
        ax.set_xlabel("Model Size Reduction (%)")
        ax.set_ylabel("Output Divergence")
        ax.set_title(f"{self.model_version} TextDecoder Palettization Response Curves")
        ax.legend()


class WhisperAudioEncoderPalettizer(Palettizer):
    """ Palettizer for WhisperAudioEncoder
    """
    default_dtype: torch.dtype = torch.float16

    def init_model_and_test_data(self,
                                 model_version: str,
                                 ) -> Tuple[
                                    audio_encoder.WhisperAudioEncoder,
                                    Dict[str, torch.Tensor]
                                ]:
        """ Initialize WhisperAudioEncoder and obtain a test mini-batch of random data
        """
        hf_model = WhisperForConditionalGeneration.from_pretrained(
            model_version,
            torch_dtype=self.default_dtype,
        ).model.encoder

        argmax_model = audio_encoder.WhisperAudioEncoder(
            hf_model.config).to(self.default_dtype).eval()
        argmax_model.load_state_dict(hf_model.state_dict())

        test_data = whisperkit_test_utils._prepare_test_inputs_for_encoder_from_cfg(
            TEST_BATCH_SIZE, hf_model.config
        )[0]

        return argmax_model, test_data

    def divergence_fn(self, reference: torch.Tensor, proxy: torch.Tensor) -> float:
        """ WhisperEncoder emits embeddings. The function
        used to quantify output change is negative PSNR (lower the better)
        """
        return -compute_psnr(reference, proxy)

    def plot_specs(self, f, ax):
        ax.set_xlabel("Model Size Reduction (%)")
        ax.set_ylabel("Output Divergence")
        ax.set_title(f"{self.model_version} AudioEncoder Palettization Response Curves")
        ax.legend()
