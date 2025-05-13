#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import json
import os
import unittest

import torch
from argmaxtools import _sdpa, compress
from argmaxtools import test_utils as argmaxtools_test_utils
from argmaxtools.utils import get_fastest_device, get_logger
from huggingface_hub import hf_hub_download
from transformers.models.whisper import modeling_whisper

from whisperkit import audio_encoder, test_utils
from whisperkit.compress import palettize

torch.set_grad_enabled(False)
logger = get_logger(__name__)

TEST_WHISPER_VERSION = os.getenv("TEST_WHISPER_VERSION", None) or "openai/whisper-tiny"
TEST_CACHE_DIR = os.getenv("TEST_CACHE_DIR", None) or "/tmp"
TEST_DEV = os.getenv("TEST_DEV", None) or get_fastest_device()
TEST_TORCH_DTYPE = torch.float32
TEST_PSNR_THR = 35
argmaxtools_test_utils.TEST_PSNR_THR = TEST_PSNR_THR
TEST_FORCE_RECIPE_NBITS = True

argmaxtools_test_utils.TEST_MIN_SPEEDUP_VS_CPU = 0.95
argmaxtools_test_utils.TEST_SKIP_SPEED_TESTS = True

# WhisperMelSpectrogram constants
# TEST_N_MELS = [80, 128]
TEST_N_SAMPLES = 480000  # 16kHz * 30s


class TestWhisperAudioEncoder(
    argmaxtools_test_utils.CoreMLTestsMixin, unittest.TestCase
):
    """Unit tests for whisperkit.audio_encoder.WhisperAudioEncoder"""

    @classmethod
    def setUpClass(cls):
        cls.model_name = "AudioEncoder"
        cls.test_output_names = ["encoder_output_embeds"]
        cls.test_cache_dir = TEST_CACHE_DIR

        # Original model
        orig_torch_model = (
            modeling_whisper.WhisperForConditionalGeneration.from_pretrained(
                TEST_WHISPER_VERSION,
                torch_dtype=TEST_TORCH_DTYPE,
            )
        )

        cls.orig_torch_model = (
            orig_torch_model.model.encoder.to(TEST_DEV).to(TEST_TORCH_DTYPE).eval()
        )

        # Base test model
        cls.test_torch_model = audio_encoder.WhisperAudioEncoder(
            cls.orig_torch_model.config
        )
        cls.test_torch_model.load_state_dict(cls.orig_torch_model.state_dict())
        cls.test_torch_model = (
            cls.test_torch_model.to(TEST_DEV).to(TEST_TORCH_DTYPE).eval()
        )

        # Elaboration: I/O and architecture config
        cfg = cls.orig_torch_model.config
        cls.cfg = dict(
            embed_dim=cfg.d_model,
            batch_size=1,
            melspectrogram_seq_len=cfg.max_source_positions * 2,
            n_mels=cfg.num_mel_bins,
            num_layers=cfg.encoder_layers,
        )

        (
            cls.test_torch_inputs,
            cls.orig_torch_inputs,
        ) = test_utils._prepare_test_inputs_for_encoder(**cls.cfg)

        # Do casting and device placement per test config
        def place(t):
            return (
                t.to(TEST_DEV).to(TEST_TORCH_DTYPE)
                if t.dtype.is_floating_point
                else t.to(TEST_DEV)
            )

        cls.test_torch_inputs = {k: place(v) for k, v in cls.test_torch_inputs.items()}
        cls.orig_torch_inputs = {k: place(v) for k, v in cls.orig_torch_inputs.items()}

        cls.test_torch_output = cls.test_torch_model(**cls.test_torch_inputs).squeeze(2)
        cls.orig_torch_output = cls.orig_torch_model(**cls.orig_torch_inputs)[
            0
        ].transpose(1, 2)

        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        # Models
        cls.orig_torch_model = None
        cls.test_torch_model = None

        # Data
        cls.cfg = None
        cls.test_torch_inputs = None
        cls.orig_torch_inputs = None
        cls.test_torch_output = None
        cls.orig_torch_output = None

        super().tearDownClass()

    def test_torch2torch_correctness(self):
        """Coverage:
        - torch2torch parity transformers.models.whisper.modeling_whisper.WhisperEncoder
         and  argmax.ane.whisper.WhisperEncoder
        """
        with self.subTest(phase="torch_correctness_logits"):
            psnr = argmaxtools_test_utils.compute_psnr(
                self.orig_torch_output, self.test_torch_output
            )
            logger.info(f"torch2torch logits PSNR={psnr:.3g}")
            self.assertGreater(psnr, TEST_PSNR_THR)


class TestWhisperMelSpectrogram(
    argmaxtools_test_utils.CoreMLTestsMixin, unittest.TestCase
):
    """Unit tests for whisperkittools.audio_encoder.WhisperMelSpectrogram"""

    @classmethod
    def setUpClass(cls):
        # Try loading config from local path first
        config_path = os.path.join(TEST_WHISPER_VERSION, "config.json")
        if os.path.exists(config_path):
            logger.info(f"Loading config from local path: {config_path}")
            with open(config_path, "r") as f:
                n_mels = json.load(f)["num_mel_bins"]
        else:
            # Fall back to downloading from HF hub
            logger.info(f"Loading config from Hugging Face hub: {TEST_WHISPER_VERSION}")
            with open(
                hf_hub_download(repo_id=TEST_WHISPER_VERSION, filename="config.json"), "r"
            ) as f:
                n_mels = json.load(f)["num_mel_bins"]

        logger.info(
            f"WhisperMelSpectrogram: n_mels={n_mels} for {TEST_WHISPER_VERSION}"
        )
        cls.model_name = "MelSpectrogram"
        cls.test_output_names = ["melspectrogram_features"]
        cls.test_cache_dir = TEST_CACHE_DIR

        cls.test_torch_model = audio_encoder.WhisperMelSpectrogram(n_mels=n_mels)
        cls.test_torch_inputs = {
            "audio": torch.randn([TEST_N_SAMPLES], dtype=TEST_TORCH_DTYPE)
        }
        cls.test_torch_output = cls.test_torch_model(**cls.test_torch_inputs)

        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        # Models
        cls.test_torch_model = None

        # Data
        cls.test_torch_inputs = None
        cls.test_torch_output = None

        super().tearDownClass()


argmaxtools_test_utils.TEST_DONT_PALETTIZE_TOP_K = 0
argmaxtools_test_utils.TEST_ALLOWED_NBITS = [4]
compress.palettize.NUM_MIXED_BIT_RECIPES = 1
compress.palettize.TEST_BATCH_SIZE = 1
compress.palettize.INVERTED_RESULT_THR = 0.25
compress.palettize.SPARSE_OUTLIER_DECOMPOSITION = True
compress.sparse_outlier.OUTLIER_NUM_STD = 3.0
compress.palettize.PALETTIZATION_GROUP_SIZE = None


class TestWhisperAudioEncoderPalettizer(
    argmaxtools_test_utils.CoreMLPalettizerTestsMixin, unittest.TestCase
):
    """
    Unit tests for argmaxtools.ane.palettize_whisper.WhisperEncoderPalettizer

    Coverage:
    - Per-layer palettization
    - Cumulative palettization
    - Mixed-bit palettization
    - Core ML model compression and correctness

    """

    @classmethod
    def setUpClass(cls):
        cls.model_name = "AudioEncoder"
        cls.output_names = ["encoder_output_embeds"]
        cls.palettizer = palettize.WhisperAudioEncoderPalettizer(
            model_version=TEST_WHISPER_VERSION,
            cache_dir=os.path.join(
                TEST_CACHE_DIR, "compression_artifacts", "AudioEncoder"
            ),
        )

    @classmethod
    def tearDownClass(cls):
        cls.output_names = None
        cls.palettizer = None


def main(args):
    global TEST_WHISPER_VERSION, TEST_CACHE_DIR, TEST_KV_CACHE_OUTPUTS, CREATE_FAST_LOAD_ASSET

    # Quantization variables
    argmaxtools_test_utils.TEST_ALLOWED_NBITS = args.allowed_nbits
    logger.info(f"Allowed nbits: {argmaxtools_test_utils.TEST_ALLOWED_NBITS}")
    compress.palettize.SPARSE_OUTLIER_DECOMPOSITION = args.outlier_decomp
    logger.info(f"Outlier decomposition: {compress.palettize.SPARSE_OUTLIER_DECOMPOSITION}")
    compress.palettize.PALETTIZATION_GROUP_SIZE = args.palettization_group_size
    logger.info(f"Palettization group size: {compress.palettize.PALETTIZATION_GROUP_SIZE}")
    TEST_FORCE_RECIPE_NBITS = args.force_recipe_nbits
    logger.info(f"Force recipe nbits: {TEST_FORCE_RECIPE_NBITS}")

    if (
        3 in argmaxtools_test_utils.TEST_ALLOWED_NBITS
    ) and (
        argmaxtools_test_utils.TEST_MIN_DEPLOYMENT_TARGET < ct.target.macOS15
    ):
        logger.info(
            "3-bit palettization requires iOS18/macOS15 or later. "
            "Setting minimum deployment target to macOS15 and iOS18"
        )
        argmaxtools_test_utils.TEST_MIN_DEPLOYMENT_TARGET = ct.target.macOS15
    if (
        compress.palettize.PALETTIZATION_GROUP_SIZE is not None
    ) and (
        argmaxtools_test_utils.TEST_MIN_DEPLOYMENT_TARGET < ct.target.macOS15
    ):
        logger.info(
            "`per_grouped_channel` palettization requires iOS18/macOS15 or later. "
            "Setting minimum deployment target to macOS15 and iOS18"
        )
        argmaxtools_test_utils.TEST_MIN_DEPLOYMENT_TARGET = ct.target.macOS15

    TEST_WHISPER_VERSION = args.test_model_version
    logger.info(f"Testing {TEST_WHISPER_VERSION}")

    audio_encoder.SDPA_IMPL = getattr(_sdpa, args.sdpa_implementation)
    logger.info(f"Set SDPA implementation to: {audio_encoder.SDPA_IMPL}")

    with argmaxtools_test_utils._get_test_cache_dir(
        args.persistent_cache_dir
    ) as TEST_CACHE_DIR:
        suite = unittest.TestSuite()

        if not args.disable_default_tests:
            suite.addTest(TestWhisperAudioEncoder("test_torch2torch_correctness"))
            suite.addTest(
                TestWhisperAudioEncoder("test_torch2coreml_correctness_and_speedup")
            )
        else:
            logger.info("Skipped default tests")

        if args.melspectrogram_tests:
            suite.addTest(
                TestWhisperMelSpectrogram("test_torch2coreml_correctness_and_speedup")
            )

        if args.palettizer_tests:
            if TEST_FORCE_RECIPE_NBITS:
                logger.info(f"Forcing recipe nbits to {argmaxtools_test_utils.TEST_ALLOWED_NBITS}")
                suite.addTest(TestWhisperAudioEncoderPalettizer("test_create_recipe_with_forced_nbits"))
            else:
                suite.addTest(TestWhisperAudioEncoderPalettizer("test_profile_response"))
            suite.addTest(
                TestWhisperAudioEncoderPalettizer(
                    "test_palettized_torch2coreml_conversion_and_correctness"
                )
            )

        if os.getenv("DEBUG", False):
            suite.debug()
        else:
            runner = unittest.TextTestRunner()
            runner.run(suite)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent-cache-dir", default=None, type=str)
    parser.add_argument("--palettizer-tests", action="store_true")
    parser.add_argument("--disable-default-tests", action="store_true")
    parser.add_argument("--melspectrogram-tests", action="store_true")
    parser.add_argument(
        "--sdpa-implementation",
        default="Cat",
        choices=tuple(_sdpa.__all__)
    )
    parser.add_argument(
        "--test-model-version",
        default=TEST_WHISPER_VERSION,
    )
    args = parser.parse_args()

    main(args)
