#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.
#

import coremltools as ct
import os
import unittest

import torch

from argmaxtools import test_utils as argmaxtools_test_utils
from argmaxtools.utils import get_logger

from whisperkit.diarization.diarization_utils import register_torch_ops_for_speaker_segmenter
from whisperkit.diarization.sseriouss_speaker_segmenter import SSeRiouSS as ArgmaxSSeRiouSS

torch.set_grad_enabled(False)
logger = get_logger(__name__)

# Register the torch ops for the speaker segmenter
register_torch_ops_for_speaker_segmenter()

TEST_CHECKPOINT_PATH = "pytorch_model.bin"
TEST_SLIDING_WINDOW_STRIDE = 5  # s
TEST_CACHE_DIR = os.getenv("TEST_CACHE_DIR", None) or "/tmp"
TEST_TORCH_DTYPE = torch.float32
TEST_PSNR_THR = 35

# Quantization bits for Core ML
argmaxtools_test_utils.TEST_DEFAULT_NBITS = None

# Core ML precision
argmaxtools_test_utils.TEST_COREML_PRECISION = ct.precision.FLOAT16

# Audio input spec
TEST_SAMPLE_RATE = 16_000
TEST_MAX_INPUT_SECS = 30
TEST_MIN_INPUT_SECS = 10

TEST_SEQ_LEN = 999  # 20s of audio at 16kHz yields 999 seqlen after the Convolutional preprocessor

# Note: SpeakerSegmenter models are optimized for CPU inference at the moment
argmaxtools_test_utils.TEST_MIN_SPEEDUP_VS_CPU = -1
argmaxtools_test_utils.TEST_SKIP_SPEED_TESTS = True
argmaxtools_test_utils.TEST_COMPUTE_UNIT = ct.ComputeUnit.CPU_AND_NE
argmaxtools_test_utils.TEST_MIN_DEPLOYMENT_TARGET = ct.target.macOS14
argmaxtools_test_utils.TEST_COMPILE_COREML = True
argmaxtools_test_utils.TEST_DEFAULT_NBITS = 8


class TestSpeakerSegmenter(argmaxtools_test_utils.CoreMLTestsMixin, unittest.TestCase):
    """Unit tests for SSeRiouSS speaker segmenter
    - Test the torch2torch correctness of the Argmax SSeRiouSS model
    - Test the torch2coreml correctness and speedup of the Argmax SSeRiouSS model
    """

    @classmethod
    def setUpClass(cls):
        cls.model_name = "SpeakerSegmenter"
        cls.test_output_names = [
            "speaker_probs",
            "speaker_ids",
            "speaker_activity",
            "overlapped_speaker_activity",
            "voice_activity",
            "sliding_window_waveform"
            ]
        cls.test_cache_dir = TEST_CACHE_DIR

        # Load the Argmax SSeRiouSS model
        cls.test_torch_model = ArgmaxSSeRiouSS.from_pretrained(
            TEST_CHECKPOINT_PATH,
            int(TEST_SLIDING_WINDOW_STRIDE * TEST_SAMPLE_RATE)
        )

        input_num_frames = TEST_MAX_INPUT_SECS * TEST_SAMPLE_RATE
        cls.test_torch_model.eval()
        cls.test_torch_inputs = dict(
            waveform=torch.randn(input_num_frames).to(TEST_TORCH_DTYPE)
        )

        # Cache the outputs of Argmax model
        cls.test_torch_outputs = cls.test_torch_model(**cls.test_torch_inputs)

        super().setUpClass()


def main(args):
    global TEST_CHECKPOINT_PATH, TEST_CACHE_DIR, TEST_SLIDING_WINDOW_STRIDE

    TEST_CHECKPOINT_PATH = args.local_checkpoint_path
    TEST_SLIDING_WINDOW_STRIDE = args.test_sliding_window_stride
    logger.info(f"Testing {TEST_CHECKPOINT_PATH}")
    logger.info(f"Sliding window stride: {TEST_SLIDING_WINDOW_STRIDE}s")

    with argmaxtools_test_utils._get_test_cache_dir(
        args.persistent_cache_dir
    ) as TEST_CACHE_DIR:
        logger.info(f"Using cache dir: {TEST_CACHE_DIR}")
        suite = unittest.TestSuite()
        suite.addTest(TestSpeakerSegmenter("test_torch2coreml_correctness_and_speedup"))

        if os.getenv("DEBUG", False):
            suite.debug()
        else:
            runner = unittest.TextTestRunner()
            runner.run(suite)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent-cache-dir", default=None, type=str)
    parser.add_argument(
        "--local-checkpoint-path",
        default=TEST_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--test-sliding-window-stride",
        default=TEST_SLIDING_WINDOW_STRIDE,
        type=int,
        help="Sliding window stride in seconds",
    )
    args = parser.parse_args()

    main(args)
