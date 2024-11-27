#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import os
import torch
import unittest

from argmaxtools import test_utils as argmaxtools_test_utils
from argmaxtools.utils import get_logger

from whisperkit.android import models as android
from whisperkit.android import utils as aihub_utils
from whisperkit import audio_encoder as apple

from tests.test_audio_encoder import TEST_N_SAMPLES

TEST_VOCAB_SIZE = 51865
TEST_PSNR_THR = 40
TEST_CACHE_DIR = "/tmp"

logger = get_logger(__name__)


# TODO(atila): Refactor this testing pattern into `argmaxtools.test_utils`
class TestAIHubModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {
            # "melspectrogram": {
            #     "reference": apple.WhisperMelSpectrogram(),
            #     "test": android.WhisperMelSpectrogram(),
            #     "inputs": {"audio": torch.randn(TEST_N_SAMPLES)}
            # },
            "decoder_postproc": {
                "reference": None,
                "test": android.WhisperDecoderPostProc(),
                "inputs": {"logits": torch.randn(TEST_VOCAB_SIZE)},
            },
            "energy_vad": {
                "reference": None,
                "test": android.EnergyVAD(),
                "inputs": {
                    "input_frames": torch.randn(150, 1600),  # 15 second batch of 100ms segments at 16kHz
                    "energy_threshold": torch.randn(1),
                }
            }
        }
        super().setUpClass()
    
    @classmethod
    def tearDownClass(cls):
        cls.models = None
        super().tearDownClass()


    def test_torch2torch_correctness(self):
        """ Test forward pass functionality and correctness of PyTorch models
        """
        for model_key, model_value in self.models.items():
            with self.subTest(phase=model_key):
                test_out = model_value["test"](**model_value["inputs"])
                if model_value["reference"] is not None:
                    psnr = argmaxtools_test_utils.compute_psnr(
                        model_value["reference"](**model_value["inputs"]),
                        test_out
                    )
                    self.assertGreater(psnr, TEST_PSNR_THR)
                    logger.info(f"torch2torch model={model_key} PSNR={psnr:.3g}")
                else:
                    logger.info(
                        f"torch2torch correctness test skipped: Reference model does not exist for {model_key}")


    def test_torch2aihub_performance_and_correctness(self):
        """ Test AI Hub compilation and inference job results against local PyTorch test results
        """
        for model_key, model_value in self.models.items():
            results = aihub_utils.convert_via_aihub(
                model_key,
                model_value["test"],
                model_value["inputs"],
                target_runtime="tflite",
                benchmark_chipset="gen1"
            )

            logger.info(f"Results for {model_key}")
            print(aihub_utils.summarize_performance(results["profile"])[1])

            torch_local_output = model_value["test"](**model_value["inputs"])
            remote_device_output = torch.from_numpy(list(results["inference"].values())[0][0])
            psnr = argmaxtools_test_utils.compute_psnr(torch_local_output, remote_device_output)

            logger.info(f"torch2tflite PSNR={psnr:.3g}")
            self.assertGreater(psnr, TEST_PSNR_THR)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent-cache-dir", default=None, type=str)
    parser.add_argument(
        "--skip-qai-hub-submission",
        action="store_true",
        help="If set, skips the download the compiled model and exit.",
    )
    parser.add_argument(
        "--model-version",
        choices=aihub_utils.ANDROID_SUPPORTED_WHISPER_VERSIONS,
        default="tiny",
    )
    # parser.add_argument(
    #     "-t",
    #     "--target-runtime",
    #     choices=tuple(TARGET_TABLE),
    #     default="tflite",
    #     type=str,
    # )
    # parser.add_argument(
    #     "-c",
    #     "--chipset",
    #     help="888, gen1, gen2, gen3",
    #     default="gen2",
    #     type=str,
    # )
    args = parser.parse_args()

    with argmaxtools_test_utils._get_test_cache_dir(
        args.persistent_cache_dir
    ) as TEST_CACHE_DIR:
        suite = unittest.TestSuite()
        suite.addTest(TestAIHubModels("test_torch2torch_correctness"))
        if not args.skip_qai_hub_submission:
            suite.addTest(TestAIHubModels("test_torch2aihub_performance_and_correctness"))

        if os.getenv("DEBUG", False):
            suite.debug()
        else:
            runner = unittest.TextTestRunner()
            runner.run(suite)
