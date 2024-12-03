#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import datetime
import json
import os
import pprint
import subprocess
import unittest

from argmaxtools import test_utils as argmaxtools_test_utils
from argmaxtools.utils import get_logger
from huggingface_hub import HfApi

from whisperkit._constants import EVALS_REPO_ID, MODEL_REPO_ID
from whisperkit.evaluate.datasets import EVAL_DATASETS
from whisperkit.evaluate.evaluate import evaluate
import whisperkit.evaluate.evaluate
from whisperkit.pipelines import get_pipeline_cls
from whisperkit.test_utils import BenchmarkContext

logger = get_logger(__name__)

# Test configuration
TEST_REFERENCE_RESULTS = os.getenv("TEST_REFERENCE_RESULTS", None) or None
TEST_PIPELINE = os.getenv("TEST_PIPELINE", None) or "WhisperKit"
TEST_DATASET_NAME = os.getenv("TEST_DATASET_NAME", None) or "librispeech"
TEST_NUM_SAMPLES = os.getenv("TEST_NUM_SAMPLES", None) or -1  # -1 = all
TEST_NUM_PROC = os.getenv("TEST_NUM_PROC", None) or 1
TEST_CACHE_DIR = os.getenv("TEST_CACHE_DIR", None) or "./external"
TEST_CODE_COMMIT_HASH = os.getenv("TEST_CODE_COMMIT_HASH", None) or None
TEST_MODEL_COMMIT_HASH = os.getenv("TEST_MODEL_COMMIT_HASH", None) or None
TEST_MODEL_VERSION = os.getenv("TEST_MODEL_VERSION", None) or \
    "openai/whisper-tiny.en"
TEST_UPLOAD_RESULTS = os.getenv("TEST_UPLOAD_RESULTS", None) or False
TEST_QOI_REFERENCE = os.getenv("TEST_QOI_REFERENCE", None) or None  # TODO
AVG_WER_SANITY_CHECK_THR = 0.5
LANGUAGE_SUBSET = None


class TestWhisperPipelineEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = get_pipeline_cls(TEST_PIPELINE)(
            whisper_version=TEST_MODEL_VERSION,
            out_dir=TEST_CACHE_DIR,
            code_commit_hash=TEST_CODE_COMMIT_HASH,
            model_commit_hash=TEST_MODEL_COMMIT_HASH,
        )

        cls.inference_context = BenchmarkContext(
            code_commit_hash=cls.pipeline.code_commit_hash,
            model_commit_hash=cls.pipeline.model_commit_hash,
        )

        # Get whisperkittools commit hash
        wkt_commit_hash = subprocess.run(
            "git rev-parse HEAD",
            stdout=subprocess.PIPE,
            shell=True
        ).stdout.decode('utf-8').strip()[:7]

        cls.results = {
            "results": evaluate(
                cls.pipeline,
                dataset_name=TEST_DATASET_NAME,
                num_samples=TEST_NUM_SAMPLES,
                cache_dir=TEST_CACHE_DIR,
                num_proc=TEST_NUM_PROC,
                language_subset=LANGUAGE_SUBSET),
            "metadata": {
                "num_samples": TEST_NUM_SAMPLES,
                "num_proc": TEST_NUM_PROC,
                "pipeline": TEST_PIPELINE,
                "dataset_name": TEST_DATASET_NAME,
                "model_version": TEST_MODEL_VERSION,
                "whisperkittools_commit_hash": wkt_commit_hash,
                "inference_context": cls.inference_context.spec_dict(),
                "model_repo_id": MODEL_REPO_ID
            }
        }

        # Sanity check results
        sample_results = cls.results["results"]
        average_wer = sum([sample["wer"] for sample in sample_results]) / len(sample_results)
        if average_wer > AVG_WER_SANITY_CHECK_THR:
            raise ValueError(
                "Average WER failed sanity check: "
                f"{average_wer} > {AVG_WER_SANITY_CHECK_THR}")

        out_path = os.path.join(TEST_CACHE_DIR, "results.json")
        with open(out_path, "w") as f:
            json.dump(cls.results, f, indent=2)
        logger.info(f"Saved results to {out_path}")

        if TEST_UPLOAD_RESULTS:
            results_dir = os.path.join(
                TEST_PIPELINE,
                TEST_MODEL_VERSION.replace("/", "_"),
                TEST_DATASET_NAME,
                "forced" if whisperkit.evaluate.evaluate.FORCE_LANGUAGE else "",
                LANGUAGE_SUBSET if LANGUAGE_SUBSET else ""
            )
            results_fname = datetime.datetime.now().astimezone(
                ).strftime("%Y-%m-%d_%H:%M:%S_GMT%z") + ".json"

            api = HfApi()
            logger.info(f"Uploading results to hf.co/datasets/{EVALS_REPO_ID}")
            api.upload_file(
                path_or_fileobj=out_path,
                path_in_repo=os.path.join(results_dir, results_fname),
                repo_id=EVALS_REPO_ID,
                repo_type="dataset",
                commit_message=f"whisperkittools {wkt_commit_hash}: "
                               f"Eval {TEST_MODEL_VERSION} on {TEST_DATASET_NAME}",
            )
        else:
            logger.info(f"Skipped uploading results to hf.co/datasets/{EVALS_REPO_ID} ")
            # pprint.pprint(cls.results)

    def test_evaluate(self):
        pass


def main(args):
    global TEST_DATASET_NAME, TEST_PIPELINE, TEST_NUM_SAMPLES, TEST_CACHE_DIR, \
           TEST_MODEL_VERSION, TEST_CODE_COMMIT_HASH, TEST_MODEL_COMMIT_HASH, \
           TEST_NUM_PROC, TEST_UPLOAD_RESULTS, TEST_QOI_REFERENCE, LANGUAGE_SUBSET
    TEST_DATASET_NAME = args.dataset
    TEST_PIPELINE = args.pipeline
    TEST_NUM_SAMPLES = args.num_samples
    TEST_CACHE_DIR = args.persistent_cache_dir
    TEST_MODEL_VERSION = args.model_version
    TEST_CODE_COMMIT_HASH = args.code_commit_hash
    TEST_MODEL_COMMIT_HASH = args.model_commit_hash
    TEST_NUM_PROC = args.num_proc
    TEST_UPLOAD_RESULTS = args.upload_results
    LANGUAGE_SUBSET = args.language_subset

    # Force language option
    whisperkit.evaluate.evaluate.FORCE_LANGUAGE = args.force_language

    with argmaxtools_test_utils._get_test_cache_dir(
        args.persistent_cache_dir
    ) as TEST_CACHE_DIR:
        suite = unittest.TestSuite()
        suite.addTest(TestWhisperPipelineEvaluate("test_evaluate"))

    if os.getenv("DEBUG", False):
        suite.debug()
    else:
        runner = unittest.TextTestRunner()
        runner.run(suite)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=tuple(EVAL_DATASETS)
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=("WhisperKit", "whisper.cpp", "WhisperMLX", "WhisperOpenAIAPI"),
        required=True
    )
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--persistent-cache-dir", type=str, default="./external")
    parser.add_argument("--model-version", type=str, default="openai/whisper-tiny")
    parser.add_argument("--code-commit-hash", type=str, default=None)
    parser.add_argument("--model-commit-hash", type=str, default=None)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--upload-results", action="store_true")
    args = parser.parse_args()

    main(args)
