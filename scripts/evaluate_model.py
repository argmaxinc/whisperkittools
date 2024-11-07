#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import argparse
import glob
import os

from argmaxtools.utils import get_logger
from huggingface_hub import snapshot_download

from tests import test_evaluate
from whisperkit._constants import EVAL_DATASETS, EVALS_REPO_ID, MODEL_REPO_ID

logger = get_logger(__name__)


def cli():
    f""" Evaluates models from {MODEL_REPO_ID} on benchmark datasets
    and publishes results to hf.co/datasets/{EVALS_REPO_ID}
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local directory to save the generated model files and other test artifacts"
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help="Whisper model version string that matches Hugging Face model hub name, "
             "e.g. openai/whisper-tiny.en",
    )
    parser.add_argument(
        "--evaluation-dataset",
        default=[],
        action="append",
        required=True,
        choices=EVAL_DATASETS,
        help="Dataset to use for evaluation, can be specified multiple times. "
             "Use `*-debug` datasets for quick local testing."
    )
    parser.add_argument(
        "--code-commit-hash",
        type=str,
        default=None,
        help="https://github.com/argmaxinc/WhisperKit code commit hash. "
             "Uses main branch if not specified")
    parser.add_argument(
        "--model-commit-hash",
        type=str,
        default=None,
        help=f"hf.co/{MODEL_REPO_ID} commit hash. "
             "Uses main branch if not specified")
    # TODO (atiorh): Remove after Swift async batching is implemented
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of async processes to launch in parallel (to saturate compute resources)"
    )
    parser.add_argument(
        "--upload-results",
        action="store_true",
        help="If specified, runs on full dataset and uploads the evaluation results to "
             f"hf.co/datasets/{EVALS_REPO_ID}"
    )
    parser.add_argument(
        "--evaluate-all-variants",
        action="store_true",
        help="If specified, evaluates all variants of the model with matching"
             "model version prefix, e.g. openai/whisper-large-v3_*"
    )

    # Alias the CLI args to match the test scripts
    args = parser.parse_args()

    model_versions = [args.model_version]
    persistent_cache_dirs = [
        os.path.join(args.output_dir, args.model_version.replace("/", "_"))]

    if args.evaluate_all_variants:
        glob_pattern = args.model_version.replace("/", "_") + "*"
        _ = snapshot_download(
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            local_dir=args.output_dir,
            allow_patterns=glob_pattern
        )

        persistent_cache_dirs = glob.glob(os.path.join(args.output_dir, glob_pattern))
        model_versions = [p.rsplit("/")[-1] for p in persistent_cache_dirs]

    logger.info(f"Evaluating the following models: {model_versions}")

    for model_version, persistent_cache_dir in zip(model_versions, persistent_cache_dirs):
        args.persistent_cache_dir = persistent_cache_dir
        args.test_model_version = model_version
        # Evaluate Whisper on benchmark tests
        for dataset in args.evaluation_dataset:
            logger.info(f"Evaluating {model_version} on {dataset}")
            args.dataset = dataset
            args.pipeline = "WhisperKit"
            args.num_samples = -1
            test_evaluate.main(args)
