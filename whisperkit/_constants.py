#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#
import os

WHISPERKIT_REPO_OWNER = "argmaxinc"
WHISPERKIT_REPO_NAME = "WhisperKit"
COMPRESSION_REPO_ID = "argmaxinc/compression_artifacts"
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", None) or "argmaxinc/whisperkit-coreml"
EVALS_REPO_ID = os.getenv("EVALS_REPO_ID", None) or "argmaxinc/whisperkit-evals"

# Override these to register your own dataset
DATASET_REPO_OWNER = os.getenv("DATASET_REPO_OWNER", None) or "argmaxinc"
EVAL_DATASETS = [
    "earnings22", "librispeech", "librispeech-200",
    "earnings22-debug", "librispeech-debug",
    "earnings22-12hours"
]
CUSTOM_EVAL_DATASET = os.getenv("EVAL_DATASET", None)
if CUSTOM_EVAL_DATASET is not None:
    EVAL_DATASETS.append(CUSTOM_EVAL_DATASET)

# Tests
OPENAI_API_MODEL_VERSION = "openai/whisper-large-v2"
OPENAI_API_MAX_FILE_SIZE = 25e6  # bytes
OPENAI_API_COMPRESSED_UPLOAD_BIT_RATE = "50k"  # kbps
TEST_DATA_REPO = "argmaxinc/whisperkit-test-data"
