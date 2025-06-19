#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#
import os

WHISPERKIT_REPO_OWNER = "argmaxinc"
WHISPERKIT_REPO_NAME = "WhisperKit"
COMPRESSION_REPO_ID = "argmaxinc/compression_artifacts"
_DEFAULT_MODEL_REPO_ID = "argmaxinc/whisperkit-coreml"
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", None) or "argmaxinc/whisperkit-coreml"
EVALS_REPO_ID = os.getenv("EVALS_REPO_ID", None) or "argmaxinc/whisperkit-evals"

# Override these to register your own dataset
DATASET_REPO_OWNER = os.getenv("DATASET_REPO_OWNER", None) or "argmaxinc"
EVAL_DATASETS = [
    "earnings22", "librispeech", "librispeech-200",
    "earnings22-debug", "librispeech-debug",
    "earnings22-12hours",
    "common_voice_17_0-debug-zip",
    "common_voice_17_0-argmax_subset-400"
]

IS_LOCAL_DATASET = False
IS_LOCAL_MODEL = False
CUSTOM_EVAL_DATASET = os.getenv("EVAL_DATASET", None)
if CUSTOM_EVAL_DATASET is not None:
    EVAL_DATASETS.append(CUSTOM_EVAL_DATASET)

# Tests
OPENAI_API_MODEL_VERSION = "openai_whisper-large-v2"
OPENAI_API_MAX_FILE_SIZE = 25e6  # bytes
OPENAI_API_COMPRESSED_UPLOAD_BIT_RATE = "12k"  # kbps
TEST_DATA_REPO = "argmaxinc/whisperkit-test-data"

# Supported Languages
SUPPORTED_LANGUAGES = [
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "yue",
    "zh",
]
