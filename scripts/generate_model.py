#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import argparse
import glob
import json
import os
import shutil
import subprocess
from pprint import pprint

from argmaxtools import _sdpa
from argmaxtools.utils import get_logger
from argmaxtools import test_utils
from huggingface_hub import HfApi, hf_hub_download

from tests import test_audio_encoder, test_text_decoder
from whisperkit._constants import COMPRESSION_REPO_ID, MODEL_REPO_ID

logger = get_logger(__name__)

test_utils.TEST_MIN_SPEEDUP_VS_CPU = 0.3


def cli():
    f""" Generates Whisper models and publishes them to hf.co/{MODEL_REPO_ID} """
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
        "--generate-quantized-variants",
        action="store_true",
        help="If specified, generates several variants of the model with varying bit precision"
    )
    parser.add_argument(
        "--generate-decoder-context-prefill-data",
        action="store_true",
        help="If specified, pre-computes the KV cache for the first 3 tokens of WhisperTextDecoder"
    )
    parser.add_argument(
        "--audio-encoder-sdpa-implementation",
        default="SplitHeadsQ",
        choices=tuple(_sdpa.__all__),
        help="Scaled Dot Product Attention (SDPA) implementation to use for WhisperAudioEncoder"
    )
    parser.add_argument(
        "--text-decoder-sdpa-implementation",
        default="Cat",
        choices=tuple(_sdpa.__all__),
        help="Scaled Dot Product Attention (SDPA) implementation to use for WhisperTextDecoder"
    )
    parser.add_argument(
        "--text-decoder-max-sequence-length",
        default=None,
        type=int,
        help="If specified, overrides the default max sequence length for WhisperTextDecoder"
    )
    parser.add_argument(
        "--repo-path-suffix",
        default=None,
        type=str,
        help=f"If specified, this string gets appended to the folder name on hf.co/{MODEL_REPO_ID}"
    )
    parser.add_argument(
        "--disable-default-tests",
        action="store_true",
        help="If specified, disables default tests for WhisperAudioEncoder and WhisperTextDecoder"
    )
    parser.add_argument(
        "--upload-results",
        action="store_true",
        help=f"If specified, uplaods the generated models to hf.co/{MODEL_REPO_ID}"
    )

    # Alias the CLI args to match the test scripts
    args = parser.parse_args()
    args.test_model_version = args.model_version
    args.palettizer_tests = args.generate_quantized_variants
    args.context_prefill_tests = args.generate_decoder_context_prefill_data
    args.persistent_cache_dir = os.path.join(
        args.output_dir, args.model_version.replace("/", "_")
    )
    if args.repo_path_suffix is not None:
        args.persistent_cache_dir += f"_{args.repo_path_suffix}"

    logger.info(f"Generating {args.model_version} files")

    # FIXME(atiorh): Remove this once distil-whisper-* models are updated
    args.disable_token_timestamps = False
    if "distil" in args.model_version and "distil-large-v3" not in args.model_version:
        logger.info(
            "Disabling token-level timestamps due to missing alignment_heads in distil-whisper-* models"
        )
        args.disable_token_timestamps = True

    # Generate WhisperTextDecoder
    args.test_seq_len = args.text_decoder_max_sequence_length
    args.sdpa_implementation = args.text_decoder_sdpa_implementation
    test_text_decoder.main(args)

    # Generate WhisperAudioEncoder
    args.sdpa_implementation = args.audio_encoder_sdpa_implementation
    args.melspectrogram_tests = True
    test_audio_encoder.main(args)

    # Rearrange quantized variants
    folders_to_upload = [args.persistent_cache_dir]
    if args.generate_quantized_variants:
        logger.info("Rearranging quantized variants")
        folders_to_upload.extend(rearrange_quantized_variants(args))

    if args.upload_results:
        # Upload generated models to model repository
        for folder_path in folders_to_upload:
            upload_version(folder_path, args.model_version)

        if args.generate_quantized_variants:
            upload_compression_artifacts(args.persistent_cache_dir, args.model_version)


def upload_version(local_folder_path, model_version):
    path_in_repo = local_folder_path.split("/")[-1]
    logger.info(f"Uploading results to hf.co/models/{MODEL_REPO_ID}/{path_in_repo}")

    # Dump required metadata before upload
    for filename in ["config.json", "generation_config.json"]:
        with open(hf_hub_download(repo_id=model_version,
                                  filename=filename), "r") as f:
            model_file = json.load(f)
        with open(os.path.join(local_folder_path, filename), "w") as f:
            json.dump(model_file, f)
        logger.info(f"Copied over {filename} from the original {model_version} repo")

    # Get whisperkittools commit hash
    wkt_commit_hash = subprocess.run(
        "git rev-parse HEAD",
        stdout=subprocess.PIPE,
        shell=True
    ).stdout.decode('utf-8').strip()

    # Upload model files
    api = HfApi()
    api.upload_folder(
        folder_path=local_folder_path,
        repo_id=MODEL_REPO_ID,
        path_in_repo=path_in_repo,
        repo_type="model",
        commit_message=f"whisperkittools-{wkt_commit_hash} generated files: {path_in_repo}",
        ignore_patterns="compression_artifacts/*"
    )


def upload_compression_artifacts(local_folder_path, model_version):
    # Get whisperkittools commit hash
    wkt_commit_hash = subprocess.run(
        "git rev-parse HEAD",
        stdout=subprocess.PIPE,
        shell=True
    ).stdout.decode('utf-8').strip()

    # Upload compression artifacts
    api = HfApi()
    compression_artifacts_dir = os.path.join(local_folder_path, "compression_artifacts")
    if os.path.exists(compression_artifacts_dir):
        api.upload_folder(
            folder_path=compression_artifacts_dir,
            repo_id=COMPRESSION_REPO_ID,
            path_in_repo="palettization",
            repo_type="model",
            commit_message=f"whisperkittools-{wkt_commit_hash} generated files: {model_version}",
            ignore_patterns="**/*.mlmodelc/*"
        )
    else:
        logger.warning(f"No compression artifacts found at {compression_artifacts_dir}")


def rearrange_quantized_variants(args):
    """ Move quantized variants from nested folders into publishable structure
    """
    audio_encoder_variants = sorted([
        (float(k.rsplit("_")[-1][:-13]), k) for k in
        glob.glob(os.path.join(
            args.persistent_cache_dir,
            "compression_artifacts", "AudioEncoder", args.model_version, "*mlmodelc"))
    ], key=lambda x: x[0])
    logger.info(f"Found {len(audio_encoder_variants)} quantized variants for AudioEncoder: ")
    pprint(audio_encoder_variants)

    if len(audio_encoder_variants) == 0:
        FileNotFoundError("No quantized variants found for AudioEncoder")
    elif len(audio_encoder_variants) > 1:
        audio_encoder_variants = [audio_encoder_variants[0], audio_encoder_variants[-1]]

    text_decoder_variants = sorted([
        (float(k.rsplit("_")[-1][:-13]), k) for k in
        glob.glob(os.path.join(
            args.persistent_cache_dir,
            "compression_artifacts", "TextDecoder", args.model_version, "*mlmodelc"))
    ], key=lambda x: x[0])
    logger.info(f"Found {len(text_decoder_variants)} quantized variants for TextDecoder: ")
    pprint(text_decoder_variants)

    if len(text_decoder_variants) == 0:
        raise FileNotFoundError("No quantized variants found for TextDecoder")
    elif len(text_decoder_variants) == 1:
        text_decoder_variants = [text_decoder_variants[0]]
    else:
        text_decoder_variants = [text_decoder_variants[0], text_decoder_variants[-1]]

    if len(text_decoder_variants) != len(audio_encoder_variants):
        raise ValueError(f"({len(text_decoder_variants)} != {len(audio_encoder_variants)})")

    quantized_asset_folders = []
    for aev, tdv in zip(audio_encoder_variants, text_decoder_variants):
        size_in_mb = get_total_size_in_mb(aev[1], tdv[1])
        # Name model variants with total size (instead of average number of bits per submodel)
        quantized_assets_folder = os.path.join(
            args.output_dir,
            args.persistent_cache_dir.rsplit("/")[-1] + f"_{int(size_in_mb)}MB"
        )
        os.makedirs(quantized_assets_folder, exist_ok=True)
        quantized_asset_folders.append(quantized_assets_folder)

        shutil.copytree(aev[1], os.path.join(quantized_assets_folder, "AudioEncoder.mlmodelc"))
        shutil.copytree(tdv[1], os.path.join(quantized_assets_folder, "TextDecoder.mlmodelc"))
        shutil.copytree(
            os.path.join(args.persistent_cache_dir, "MelSpectrogram.mlmodelc"),
            os.path.join(quantized_assets_folder, "MelSpectrogram.mlmodelc"))

        decoder_prefill_path = os.path.join(
            args.persistent_cache_dir, "TextDecoderContextPrefill.mlmodelc")
        if os.path.exists(decoder_prefill_path):
            shutil.copytree(
                decoder_prefill_path,
                os.path.join(quantized_assets_folder, "TextDecoderContextPrefill.mlmodelc"))
        logger.info(f"Rearranging variants: {aev[1]} & {tdv[1]}")

    return quantized_asset_folders


def get_total_size_in_mb(*dirs):
    assert all(os.path.exists(_dir) for _dir in dirs)
    return sum(get_dir_size(_dir) for _dir in dirs)


def get_dir_size(root_dir):
    size_in_mb = 0
    for parent, _, files in os.walk(root_dir):
        for f in files:
            path = os.path.join(parent, f)
            if not os.path.islink(path):
                size_in_mb += os.path.getsize(path)
    return size_in_mb / 1e6
