#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import json
import os
from pathlib import Path

from argmaxtools.utils import get_logger
from huggingface_hub import snapshot_download

from whisperkit._constants import DATASET_REPO_OWNER, EVAL_DATASETS, SUPPORTED_LANGUAGES
from whisperkit.evaluate.normalize_en import EnglishTextNormalizer

logger = get_logger(__name__)

text_normalizer = EnglishTextNormalizer()


def get_dataset(dataset_name, cache_dir, max_num_samples=-1, language_subset=None):
    if dataset_name not in EVAL_DATASETS:
        raise ValueError(f"Dataset not yet registered: {dataset_name}")

    if language_subset is not None:
        assert language_subset in SUPPORTED_LANGUAGES, f"Unsupported language: {language_subset}"
        logger.info(f"Filtering dataset for language: {language_subset}")

    logger.info(f"""\n
        =======================================================
        Downloading and preprocessing '{dataset_name}' dataset
        (if not cached)
        =======================================================
        """)

    cache_dir = os.path.join(cache_dir, "datasets", dataset_name)
    os.makedirs(Path(cache_dir).parent, exist_ok=True)

    if not os.path.exists(cache_dir):
        snapshot_download(
            repo_id=f"{DATASET_REPO_OWNER}/{dataset_name}",
            repo_type="dataset",
            allow_patterns="*",
            local_dir=cache_dir,
            local_dir_use_symlinks=True
        )

        # Unzip if necessary
        zip_files = [f for f in os.listdir(cache_dir) if f.endswith('.zip')]
        if len(zip_files) > 0:
            logger.info(f"Unzipping {len(zip_files)} files")
            for zip_file in zip_files:
                zip_path = os.path.join(cache_dir, zip_file)
                os.system(f"unzip -q -o {zip_path} -d {cache_dir}")
                os.remove(zip_path)

    has_folders = False
    for path in os.listdir(cache_dir):
        if os.path.isdir(os.path.join(cache_dir, path)) and not path.startswith("."):
            has_folders = True
            break

    audio_paths = _get_audio_paths(cache_dir)
    audio_paths = {path.split("/")[-1]: path for path in audio_paths}

    metadata_path = os.path.join(cache_dir, "metadata.json")
    assert os.path.exists(metadata_path), \
        f"Missing metadata file: {metadata_path}"

    with open(metadata_path, "r") as f:
        dataset = json.load(f)

    def preprocess_fn(batch):
        # Match potentially nested paths
        possible_keys = ["path", "audio", "audio_path"]
        for key in possible_keys:
            if key in batch:
                break
        current_path = batch[key]
        current_fname = current_path.rsplit("/")[-1]
        try:
            current_path = audio_paths[current_fname]
        except KeyError:
            current_path = audio_paths[
                current_fname.split(".")[0] + ".wav"]
        batch["norm_path"] = current_path

        # Normalize text
        possible_keys = [
            "text", "sentence", "normalized_text", "transcript", "transcription"]
        for key in possible_keys:
            if key in batch:
                break
        batch["original_text"] = batch[key]
        if not isinstance(batch[key], str):
            logger.warning(f"non-string text dectected: {batch[key]} | Class: {type(batch[key])}")
            logger.warning(f"Conversion to string: {str(batch[key])}")
        batch["norm_text"] = text_normalizer(str(batch[key]))

        # Remove invalid samples
        drop = batch["norm_text"].strip() == "ignore time segment in scoring"
        drop = drop or batch["norm_text"].strip() == ""

        # Filter by language
        if language_subset is not None:
            drop = drop or batch.get("language", None) != language_subset

        if drop:
            return None
        if has_folders:
            batch["norm_folder"] = "/".join(batch["norm_path"].split("/")[:-1])
        return batch

    original_num_samples = len(dataset)
    if max_num_samples > 0:
        dataset = dataset[:max_num_samples]

    dataset = list(filter(lambda x: x is not None, map(preprocess_fn, dataset)))
    logger.info(f"Running evaluation on {len(dataset)}/{original_num_samples} samples")

    return dataset


def _get_audio_paths(source_dir):
    AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3"]
    def filter_audio_ext(f): return os.path.splitext(f)[1] in AUDIO_EXTENSIONS

    audio_files = []
    count = 0
    for parent, _, files in os.walk(source_dir):
        def map_join(p): return os.path.join(parent, p)
        matches = list(map(map_join, filter(filter_audio_ext, files)))

        count += len(matches)
        audio_files.extend(matches)
        assert count == len(set(audio_files)), "Duplicate audio file names found"

    logger.info(f"Found {count} audio files in {source_dir} directory tree")
    return audio_files
