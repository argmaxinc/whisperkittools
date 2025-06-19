#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#
import time
from functools import partial
from multiprocessing import Pool
from typing import Union, Optional
from difflib import Differ

import evaluate
import tqdm
from argmaxtools.utils import get_logger

from whisperkit import pipelines
from whisperkit.evaluate.datasets import get_dataset
from whisperkit.evaluate.normalize_en import (
    EnglishTextNormalizer,
    BasicTextNormalizer,
    Cyrillic2LatinTextNormalizer
)

logger = get_logger(__name__)


"""All languages use the default BasicTextNormalizer except for the denoted languages below.
- EnglishTextNormalizer:
    Used for English and unlabeled languages (for consistency with the previous evaluation script)
- BasicTextNormalizer(split_letters=True):
    Used for Chinese, Japanese, Thai, Lao, Burmese, and Cantonese
- Cyrillic2LatinTextNormalizer:
    Some languages require special normalization steps, such as converting Cyrillic to Latin, when
    the reference text and predicted text use different scripts.
"""
TEXT_NORMALIZER = {
    (None, "en"): EnglishTextNormalizer(),
    ("zh", "ja", "th", "lo", "my", "yue"): BasicTextNormalizer(split_letters=True),
    ("ba", "sr", "tk"): Cyrillic2LatinTextNormalizer(),
}
TEXT_NORMALIZER = {
    lang: normalizer
    for langs, normalizer in TEXT_NORMALIZER.items()
    for lang in langs
}

basic_text_normalizer = BasicTextNormalizer()
text_normalizer = EnglishTextNormalizer()

wer_metric = evaluate.load("argmaxinc/detailed-wer")


# Force the pipeline to use the language in the dataset (increases the number of fallbacks)
FORCE_LANGUAGE = False

# Prompt to use for the pipeline
PROMPT = None

# Unsupported languages except for large-v3
UNSUPPORTED_LANGUAGES = ["yue"]


def evaluate(whisper_pipeline: Union[pipelines.WhisperPipeline, pipelines.WhisperOpenAIAPI],
             dataset_name: str,
             num_samples: int,
             cache_dir: str,
             num_proc: int,
             language_subset: Optional[str] = None,
             ) -> None:
    """ Evaluate the given whisper pipeline implementation on a registered dataset.
    """
    dataset = get_dataset(
        dataset_name,
        cache_dir,
        max_num_samples=num_samples,
        language_subset=language_subset,
    )

    use_folders = "norm_folder" in dataset[0]
    if use_folders:
        for sample in dataset:
            if "norm_folder" not in sample:
                use_folders = False
                break
        logger.info("Folder structure detected in dataset. Transcribing each folder separately.")

    if use_folders:
        folder_dataset = {}
        for sample in dataset:
            folder = sample["norm_folder"]
            if folder not in folder_dataset:
                folder_dataset[folder] = []
            folder_dataset[folder].append(sample)
        folder_dataset = [
            {
                "norm_folder": folder, "samples": samples
            } for folder, samples in folder_dataset.items()
        ]

    begin = time.time()

    # Warm-up (model compilation without multi-processing)
    logger.info("Warmup the pipeline on a single sample (may trigger model download)")
    _ = evaluate_sample(dataset[0], whisper_pipeline)

    if num_proc > len(dataset):
        logger.info(
            f"--num-proc ({num_proc}) is larger than the dataset size "
            f"({len(dataset)}), setting it to the dataset size")
        num_proc = len(dataset)

    if use_folders:
        if num_proc > 1:
            if isinstance(whisper_pipeline, pipelines.WhisperKit):
                whisper_pipeline._randomize_dispatch = True
                logger.info("Randomizing dispatch for WhisperKit for num_proc>1")

            logger.info(f"Launching {num_proc} processes to run {whisper_pipeline.__class__.__name__}")
            with Pool(num_proc) as pool:
                results = list(tqdm.tqdm(pool.map(
                    partial(evaluate_folder, whisper_pipeline=whisper_pipeline), folder_dataset),
                    total=len(folder_dataset)
                ))

                # Flatten list of lists to a single list
                results = [
                    result
                    for lang_results in results
                    for result in lang_results
                ]
        else:
            results = []
            eval_folder = partial(evaluate_folder, whisper_pipeline=whisper_pipeline)
            for folder in folder_dataset:
                results.extend(eval_folder(folder))
    else:
        if num_proc > 1:
            if isinstance(whisper_pipeline, pipelines.WhisperKit):
                whisper_pipeline._randomize_dispatch = True
                logger.info("Randomizing dispatch for WhisperKit for num_proc>1")

            logger.info(f"Launching {num_proc} processes to run {whisper_pipeline.__class__.__name__}")
            with Pool(num_proc) as pool:
                results = list(tqdm.tqdm(pool.map(
                    partial(evaluate_sample, whisper_pipeline=whisper_pipeline), dataset),
                    total=len(dataset)
                ))
        else:
            results = []
            eval_sample = partial(evaluate_sample, whisper_pipeline=whisper_pipeline)
            for sample in tqdm.tqdm(dataset):
                results.append(eval_sample(sample))

    total_elapsed = time.time() - begin

    avg_wer_result = wer_metric.compute(
        references=[result["reference"] for result in results],
        predictions=[result["prediction"] for result in results],
        detailed=True,
    )

    # Get average WER and its breakdown
    keys = ["wer", "substitution_rate", "deletion_rate", "insertion_rate"]
    avg_wer = {k: round(100 * avg_wer_result[k], 2) for k in keys}

    tot_audio_duration = sum([result["audio_duration"] for result in results])
    tot_prediction_duration = sum([result["prediction_duration"] for result in results])

    sample_average_rtf = tot_prediction_duration / tot_audio_duration
    global_rtf = total_elapsed / tot_audio_duration

    speed_factor = 1 / sample_average_rtf  # RTFx = (1/RTF) is the speed factor

    # Fallback bookkeeping
    num_fallbacks = list(filter(
        lambda x: x is not None,
        [result["num_fallbacks"] for result in results]
    ))
    fallback_str = ""
    if len(num_fallbacks) > 0:
        total_fallbacks = sum(num_fallbacks)
        samples_with_fallback_percent = sum([
            int(bool(_num_fallbacks)) for _num_fallbacks in num_fallbacks
        ]) / len(num_fallbacks)
        fallback_str = "-------------------------------------------------------"
        fallback_str += f"\n    Total fallbacks: {total_fallbacks}"
        fallback_str += "\n    Samples with fallback: "
        fallback_str += f"{samples_with_fallback_percent * 100.:.3g}%"

    # Failed example bookkeeping
    num_failed = len(list(filter(
        lambda x: "failed" in x,
        results,
    )))

    logger.info(f"""\n\n
    =======================================================
    Evaluation Results
    =======================================================

    Pipeline:\t{whisper_pipeline.__class__.__name__}
    Precision:\t{whisper_pipeline.precision.value}
    Dataset:\t{dataset_name} {'(num_samples=' + str(num_samples) + ')' if num_samples > 1 else ''}
    Model:\t{whisper_pipeline.whisper_version}
    Prompt:\t{PROMPT}
    -------------------------------------------------------
           WER = Substitution + Deletion + Insertion
    WER:\t\t{avg_wer["wer"]}
    Substitutions:\t{avg_wer["substitution_rate"]}
    Deletions:\t\t{avg_wer["deletion_rate"]}
    Insertions:\t\t{avg_wer["insertion_rate"]}
    -------------------------------------------------------
    RTF (per-clip average):\t{sample_average_rtf:.3g}
    RTF (global average):\t{global_rtf:.3g}
    Speed factor (RTFx):\t{speed_factor:.3g}
    -------------------------------------------------------
    Average audio duration:\t{tot_audio_duration/len(results):.3g}
    Average prediction time:\t{tot_prediction_duration/len(results):.3g}
    {fallback_str}
    Failed to transcribe: {num_failed}/{len(results)}
    =======================================================
    """)
    return results


def evaluate_sample(sample, whisper_pipeline):
    """ Evaluate a single audio file with whisper_pipeline
    """
    audio_file_path = sample["norm_path"]
    logger.debug("Transcribing: " + audio_file_path.rsplit("/")[-1])
    forced_language = sample["language"] if "language" in sample else None

    forced_language = forced_language if FORCE_LANGUAGE else None

    start = time.time()
    prediction = whisper_pipeline(audio_file_path, forced_language=forced_language, prompt=PROMPT)
    assert "text" in prediction, list(prediction)

    if isinstance(whisper_pipeline, pipelines.WhisperKit):
        num_fallbacks = prediction["timings"]["totalDecodingFallbacks"]
    else:
        num_fallbacks = None

    duration = time.time() - start
    normalized_predicted_text = text_normalizer(prediction["text"])

    if "file_length" in sample:
        audio_duration = int(sample["file_length"])
    elif "duration" in sample:
        audio_duration = int(sample["duration"])
    else:
        logger.warning(f"Missing audio duration in sample: {sample['norm_path']}, imputed with 0")
        audio_duration = 0

    wer_result = wer_metric.compute(
        references=[sample["norm_text"]],
        predictions=[normalized_predicted_text],
        detailed=True  # Return detailed results
    )

    # round to 3 decimal places
    wer_result = {k: round(v, 3) for k, v in wer_result.items()}

    # get text diffs
    text_diffs = get_text_diffs(sample["norm_text"], normalized_predicted_text)
    # filter out Nones in diffs
    text_diffs = [diff for diff in text_diffs if diff[1] is not None]

    return dict(
        audio_duration=audio_duration,
        reference=sample["norm_text"],
        prediction=normalized_predicted_text,
        prediction_duration=duration,
        file=audio_file_path .split('/')[-1],
        **wer_result,
        num_fallbacks=num_fallbacks,
        diffs=text_diffs,
    )


def evaluate_folder(folder, whisper_pipeline: pipelines.WhisperKit):
    """ Evaluate a single audio folder with whisper_pipeline
    """

    if not isinstance(whisper_pipeline, pipelines.WhisperKit):
        raise NotImplementedError("evaluate_folder is only implemented and tested for WhisperKit")

    audio_folder_path = folder["norm_folder"]
    logger.info(
        f"""\n
        =======================================================
        Beginning to transcribe folder: {audio_folder_path.rsplit("/")[-1]}
        Number of samples: {len(folder["samples"])}
        -------------------------------------------------------
        =======================================================
        """
    )

    forced_language = folder["samples"][0].get("language", None)
    for sample in folder["samples"]:
        if sample.get("language", None) != forced_language:
            forced_language = None
            break

    forced_language = forced_language if FORCE_LANGUAGE else None
    # Check if the forced language is unsupported
    if forced_language in UNSUPPORTED_LANGUAGES and ("large-v3" not in whisper_pipeline.whisper_version):
        logger.warning(
            f"Unsupported language: {forced_language} for model version: {whisper_pipeline.whisper_version}"
        )
        forced_language = None
    logger.info(f"Forced language: {forced_language}")

    start = time.time()
    predictions = whisper_pipeline.transcribe_folder(audio_folder_path, forced_language=forced_language)
    results = []
    for sample in folder["samples"]:
        audio_file_path = sample["norm_path"]
        prediction = predictions[audio_file_path]
        language = sample.get("language", None)
        assert "text" in prediction, list(prediction)

        if isinstance(whisper_pipeline, pipelines.WhisperKit):
            num_fallbacks = prediction["timings"]["totalDecodingFallbacks"]
        else:
            num_fallbacks = None

        duration = time.time() - start

        extra_text_normalizer = TEXT_NORMALIZER.get(language, basic_text_normalizer)
        normalized_reference_text = extra_text_normalizer(sample["norm_text"])
        normalized_predicted_text = extra_text_normalizer(text_normalizer(prediction["text"]))

        if "file_length" in sample:
            audio_duration = int(sample["file_length"])
        elif "duration" in sample:
            audio_duration = int(sample["duration"])
        else:
            logger.warning(f"Missing audio duration in sample: {sample['norm_path']}, imputed with 0")
            audio_duration = 0

        results.append(
            dict(
                audio_duration=audio_duration,
                reference=normalized_reference_text,
                prediction=normalized_predicted_text,
                prediction_duration=duration,
                file=audio_file_path.split('/')[-1],
                wer=wer_metric.compute(
                    references=[normalized_reference_text],
                    predictions=[normalized_predicted_text]
                ),
                num_fallbacks=num_fallbacks,
                text_normalizer=extra_text_normalizer.__class__.__name__,
                reference_language=language,
                predicted_language=prediction.get("language", None),
                model_version=whisper_pipeline.whisper_version
            )
        )

    logger.info(
        f"""\n
        =======================================================
        Completed folder transcription: {audio_folder_path.rsplit("/")[-1]}
        Number of samples: {len(folder["samples"])}
        -------------------------------------------------------
        =======================================================
        """
    )

    return results


def get_text_diffs(reference, prediction):
    d = Differ()
    reference_words = reference.split()
    prediction_words = prediction.split()

    diffs = []
    for token in d.compare(reference_words, prediction_words):
        if token.startswith('?'):
            continue

        status = token[0]
        word = token[2:].strip()

        if word:
            if status == ' ':
                diffs.append((word, None))
            else:
                diffs.append((word, status))

    return diffs
