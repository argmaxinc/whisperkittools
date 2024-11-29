#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#
import time
from functools import partial
from multiprocessing import Pool
from typing import Union

import evaluate
import tqdm
from argmaxtools.utils import get_logger

from whisperkit import pipelines
from whisperkit.evaluate.datasets import get_dataset
from whisperkit.evaluate.normalize_en import EnglishTextNormalizer

logger = get_logger(__name__)

text_normalizer = EnglishTextNormalizer()
wer_metric = evaluate.load("wer")


def evaluate(whisper_pipeline: Union[pipelines.WhisperPipeline, pipelines.WhisperOpenAIAPI],
             dataset_name: str,
             num_samples: int,
             cache_dir: str,
             num_proc: int,
             ) -> None:
    """ Evaluate the given whisper pipeline implementation on a registered dataset.
    """
    dataset = get_dataset(
        dataset_name,
        cache_dir,
        max_num_samples=num_samples,
    )

    begin = time.time()

    # Warm-up (model compilation without multi-processing)
    logger.info("Warmup the pipeline on a single sample (may trigger model download)")
    _ = evaluate_sample(dataset[0], whisper_pipeline)

    if num_proc > len(dataset):
        logger.info(
            f"--num-proc ({num_proc}) is larger than the dataset size "
            f"({len(dataset)}), setting it to the dataset size")
        num_proc = len(dataset)

    if num_proc > 1:
        if isinstance(whisper_pipeline, pipelines.WhisperKit):
            whisper_pipeline._randomize_dispatch = True
            logger.info("Randomizing dispatch for WhisperKit for num_proc>1")

        logger.info(f"Launching {num_proc} processes to run {whisper_pipeline.__class__.__name__}")
        with Pool(num_proc) as pool:
            results = list(tqdm.tqdm(pool.imap(
                partial(evaluate_sample, whisper_pipeline=whisper_pipeline), dataset),
                total=len(dataset)
            ))
    else:
        results = []
        eval_sample = partial(evaluate_sample, whisper_pipeline=whisper_pipeline)
        for sample in dataset:
            results.append(eval_sample(sample))

    total_elapsed = time.time() - begin

    avg_wer = wer_metric.compute(
        references=[result["reference"] for result in results],
        predictions=[result["prediction"] for result in results],
    )
    avg_wer = round(100 * avg_wer, 2)

    tot_audio_duration = sum([result["audio_duration"] for result in results])
    tot_prediction_duration = sum([result["prediction_duration"] for result in results])

    sample_average_rtf = tot_prediction_duration / tot_audio_duration
    global_rtf = total_elapsed / tot_audio_duration

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
    Dataset:\t{dataset_name} {'(num_samples=' + str(num_samples) + ')' if num_samples > 1 else ''}
    Model:\t{whisper_pipeline.whisper_version}
    -------------------------------------------------------
    WER:\t{avg_wer}
    -------------------------------------------------------
    RTF (per-clip average):\t{sample_average_rtf:.3g}
    RTF (global average):\t{global_rtf:.3g}
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

    start = time.time()
    prediction = whisper_pipeline(audio_file_path)
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

    return dict(
        audio_duration=audio_duration,
        reference=sample["norm_text"],
        prediction=normalized_predicted_text,
        prediction_duration=duration,
        file=audio_file_path .split('/')[-1],
        wer=wer_metric.compute(
            references=[sample["norm_text"]],
            predictions=[normalized_predicted_text]
        ),
        num_fallbacks=num_fallbacks,
    )
