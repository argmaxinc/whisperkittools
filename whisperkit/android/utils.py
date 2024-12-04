# For licensing see accompanying LICENSE.txt file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

import json
import numpy as np
import os
import torch

import qai_hub

from argmaxtools.utils import get_logger
from collections import Counter
from qai_hub.client import Device
from qai_hub_models.utils.base_model import TargetRuntime
from typing import Optional

logger = get_logger(__name__)

ANDROID_SUPPORTED_WHISPER_VERSIONS = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en"
]

# Reference: 
# https://app.aihub.qualcomm.com/docs/hub/compile_examples.html#compiling-pytorch-model-to-a-qnn-model-library

TARGET_TABLE = {
    "qnn": TargetRuntime.QNN,
    "onnx": TargetRuntime.ONNX,
    "tflite": TargetRuntime.TFLITE,
}

CHIPSET_TABLE = {
    "888": "qualcomm-snapdragon-888",
    "gen1": "qualcomm-snapdragon-8gen1",
    "gen2": "qualcomm-snapdragon-8gen2",
    "gen3": "qualcomm-snapdragon-8gen3",
}


def get_hub_compile_options(
    delegate: str = "qnn",
    other_compile_options: str = "",
    device: Optional[Device] = None,
) -> str:
    """ AI Hub compile options recommended for the model
    """
    target_runtime = TARGET_TABLE[delegate]
    target_runtime_flag = None
    extra_target_options = ""
    if "--target_runtime" not in other_compile_options:
        if target_runtime == TargetRuntime.QNN:
            if device:
                if not device.attributes:
                    # Only name/os specified
                    devices = qai_hub.get_devices(device.name, device.os)
                elif not device.name:
                    # Only attribute specified
                    devices = qai_hub.get_devices(attributes=device.attributes)
                else:
                    devices = [device]

                for device in devices:
                    if (
                        "os:android" not in device.attributes
                        or "format:iot" in device.attributes
                        or "format:auto" in device.attributes
                    ):
                        target_runtime_flag = "qnn_context_binary"
                        break

            target_runtime_flag = target_runtime_flag or "qnn_lib_aarch64_android"
            extra_target_options = " --quantize_full_type float16"
        elif target_runtime == TargetRuntime.ONNX:
            target_runtime_flag = "onnx"
        elif target_runtime == TargetRuntime.TFLITE:
            target_runtime_flag = "tflite"
            extra_target_options = " --quantize_io"
        elif target_runtime == TargetRuntime.PRECOMPILED_QNN_ONNX:
            target_runtime_flag = "precompiled_qnn_onnx"
        else:
            raise NotImplementedError()

    compile_options = (
        f"--target_runtime {target_runtime_flag}" if target_runtime_flag else ""
    )

    # compile_options += f" --output_names k_cache,v_cache --quantize_full_type float16 --quantize_io"
    compile_options += extra_target_options

    if other_compile_options != "":
        return compile_options + " " + other_compile_options

    return compile_options


def convert_via_aihub(
        model_name,
        torch_model,
        inputs,
        target_runtime="tflite",
        benchmark_chipset="gen2",
        output_dir="."
):
    traced_torch_model = torch.jit.trace(
        torch_model,
        tuple(inputs.values()),
        strict=True
    )

    # Snapdragon HTP has 32-bit arch
    DTYPE_TABLE = {
        torch.float32: "float32",
        torch.int32: "int32",
        torch.int64: "int32",
    }

    input_specs = {k: (v.shape, DTYPE_TABLE[v.dtype]) for k, v in inputs.items()}
    hub_device = qai_hub.Device(attributes=f"chipset:{CHIPSET_TABLE[benchmark_chipset]}")

    compile_options = "--truncate_64bit_tensors"
    model_compile_options = get_hub_compile_options(
        target_runtime,
        compile_options,
        hub_device
    )

    logger.info(f"Compiling model with options: {model_compile_options}")
    compile_job = qai_hub.submit_compile_job(
        model=traced_torch_model,
        device=hub_device,
        options=model_compile_options,
        input_specs=input_specs,
    )

    logger.info("Downloading model from AI Hub")
    compile_job.download_target_model(model_name)

    file_name = f"{model_name}.{target_runtime}"
    os.makedirs(output_dir, exist_ok=True)
    os.rename(os.path.join(os.getcwd(), file_name), os.path.join(output_dir, file_name))

    logger.info("Profiling model on AI Hub devices")
    target_model = compile_job.get_target_model()
    profile_job = qai_hub.submit_profile_job(
        model=target_model,
        device=hub_device,
    )

    inference_job = qai_hub.submit_inference_job(
        model=target_model,
        device=hub_device,
        inputs={k: [v.numpy()] for k, v in inputs.items()}
    )

    return {
        "inference": inference_job.download_output_data(),
        "profile": profile_job.download_profile(),
    }


def summarize_performance(ai_hub_profile_report):
    """ Summarize the profile job results from AI Hub
    """
    forward_latencies = np.array(ai_hub_profile_report["execution_summary"]["all_inference_times"])

    mean = np.mean(forward_latencies)
    std = np.std(forward_latencies)
    outlier_latencies = forward_latencies[forward_latencies > mean + 3*std]
    inlier_latencies = forward_latencies[forward_latencies < mean + 3*std]

    outlier_incidence_rate = len(outlier_latencies) / len(forward_latencies)
    outlier_overhead_rate = sum(outlier_latencies) / (sum(inlier_latencies) + len(outlier_latencies) * mean)

    load_time = round(ai_hub_profile_report["execution_summary"]["all_warm_load_times"][0] / 1e3, 1)
    on_device_compile_time = round(
        ai_hub_profile_report["execution_summary"]["all_first_load_times"][0] / 1e3,
        1
    )

    compute_unit_dispatch = dict(
        Counter(layer["compute_unit"] for layer in ai_hub_profile_report["execution_detail"])
    )

    summary_str = f"""\n
================================================
============ AI Hub Profile Summary ============
================================================
Layer dispatch: {compute_unit_dispatch}

On-device compile time: {on_device_compile_time} ms
Load Time {load_time} ms

Latency average = {round(mean / 1e3, 1):.1f} ms (std={round(std / 1e3, 1)})
Latency outlier incidence = {outlier_incidence_rate * 100:.1f}%
                overhead  = {outlier_overhead_rate * 100:.1f}%
================================================
"""

    return dict(
        compute_unit_dispatch=compute_unit_dispatch,
        on_device_compile_time=on_device_compile_time,
        load_time=load_time,
        outlier_incidence_rate=outlier_incidence_rate,
        outlier_overhead_rate=outlier_overhead_rate,
        prediction_latency=round(mean / 1e3, 1),
        prediction_std=round(std / 1e3, 1),
    ), summary_str


def convert_tokenizer(tokenizer_path):
    """ Temporary function to reformat the original Whisper tokenizer
    """
    with open(tokenizer_path, "r") as file:
        tokenizer = json.load(file)
        added_tokens = tokenizer["added_tokens"]
        special_tokens = tokenizer["post_processor"]["special_tokens"]
        vocab = tokenizer["model"]["vocab"]

        output = {}
        for key, value in vocab.items():
            key = key.replace("\u0120", "")
            output[value] = key

        for token in added_tokens:
            text = token["content"]
            text = text.replace("\u0120", "")
            output[token["id"]] = text

        for token, id in special_tokens.items():
            text = text.replace("\u0120", "")
            output[id["ids"][0]] = token

        with open("converted_tokenizer.json", "w") as file:
            json.dump(output, file)
