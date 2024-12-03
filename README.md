<div align="center">

<a href="https://github.com/argmaxinc/WhisperKit#gh-light-mode-only">
  <img src="https://github.com/user-attachments/assets/f0699c07-c29f-45b6-a9c6-f6d491b8f791" alt="WhisperKit" width="20%" />
</a>

<a href="https://github.com/argmaxinc/WhisperKit#gh-dark-mode-only">
  <img src="https://github.com/user-attachments/assets/1be5e31c-de42-40ab-9b85-790cb911ed47" alt="WhisperKit" width="20%" />
</a>

# whisperkittools
</div>

+![Unit and Functional Tests](https://github.com/argmaxinc/whisperkittools/actions/workflows/public-ci.yml/badge.svg)

Python tools for [WhisperKit](https://github.com/argmaxinc/whisperkit) and [WhisperKitAndroid](https://github.com/argmaxinc/WhisperKitAndroid)
- Convert PyTorch Whisper models to WhisperKit format
- Apply custom inference optimizations and model compression
- Evaluate Whisper using WhisperKit and other Whisper implementations on benchmarks

[!IMPORTANT]
If you are looking for more features such as speaker diarization and upgraded performance, check out [WhisperKit Pro](https://huggingface.co/argmaxinc/whisperkit-pro)!

## Table of Contents
- [Installation](#installation)
- [Model Generation (Apple)](#model-generation)
- [Model Generation (Android)](#model-generation-android)
- [Model Evaluation (Apple)](#evaluate)
- [Python Inference](#python-inference)
- [Example SwiftUI App](#example-swiftui-app)
- [Quality-of-Inference](#qoi)
- [FAQ](#faq)
- [Citation](#citation)


## Installation
- **Step 1:** [Fork this repository](https://github.com/argmaxinc/whisperkittools/fork)
- **Step 2:** Create a Python virtual environment, e.g.:
```shell
conda create -n whisperkit python=3.11 -y && conda activate whisperkit
```
- **Step 3:** Install the base package as editable
```shell
cd WHISPERKIT_ROOT_DIR && pip install -e .
```

## <a name="model-generation-apple"></a> Model Generation (Apple)
Convert [Hugging Face Whisper Models](https://huggingface.co/models?search=whisper) (PyTorch) to [WhisperKit](https://github.com/argmaxinc/whisperkit) (Core ML) format:

```shell
whisperkit-generate-model --model-version <model-version> --output-dir <output-dir>
```
For optional arguments related to model optimizations, please see the help menu with `-h`

### <a name="publish-custom-model"></a> Publishing Models
We host several popular Whisper model versions [here](https://huggingface.co/argmaxinc/whisperkit-coreml/tree/main). These hosted models are automatically over-the-air deployable to apps integrating WhisperKit such as our example app [WhisperAX on TestFlight](https://testflight.apple.com/join/LPVOyJZW). If you would like to publish custom Whisper versions that are not already published, you can do so as follows:
- **Step 1**: Find the user or organization name that you have write access to on [Hugging Face Hub](https://huggingface.co/settings/profile). If you are logged into `huggingface-cli` locally, you may simply do:
```shell
huggingface-cli whoami
```

If you don't have a write token yet, you can generate it [here](https://huggingface.co/settings/tokens).

- **Step 2**: Point to the model repository that you would like to publish to, e.g. `my-org/my-whisper-repo-name`, with the `MODEL_REPO_ID` environment variable and specify the name of the source PyTorch Whisper repository (e.g. [distil-whisper/distil-small.en](https://huggingface.co/distil-whisper/distil-small.en))
```shell
MODEL_REPO_ID=my-org/my-whisper-repo-name whisperkit-generate-model --model-version distil-whisper/distil-small.en --output-dir <output-dir>
```

If the above command is successfuly executed, your model will have been published to `hf.co/my-org/my-whisper-repo-name/distil-whisper_distil-small.en`!


## <a name="model-generation-android"></a> Model Generation (Android)
WhisperKit currently only supports Qualcomm AI Hub Whisper models on Hugging Face:
- [tiny.en](https://huggingface.co/qualcomm/Whisper-Tiny-En)
- [base.en](https://huggingface.co/qualcomm/Whisper-Base-En)
- [small.en](https://huggingface.co/qualcomm/Whisper-Small-En)

whisperkittools generates 3 more support models for input preprocessing and output postprocessing used in the WhisperKitAndroid pipeline. These are all published on Hugging Face [here](https://huggingface.co/argmaxinc/whisperkit-android/tree/main). Nonetheless, you may regenerate these models if you wish by following these steps:
- **Step 1**: Create an account at [aihub.qualcomm.com](aihub.qualcomm.com)
- **Step 2**: Set your API key locally as `qai-hub configure --api_token`
- **Step 3**: Install extra dependencies via `pip install -e '.[android]'` (Note that this requires `python<3.11`)
- **Step 4**: Execute `python tests/test_aihub.py --persistent-cache-dir <output-path>`

Stay tuned for more options for generating models without creating an account and more model version coverage!

## <a name="evaluate-apple"></a> Model Evaluation (Apple)

Evaluate ([Argmax-](https://huggingface.co/argmaxinc/whisperkit-coreml) or developer-published) models on speech recognition datasets:

```shell
whisperkit-evaluate-model --model-version <model-version> --output-dir <output-dir> --evaluation-dataset {librispeech-debug,librispeech,earnings22}
```

Install additional dependencies via:
```shell
pip install -e '.[evals,pipelines]'
```

By default, this command uses the latest `main` branch commits from `WhisperKit` and searches within [Argmax-published](https://huggingface.co/argmaxinc/whisperkit-coreml) model repositories. For optional arguments related to code and model versioning, please see the help menu with `-h`

We continually publish the evaluation results of Argmax-hosted models [here](https://huggingface.co/datasets/argmaxinc/whisperkit-evals) as part of our continuous integration tests.

### <a name="evaluate-on-custom-dataset"></a> Model Evaluation on Custom Dataset
If you would like to evaluate WhisperKit models on your own dataset:
- **Step 1**: Publish a dataset on the [Hub](https://huggingface.co/new-dataset) with the same simple structure as this [toy dataset](https://huggingface.co/datasets/argmaxinc/librispeech-debug) (audio files + `metadata.json`)
- **Step 2:** Run evaluation with environment variables as follows:

```shell
export CUSTOM_EVAL_DATASET="my-dataset-name-on-hub"
export DATASET_REPO_OWNER="my-user-or-org-name-on-hub"
export MODEL_REPO_ID="my-org/my-whisper-repo-name" # if evaluating self-published models
whisperkit-evaluate-model --model-version <model-version> --output-dir <output-dir> --evaluation-dataset my-dataset-name-on-hub
```

## Python Inference

Use the unified Python wrapper for several on-device Whisper frameworks:
- [WhisperKit](https://github.com/argmaxinc/whisperkit)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [mlx-examples/whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [WhisperOpenAIAPI (Server-side)](https://platform.openai.com/docs/guides/speech-to-text)

Install additional dependencies via:
```shell
pip install -e '.[pipelines]'
```

```python
from whisperkit.pipelines import WhisperKit, WhisperCpp, WhisperMLX, WhisperOpenAIAPI

pipe = WhisperKit(whisper_version="openai/whisper-large-v3", out_dir="/path/to/out/dir")
print(pipe("audio.{wav,flac,mp3}"))
```

**Note:** `WhisperCpp` requires `ffmpeg` to be installed. Recommended installation is with `brew install ffmpeg`
**Note:** `WhisperOpenAIAPI` requires setting `OPENAI_API_KEY` as an environment variable

## Example SwiftUI App
[TestFlight](https://testflight.apple.com/join/LPVOyJZW)

[Source Code (MIT License)](https://github.com/argmaxinc/whisperkit/tree/main/Examples/WhisperAX)

This app serves two purposes:
- Base template for developers to freely customize and integrate parts into their own app
- Real-world testing/debugging utility for custom Whisper versions or WhisperKit features before/without building an app.

Note that the app is in beta and we are actively seeking feedback to improve it before widely distributing it.

## <a name="qoi"></a> WhisperKit Quality and Performance Benchmarks

Please visit the [WhisperKit Benchmarks](https://huggingface.co/spaces/argmaxinc/whisperkit-benchmarks) Hugging Face Space for detailed benchmark results. Here is a [brief explanation](https://x.com/argmaxinc/status/1851723587423756680) to help with navigation of the results. This benchmark is updated for every non-patch release on virtually all supported devices.


## FAQ

**Q1**: `xcrun: error: unable to find utility "coremlcompiler", not a developer tool or in PATH`
**A1**: Ensure Xcode is installed on your Mac and run `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`.


## Citation
If you use WhisperKit for something cool or just find it useful, please drop us a note at [info@argmaxinc.com](mailto:info@argmaxinc.com)!

If you use WhisperKit for academic work, here is the BibTeX:

```
@misc{whisperkit-argmax,
title = {WhisperKit},
author = {Argmax, Inc.},
year = {2024},
URL = {https://github.com/argmaxinc/WhisperKit}
}
```
