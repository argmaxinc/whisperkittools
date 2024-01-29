# whisperkittools

Python tools for [WhisperKit](https://github.com/argmaxinc/whisperkit)
- Convert PyTorch Whisper models to WhisperKit format
- Apply custom inference optimizations and model compression
- Evaluate Whisper using WhisperKit and other Whisper implementations on benchmarks

## Table of Contents
- [Installation](#installation)
- [Model Generation](#model-generation)
- [Model Evaluation](#evaluate)
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
- **Step 3:** Install as editable
```shell
cd WHISPERKIT_ROOT_DIR && pip install -e .
```

## Model Generation
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

## <a name="evaluate"></a> Model Evaluation

Evaluate ([Argmax-](https://huggingface.co/argmaxinc/whisperkit-coreml) or developer-published) models on speech recognition datasets:

```shell
whisperkit-evaluate-model --model-version <model-version> --output-dir <output-dir> --dataset {librispeech-debug,librispeech,earnings22}
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
whisperkit-evaluate-model --model-version <model-version> --output-dir <output-dir> --dataset my-dataset-name-on-hub
```

## Python Inference

Use the unified Python wrapper for several Whisper frameworks:
- [WhisperKit](https://github.com/argmaxinc/whisperkit)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [mlx-examples/whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)

```python
from whisperkit.pipelines import WhisperKit, WhisperCpp, WhisperMLX

pipe = WhisperKit(whisper_version="openai/whisper-large-v3", out_dir="/path/to/out/dir")
print(pipe("audio.{wav,flac,mp3}"))
```

**Note:** Using `WhisperCpp` requires `ffmpeg` to be installed. Recommended installation is with `brew install ffmpeg`

## Example SwiftUI App
[TestFlight](https://testflight.apple.com/join/LPVOyJZW)

[Source Code (MIT License)](https://github.com/argmaxinc/whisperkit/tree/main/Examples/WhisperAX)

This app serves two purposes:
- Base template for developers to freely customize and integrate parts into their own app
- Real-world testing/debugging utility for custom Whisper versions or WhisperKit features before/without building an app.

Note that the app is in beta and we are actively seeking feedback to improve it before widely distributing it.

## <a name="qoi"></a> Quality-of-Inference

We believe that rigorously measuring the quality of inference is necessary for developers and
enterprises to make informed decisions when opting to use optimized or compressed variants of
Whisper models in production. The current measurements are between reference and optimized
WhisperKit models. We are going to extend the scope of this measurement to other Whisper
implementations soon so developers can certify the behavior change (if any) caused by
alternating use of WhisperKit with (or migration from) these implementations.

In all measurements, we care primarily about per-example no-regressions (quantified as `qoi` below)
which is a stricter metric compared to dataset average WER. A 100% `qoi` preserves perfect
backwards-compatibility on the test distribution and avoids "perceived regressions", the phenomenon
where per-example known behavior changes after a code/model update and causes divergence in
downstream code or breaks the user experience itself (even if dataset averages might stay flat
across updates). Pseudocode for `qoi`:

```python
qoi = []
for example in dataset:
    no_regression = wer(optimized_model(example)) <= wer(reference_model(example))
    qoi.append(no_regression)
qoi = (sum(qoi) / len(qoi)) * 100.
```

We define the reference model as the default float16 precision Core ML model that is generated by
whisperkittools. This reference model matches the accuracy of the original PyTorch model
on the specified test sets. We use `librispeech/test.clean` (5 hours of short English audio clips)
as our testing set for Whisper. We are actively expanding our test set coverage to `earnings22`
(120 hours of long English audio clips with various accents). We anticipate developers that use Whisper in production to have
their own Quality Assurance test sets and whisperkittools offers the tooling necessary to run the
same measurements on such custom test sets, please see the [Model Evaluation on Custom Dataset](#evaluate-on-custom-dataset)
for details.

### Reproducing Results
Results in this page are generated by our cluster of Apple Silicon Macs. We use them as self-hosted runners on
Github Actions as our CI infrastructure. Due to [security concerns](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions#hardening-for-self-hosted-runners),
we are unable to open up the cluster to the public. However, any Apple Silicon Mac (even with 8GB RAM) can be used to
run identical [evaluation jobs](#evaluation)
locally. For reference, our M2 Ultra devices complete a `librispeech` + `openai/whisper-large-v3`
evaluation in under 1 hour regardless of the Whisper implementation. Older Apple Silicon Macs should take less than
1 day to complete the same evaluation.


### Evaluation Results on Librispeech

#### WhisperKit + `openai_whisper-large-v3`

|                                                                                                                                                     |   WER |   QoI (%) |   File Size (MB) |
|:----------------------------------------------------------------------------------------------------------------------------------------------------|------:|----------:|-----------------:|
| [openai_whisper-large-v3](https://huggingface.co/argmaxinc/whisperkit-coreml-1-25-test/tree/main/openai_whisper-large-v3)                           |  2.44 |     100   |             3100 |
| [openai_whisper-large-v3_turbo](https://huggingface.co/argmaxinc/whisperkit-coreml-1-25-test/tree/main/openai_whisper-large-v3_turbo)               |  2.54 |      99.5 |             3100 |
| [openai_whisper-large-v3_turbo_1128MB](https://huggingface.co/argmaxinc/whisperkit-coreml-1-25-test/tree/main/openai_whisper-large-v3_turbo_1128MB) |  3.27 |      93   |             1128 |
| [openai_whisper-large-v3_turbo_1398MB](https://huggingface.co/argmaxinc/whisperkit-coreml-1-25-test/tree/main/openai_whisper-large-v3_turbo_1398MB) |  2.57 |      97.8 |             1398 |

#### Different Projects + `openai_whisper-large-v3`

|                                                                                                  |   WER | Commit Hash   | Model Format   |
|:-------------------------------------------------------------------------------------------------|------:|:--------------|:---------------|
| [WhisperKit](https://github.com/argmaxinc/whisperkit)                                            |  2.44 | 14e705e       | Core ML        |
| [WhisperCpp](https://github.com/ggerganov/whisper.cpp)                                           |  2.57 | 4bbb60e       | Core ML + GGUF |
| [WhisperMLX](https://github.com/ml-explore/mlx-examples/blob/main/whisper/whisper/transcribe.py) |  2.57 | 854ad87       | MLX (Numpy)    |


- `_turbo`: Indicates the presence of additional optimizations (not compression) to unlock streaming transcription as described in our [Blog Post](https://www.takeargmax.com/blog/whisperkit).

- `_*MB`: Indicates the presence of mixed-bit quantization. Instead of cluttering the filename with details like `_AudioEncoder-5.8bits_TextDecoder-6.1bits`, we choose to summarize the compression spec as the resulting total file size since this is what matters to developers in production.


## FAQ

**Q1**: `xcrun: error: unable to find utility "coremlcompiler", not a developer tool or in PATH`
**A1**: Ensure Xcode is installed on your Mac and run `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`.


## Citation
If you use WhisperKit for something cool or just find it useful, please drop us a note at [info@takeargmax.com](mailto:info@takeargmax.com)!

If you use WhisperKit for academic work, here is the BibTeX:

```
@misc{whisperkit-argmax,
title = {WhisperKit},
author = {Argmax, Inc.},
year = {2024},
URL = {https://github.com/argmaxinc/WhisperKit}
}
```
