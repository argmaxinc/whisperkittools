#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import json
import os
import random
import subprocess
from abc import ABC, abstractmethod
from typing import Optional

from argmaxtools.utils import _maybe_git_clone, get_logger
from huggingface_hub import snapshot_download

from whisperkit import _constants
from whisperkit.test_utils import CoreMLSwiftComputeUnit

logger = get_logger(__name__)


class WhisperPipeline(ABC):
    """ Abstract base class for Whisper CLI pipelines """
    def __init__(self,
                 whisper_version: str,
                 out_dir: Optional[str],
                 code_commit_hash: Optional[str] = None,
                 model_commit_hash: Optional[str] = None) -> None:
        self.whisper_version = whisper_version
        self.out_dir = out_dir
        self.code_commit_hash = code_commit_hash
        self.model_commit_hash = model_commit_hash

        logger.info(f"""\n
        =======================================================
        Cloning {self.__class__.__name__} repo:
        revision={self.code_commit_hash if self.code_commit_hash else 'main'}
        =======================================================
        """)
        self.clone_repo()
        assert hasattr(self, "repo_dir"), "clone_repo() must set self.repo_dir"
        self.repo_dir = os.path.abspath(self.repo_dir)

        logger.info(f"""\n
        =======================================================
        Building {self.__class__.__name__} CLI
        =======================================================
        """)
        self.build_cli()
        assert hasattr(self, "cli_path"), "build_cli() must set self.cli_path"

        logger.info(f"""\n
        =======================================================
        Downloading {self.__class__.__name__} models
        (whisper_version={whisper_version})
        =======================================================
        """)
        self.clone_models()
        assert hasattr(self, "models_dir"), "clone_models() must set self.models_dir"

    @abstractmethod
    def clone_repo(self):
        pass

    @abstractmethod
    def build_cli(self):
        pass

    @abstractmethod
    def clone_models(self):
        pass

    @abstractmethod
    def transcribe(self, audio_file_path: str) -> str:
        """ Transcribe an audio file using the Whisper pipeline
        """
        pass

    def __call__(self, audio_file_path: str) -> str:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(audio_file_path)

        logger.info(f"""\n
        =======================================================
        Beginning to transcribe {audio_file_path.rsplit("/")[-1]}:
        -------------------------------------------------------
        =======================================================
        """)
        cli_result = self.transcribe(audio_file_path)
        logger.info(f"""\n
        =======================================================
        Transcription result for {audio_file_path.rsplit("/")[-1]}:
        -------------------------------------------------------\n\n{cli_result}
        =======================================================
        """)

        return cli_result


class WhisperKit(WhisperPipeline):
    """ Pipeline to clone, build and run the CLI from
    https://github.com/argmaxinc/WhisperKit
    """
    _compute_unit: CoreMLSwiftComputeUnit = CoreMLSwiftComputeUnit.ANE
    _randomize_dispatch: bool = False

    def clone_repo(self):
        self.repo_dir, self.code_commit_hash = _maybe_git_clone(
            out_dir=self.out_dir,
            hub_url="github.com",
            repo_name=_constants.WHISPERKIT_REPO_NAME,
            repo_owner=_constants.WHISPERKIT_REPO_OWNER,
            commit_hash=self.code_commit_hash)

    def build_cli(self):
        self.product_name = "transcribe"
        if subprocess.check_call(f"swift build -c release --product {self.product_name}",
                                 cwd=self.repo_dir, shell=True):
            raise subprocess.CalledProcessError(f"Failed to build {self.product_name}")
        logger.info(f"Successfuly built {self.product_name} CLI")

        build_dir = subprocess.run(
            f"swift build -c release --product {self.product_name} --show-bin-path",
            cwd=self.repo_dir, stdout=subprocess.PIPE, shell=True, text=True).stdout.strip()
        self.cli_path = os.path.join(build_dir, self.product_name)

    def clone_models(self):
        """ Download WhisperKit model files from Hugging Face Hub
        (only the files needed for `self.whisper_version`)
        """
        self.models_dir = os.path.join(
            self.repo_dir, "Models", self.whisper_version.replace("/", "_"))

        os.makedirs(self.models_dir, exist_ok=True)

        snapshot_download(
            repo_id=_constants.MODEL_REPO_ID,
            allow_patterns=f"{self.whisper_version.replace('/', '_')}/*",
            revision=self.model_commit_hash,
            local_dir=os.path.dirname(self.models_dir),
            local_dir_use_symlinks=True
        )

        if self.model_commit_hash is None:
            self.model_commit_hash = subprocess.run(
                f"git ls-remote git@hf.co:{_constants.MODEL_REPO_ID}",
                shell=True, stdout=subprocess.PIPE
            ).stdout.decode("utf-8").rsplit("\n")[0].rsplit("\t")[0]
            logger.info(
                "--model-commit-hash not specified, "
                f"imputing with HEAD={self.model_commit_hash}")

        self.results_dir = os.path.join(self.models_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

    @property
    def compute_unit(self) -> CoreMLSwiftComputeUnit:
        return self._compute_unit

    @compute_unit.setter
    def compute_unit(self, value: str):
        if value.lower() == "ane":
            self._compute_unit = CoreMLSwiftComputeUnit.ANE
        elif value.lower() == "gpu":
            self._compute_unit = CoreMLSwiftComputeUnit.GPU
        elif value.lower() == "ane_or_gpu":
            # FIXME(atiorh): Remove this once we have an integrated multi-engine async dispatch
            self._compute_unit = CoreMLSwiftComputeUnit.ANE if random.random() > 0.5 \
                    else CoreMLSwiftComputeUnit.GPU
            logger.info("Randomizing compute unit")

        logger.info(f"Set compute unit: {self._compute_unit}")

    def transcribe(self, audio_file_path: str) -> str:
        """ Transcribe an audio file using the WhisperKit CLI
        """
        if self._randomize_dispatch:
            self.compute_unit = "ane_or_gpu"

        cmd = " ".join([
            self.cli_path,
            "--audio-path", audio_file_path,
            "--model-path", self.models_dir,
            "--text-decoder-compute-units", self.compute_unit.value,
            "--audio-encoder-compute-units", self.compute_unit.value,
            "--report-path", self.results_dir, "--report",
        ])

        logger.debug(f"Executing command: {cmd}")
        if subprocess.check_call(cmd, stdout=subprocess.PIPE, shell=True, text=True) != 0:
            raise subprocess.CalledProcessError(f"Failed to transcribe {audio_file_path}")

        result_path = os.path.join(
            self.results_dir,
            os.path.splitext(audio_file_path.rsplit("/")[-1])[0] + ".json"
        )

        if not os.path.exists(result_path):
            results = None
            logger.warning(f"Result not found at {result_path}")

        with open(result_path, "r") as f:
            results = json.load(f)

        if results is None or "text" not in results:
            logger.warning(f"No text found in results: {results}")
            results = {"text": "", "timings": {"totalDecodingFallbacks": 0}, "failed": True}

        return results


class WhisperCpp(WhisperPipeline):
    """ Pipeline to clone, build and run the CLI from
    https://github.com/ggerganov/whisper.cpp
    """
    def clone_repo(self):
        self.repo_dir, self.code_commit_hash = _maybe_git_clone(
            out_dir=self.out_dir,
            hub_url="github.com",
            repo_name="whisper.cpp",
            repo_owner="ggerganov",
            commit_hash=self.code_commit_hash)

    def build_cli(self):
        self.cli_path = os.path.join(self.repo_dir, "main")
        if not os.path.exists(self.cli_path):
            # commands = ["make clean", "WHISPER_COREML=1 make -j"]
            commands = ["make clean", "WHISPER_COREML=1 make -j"]
            for command in commands:
                if subprocess.check_call(command, cwd=self.repo_dir, shell=True):
                    raise subprocess.CalledProcessError(f"Failed to run: `{command}`")
            logger.info("Successfuly built whisper.cpp CLI")
        else:
            logger.info("Reusing cached CLI binary")

    def clone_models(self):
        """ Download whisper.cpp model files from Hugging Face Hub
        (only the ones needed for `self.whisper_version`)
        """
        self.models_dir = os.path.join(self.repo_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        model_version_str = self.whisper_version.rsplit('/')[-1].replace("whisper-", "")

        # Download pre-compiled Core ML models from Hugging Face
        mlmodelc_fname = f"ggml-{model_version_str}-encoder.mlmodelc"
        snapshot_download(
            repo_id="ggerganov/whisper.cpp",
            allow_patterns=mlmodelc_fname + ".zip",
            revision=self.model_commit_hash,
            local_dir=self.models_dir,
            local_dir_use_symlinks=True
        )

        # Unzip mlmodelc.zip
        if not os.path.exists(os.path.join(self.models_dir, mlmodelc_fname)):
            if subprocess.check_call(" ".join([
                "unzip", os.path.join(self.models_dir, mlmodelc_fname + ".zip"),
                "-d", self.models_dir,
            ]), cwd=self.repo_dir, shell=True):
                raise subprocess.CalledProcessError("Failed to unzip Core ML model")

        # Download other model files (Only the encoder is a Core ML model)
        self.ggml_model_path = os.path.join(
            self.models_dir, f"ggml-{model_version_str}.bin")

        if not os.path.exists(self.ggml_model_path):
            download_script = os.path.join(self.repo_dir, "models", "download-ggml-model.sh")
            if subprocess.check_call(" ".join([
                download_script,
                model_version_str,
            ]), cwd=self.repo_dir, shell=True):
                raise subprocess.CalledProcessError("Failed to download ggml model")

            logger.info("Downloaded ggml model: ", self.ggml_model_path)

    def preprocess_audio_file(self, audio_file_path: str) -> str:
        if os.path.splitext(audio_file_path)[-1] == "wav":
            import contextlib
            return contextlib.nullcontext(enter_result=os.path.dirname(audio_file_path))

        else:
            import tempfile
            tempfile_context = tempfile.TemporaryDirectory(prefix="whispercpp_wav_conversions")
            temp_path = os.path.join(
                tempfile_context.name,
                audio_file_path.rsplit("/")[-1].rsplit(".")[0] + ".wav"
            )

            # Resample to 16kHz and convert to wav
            if subprocess.check_call(" ".join([
                "ffmpeg",
                "-nostdin",
                "-threads", "0",
                "-i", audio_file_path,
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                temp_path
            ]), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True):
                raise subprocess.CalledProcessError(
                    "Failed to resample audio file. Make sure ffmpeg is installed.")

            logger.info(f"Resampled {audio_file_path} to temporary file ({temp_path})")
            return tempfile_context

    def transcribe(self, audio_file_path: str) -> str:
        """ Transcribe an audio file using the whisper.cpp CLI
        """
        with self.preprocess_audio_file(audio_file_path) as processed_file_dir:
            if hasattr(processed_file_dir, "name"):
                processed_file_dir = processed_file_dir.name

            processed_file_path = os.path.join(
                processed_file_dir,
                audio_file_path.rsplit("/")[-1].rsplit(".")[0] + ".wav"
            )

            cli_result = subprocess.run(" ".join([
                    self.cli_path,
                    "-m", self.ggml_model_path,
                    "--beam-size", "1",
                    "--no-timestamps",
                    "-f", processed_file_path,
                ]),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                shell=True, text=True
            ).stdout.strip()

            if not cli_result:
                raise RuntimeError("Failed to transcribe audio file")

            return {"text": cli_result}


class WhisperMLX(WhisperPipeline):
    """ Pipeline to clone, build and run the Python pipeline from
    git@github.com:ml-explore/mlx-examples.git
    """
    def clone_repo(self):
        self.repo_dir, self.code_commit_hash = _maybe_git_clone(
            out_dir=self.out_dir,
            hub_url="github.com",
            repo_name="mlx-examples",
            repo_owner="ml-explore",
            commit_hash=self.code_commit_hash)

    def build_cli(self):
        self.cli_path = "N/A"

    def clone_models(self):
        self.models_dir = "N/A"

    def transcribe(self, audio_file_path: str) -> str:
        """ Transcribe an audio file using MLX
        """
        text = subprocess.run(" ".join([
            "python", "-c",
            "'import sys;",
            "sys.path.append(\"" + self.repo_dir + "/whisper\");",
            "import whisper;",
            "print(whisper.transcribe(\"" + audio_file_path + "\",",
            "path_or_hf_repo = \"mlx-community/" + self.whisper_version.rsplit('/')[-1] + "-mlx\",",
            "fp16=True,",
            # "temperature=(0,),",
            ")[\"text\"])'"
        ]), stdout=subprocess.PIPE, shell=True, text=True).stdout.strip()
        return {"text": text}


def get_pipeline_cls(cls_name):
    if cls_name == "WhisperKit":
        return WhisperKit
    elif cls_name == "whisper.cpp":
        return WhisperCpp
    elif cls_name == "WhisperMLX":
        return WhisperMLX
    else:
        raise ValueError(f"Unknown pipeline: {cls_name}")
