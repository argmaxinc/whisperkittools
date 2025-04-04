#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.
#

import json
import mlx_whisper
import openai
import os
import subprocess

from abc import ABC, abstractmethod
from typing import Optional, Any

from argmaxtools.utils import _maybe_git_clone, get_logger
from huggingface_hub import snapshot_download

from whisperkit import _constants


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
    def transcribe(self, audio_file_path: str, forced_language: Optional[str] = None) -> dict[str, Any]:
        """ Transcribe an audio file using the Whisper pipeline
        """
        pass

    def __call__(self, audio_file_path: str, forced_language: Optional[str] = None) -> dict[str, Any]:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(audio_file_path)

        logger.info(f"""\n
        =======================================================
        Beginning to transcribe {audio_file_path.rsplit("/")[-1]}:
        -------------------------------------------------------
        =======================================================
        """)
        cli_result = self.transcribe(audio_file_path, forced_language=forced_language)
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
    _word_timestamps: bool = False

    # Different compute unit configurations may be faster on different devices
    _text_decoder_compute_units = "cpuAndNeuralEngine"
    _audio_encoder_compute_units = "cpuAndNeuralEngine"

    def clone_repo(self):
        self.repo_dir, self.code_commit_hash = _maybe_git_clone(
            out_dir=self.out_dir,
            hub_url="github.com",
            repo_name=_constants.WHISPERKIT_REPO_NAME,
            repo_owner=_constants.WHISPERKIT_REPO_OWNER,
            commit_hash=self.code_commit_hash)

    def build_cli(self):
        self.product_name = "whisperkit-cli"
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

    def transcribe(self, audio_file_path: str, forced_language: Optional[str] = None) -> dict[str, Any]:
        """ Transcribe an audio file using the WhisperKit CLI
        """
        cmd = " ".join([
            self.cli_path,
            "transcribe",
            "--audio-path", audio_file_path,
            "--model-path", self.models_dir,
            "--text-decoder-compute-units", self._text_decoder_compute_units,
            "--audio-encoder-compute-units", self._audio_encoder_compute_units,
            # "--chunking-strategy", "vad",
            "--report-path", self.results_dir, "--report",
            "--word-timestamps" if self._word_timestamps else "",
            "" if forced_language is None else f"--use-prefill-prompt --language {forced_language}",
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

    def transcribe_folder(self, audio_folder_path: str, forced_language: Optional[str] = None) -> dict[str, Any]:
        """ Transcribe an audio folder using the WhisperKit CLI
        """
        cmd = " ".join([
            self.cli_path,
            "transcribe",
            "--audio-folder", audio_folder_path,
            "--model-path", self.models_dir,
            "--text-decoder-compute-units", self._text_decoder_compute_units,
            "--audio-encoder-compute-units", self._audio_encoder_compute_units,
            "--report-path", self.results_dir, "--report",
            "--word-timestamps" if self._word_timestamps else "",
            # "--concurrent-worker-count 1",
            "--task transcribe",
            "" if forced_language is None else f"--use-prefill-prompt --language {forced_language}",
        ])

        logger.debug(f"Executing command: {cmd}")
        if subprocess.check_call(cmd, stdout=subprocess.PIPE, shell=True, text=True) != 0:
            raise subprocess.CalledProcessError(f"Failed to transcribe folder: {audio_folder_path}")

        audio_file_paths = os.listdir(audio_folder_path)
        folder_results = {}
        for audio_file_path in audio_file_paths:
            result_path = os.path.join(
                self.results_dir,
                os.path.splitext(audio_file_path)[0] + ".json"
            )

            if not os.path.exists(result_path):
                results = None
                logger.warning(f"Result not found at {result_path}")

            with open(result_path, "r") as f:
                results = json.load(f)

            if results is None or "text" not in results:
                logger.warning(f"No text found in results: {results}")
                results = {"text": "", "timings": {"totalDecodingFallbacks": 0}, "failed": True}

            logger.info(
                f"""\n
                =======================================================
                Transcription result for {audio_file_path.rsplit("/")[-1]}:
                -------------------------------------------------------\n\n{results}
                =======================================================
                """
            )

            folder_results[os.path.join(audio_folder_path, audio_file_path)] = results

        return folder_results


class WhisperCpp(WhisperPipeline):
    """ Pipeline to clone, build and run the CLI from
    https://github.com/ggerganov/whisper.cpp
    """
    QUANT_SUFFIXES = ["-q5_0", "-q8_0", "-q5_1", "-q8_1"]

    def clone_repo(self):
        self.repo_dir, self.code_commit_hash = _maybe_git_clone(
            out_dir=self.out_dir,
            hub_url="github.com",
            repo_name="whisper.cpp",
            repo_owner="ggerganov",
            commit_hash=self.code_commit_hash)

    def quant_variant(self):
        for suffix in self.QUANT_SUFFIXES:
            if self.whisper_version.endswith(suffix):
                return suffix
        return None

    def build_cli(self):
        ENV_PREFIX = ""

        self.model_version_str = self.whisper_version.rsplit('/')[-1].replace("whisper-", "")

        self.cli_path = os.path.join(self.repo_dir, "main")
        if not os.path.exists(self.cli_path):
            commands = ["make clean", f"{ENV_PREFIX} make -j {self.model_version_str}"]
            for command in commands:
                print(command)
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

        # Download other model files (Only the encoder is a Core ML model)
        self.ggml_model_path = os.path.join(
            self.models_dir, f"ggml-{self.model_version_str}.bin")

    def preprocess_audio_file(self, audio_file_path: str) -> Any:
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

    def transcribe(self, audio_file_path: str, forced_language: Optional[str] = None) -> dict[str, Any]:
        """ Transcribe an audio file using the whisper.cpp CLI
        """
        with self.preprocess_audio_file(audio_file_path) as processed_file_dir:
            if hasattr(processed_file_dir, "name"):
                processed_file_dir = processed_file_dir.name

            processed_file_path = os.path.join(
                processed_file_dir,
                audio_file_path.rsplit("/")[-1].rsplit(".")[0] + ".wav"
            )

            cmd = " ".join([
                self.cli_path,
                "-m", self.ggml_model_path,
                "--beam-size", "1",
                "--no-timestamps",
                "--flash-attn",
                "-f", processed_file_path,
                "" if forced_language is None else f"-l {forced_language}"
            ])
            print(cmd)

            cli_result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                shell=True, text=True
            ).stdout.strip()

            if not cli_result:
                raise RuntimeError("Failed to transcribe audio file")

            return {"text": cli_result}


MLX_HF_REPO_MAP = {
    "openai/whisper-tiny": "mlx-community/whisper-tiny-mlx",
    "openai/whisper-tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "openai/whisper-base": "mlx-community/whisper-base-mlx",
    "openai/whisper-small": "mlx-community/whisper-small-mlx",
    "openai/whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
    "openai/whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


class WhisperMLX(WhisperPipeline):
    """ Pipeline to run MLX Whisper
    """
    def clone_repo(self):
        self.repo_dir = "."

    def build_cli(self):
        self.cli_path = None

    def clone_models(self):
        self.models_dir = None

    def transcribe(self, audio_file_path: str, forced_language: Optional[str] = None) -> dict[str, Any]:
        """ Transcribe an audio file using MLX
        """
        # Note 1: `condition_on_previous_text=True` causes repetitions
        # Note 2: beam_search is not implemented so no need to set it to 1
        return mlx_whisper.transcribe(
            audio_file_path,
            path_or_hf_repo=MLX_HF_REPO_MAP[self.whisper_version],
            condition_on_previous_text=False,
            language=None if forced_language is None else forced_language
        )


class WhisperOpenAIAPI:
    """ Pipeline to use the OpenAI API for transcription

    See https://platform.openai.com/docs/guides/speech-to-text
    """
    def __init__(self,
                 whisper_version: str = _constants.OPENAI_API_MODEL_VERSION,
                 out_dir: Optional[str] = ".",
                 **kwargs) -> None:

        if whisper_version != _constants.OPENAI_API_MODEL_VERSION:
            raise ValueError(f"OpenAI API only supports '{_constants.OPENAI_API_MODEL_VERSION}'")
        self.whisper_version = whisper_version

        self.client = None

        if len(kwargs) > 0:
            logger.warning(f"Unused kwargs: {kwargs}")

        self.out_dir = out_dir
        self.results_dir = os.path.join(out_dir, "OpenAI-API")
        os.makedirs(self.results_dir, exist_ok=True)

        # Can not version OpenAI API
        self.code_commit_hash = None
        self.model_commit_hash = None

        logger.info("""\n
        =======================================================
        Using OpenAI API
        =======================================================
        """)

    def _maybe_init_client(self):
        if self.client is None:
            api_key = os.getenv("OPENAI_API_KEY", None)
            assert api_key is not None
            self.client = openai.Client(api_key=api_key)

    def _maybe_compress_audio_file(self, audio_file_path: str) -> str:
        """ If size of file at `audio_file_path` is larger than OpenAI API max file size, compress with ffmpeg
        """
        audio_file_size = os.path.getsize(audio_file_path)
        if audio_file_size > _constants.OPENAI_API_MAX_FILE_SIZE:
            logger.info(
                f"Compressing {audio_file_path.rsplit('/')[-1]} with size {audio_file_size / 1e6:.1f} MB > "
                f"{_constants.OPENAI_API_MAX_FILE_SIZE / 1e6:.1f} MB (OpenAI API max file size)")

            compressed_audio_file_path = os.path.splitext(audio_file_path)[0] + ".ogg"
            # if not os.path.exists(compressed_audio_file_path):
            if subprocess.check_call(" ".join([
                "ffmpeg",
                "-i", audio_file_path,
                "-vn",
                "-map_metadata", "-1",
                "-ac", "1", "-c:a", "libopus", "-b:a", _constants.OPENAI_API_COMPRESSED_UPLOAD_BIT_RATE,
                "-application", "voip",
                "-y",  # Overwrite
                compressed_audio_file_path
            ]), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True):
                raise subprocess.CalledProcessError(
                    "Failed to compress audio file. Make sure ffmpeg is installed.")

            audio_file_path = compressed_audio_file_path
            compressed_size = os.path.getsize(audio_file_path)

            if compressed_size > _constants.OPENAI_API_MAX_FILE_SIZE:
                raise ValueError(
                    f"Compressed file size {compressed_size / 1e6:.1f} MB exceeds OpenAI API max file size "
                    f"({_constants.OPENAI_API_MAX_FILE_SIZE / 1e6:.1f}) MB. Either (a) override "
                    "whisperkit._constants.OPENAI_API_COMPRESSED_UPLOAD_BIT_RATE with a lower value or (2) "
                    "follow https://platform.openai.com/docs/guides/speech-to-text/longer-inputs"
                )

            logger.info(
                f"Compressed  {audio_file_path.rsplit('/')[-1]} to {compressed_size / 1e6:.1f} MB < "
                f"{_constants.OPENAI_API_MAX_FILE_SIZE / 1e6:.1f} MB"
            )

        return audio_file_path

    def __call__(self, audio_file_path: str) -> dict[str, Any]:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(audio_file_path)

        logger.info(f"""\n
        =======================================================
        Beginning to transcribe {audio_file_path.rsplit("/")[-1]}:
        -------------------------------------------------------
        =======================================================
        """)
        result_fname = f"{audio_file_path.rsplit('/')[-1].rsplit('.')[0]}.json"

        if not os.path.exists(os.path.join(self.results_dir, result_fname)):
            audio_file_path = self._maybe_compress_audio_file(audio_file_path)
            self._maybe_init_client()

            with open(audio_file_path, "rb") as file_handle:
                api_result = json.loads(self.client.audio.transcriptions.create(
                    model="whisper-1",
                    timestamp_granularities=["word", "segment"],
                    response_format="verbose_json",
                    file=file_handle,
                ).json())

            # result_fname = f"{audio_file_path.rsplit('/')[-1].rsplit('.')[0]}.json"
            with open(os.path.join(self.results_dir, result_fname), "w") as f:
                json.dump(api_result, f, indent=4)
        else:
            with open(os.path.join(self.results_dir, result_fname), "r") as f:
                api_result = json.load(f)

        logger.info(f"""\n
        =======================================================
        Transcription result for {audio_file_path.rsplit("/")[-1]}:
        -------------------------------------------------------\n\n{api_result}
        =======================================================
        """)

        return api_result


def get_pipeline_cls(cls_name):
    if cls_name == "WhisperKit":
        return WhisperKit
    elif cls_name == "whisper.cpp":
        return WhisperCpp
    elif cls_name == "WhisperMLX":
        return WhisperMLX
    elif cls_name == "WhisperOpenAIAPI":
        return WhisperOpenAIAPI
    else:
        raise ValueError(f"Unknown pipeline: {cls_name}")
