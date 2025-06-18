from setuptools import find_packages, setup

from whisperkit._version import __version__

with open('README.md') as f:
    readme = f.read()

setup(
    name='whisperkit',
    version=__version__,
    url='https://github.com/argmaxinc/whisperkittools',
    description="Argmax WhisperKit Optimization Toolkit",
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Argmax, Inc.',
    install_requires=[
        "argmaxtools",
        "transformers",
        "huggingface-hub",
        "torch==2.6.0",
    ],
    extras_require={
        "pipelines": [
            "openai",
            "mlx-whisper",
        ],
        "diarization": [
            "pyannote-audio",
        ],
        "evals": [
            "jiwer",
            "soundfile",
            "librosa",
            "datasets",
            "evaluate",
            "transliterate",
            "openai",
            "mlx-whisper",
            "openai-whisper",
        ],
        "android": [
            "qai-hub",
            "qai_hub_models"
        ]
    },
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "whisperkit-generate-model=scripts.generate_model:cli",
            "whisperkit-evaluate-model=scripts.evaluate_model:cli",
            "whisperkit-generate-readme=scripts.generate_readme:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
