from argmaxtools.utils import get_logger

logger = get_logger(__name__)

try:
    import pyannote.audio
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "`pyannote` not found. Please install diarization extras via: `pip install -e '.[diarization]'`"
    ) from e
