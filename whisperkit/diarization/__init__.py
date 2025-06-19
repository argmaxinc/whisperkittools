from argmaxtools.utils import get_logger

logger = get_logger(__name__)

try:
    import pyannote.audio   # noqa: F401  (imported for its side-effect)
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "`pyannote` not found. Please install diarization extras via: `pip install -e '.[diarization]'`"
    ) from e
