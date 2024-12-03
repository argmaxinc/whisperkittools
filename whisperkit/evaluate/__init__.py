from argmaxtools.utils import get_logger

logger = get_logger(__name__)

try:
    import evaluate
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("`evaluate` not found. Please install evals extras via: `pip install -e '.[evals]'`" ) from e