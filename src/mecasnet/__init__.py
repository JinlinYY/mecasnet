"""Public package interface for MeCaSNet."""

from .config import Config
from .factory import apply_profile, build_mecasnet

__all__ = ["Config", "apply_profile", "build_mecasnet"]
__version__ = "0.1.0"

