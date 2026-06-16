"""Stellar Predictor - Predict unknown celestial bodies through gravitational modeling."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("stellar-predictor")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.6.0"

