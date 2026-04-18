"""Analytics modules: bout decomposition and matching-law fitting."""

from .bout import BoutMetrics, analyze_bouts
from .matching import MatchingParameters, estimate_matching_parameters

__all__ = [
    "BoutMetrics",
    "MatchingParameters",
    "analyze_bouts",
    "estimate_matching_parameters",
]
