"""Cumulative-record visualization, statistical analysis, and behavioral model fitting.

Consumes OKL v1 session logs (typically produced by ``session-recorder``) and
exposes visualization and analytics modules for cumulative records, bout
decomposition, and Generalized Matching Law fitting.
"""

from .visualizer.cumulative_record import plot_cumulative_record

__all__ = ["plot_cumulative_record"]
