"""Shared pytest configuration for session-analyzer.

Forces the matplotlib ``Agg`` backend so visualizer tests run in headless
environments (CI, ssh sessions without DISPLAY).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
