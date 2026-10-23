"""Spectral and autocorrelation analyses of inter-response time series.

When IRTs show non-trivial temporal structure (rhythms,
schedule-induced periodicities, drift), simple summary statistics
miss it. This module provides:

- :func:`compute_irt_autocorrelation` — biased autocorrelation function
  ``rho(k) = E[(x_t - mu)(x_{t+k} - mu)] / sigma^2`` for ``k = 0..max_lag``.
- :func:`compute_irt_spectrum` — raw periodogram (FFT) of the
  mean-centered IRT series (see :class:`IRTSpectrum` for its scaling).

References
----------
Wei, W. W. S. (2006). *Time series analysis: Univariate and multivariate
    methods* (2nd ed.). Pearson.

Schoenfeld, W. N., & Cole, B. K. (1972). *Stimulus schedules: The
    t-tau systems*. Harper & Row. [periodicity in operant series]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class IRTAutocorrelation:
    """Autocorrelation function of an IRT series.

    Attributes
    ----------
    lags : tuple of int
        Lag values (0, 1, ..., max_lag).
    values : tuple of float
        ``rho(k)`` at each lag, with ``rho(0) == 1.0``.
    n : int
        Number of IRTs used.
    """

    lags: tuple[int, ...]
    values: tuple[float, ...]
    n: int


@dataclass(frozen=True)
class IRTSpectrum:
    """Periodogram-based power spectral density of an IRT series.

    Attributes
    ----------
    frequencies : tuple of float
        Frequencies in cycles per IRT-step (or, if ``sampling_rate``
        is supplied, cycles per second).
    power : tuple of float
        Raw periodogram ``|X_k|^2 / n`` of the mean-centred series at each
        non-negative frequency: two-sided and not divided by the sampling
        rate. ``scipy.signal.periodogram(x, fs, scaling="density")`` (one
        sided) equals ``2 * power / fs`` at interior frequencies and
        ``power / fs`` at 0 and, for even ``n``, at ``fs / 2``.
    peak_frequency : float
        Frequency of maximum power (excluding the DC component).
    n : int
    sampling_rate : float | None
    """

    frequencies: tuple[float, ...]
    power: tuple[float, ...]
    peak_frequency: float
    n: int
    sampling_rate: float | None


def compute_irt_autocorrelation(
    irts: Sequence[float],
    *,
    max_lag: int,
) -> IRTAutocorrelation:
    """Compute the biased autocorrelation function of an IRT series.

    Parameters
    ----------
    irts : sequence of float
        IRT series (positive values).
    max_lag : int
        Maximum lag (inclusive). Must satisfy ``1 <= max_lag <= n - 1``.

    Returns
    -------
    IRTAutocorrelation
    """
    if len(irts) < 3:
        raise ValueError(
            f"autocorrelation requires at least 3 IRTs, got {len(irts)}"
        )
    if max_lag < 1 or max_lag > len(irts) - 1:
        raise ValueError(
            f"max_lag must be in [1, {len(irts) - 1}], got {max_lag}"
        )

    arr = np.array(irts, dtype=np.float64)
    n = arr.size
    centered = arr - float(np.mean(arr))
    var0 = float(np.dot(centered, centered) / n)
    if var0 == 0.0:
        raise ValueError("autocorrelation undefined for constant series")

    lags = list(range(0, max_lag + 1))
    rhos: list[float] = [1.0]
    for k in range(1, max_lag + 1):
        cov = float(np.dot(centered[: n - k], centered[k:]) / n)
        rhos.append(cov / var0)

    return IRTAutocorrelation(
        lags=tuple(lags),
        values=tuple(rhos),
        n=int(n),
    )


def compute_irt_spectrum(
    irts: Sequence[float],
    *,
    sampling_rate: float | None = None,
) -> IRTSpectrum:
    """Compute the periodogram of an IRT series.

    Mean-centers the input, applies an unwindowed FFT, returns the
    one-sided spectrum (frequencies in ``[0, fs/2]`` for even ``n`` or
    ``[0, fs/2)`` for odd ``n``).

    Parameters
    ----------
    irts : sequence of float
        IRT series.
    sampling_rate : float, optional
        Samples per second. When omitted, frequencies are reported in
        cycles per IRT-step.

    Returns
    -------
    IRTSpectrum
    """
    if len(irts) < 4:
        raise ValueError(
            f"spectrum requires at least 4 IRTs, got {len(irts)}"
        )
    if sampling_rate is not None and sampling_rate <= 0.0:
        raise ValueError(
            f"sampling_rate must be positive when supplied, got {sampling_rate}"
        )

    arr = np.array(irts, dtype=np.float64)
    n = arr.size
    centered = arr - float(np.mean(arr))
    fft_vals = np.fft.rfft(centered)
    power = (np.abs(fft_vals) ** 2) / n
    fs = sampling_rate if sampling_rate is not None else 1.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    if power.size > 1:
        peak_idx = int(np.argmax(power[1:])) + 1
    else:
        peak_idx = 0
    peak_freq = float(freqs[peak_idx])

    return IRTSpectrum(
        frequencies=tuple(float(f) for f in freqs),
        power=tuple(float(p) for p in power),
        peak_frequency=peak_freq,
        n=int(n),
        sampling_rate=sampling_rate,
    )
