"""Aversive control: free-operant avoidance and punishment summaries.

Both functions take event times in seconds rather than a session log, because
the log format does not separate shocks from positive reinforcers. Rates are
responses (or shocks) per minute.

Avoidance follows the free-operant (Sidman) procedure: a shock every S-S
seconds unless a response postpones it by R-S seconds. The summary reports the
shock rate, the response rate, the share of time bins without a shock, the
ratio of responses to shocks, and, when the S-S interval is given, the share
of the shocks the schedule would have delivered without responding that were
avoided.

Punishment compares the response rate in a punished phase with the rate in an
unpunished baseline, and optionally a recovery phase after punishment is
removed. The suppression ratio is punished rate / baseline rate: 1 means no
suppression and 0 means complete suppression. The rate-ratio form follows the
suppression ratios of conditioned suppression (Lyon, 1963); the comparison of
a punished phase with its unpunished baseline follows Azrin (1960).

References:
    Azrin, N. H. (1960). Effects of punishment intensity during
    variable-interval reinforcement. Journal of the Experimental Analysis of
    Behavior, 3(2), 123-142. https://doi.org/10.1901/jeab.1960.3-123

    Clark, F. C., & Hull, L. D. (1966). Free operant avoidance as a function
    of the response-shock = shock-shock interval. Journal of the Experimental
    Analysis of Behavior, 9(6), 641-647. https://doi.org/10.1901/jeab.1966.9-641

    Lyon, D. O. (1963). Frequency of reinforcement as a parameter of
    conditioned suppression. Journal of the Experimental Analysis of
    Behavior, 6(1), 95-98. https://doi.org/10.1901/jeab.1963.6-95

    Sidman, M. (1953). Two temporal parameters of the maintenance of avoidance
    behavior by the white rat. Journal of Comparative and Physiological
    Psychology, 46(4), 253-261. https://doi.org/10.1037/h0060730
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "AvoidanceMetrics",
    "PunishmentMetrics",
    "analyze_avoidance",
    "analyze_punishment",
]


@dataclass(frozen=True)
class AvoidanceMetrics:
    """Summary of one free-operant avoidance session.

    Attributes:
        shock_rate: shocks per minute.
        response_rate: responses per minute.
        shock_free_fraction: share of ``bin_width`` bins of ``[0, duration)``
            that hold no shock (the trailing partial bin counts as a bin).
        responses_per_shock: responses / shocks, the ratio of responses
            emitted to shocks received (Clark & Hull, 1966); ``inf`` when
            no shock was received.
        shocks_avoided_fraction: ``1 - shocks / floor(duration / S-S)``,
            clipped at 0; ``None`` when no S-S interval was given.
        n_shocks: number of shocks.
        n_responses: number of responses.
        duration: session length in seconds.
    """

    shock_rate: float
    response_rate: float
    shock_free_fraction: float
    responses_per_shock: float
    shocks_avoided_fraction: float | None
    n_shocks: int
    n_responses: int
    duration: float


@dataclass(frozen=True)
class PunishmentMetrics:
    """Summary of a baseline, punished and optional recovery phase.

    Attributes:
        baseline_rate: responses per minute in the unpunished baseline.
        punished_rate: responses per minute under punishment.
        suppression_ratio: ``punished_rate / baseline_rate``; ``nan`` when the
            baseline rate is 0.
        recovery_rate: responses per minute after punishment is removed;
            ``None`` without a recovery phase.
        recovery_ratio: ``recovery_rate / baseline_rate``; ``None`` without a
            recovery phase, ``nan`` when the baseline rate is 0.
        n_punishers: number of punishers delivered (0 when not given).
    """

    baseline_rate: float
    punished_rate: float
    suppression_ratio: float
    recovery_rate: float | None
    recovery_ratio: float | None
    n_punishers: int


def _check_duration(duration: float, name: str) -> float:
    value = float(duration)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number of seconds, got {duration!r}")
    return value


def _check_times(times: Sequence[float], duration: float, name: str) -> list[float]:
    values = [float(t) for t in times]
    for t in values:
        if not math.isfinite(t) or t < 0.0 or t > duration:
            raise ValueError(f"{name} must lie in [0, {duration}], got {t!r}")
    return values


def _per_minute(count: int, duration: float) -> float:
    return count * 60.0 / duration


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else math.nan


def analyze_avoidance(
    response_times: Sequence[float],
    shock_times: Sequence[float],
    duration: float,
    *,
    shock_shock_interval: float | None = None,
    bin_width: float = 60.0,
) -> AvoidanceMetrics:
    """Summarize one free-operant avoidance session.

    Args:
        response_times: response times in seconds, within ``[0, duration]``.
        shock_times: shock times in seconds, within ``[0, duration]``.
        duration: session length in seconds.
        shock_shock_interval: the S-S interval in seconds; when given, the
            avoided share is computed against ``floor(duration / S-S)``.
        bin_width: bin length in seconds for ``shock_free_fraction``.

    Raises:
        ValueError: on a non-positive duration, interval or bin width, or a
            time outside the session.
    """
    total = _check_duration(duration, "duration")
    width = _check_duration(bin_width, "bin_width")
    responses = _check_times(response_times, total, "response_times")
    shocks = _check_times(shock_times, total, "shock_times")

    n_bins = math.ceil(total / width)
    shocked_bins = {min(int(t // width), n_bins - 1) for t in shocks}

    avoided: float | None = None
    if shock_shock_interval is not None:
        ss = _check_duration(shock_shock_interval, "shock_shock_interval")
        possible = math.floor(total / ss)
        avoided = max(0.0, 1.0 - len(shocks) / possible) if possible > 0 else math.nan

    return AvoidanceMetrics(
        shock_rate=_per_minute(len(shocks), total),
        response_rate=_per_minute(len(responses), total),
        shock_free_fraction=1.0 - len(shocked_bins) / n_bins,
        responses_per_shock=len(responses) / len(shocks) if shocks else math.inf,
        shocks_avoided_fraction=avoided,
        n_shocks=len(shocks),
        n_responses=len(responses),
        duration=total,
    )


def analyze_punishment(
    baseline_response_times: Sequence[float],
    punished_response_times: Sequence[float],
    baseline_duration: float,
    punished_duration: float,
    *,
    recovery_response_times: Sequence[float] | None = None,
    recovery_duration: float | None = None,
    punisher_times: Sequence[float] | None = None,
) -> PunishmentMetrics:
    """Compare response rates under punishment with an unpunished baseline.

    Each phase's times are measured from that phase's own start.

    Args:
        baseline_response_times: response times in the baseline phase.
        punished_response_times: response times in the punished phase.
        baseline_duration: baseline length in seconds.
        punished_duration: punished-phase length in seconds.
        recovery_response_times: response times after punishment is removed;
            must be given together with ``recovery_duration``.
        recovery_duration: recovery-phase length in seconds.
        punisher_times: punisher times within the punished phase.

    Raises:
        ValueError: on a non-positive duration, a time outside its phase, or
            only one of the two recovery arguments.
    """
    base_dur = _check_duration(baseline_duration, "baseline_duration")
    pun_dur = _check_duration(punished_duration, "punished_duration")
    baseline = _check_times(baseline_response_times, base_dur, "baseline_response_times")
    punished = _check_times(punished_response_times, pun_dur, "punished_response_times")
    punishers: list[float] = []
    if punisher_times is not None:
        punishers = _check_times(punisher_times, pun_dur, "punisher_times")

    if (recovery_response_times is None) != (recovery_duration is None):
        raise ValueError("recovery_response_times and recovery_duration must be given together")

    baseline_rate = _per_minute(len(baseline), base_dur)
    punished_rate = _per_minute(len(punished), pun_dur)

    recovery_rate: float | None = None
    recovery_ratio: float | None = None
    if recovery_response_times is not None and recovery_duration is not None:
        rec_dur = _check_duration(recovery_duration, "recovery_duration")
        recovery = _check_times(recovery_response_times, rec_dur, "recovery_response_times")
        recovery_rate = _per_minute(len(recovery), rec_dur)
        recovery_ratio = _ratio(recovery_rate, baseline_rate)

    return PunishmentMetrics(
        baseline_rate=baseline_rate,
        punished_rate=punished_rate,
        suppression_ratio=_ratio(punished_rate, baseline_rate),
        recovery_rate=recovery_rate,
        recovery_ratio=recovery_ratio,
        n_punishers=len(punishers),
    )
