"""Renewal analysis (ABA / ABC / AAB).

Renewal is the recovery of an extinguished response when the test
context differs from the extinction context (Bouton, 2004). The three
canonical designs:

- **ABA renewal**: train in context A, extinguish in B, test in A.
  Recovery driven by return to the original training context.
- **ABC renewal**: train in A, extinguish in B, test in a novel C.
  Recovery driven by leaving the extinction context.
- **AAB renewal**: train and extinguish in A, test in B.
  Recovery driven by leaving the extinction context (variant).

This module summarises a 3-phase session, classifies the design from
context labels, and computes renewal magnitude as the deviation of the
test response rate from the asymptotic extinction rate.

The phase-and-context substrate is provided by ``contingency-dsl``
(`spec/en/experiment/context.md`); a ``session-recorder`` OKL v1 log
emits ``phase_enter``/``phase_exit`` records that downstream callers
turn into the :class:`PhaseSummary` list this analyzer consumes.

References
----------
Bouton, M. E. (2004). Context and behavioral processes in extinction.
*Learning & Memory*, 11(5), 485-494.
https://doi.org/10.1101/lm.78804

Bouton, M. E., & Bolles, R. C. (1979). Contextual control of the
extinction of conditioned fear. *Learning and Motivation*, 10(4),
445-466. https://doi.org/10.1016/0023-9690(79)90057-2
"""

from __future__ import annotations

from dataclasses import dataclass

_RenewalContexts = tuple[str, str, str]


@dataclass(frozen=True)
class PhaseSummary:
    """One-phase summary used as input to :func:`analyze_renewal`.

    Attributes
    ----------
    label : str
        Human-readable phase label (``"Acquisition"``, ``"Extinction"``,
        ``"Test"``, ...).
    context : str or None
        Context identifier (typically ``"A"``, ``"B"``, ``"C"``, or a
        descriptive name). ``None`` is permitted but coerced to the
        sentinel ``"_"`` so all-None designs collapse to ``AAA``.
    response_count : int
        Total target-response count emitted within the phase.
    duration_sec : float
        Phase duration (seconds). Must be > 0.
    """

    label: str
    context: str | None
    response_count: int
    duration_sec: float

    @property
    def rate(self) -> float:
        """Response rate in responses / second."""
        if self.duration_sec <= 0.0:
            raise ValueError("duration_sec must be > 0")
        return self.response_count / self.duration_sec


@dataclass(frozen=True)
class RenewalMetrics:
    """Renewal analysis result.

    Attributes
    ----------
    design : {"ABA", "ABC", "AAB", "AAA", "OTHER"}
        Canonical design code, or ``"OTHER"`` if the (train, ext, test)
        context tuple does not match any canonical pattern.
    contexts : tuple[str, str, str]
        The (train, extinction, test) context tuple.
    train_rate : float
    extinction_terminal_rate : float
        Response rate during the extinction phase (assumed to be the
        asymptote of suppression).
    test_rate : float
    renewal_magnitude : float
        ``test_rate - extinction_terminal_rate``.
    renewal_ratio : float
        ``test_rate / train_rate`` if ``train_rate > 0`` else 0.
    renewed : bool
        ``True`` if ``renewal_magnitude > 0`` and ``renewal_ratio`` is
        at least the threshold (default 10 % of baseline).
    """

    design: str
    contexts: _RenewalContexts
    train_rate: float
    extinction_terminal_rate: float
    test_rate: float
    renewal_magnitude: float
    renewal_ratio: float
    renewed: bool


def classify_renewal_design(contexts: _RenewalContexts) -> str:
    """Classify a (train, extinction, test) context tuple."""
    train, ext, test = contexts
    if train == ext == test:
        return "AAA"
    if train == test and train != ext:
        return "ABA"
    if train == ext and ext != test:
        return "AAB"
    if train != ext and ext != test and train != test:
        return "ABC"
    return "OTHER"


def analyze_renewal(
    phases: list[PhaseSummary],
    *,
    threshold: float = 0.10,
) -> RenewalMetrics:
    """Quantify renewal from three phases (train, extinction, test).

    Parameters
    ----------
    phases : list of PhaseSummary
        Exactly three phases in chronological order.
    threshold : float
        Fraction of baseline that the test rate must exceed for
        ``renewed`` to be ``True`` (default 0.10).
    """
    if len(phases) != 3:
        raise ValueError(
            f"renewal analysis requires three phases (train, extinction, test); "
            f"got {len(phases)}"
        )
    for p in phases:
        if p.duration_sec <= 0.0:
            raise ValueError(
                f"phase {p.label!r} has non-positive duration {p.duration_sec}"
            )

    train, extinct, test = phases
    contexts: _RenewalContexts = (
        train.context if train.context is not None else "_",
        extinct.context if extinct.context is not None else "_",
        test.context if test.context is not None else "_",
    )
    design = classify_renewal_design(contexts)

    train_rate = train.rate
    ext_rate = extinct.rate
    test_rate = test.rate
    magnitude = test_rate - ext_rate
    ratio = test_rate / train_rate if train_rate > 0.0 else 0.0
    renewed = magnitude > 0.0 and ratio >= threshold

    return RenewalMetrics(
        design=design,
        contexts=contexts,
        train_rate=round(train_rate, 6),
        extinction_terminal_rate=round(ext_rate, 6),
        test_rate=round(test_rate, 6),
        renewal_magnitude=round(magnitude, 6),
        renewal_ratio=round(ratio, 6),
        renewed=renewed,
    )
