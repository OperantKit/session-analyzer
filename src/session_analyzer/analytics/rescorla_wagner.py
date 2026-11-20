"""Rescorla-Wagner (1972) associative-strength simulation and fitting.

On each trial, for every stimulus ``i`` in the presented set ``S``:

    Delta V_i = alpha_i * beta * (lambda_t - sum_{j in S} V_j)

where ``alpha_i`` is the salience of stimulus ``i``, ``beta`` is US salience,
``lambda_t`` is the asymptotic strength supported by the US on trial ``t``
(positive on reinforced trials, zero on trials without the US), and the sum
is the associative strength of the presented compound entering the trial.
Stimuli not in ``S`` are not updated.

``simulate_compound_rescorla_wagner`` runs this over compound trials.
Associative strength is kept separately per US label (one system per US):
a trial delivering US ``u`` drives the ``u`` system toward
``lambda_by_us[u]`` and every other system toward 0. A US label without an
asymptote in ``lambda_by_us`` counts as a trial without a US.

``simulate_rescorla_wagner`` and ``fit_rescorla_wagner`` are the single-CS
case: one stimulus, one US, and ``alpha * beta`` as one combined rate.

Reference
---------
Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian
    conditioning: Variations in the effectiveness of reinforcement and
    nonreinforcement. In A. H. Black & W. F. Prokasy (Eds.),
    *Classical conditioning II: Current research and theory*
    (pp. 64-99). Appleton-Century-Crofts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class RescorlaWagnerParameters:
    """Fit result for the single-CS Rescorla-Wagner model.

    Attributes
    ----------
    alpha_beta : float
        Combined learning rate ``alpha * beta`` (these enter the update
        only as a product, so they are not separately identifiable from
        a single CS without independent manipulation).
    lambda_ : float
        Asymptotic associative strength supported by the US on
        reinforced trials.
    initial_v : float
        Associative strength assumed at the start of the simulated
        trace.
    sse : float
        Sum of squared errors between simulated and observed trace.
    r_squared : float
        Coefficient of determination.
    n_trials : int
        Number of trials used.
    """

    alpha_beta: float
    lambda_: float
    initial_v: float
    sse: float
    r_squared: float
    n_trials: int


def simulate_rescorla_wagner(
    us_present: Sequence[bool],
    *,
    alpha_beta: float,
    lambda_: float,
    initial_v: float = 0.0,
) -> list[float]:
    """Simulate associative strength over trials under a single CS.

    Returns the post-trial ``V`` value for each trial. ``V`` is updated
    by ``alpha_beta * (lambda - V)`` on reinforced trials and by
    ``alpha_beta * (0 - V)`` on extinction trials. This is the one-stimulus
    case of :func:`simulate_compound_rescorla_wagner` with ``alpha =
    alpha_beta`` and ``beta = 1`` and gives the same values; it is kept as a
    separate loop because it does not record every system on every trial.
    """
    if not 0.0 < alpha_beta <= 1.0:
        raise ValueError(
            f"alpha_beta must lie in (0, 1], got {alpha_beta}"
        )
    v = float(initial_v)
    trace: list[float] = []
    for present in us_present:
        target = lambda_ if present else 0.0
        v = v + alpha_beta * (target - v)
        trace.append(v)
    return trace


def _simulate_array(
    us_present: np.ndarray, alpha_beta: float, lambda_: float, initial_v: float
) -> np.ndarray:
    v = initial_v
    out = np.empty(us_present.size, dtype=np.float64)
    for i in range(us_present.size):
        target = lambda_ if us_present[i] else 0.0
        v = v + alpha_beta * (target - v)
        out[i] = v
    return out


def fit_rescorla_wagner(
    observed: Sequence[float],
    us_present: Sequence[bool],
    *,
    initial_v: float = 0.0,
) -> RescorlaWagnerParameters:
    """Fit ``alpha_beta`` and ``lambda_`` to an observed trace.

    Parameters
    ----------
    observed : sequence of float
        Observed associative strength (or any monotone proxy) at the
        end of each trial.
    us_present : sequence of bool
        Whether the US occurred on each trial.
    initial_v : float
        Initial associative strength prior to trial 0.

    Returns
    -------
    RescorlaWagnerParameters

    Raises
    ------
    ValueError
        On length mismatch or fewer than 2 trials.
    """
    if len(observed) != len(us_present):
        raise ValueError(
            "observed and us_present must have the same length "
            f"(got {len(observed)} and {len(us_present)})"
        )
    if len(observed) < 2:
        raise ValueError(
            f"Rescorla-Wagner fitting requires at least 2 trials, got {len(observed)}"
        )

    obs = np.array(observed, dtype=np.float64)
    us = np.array(us_present, dtype=bool)

    def negative_log_loss(params: np.ndarray) -> float:
        ab, lam = params
        if not (0.0 < ab <= 1.0):
            return 1e12
        if lam <= 0.0:
            return 1e12
        predicted = _simulate_array(us, float(ab), float(lam), float(initial_v))
        return float(np.sum((predicted - obs) ** 2))

    init_lam = float(np.max(obs)) if obs.size else 1.0
    init_lam = max(init_lam, 1e-3)
    result = minimize(
        negative_log_loss,
        x0=np.array([0.3, init_lam]),
        method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 5000},
    )
    ab_fit, lam_fit = float(result.x[0]), float(result.x[1])
    predicted = _simulate_array(us, ab_fit, lam_fit, float(initial_v))
    sse = float(np.sum((predicted - obs) ** 2))
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0.0 else 0.0
    return RescorlaWagnerParameters(
        alpha_beta=ab_fit,
        lambda_=lam_fit,
        initial_v=float(initial_v),
        sse=sse,
        r_squared=float(r2),
        n_trials=len(obs),
    )


# ---------------------------------------------------------------------------
# Compound CS
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM = ""
"""System name used when no trial delivers a US that has an asymptote."""


def _frozen(m: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(m))


def _check_rate(name: str, value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1], got {value}")


@dataclass(frozen=True)
class CompoundTrial:
    """One trial of a compound-CS simulation.

    Attributes
    ----------
    stimuli : frozenset of str
        Stimuli presented on the trial (the CS set, plus the context when
        the context is treated as an element).
    us : str or None
        Label of the US delivered on the trial, or ``None``.
    stage : str
        Name of the experimental stage the trial belongs to.
    """

    stimuli: frozenset[str]
    us: str | None = None
    stage: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.stimuli, str):
            raise TypeError("stimuli must be a set of stimulus names, not a string")
        object.__setattr__(self, "stimuli", frozenset(self.stimuli))


@dataclass(frozen=True, init=False)
class CompoundRescorlaWagner:
    """Parameters of the compound-CS Rescorla-Wagner model.

    ``alpha_by_cs`` gives per-stimulus salience; stimuli not listed use
    ``alpha_default``. ``lambda_by_us`` gives the asymptote of each US
    label. ``beta_nonreinforced`` is the US-salience term on trials
    without the US (``beta`` when ``None``). ``v0`` is the strength of
    every stimulus before the first trial.
    """

    alpha_by_cs: Mapping[str, float]
    lambda_by_us: Mapping[str, float]
    alpha_default: float = 0.3
    beta: float = 1.0
    beta_nonreinforced: float | None = None
    v0: float = 0.0

    def __init__(
        self,
        alpha_by_cs: Mapping[str, float],
        *,
        lambda_by_us: Mapping[str, float],
        alpha_default: float = 0.3,
        beta: float = 1.0,
        beta_nonreinforced: float | None = None,
        v0: float = 0.0,
    ) -> None:
        for name, a in alpha_by_cs.items():
            _check_rate(f"alpha for {name!r}", a)
        _check_rate("alpha_default", alpha_default)
        _check_rate("beta", beta)
        if beta_nonreinforced is not None:
            _check_rate("beta_nonreinforced", beta_nonreinforced)
        object.__setattr__(self, "alpha_by_cs", _frozen(alpha_by_cs))
        object.__setattr__(self, "lambda_by_us", _frozen(lambda_by_us))
        object.__setattr__(self, "alpha_default", float(alpha_default))
        object.__setattr__(self, "beta", float(beta))
        object.__setattr__(
            self,
            "beta_nonreinforced",
            None if beta_nonreinforced is None else float(beta_nonreinforced),
        )
        object.__setattr__(self, "v0", float(v0))

    def alpha(self, stimulus: str) -> float:
        """Salience of ``stimulus``."""
        return self.alpha_by_cs.get(stimulus, self.alpha_default)

    def asymptote(self, us: str | None) -> float | None:
        """``lambda`` of ``us``; ``None`` when the label has no asymptote."""
        if us is None:
            return None
        return self.lambda_by_us.get(us)


@dataclass(frozen=True)
class CompoundTrialRecord:
    """Associative strengths before and after one trial.

    ``v_before`` / ``v_after`` are the primary system (the first US label
    of ``lambda_by_us`` delivered in the run); ``*_by_us`` hold every
    system.
    """

    index: int
    trial: CompoundTrial
    v_before_by_us: Mapping[str, Mapping[str, float]]
    v_after_by_us: Mapping[str, Mapping[str, float]]
    primary: str

    @property
    def stage(self) -> str:
        return self.trial.stage

    @property
    def us(self) -> str | None:
        return self.trial.us

    @property
    def elements(self) -> frozenset[str]:
        return self.trial.stimuli

    @property
    def v_before(self) -> dict[str, float]:
        return dict(self.v_before_by_us[self.primary])

    @property
    def v_after(self) -> dict[str, float]:
        return dict(self.v_after_by_us[self.primary])

    @property
    def sum_v_before(self) -> float:
        """Summed strength of the presented stimuli entering the trial."""
        before = self.v_before_by_us[self.primary]
        return sum(before[s] for s in self.elements)


@dataclass(frozen=True)
class CompoundTrace:
    """Trial-by-trial associative strengths of one compound-CS run."""

    records: tuple[CompoundTrialRecord, ...]
    model: CompoundRescorlaWagner
    systems: tuple[str, ...] = field(default=())

    def v_at_stage_start(self, stage: str, *, us: str | None = None) -> dict[str, float]:
        """Strengths entering the first trial of ``stage``."""
        for r in self.records:
            if r.stage == stage:
                return dict(r.v_before_by_us[us if us is not None else r.primary])
        raise KeyError(f"stage {stage!r} not in trace")

    def final(self, *, us: str | None = None) -> dict[str, float]:
        """Strengths after the last trial (empty when there are no trials)."""
        if not self.records:
            return {}
        last = self.records[-1]
        return dict(last.v_after_by_us[us if us is not None else last.primary])


def _systems(trials: Sequence[CompoundTrial], model: CompoundRescorlaWagner) -> tuple[str, ...]:
    used = {t.us for t in trials if t.us is not None}
    systems = tuple(u for u in model.lambda_by_us if u in used)
    return systems or (DEFAULT_SYSTEM,)


def simulate_compound_rescorla_wagner(
    trials: Iterable[CompoundTrial],
    model: CompoundRescorlaWagner,
) -> CompoundTrace:
    """Run the compound-CS Rescorla-Wagner model over ``trials``.

    Every stimulus that appears in any trial starts at ``model.v0`` in
    every system. On each trial the presented stimuli of each system are
    updated by ``alpha_i * beta * (lambda - sum V)``, with ``lambda`` and
    ``beta`` of the reinforced system for the delivered US and ``0`` /
    ``beta_nonreinforced`` for the other systems.
    """
    trial_list = tuple(trials)
    stimuli: dict[str, None] = {}
    for t in trial_list:
        for s in sorted(t.stimuli):
            stimuli.setdefault(s, None)
    systems = _systems(trial_list, model)
    v: dict[str, dict[str, float]] = {u: dict.fromkeys(stimuli, model.v0) for u in systems}
    beta_off = model.beta if model.beta_nonreinforced is None else model.beta_nonreinforced
    records: list[CompoundTrialRecord] = []
    for index, t in enumerate(trial_list):
        elements = t.stimuli
        before = {u: _frozen(vs) for u, vs in v.items()}
        reinforced = t.us if model.asymptote(t.us) is not None else None
        nxt: dict[str, dict[str, float]] = {}
        for u, vs in v.items():
            if u == reinforced:
                lam, beta = model.lambda_by_us[u], model.beta
            else:
                lam, beta = 0.0, beta_off
            error = lam - sum(vs[s] for s in elements)
            nxt[u] = {
                s: (x + model.alpha(s) * beta * error if s in elements else x)
                for s, x in vs.items()
            }
        v = nxt
        records.append(
            CompoundTrialRecord(
                index=index,
                trial=t,
                v_before_by_us=MappingProxyType(before),
                v_after_by_us=MappingProxyType({u: _frozen(vs) for u, vs in v.items()}),
                primary=systems[0],
            )
        )
    return CompoundTrace(tuple(records), model, systems)
