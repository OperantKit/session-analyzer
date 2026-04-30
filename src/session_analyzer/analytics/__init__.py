"""Analytics modules: bout decomposition, matching-law fitting, choice
allocation, response latency, progressive-ratio breakpoint detection,
delay discounting, demand-curve fitting, single-alternative hyperbola,
and resistance-to-change (behavioral momentum)."""

from .bout import BoutMetrics, analyze_bouts
from .breakpoint import BreakpointResult, detect_breakpoint
from .log_survivor import (
    LogSurvivorBilinearFit,
    LogSurvivorCurve,
    compute_log_survivor,
    fit_bilinear_log_survivor,
)
from .renewal import (
    PhaseSummary,
    RenewalMetrics,
    analyze_renewal,
    classify_renewal_design,
)
from .resurgence import ResurgenceMetrics, analyze_resurgence
from .signal_detection import (
    ROCCurve,
    SignalDetectionMetrics,
    compute_roc,
    compute_sdt,
)
from .stimulus_control import (
    DiscriminationRatio,
    PeakShiftResult,
    compute_discrimination_ratio,
    detect_peak_shift,
)
from .indifference import IndifferencePoint, estimate_indifference_point
from .lag_variability import (
    LagVariabilityMetrics,
    analyze_lag_variability,
    compute_lag_pass_rate,
)
from .observing import (
    ConditionedReinforcementStrength,
    ObservingMetrics,
    compute_conditioned_reinforcement_strength,
    compute_observing_metrics,
)
from .time_matching import (
    TimeMatchingParameters,
    estimate_time_matching_parameters,
)
from .unit_price import (
    CrossPriceElasticity,
    UnitPriceElasticity,
    compute_cross_price_elasticity,
    compute_unit_prices,
    fit_unit_price_elasticity,
)
from .within_session import (
    WithinSessionDynamics,
    analyze_within_session,
    bin_response_rates,
)
from .choice import (
    ChoiceAllocation,
    ChoiceMode,
    ExclusiveChoiceResult,
    classify_choice_mode,
    compute_choice_allocation,
    detect_exclusive_choice,
)
from .demand import DemandParameters, fit_exponential_demand
from .discounting import (
    HyperbolicDiscountingParameters,
    HyperboloidDiscountingParameters,
    fit_hyperbolic_discounting,
    fit_hyperboloid_discounting,
)
from .generalization import (
    ExponentialGradientParameters,
    GaussianGradientParameters,
    fit_exponential_gradient,
    fit_gaussian_gradient,
)
from .latency import LatencyMetrics, analyze_latencies, compute_latencies
from .matching import (
    MatchingParameters,
    SingleAlternativeMatchingParameters,
    estimate_matching_parameters,
    fit_single_alternative_matching,
)
from .momentum import MomentumParameters, estimate_resistance_to_change
from .peak import (
    PeakTrialBinnedMetrics,
    PeakTrialMLEMetrics,
    analyze_peak_trial_binned,
    analyze_peak_trial_mle,
)
from .concurrent_chains import (
    ConcurrentChainsPrediction,
    predict_concurrent_chains_choice,
)
from .prp import PRPMetrics, analyze_prps, compute_prps, compute_quarter_life
from .run_length import RunMetrics, analyze_fr_runs
from .survival import SurvivalCurve, kaplan_meier
from .tracking import TrackingIndex, TrackingType, compute_tracking_index
from .variability import VariabilityMetrics, compute_u_value

__all__ = [
    "BoutMetrics",
    "BreakpointResult",
    "ChoiceAllocation",
    "ChoiceMode",
    "ConcurrentChainsPrediction",
    "ConditionedReinforcementStrength",
    "CrossPriceElasticity",
    "DemandParameters",
    "DiscriminationRatio",
    "ExclusiveChoiceResult",
    "ExponentialGradientParameters",
    "GaussianGradientParameters",
    "HyperbolicDiscountingParameters",
    "HyperboloidDiscountingParameters",
    "IndifferencePoint",
    "LagVariabilityMetrics",
    "LatencyMetrics",
    "LogSurvivorBilinearFit",
    "LogSurvivorCurve",
    "MatchingParameters",
    "MomentumParameters",
    "ObservingMetrics",
    "PRPMetrics",
    "PeakShiftResult",
    "PeakTrialBinnedMetrics",
    "PeakTrialMLEMetrics",
    "PhaseSummary",
    "ROCCurve",
    "RenewalMetrics",
    "ResurgenceMetrics",
    "RunMetrics",
    "SignalDetectionMetrics",
    "SingleAlternativeMatchingParameters",
    "SurvivalCurve",
    "TimeMatchingParameters",
    "TrackingIndex",
    "TrackingType",
    "UnitPriceElasticity",
    "VariabilityMetrics",
    "WithinSessionDynamics",
    "analyze_bouts",
    "analyze_fr_runs",
    "analyze_lag_variability",
    "analyze_latencies",
    "analyze_peak_trial_binned",
    "analyze_peak_trial_mle",
    "analyze_prps",
    "analyze_renewal",
    "analyze_resurgence",
    "analyze_within_session",
    "bin_response_rates",
    "classify_choice_mode",
    "classify_renewal_design",
    "compute_choice_allocation",
    "compute_conditioned_reinforcement_strength",
    "compute_cross_price_elasticity",
    "compute_discrimination_ratio",
    "compute_lag_pass_rate",
    "compute_latencies",
    "compute_log_survivor",
    "compute_observing_metrics",
    "compute_prps",
    "compute_quarter_life",
    "compute_roc",
    "compute_sdt",
    "compute_tracking_index",
    "compute_u_value",
    "compute_unit_prices",
    "detect_breakpoint",
    "detect_exclusive_choice",
    "detect_peak_shift",
    "estimate_indifference_point",
    "estimate_matching_parameters",
    "estimate_resistance_to_change",
    "estimate_time_matching_parameters",
    "fit_bilinear_log_survivor",
    "fit_exponential_demand",
    "fit_exponential_gradient",
    "fit_gaussian_gradient",
    "fit_hyperbolic_discounting",
    "fit_hyperboloid_discounting",
    "fit_single_alternative_matching",
    "fit_unit_price_elasticity",
    "kaplan_meier",
    "predict_concurrent_chains_choice",
]
