"""Analytics modules: bout decomposition, matching-law fitting, choice
allocation, response latency, progressive-ratio breakpoint detection,
delay discounting, demand-curve fitting, single-alternative hyperbola,
and resistance-to-change (behavioral momentum)."""

from .bout import BoutMetrics, analyze_bouts
from .breakpoint import (
    BreakpointResult,
    IRTBreakpointResult,
    IRTBreakpointStatus,
    detect_breakpoint,
    estimate_irt_breakpoint,
)
from .choice import (
    ChoiceAllocation,
    ChoiceMode,
    ExclusiveChoiceResult,
    classify_choice_mode,
    compute_choice_allocation,
    detect_exclusive_choice,
)
from .concurrent_chains import (
    ConcurrentChainsPrediction,
    predict_concurrent_chains_choice,
)
from .cross_elasticity_matrix import (
    CrossElasticityMatrix,
    compute_cross_elasticity_matrix,
)
from .ddm import EZDiffusionParameters, fit_ez_diffusion
from .extinction_burst import ExtinctionBurstResult, detect_extinction_burst
from .hierarchical_gml import HierarchicalGMLFit, fit_hierarchical_gml
from .hmm_segment import GaussianHMMFit, HMMSegment, fit_gaussian_hmm
from .irt_spectrum import (
    IRTAutocorrelation,
    IRTSpectrum,
    compute_irt_autocorrelation,
    compute_irt_spectrum,
)
from .killeen_law import KilleenLawParameters, fit_killeen_law
from .ngram_entropy import (
    BlockEntropy,
    ConditionalEntropy,
    MarkovOrderEstimate,
    compute_block_entropy,
    compute_conditional_entropy,
    estimate_markov_order,
)
from .psychometric import PsychometricFit, fit_psychometric
from .rescorla_wagner import (
    RescorlaWagnerParameters,
    fit_rescorla_wagner,
    simulate_rescorla_wagner,
)
from .scalar_timing import (
    ScalarPropertyTest,
    ScalarTimingResult,
    analyze_scalar_timing,
    assess_scalar_property,
)
from .schneider_segment import SchneiderTwoStateResult, schneider_two_state
from .stay_switch import (
    CODComplianceResult,
    StaySwitchMetrics,
    analyze_stay_switch,
    check_cod_compliance,
)
from .td_learning import TDLearningTrace, simulate_td_zero
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
from .indifference import IndifferencePoint, estimate_indifference_point
from .lag_variability import (
    LagVariabilityMetrics,
    analyze_lag_variability,
    compute_lag_pass_rate,
)
from .latency import LatencyMetrics, analyze_latencies, compute_latencies
from .log_survivor import (
    LogSurvivorBilinearFit,
    LogSurvivorCurve,
    compute_log_survivor,
    fit_bilinear_log_survivor,
)
from .laws_of_covariance import (
    LawsOfCovariancePrediction,
    predict_resurgence_laws_of_covariance,
)
from .matching import (
    MatchingParameters,
    SingleAlternativeMatchingParameters,
    estimate_matching_parameters,
    estimate_single_observation_gml,
    fit_single_alternative_matching,
)
from .matching_induction import (
    MatchingInductionParameters,
    fit_matching_with_induction,
    predict_log_ratio_with_induction,
)
from .momentum import MomentumParameters, estimate_resistance_to_change
from .observing import (
    ConditionedReinforcementStrength,
    ObservingMetrics,
    compute_conditioned_reinforcement_strength,
    compute_observing_metrics,
)
from .peak import (
    PeakTrialBinnedMetrics,
    PeakTrialMLEMetrics,
    analyze_peak_trial_binned,
    analyze_peak_trial_mle,
)
from .pr_schedule import (
    PRClassification,
    StepType,
    classify_pr_schedule,
    extract_step_ratios,
)
from .prp import PRPMetrics, analyze_prps, compute_prps, compute_quarter_life
from .renewal import (
    PhaseSummary,
    RenewalMetrics,
    analyze_renewal,
    classify_renewal_design,
)
from .resurgence import ResurgenceMetrics, analyze_resurgence
from .run_length import RunMetrics, analyze_fr_runs
from .signal_detection import (
    ROCCurve,
    SignalDetectionMetrics,
    compute_roc,
    compute_sdt,
)
from .single_case import (
    NonOverlapResult,
    SingleCaseSeries,
    TauURow,
    TauUTable,
    nap,
    nonoverlap,
    pnd,
    select_phases,
    tau,
    tau_u,
    tau_u_table,
)
from .stimulus_control import (
    DiscriminationRatio,
    PeakShiftResult,
    compute_discrimination_ratio,
    detect_peak_shift,
)
from .survival import SurvivalCurve, kaplan_meier
from .time_matching import (
    TimeMatchingParameters,
    estimate_time_matching_parameters,
)
from .tracking import TrackingIndex, TrackingType, compute_tracking_index
from .unit_price import (
    CrossPriceElasticity,
    UnitPriceElasticity,
    compute_cross_price_elasticity,
    compute_unit_prices,
    fit_unit_price_elasticity,
)
from .variability import VariabilityMetrics, compute_u_value
from .within_session import (
    WithinSessionDynamics,
    analyze_within_session,
    bin_response_rates,
)

__all__ = [
    "BlockEntropy",
    "BoutMetrics",
    "BreakpointResult",
    "CODComplianceResult",
    "ChoiceAllocation",
    "ChoiceMode",
    "ConcurrentChainsPrediction",
    "ConditionalEntropy",
    "ConditionedReinforcementStrength",
    "CrossElasticityMatrix",
    "CrossPriceElasticity",
    "DemandParameters",
    "DiscriminationRatio",
    "EZDiffusionParameters",
    "ExclusiveChoiceResult",
    "ExponentialGradientParameters",
    "ExtinctionBurstResult",
    "GaussianGradientParameters",
    "GaussianHMMFit",
    "HMMSegment",
    "HierarchicalGMLFit",
    "HyperbolicDiscountingParameters",
    "HyperboloidDiscountingParameters",
    "IRTAutocorrelation",
    "IRTBreakpointResult",
    "IRTBreakpointStatus",
    "IRTSpectrum",
    "IndifferencePoint",
    "KilleenLawParameters",
    "LagVariabilityMetrics",
    "LatencyMetrics",
    "LogSurvivorBilinearFit",
    "LogSurvivorCurve",
    "MarkovOrderEstimate",
    "MatchingParameters",
    "MomentumParameters",
    "ObservingMetrics",
    "PRClassification",
    "PRPMetrics",
    "PeakShiftResult",
    "PeakTrialBinnedMetrics",
    "PeakTrialMLEMetrics",
    "PhaseSummary",
    "PsychometricFit",
    "ROCCurve",
    "RenewalMetrics",
    "RescorlaWagnerParameters",
    "ResurgenceMetrics",
    "RunMetrics",
    "ScalarPropertyTest",
    "ScalarTimingResult",
    "SchneiderTwoStateResult",
    "SignalDetectionMetrics",
    "SingleAlternativeMatchingParameters",
    "SingleCaseSeries",
    "StaySwitchMetrics",
    "TauURow",
    "TauUTable",
    "NonOverlapResult",
    "nap",
    "nonoverlap",
    "pnd",
    "select_phases",
    "tau",
    "tau_u",
    "tau_u_table",
    "StepType",
    "LawsOfCovariancePrediction",
    "MatchingInductionParameters",
    "SurvivalCurve",
    "TDLearningTrace",
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
    "analyze_scalar_timing",
    "analyze_stay_switch",
    "analyze_within_session",
    "assess_scalar_property",
    "bin_response_rates",
    "check_cod_compliance",
    "compute_block_entropy",
    "compute_conditional_entropy",
    "compute_cross_elasticity_matrix",
    "compute_irt_autocorrelation",
    "compute_irt_spectrum",
    "classify_choice_mode",
    "classify_pr_schedule",
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
    "detect_extinction_burst",
    "detect_peak_shift",
    "estimate_markov_order",
    "estimate_indifference_point",
    "estimate_irt_breakpoint",
    "estimate_matching_parameters",
    "estimate_resistance_to_change",
    "estimate_single_observation_gml",
    "estimate_time_matching_parameters",
    "extract_step_ratios",
    "fit_bilinear_log_survivor",
    "fit_ez_diffusion",
    "fit_exponential_demand",
    "fit_exponential_gradient",
    "fit_gaussian_gradient",
    "fit_gaussian_hmm",
    "fit_hierarchical_gml",
    "fit_hyperbolic_discounting",
    "fit_hyperboloid_discounting",
    "fit_killeen_law",
    "fit_matching_with_induction",
    "fit_psychometric",
    "fit_rescorla_wagner",
    "fit_single_alternative_matching",
    "fit_unit_price_elasticity",
    "kaplan_meier",
    "predict_concurrent_chains_choice",
    "predict_log_ratio_with_induction",
    "predict_resurgence_laws_of_covariance",
    "schneider_two_state",
    "simulate_rescorla_wagner",
    "simulate_td_zero",
]
