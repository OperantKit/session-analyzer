# Reference values

[日本語版](../ja/reference-values.md)

How each analysis in `session_analyzer.analytics` is checked against its source. 46 analytics modules: 6 match published values, 5 match an independent reference implementation, 1 matches the source's formulas, 7 have published values whose text was not retrieved and 27 have no published values; those 34 are checked by closed forms, recovery of known parameters and hand computation.

Each row is a test marked `@pytest.mark.reference(module=..., verdict=..., source=...)`. `tests/test_reference_value_coverage.py` fails when an analytics module has no such test, when a `properties` or `not-retrieved` module is missing from its allowlist with a reason, or when this table disagrees with the marks.

Verdicts:

- `published`: equals values printed in the source or in a reference package's published output.
- `reference-implementation`: equals an independent package run on the same data.
- `definition`: equals the source's formulas (the source has no numeric example).
- `not-retrieved`: the source prints values, but its text was not retrieved (scanned or paywalled); checked by properties meanwhile.
- `properties`: no published values exist; checked by closed forms, recovery of known parameters and hand computation.

| Module | Verdict | Source | Reference implementation | What the test checks |
|---|---|---|---|---|
| `ddm` | `published` | Wagenmakers, van der Maas & Grasman (2007) | EZ-diffusion (R, appendix) | appendix output v 0.09993853, a 0.1399702, Ter 0.30003 |
| `demand` | `published` | Hursh & Silberberg (2008); Kaplan et al. (2019) | beezdemand (R) | beezdemand README output: Q0 10.158665 (`fit_space="log10"`), Pmaxd 8.453799, Omaxd 22.63159 |
| `pr_schedule` | `published` | Richardson & Roberts (1996) | — | ratio series steps 1-10 equal the published series (secondary protocol); 11-20 unchecked |
| `signal_detection` | `published` | Stanislaw & Todorov (1999) | — | Figure 1: d′ 2.00, c −0.50, ln β −1.00, A′ .89, B″ −0.55 |
| `single_case` | `published` | Parker, Vannest & Davis (2011) | SingleCaseES (R), scan (R) | NAP, PND, Tau-U equal SingleCaseES / scan published output |
| `visual_analysis` | `published` | Kratochwill et al. (2010) | scan (R) | trend equals scan's `trend()` output (scan-Book ch. 8) |
| `discounting` | `reference-implementation` | Myerson & Green (1995) | discountingtools (R) | recovers k from data made with the discountingtools generator |
| `hierarchical_gml` | `reference-implementation` | Gelman & Hill (2007) | statsmodels `MixedLM` | estimates, log-likelihood and fixed-effect SEs equal statsmodels (ML) |
| `hmm_segment` | `reference-implementation` | Rabiner (1989) | hmmlearn `GaussianHMM` | estimates, log-likelihood and Viterbi path equal hmmlearn |
| `psychometric` | `reference-implementation` | Wichmann & Hill (2001) | statsmodels binomial GLM | μ, σ and log-likelihood equal statsmodels (γ = 0, lapse = 0) |
| `survival` | `reference-implementation` | Kaplan & Meier (1958) | lifelines `KaplanMeierFitter` | Freireich 6-MP arm: S(t) and median 23 equal lifelines |
| `tracking` | `definition` | Meyer et al. (2012) | — | formulas and ±0.5 cut-offs of Table 1 (no numeric example) |
| `generalization` | `not-retrieved` | Guttman & Kalish (1956) | — | paywalled; equals an independent profiled least-squares fit |
| `killeen_law` | `not-retrieved` | Killeen (1994) | — | fitted-parameter table paywalled; β and γ identifiable |
| `matching` | `not-retrieved` | Baum (1974) | `numpy.polyfit` | Table 1 not retrieved; fit equals ordinary least squares, single-observation case by hand |
| `momentum` | `not-retrieved` | Nevin (1992) | `numpy.linalg.lstsq` | published values not retrieved; recovers a known m |
| `peak` | `not-retrieved` | Church, Meck & Gibbon (1994) | — | paywalled; recovers known low-high-low trials |
| `schneider_segment` | `not-retrieved` | Schneider (1969) | — | breakpoint table not retrieved; recovers an exact two-line record |
| `time_matching` | `not-retrieved` | Baum & Rachlin (1969) | `numpy.polyfit` | data table not retrieved; equals least squares and `matching` |
| `bout` | `properties` | Shull, Gaynor & Grimes (2001) | — | recovers a known two-component mixture (estimator differs from the original log-survivor method) |
| `breakpoint` | `properties` | Richardson & Roberts (1996) | — | last completed ratio on the published ratio series, threshold included |
| `choice` | `properties` | Herrnstein (1961) | — | proportions and thresholds by hand |
| `concurrent_chains` | `properties` | Mazur (2001) | — | V = A/(1 + kD) closed form |
| `cross_elasticity_matrix` | `properties` | Hursh & Roma (2013) | `numpy.polyfit` | recovers partial elasticities; agrees with `unit_price` |
| `extinction_burst` | `properties` | Lerman & Iwata (1995) | — | threshold and window boundary by hand |
| `indifference` | `properties` | Mazur (1987) | — | stability window by hand, titration converges |
| `irt_spectrum` | `properties` | Wei (2006) | `numpy.correlate`, `scipy.signal.periodogram` | equals NumPy / SciPy standard functions |
| `lag_variability` | `properties` | Page & Neuringer (1985) | — | Lag-N pass rate by hand |
| `latency` | `properties` | none cited | Python `statistics` | pairing and cut-off by hand; summary equals `statistics` |
| `laws_of_covariance` | `properties` | Podlesnik & Baum (2024) | — | closed form (the point prediction is this module's own) |
| `log_survivor` | `properties` | Shull (1991) | — | Poisson IRTs give a line with slope −rate |
| `matching_induction` | `properties` | Baum & Grace (2020) | — | induction term is tanh(ln r / 2); parameters recovered (the form is this module's own) |
| `ngram_entropy` | `properties` | Towse & Neil (1998) | `scipy.stats.entropy` | de Bruijn sequence gives H_n = n bits |
| `observing` | `properties` | Dinsmoor (1983) | — | rates and ratios by hand |
| `prp` | `properties` | Fry, Kelleher & Cook (1960) | — | quarter-life equals 0.25^(1/p) for power cumulative records |
| `renewal` | `properties` | Bouton (2004) | — | all context designs and ratios by hand |
| `rescorla_wagner` | `properties` | Rescorla & Wagner (1972) | — | V_n = λ(1 − (1 − αβ)^n) |
| `resurgence` | `properties` | Epstein (1985) | — | windowed means, differences and ratios by hand |
| `run_length` | `properties` | Schlinger, Derenne & Baron (2008) | — | pause and run by hand |
| `scalar_timing` | `properties` | Gibbon (1977) | — | coefficient of variation by hand |
| `stay_switch` | `properties` | Stubbs & Pliskoff (1969) | — | changeovers and COD compliance by hand |
| `stimulus_control` | `properties` | Hanson (1959) | — | suppression ratio and peak shift by hand |
| `td_learning` | `properties` | Sutton (1988) | — | converges to the discounted return (closed form) |
| `unit_price` | `properties` | Hursh (1980) | `numpy.polyfit` | unit-price equivalence, elasticity recovery, equals least squares |
| `variability` | `properties` | Page & Neuringer (1985) | `scipy.stats.entropy` | U = 1 / 0 / 2/3 closed forms |
| `within_session` | `properties` | McSweeney, Roll & Cannon (1994) | — | recovers a known inverted U |

Full references for each source are in the [analysis catalog](analyses/index.md) and in each module's docstring.
