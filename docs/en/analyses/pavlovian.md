# Pavlovian

[← Catalog index](index.md) · [日本語版](../../ja/analyses/pavlovian.md)

Conditioned-stimulus measures: how subjects allocate approach between
the CS and the goal location, and how strongly a stimulus has become
a conditioned reinforcer.

---

### `tracking_index` — `TrackingIndex`

- **Theme** — In lever-CS paradigms, individual rats diverge into "sign-trackers" (approach the CS lever) and "goal-trackers" (approach the food cup). The PCA tracking index averages response-bias, latency, and probability components into `[-1, +1]`.
- **Output** — `TrackingIndex(response_bias=0.42, latency_score=0.35, probability_score=0.28, index=0.35, classification=TrackingType.SIGN_TRACKER)`
- **Reference** — Meyer, P. J., et al. (2012). Quantifying individual variation in the propensity to attribute incentive salience to reward cues. *PLoS ONE*, 7(6), e38987. <https://doi.org/10.1371/journal.pone.0038987>

![Distribution of PCA tracking indices across subjects](../../assets/figures/tracking_index.png)

### Observing response — `ObservingMetrics`

- **Theme** — In an *observing-response* procedure (Wyckoff, 1952), a non-required response produces stimuli that signal whether the upcoming schedule period is reinforcement (S+) or extinction (S-). Observing rate is taken as an index of conditioned-reinforcer strength of the discriminative stimuli. Per Dinsmoor (1983), observing is maintained primarily by S+ presentations — `s_plus_proportion ≥ 0.6` is the conventional good-news threshold.
- **Output** — `ObservingMetrics(observing_rate_per_min=12.0, s_plus_proportion=0.67, good_news_dominance=True, observing_count=120, s_plus_productions=80, s_minus_productions=40, observing_duration_sec=600.0)`
- **References** — Wyckoff, L. B. (1952). The role of observing responses in discrimination learning. *Psychological Review*, 59(6), 431-442. <https://doi.org/10.1037/h0053932> — Dinsmoor, J. A. (1983). Observing and conditioned reinforcement. *Behavioral and Brain Sciences*, 6(4), 693-704. <https://doi.org/10.1017/S0140525X00017969>

![Total observing rate decomposed into S+ vs S- production](../../assets/figures/observing.png)

### Conditioned-reinforcement strength — `ConditionedReinforcementStrength`

- **Theme** — Quantifies the conditioned-reinforcer function of a stimulus as the ratio of observing rates between a fully correlated condition (S+ / S- predict reinforcement / extinction) and a yoked or uncorrelated control. Ratios `≥ 2` are heuristically taken as evidence the stimulus has acquired conditioned-reinforcer function.
- **Output** — `ConditionedReinforcementStrength(strength_index=6.0, observing_rate_correlated=12.0, observing_rate_uncorrelated=2.0, is_conditioned_reinforcer=True)`
- **Reference** — Dinsmoor, J. A. (1983). Observing and conditioned reinforcement. *Behavioral and Brain Sciences*, 6(4), 693-704. <https://doi.org/10.1017/S0140525X00017969>

### Rescorla-Wagner — `RescorlaWagnerParameters`

- **Theme** — Per-trial associative-strength update for a single CS: `ΔV_t = α·β·(λ_t - V_t)`. `simulate_rescorla_wagner` produces the trace under given parameters; `fit_rescorla_wagner` recovers the combined learning-rate `α·β` and asymptote `λ` from observed trace data via Nelder-Mead. The figure shows acquisition-extinction-reacquisition with a fitted parameter overlay.
- **Equation** — `V_{t+1} = V_t + α·β·(λ_t - V_t)`
- **Output** — `RescorlaWagnerParameters(alpha_beta=0.25, lambda_=0.80, initial_v=0.0, sse=0.001, r_squared=0.999, n_trials=60)`
- **Reference** — Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning. In *Classical conditioning II* (pp. 64-99). Appleton-Century-Crofts.

![Rescorla-Wagner trace across acquisition / extinction / reacquisition phases](../../assets/figures/rescorla_wagner.png)
