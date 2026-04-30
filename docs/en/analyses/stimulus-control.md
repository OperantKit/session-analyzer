# Stimulus control

[← Catalog index](index.md) · [日本語版](../../ja/analyses/stimulus-control.md)

Discriminative-stimulus control: generalization gradients, S+ vs S-
discrimination, peak-shift after intermixed training, and
signal-detection sensitivity / bias.

---

### Gaussian gradient — `GaussianGradientParameters`

- **Theme** — Discriminative-stimulus generalization gradients often have symmetric, bell-shaped tails around the training stimulus. A Gaussian fit summarizes width (`sigma`) and peak response.
- **Equation** — `R(d) = R0 · exp(-d² / (2σ²))`
- **Output** — `GaussianGradientParameters(peak_response=48.3, sigma=120.0, r_squared=0.98, n=9)`
- **Reference** — Guttman, N., & Kalish, H. I. (1956). *J. Exp. Psychol.*, 51(1), 79-88. <https://doi.org/10.1037/h0046219>

![Symmetric Gaussian generalization gradient](../../assets/figures/generalization_gradient.png)

### Exponential gradient — `ExponentialGradientParameters`

- **Theme** — Shepard (1987) argued the "universal" generalization law is exponential: `R(d) = R0 · exp(-d / λ)`. Use when fits to Gaussian leave systematic heavy-tail residuals.
- **Output** — `ExponentialGradientParameters(peak_response=48.3, length_scale=175.0, r_squared=0.97, n=9)`
- **Reference** — Shepard, R. N. (1987). Toward a universal law of generalization for psychological science. *Science*, 237(4820), 1317-1323. <https://doi.org/10.1126/science.3629243>

### Discrimination ratio — `DiscriminationRatio`

- **Theme** — In an S+/S- discrimination, the simplest summary is the proportion of responding emitted during S+: `DR = R(S+) / [R(S+) + R(S-)]`. Equivalently, the *suppression ratio* (Annau & Kamin, 1961) is `R(S-) / [R(S+) + R(S-)]`. `DR ≥ 0.75` is the conventional heuristic for "discriminated."
- **Output** — `DiscriminationRatio(discrimination_ratio=0.85, suppression_ratio=0.15, s_plus_count=170, s_minus_count=30, is_discriminated=True)`
- **Reference** — Annau, Z., & Kamin, L. J. (1961). The conditioned emotional response as a function of intensity of the US. *J. Comp. Physiol. Psychol.*, 54(4), 428-432. <https://doi.org/10.1037/h0042199>

### Peak shift — `PeakShiftResult`

- **Theme** — After intermixed S+ / S- training, the empirical peak of the generalization gradient typically shifts *away from* S-, often beyond S+ itself (Hanson, 1959). The function fits a free-peak Gaussian and reports the signed shift relative to S+ along the (S+ − S-) direction.
- **Equation** — `R(x) = R0 · exp(-(x - μ)² / (2σ²))` with free `μ`
- **Output** — `PeakShiftResult(empirical_peak_value=560.0, fitted_peak_value=559.7, fitted_sigma=12.4, fitted_peak_response=49.8, s_plus=550.0, s_minus=540.0, shift_magnitude=9.7, shift_direction="away_from_s_minus", r_squared=0.99, n=8)`
- **Reference** — Hanson, H. M. (1959). Effects of discrimination training on stimulus generalization. *J. Exp. Psychol.*, 58(5), 321-334. <https://doi.org/10.1037/h0042606>

### Signal detection — `SignalDetectionMetrics` / `ROCCurve`

- **Theme** — When a yes/no detection task confounds sensitivity with response bias, signal detection theory (Green & Swets, 1966) decomposes performance into `d'` (sensitivity) and `c` / `β` (bias). Non-parametric `A'` and `B''` give bias-resistant alternatives. `compute_roc` constructs the empirical ROC curve and trapezoidal AUC from a series of operating points. The Hautus (1995) log-linear correction is always applied to handle ceiling/floor cells.
- **Equation** — `d' = z(H) − z(F)`; `c = -0.5(z(H) + z(F))`; `β = exp(c · d')`
- **Output (sdt)** — `SignalDetectionMetrics(hit_rate=0.84, false_alarm_rate=0.16, d_prime=2.0, criterion=0.0, beta=1.0, a_prime=0.92, b_double_prime=0.0, n_signal_trials=100, n_noise_trials=100)`
- **Output (roc)** — `ROCCurve(fa_rates=[0.0, 0.01, 0.10, 0.30, 0.55, 0.90, 1.0], hit_rates=[0.0, 0.05, 0.30, 0.60, 0.85, 0.99, 1.0], auc=0.85)`
- **References** — Green, D. M., & Swets, J. A. (1966). *Signal detection theory and psychophysics*. Wiley. — Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on estimated values of d'. *Behav. Res. Methods Instrum. Comput.*, 27(1), 46-51. <https://doi.org/10.3758/BF03203619>
