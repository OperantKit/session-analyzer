# Temporal

[← Catalog index](index.md) · [日本語版](../../ja/analyses/temporal.md)

Time-of-response analyses: how responding tracks the temporal
structure of the schedule and how it evolves across the session.

---

### `peak_procedure_binned` — `PeakTrialBinnedMetrics`

- **Theme** — On peak trials (no reinforcer), response rate rises, peaks near the expected time, and falls — the "internal clock" signature. Binned rate analysis yields peak time, peak rate, and spread.
- **Output** — `PeakTrialBinnedMetrics(peak_time=30.2, peak_rate=1.24, start_time=18.7, stop_time=42.8, spread=24.1, middle_time=30.75, n_responses=58, bin_width=1.0)`
- **Reference** — Roberts, S. (1981). Isolation of an internal clock. *J. Exp. Psychol.: ABP*, 7(3), 242-268. <https://doi.org/10.1037/0097-7403.7.3.242>

![Peak-trial rate curve with peak and spread markers](../../assets/figures/peak_procedure.png)

### `peak_procedure_mle` — `PeakTrialMLEMetrics` (three-state MLE, **heavy tier**)

- **Theme** — Church, Meck, & Gibbon (1994) model the peak trial as three Poisson states — low, high, low — and fit two switch times `(s1, s2)` by MLE. This gives trial-level transition times rather than grand-averaged bins.
- **Output** — `PeakTrialMLEMetrics(start_time=20.1, stop_time=41.5, high_rate=1.82, low_rate=0.21, middle_time=30.8, spread=21.4, n_responses=58, log_likelihood=-132.7)`
- **Reference** — Church, R. M., Meck, W. H., & Gibbon, J. (1994). Application of scalar timing theory to individual trials. *J. Exp. Psychol.: ABP*, 20(2), 135-155. <https://doi.org/10.1037/0097-7403.20.2.135>

### Within-session dynamics — `WithinSessionDynamics`

- **Theme** — Response rate within a session typically follows an inverted-U (McSweeney, Roll, & Cannon, 1994): a brief warm-up rise, a peak, and a satiation-driven decline. The bin trajectory and the rise/fall slopes characterize this trajectory and let the user separate motivational depletion from rate-control proper.
- **Output** — `WithinSessionDynamics(rates=[0.5, 1.4, 2.1, 2.4, 2.0, 1.5, 1.0, 0.7], peak_bin=3, peak_rate=2.4, rise_slope=0.62, fall_slope=-0.34, bilinear_r_squared=0.96, n_bins=8, n_responses=240)`
- **Reference** — McSweeney, F. K., Roll, J. M., & Cannon, C. B. (1994). The generality of within-session patterns of responding. *Animal Learning & Behavior*, 22(3), 252-266. <https://doi.org/10.3758/BF03209833>
