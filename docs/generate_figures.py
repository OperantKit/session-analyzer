"""Regenerate all figures embedded in docs/{en,ja}/analyses.md.

Each figure is rendered with the ``jeab-bw`` theme (JEAB / JABA
monochrome, 3.3 x 2.5 in, 300 dpi Helvetica) so the exported PNGs are
suitable for direct paste into paper submissions. The synthetic data
are parameter-recovery caricatures — deliberately well-behaved so the
resulting panels look like canonical textbook figures, not like any
one experiment.

Output is PNG (not SVG) so the embedded images render consistently
across local Markdown previewers, GitHub's blob view, and the VS
Code Markdown preview. PNG at 300 dpi is already the dpi the ``jeab-bw``
theme targets; vector export is one ``format="svg"`` tweak away when
a journal specifically asks for it.

Run::

    .venv/bin/python docs/generate_figures.py

Outputs to ``docs/assets/figures/*.png``. No network / state; safe to
run in CI.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from session_analyzer.analytics import (  # noqa: E402
    analyze_bouts,
    analyze_fr_runs,
    analyze_latencies,
    analyze_peak_trial_binned,
    analyze_prps,
    analyze_stay_switch,
    compute_block_entropy,
    compute_choice_allocation,
    compute_cross_elasticity_matrix,
    compute_irt_autocorrelation,
    compute_irt_spectrum,
    compute_log_survivor,
    compute_roc,
    compute_sdt,
    compute_tracking_index,
    compute_u_value,
    detect_breakpoint,
    detect_extinction_burst,
    detect_peak_shift,
    estimate_indifference_point,
    estimate_matching_parameters,
    estimate_resistance_to_change,
    estimate_time_matching_parameters,
    fit_bilinear_log_survivor,
    fit_exponential_demand,
    fit_exponential_gradient,
    fit_ez_diffusion,
    fit_gaussian_gradient,
    fit_gaussian_hmm,
    fit_hierarchical_gml,
    fit_hyperbolic_discounting,
    fit_killeen_law,
    fit_psychometric,
    fit_rescorla_wagner,
    fit_single_alternative_matching,
    kaplan_meier,
    predict_concurrent_chains_choice,
    schneider_two_state,
    simulate_rescorla_wagner,
    simulate_td_zero,
)
from session_analyzer.visualizer.cumulative_record import render_cumulative_record  # noqa: E402
from session_analyzer.visualizer.themes import get_theme  # noqa: E402
from session_analyzer.visualizer.themes.mpl_adapter import (  # noqa: E402
    apply_theme_to_axes,
    theme_to_rcparams,
)

OUT = Path(__file__).parent / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
THEME = get_theme("jeab-bw")
RNG = np.random.default_rng(42)


def _new_fig(height_ratio: float = 1.0):
    plt.rcParams.update(theme_to_rcparams(THEME))
    fig, ax = plt.subplots(
        figsize=(THEME.figure_width_in, THEME.figure_height_in * height_ratio)
    )
    return fig, ax


def _save(fig, name: str) -> None:
    apply_theme_to_axes(fig.axes[0], THEME)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", format="png", dpi=THEME.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png")


# ---------------------------------------------------------------------------
# 1. Cumulative record
# ---------------------------------------------------------------------------


def fig_cumulative_record() -> None:
    # Simulate a VI-60 session with break-and-run pattern.
    rng = np.random.default_rng(1)
    times = []
    t = 0.0
    while t < 600:
        t += rng.exponential(2.0)  # baseline IRT
        times.append(t)
    times = np.array(times)
    # Reinforcers at every VI-60 interval
    reinforcers = []
    next_r = rng.exponential(60)
    for rt in times:
        if rt >= next_r:
            reinforcers.append(rt)
            next_r = rt + rng.exponential(60)
    render_cumulative_record(
        response_times=tuple(times.tolist()),
        reinforcement_times=tuple(reinforcers),
        output=OUT / "cumulative_record.png",
        theme=THEME,
        fmt="png",
    )
    print("  wrote cumulative_record.png")


# ---------------------------------------------------------------------------
# 2. Post-reinforcement pause
# ---------------------------------------------------------------------------


def fig_prp() -> None:
    rng = np.random.default_rng(2)
    prps = rng.lognormal(mean=np.log(4.0), sigma=0.35, size=40)
    metrics = analyze_prps(prps.tolist())

    fig, ax = _new_fig()
    ax.hist(prps, bins=12, color="#555555", edgecolor="black")
    ax.axvline(metrics.mean, color="black", linestyle="--", linewidth=1.0, label=f"mean = {metrics.mean:.2f}s")
    ax.set_xlabel("Post-reinforcement pause (s)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", frameon=False)
    _save(fig, "post_reinforcement_pause")


# ---------------------------------------------------------------------------
# 3. IRT distribution (bout decomposition)
# ---------------------------------------------------------------------------


def fig_irt_bouts() -> None:
    rng = np.random.default_rng(3)
    short = rng.lognormal(mean=np.log(0.3), sigma=0.4, size=800)
    long = rng.lognormal(mean=np.log(25.0), sigma=0.7, size=100)
    irts = np.concatenate([short, long])
    # Convert IRTs to response times
    resp_times = np.cumsum(irts)
    _ = analyze_bouts(resp_times.tolist())

    fig, ax = _new_fig()
    logs = np.log10(irts[irts > 0])
    ax.hist(logs, bins=40, color="#777777", edgecolor="black")
    ax.axvline(np.log10(0.3), color="black", linestyle=":", linewidth=1.0, label="within-bout")
    ax.axvline(np.log10(25.0), color="black", linestyle="--", linewidth=1.0, label="between-bout")
    ax.set_xlabel("log10 IRT (s)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", frameon=False)
    _save(fig, "irt_distribution")


# ---------------------------------------------------------------------------
# 4. FR run length
# ---------------------------------------------------------------------------


def fig_run_length() -> None:
    rng = np.random.default_rng(4)
    ratio = 20
    responses = []
    reinforcers = []
    t = 0.0
    for _ in range(30):
        # Pause
        t += rng.lognormal(mean=np.log(3.0), sigma=0.3)
        # Run
        for _ in range(ratio):
            t += rng.lognormal(mean=np.log(0.09), sigma=0.2)
            responses.append(t)
        reinforcers.append(t)
    metrics = analyze_fr_runs(responses, reinforcers)

    fig, ax = _new_fig()
    # Show a short slice of cumulative record (first 5 runs)
    sliced = [t for t in responses if t <= reinforcers[4] + 2]
    ax.step(sliced, np.arange(1, len(sliced) + 1), color="black", linewidth=1.0, where="post")
    for r in reinforcers[:5]:
        ax.axvline(r, color="black", linestyle=":", linewidth=0.75)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative responses")
    if metrics is not None:
        ax.text(
            0.02, 0.98,
            f"pause/run = {metrics.pause_to_run_ratio:.2f}\nmean run = {metrics.mean_run_duration:.2f}s",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
        )
    _save(fig, "run_length")


# ---------------------------------------------------------------------------
# 5. Latency
# ---------------------------------------------------------------------------


def fig_latency() -> None:
    rng = np.random.default_rng(5)
    latencies = rng.lognormal(mean=np.log(1.8), sigma=0.45, size=96)
    metrics = analyze_latencies(latencies.tolist())

    fig, ax = _new_fig()
    ax.hist(latencies, bins=18, color="#555555", edgecolor="black")
    ax.axvline(metrics.median, color="black", linestyle="--", linewidth=1.0, label=f"median = {metrics.median:.2f}s")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", frameon=False)
    _save(fig, "latency")


# ---------------------------------------------------------------------------
# 6. Matching law (GML)
# ---------------------------------------------------------------------------


def fig_matching_law() -> None:
    rng = np.random.default_rng(6)
    # Six points on log-log axis with true slope 0.84
    log_r_ratio = np.linspace(-1.0, 1.0, 6)
    log_b_ratio = 0.84 * log_r_ratio + rng.normal(0, 0.05, 6)
    r_ratios = 10 ** log_r_ratio
    b_ratios = 10 ** log_b_ratio
    params = estimate_matching_parameters(b_ratios.tolist(), r_ratios.tolist())

    fig, ax = _new_fig()
    xs = np.linspace(-1.2, 1.2, 50)
    ax.plot(xs, params.sensitivity * xs + params.bias, color="black", linewidth=1.0,
            label=f"a={params.sensitivity:.2f}, log b={params.bias:.2f}")
    ax.plot(log_r_ratio, log_b_ratio, "o", color="black", markersize=4, markerfacecolor="white")
    ax.plot(xs, xs, color="black", linewidth=0.5, linestyle=":", label="strict matching")
    ax.set_xlabel("log(R1 / R2)")
    ax.set_ylabel("log(B1 / B2)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    _save(fig, "matching_law")


# ---------------------------------------------------------------------------
# 7. Single-alternative matching (Herrnstein hyperbola)
# ---------------------------------------------------------------------------


def fig_single_alternative_matching() -> None:
    rng = np.random.default_rng(7)
    k_true, re_true = 80.0, 25.0
    R = np.array([5, 10, 20, 40, 80, 160, 320])
    B = k_true * R / (R + re_true) + rng.normal(0, 2.0, len(R))
    params = fit_single_alternative_matching(B.tolist(), R.tolist())

    fig, ax = _new_fig()
    Rfit = np.linspace(0, 400, 200)
    Bfit = params.k * Rfit / (Rfit + params.r_e)
    ax.plot(Rfit, Bfit, color="black", linewidth=1.0,
            label=f"k={params.k:.1f}, R_e={params.r_e:.1f}")
    ax.plot(R, B, "o", markersize=4, markerfacecolor="white", color="black")
    ax.axhline(params.k, color="black", linestyle=":", linewidth=0.75)
    ax.set_xlabel("Reinforcement rate R (per h)")
    ax.set_ylabel("Response rate B (per min)")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    _save(fig, "single_alternative_matching")


# ---------------------------------------------------------------------------
# 8. Choice allocation
# ---------------------------------------------------------------------------


def fig_choice_allocation() -> None:
    counts = [248, 152]
    alloc = compute_choice_allocation(counts)

    fig, ax = _new_fig(height_ratio=0.9)
    positions = np.arange(2)
    ax.bar(positions, alloc.proportions, color=["#333333", "#aaaaaa"], edgecolor="black")
    for i, (p, c) in enumerate(zip(alloc.proportions, alloc.counts)):
        ax.text(i, p + 0.02, f"{p:.2f}\n(n={c})", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Left", "Right"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion of responses")
    _save(fig, "choice_allocation")


# ---------------------------------------------------------------------------
# 9. Breakpoint (progressive ratio)
# ---------------------------------------------------------------------------


def fig_breakpoint() -> None:
    # Simulate a PR session: steps 1, 2, 4, 6, 9, 12, 16, 20, 25, ...
    ratios = [1, 2, 4, 6, 9, 12, 16, 20, 25, 31, 38, 46]
    rng = np.random.default_rng(8)
    t = 0.0
    responses = []
    reinforcers = []
    for i, r in enumerate(ratios):
        # Post-reinforcement pause increases with ratio
        pause = 2.0 + 0.5 * r + rng.normal(0, 1.0)
        t += max(pause, 0.5)
        # Run phase
        for _ in range(r):
            t += rng.lognormal(mean=np.log(0.4), sigma=0.2)
            responses.append(t)
        reinforcers.append(t)
        if i == 9:  # stop after ratio 31
            break
    # Append a long terminal pause
    t += 400
    session_end = t
    completed = ratios[: len(reinforcers)]
    completions = list(zip(completed, reinforcers))
    result = detect_breakpoint(
        completions,
        session_end_time=session_end,
        pause_threshold_sec=300.0,
    )

    fig, ax = _new_fig()
    ax.plot(range(1, len(completed) + 1), completed, "o-", color="black",
            markersize=4, markerfacecolor="white", linewidth=1.0)
    if result.breakpoint_ratio is not None:
        ax.axhline(result.breakpoint_ratio, color="black", linestyle="--", linewidth=1.0,
                   label=f"breakpoint = {result.breakpoint_ratio}")
    ax.set_xlabel("Completion number")
    ax.set_ylabel("Ratio requirement")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    _save(fig, "breakpoint")


# ---------------------------------------------------------------------------
# 10. Demand curve (Hursh-Silberberg)
# ---------------------------------------------------------------------------


def fig_demand_curve() -> None:
    rng = np.random.default_rng(9)
    q0_true, alpha_true, k_true = 60.0, 0.0025, 2.9
    prices = np.array([1, 3, 5, 10, 18, 30, 56, 100, 180])
    q = q0_true * 10 ** (k_true * (np.exp(-alpha_true * q0_true * prices) - 1))
    q *= rng.normal(1.0, 0.05, len(prices))
    params = fit_exponential_demand(prices.tolist(), q.tolist())

    fig, ax = _new_fig()
    pfit = np.logspace(0, np.log10(300), 100)
    qfit = params.q0 * 10 ** (params.k * (np.exp(-params.alpha * params.q0 * pfit) - 1))
    ax.loglog(pfit, qfit, color="black", linewidth=1.0,
              label=f"α={params.alpha:.4f}")
    ax.loglog(prices, q, "o", markersize=4, markerfacecolor="white", color="black")
    ax.axvline(params.p_max, color="black", linestyle=":", linewidth=0.75,
               label=f"P_max={params.p_max:.1f}")
    ax.set_xlabel("Unit price")
    ax.set_ylabel("Consumption Q")
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    _save(fig, "demand_curve")


# ---------------------------------------------------------------------------
# 11. Delay discounting (hyperbolic)
# ---------------------------------------------------------------------------


def fig_discounting() -> None:
    rng = np.random.default_rng(10)
    k_true = 0.03
    delays = np.array([0, 2, 7, 30, 90, 180, 365])
    values = 100 / (1 + k_true * delays) + rng.normal(0, 1.5, len(delays))
    params = fit_hyperbolic_discounting(delays.tolist(), values.tolist(), amount=100.0)

    fig, ax = _new_fig()
    dfit = np.linspace(0, 400, 200)
    vfit = 100 / (1 + params.k * dfit)
    ax.plot(dfit, vfit, color="black", linewidth=1.0, label=f"k={params.k:.4f}")
    ax.plot(delays, values, "o", markersize=4, markerfacecolor="white", color="black")
    ax.set_xlabel("Delay (days)")
    ax.set_ylabel("Subjective value")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "delay_discounting")


# ---------------------------------------------------------------------------
# 12. Peak procedure (binned)
# ---------------------------------------------------------------------------


def fig_peak_procedure() -> None:
    rng = np.random.default_rng(11)
    mu, sigma = 30.0, 6.0
    trial_duration = 90.0
    # Generate response times biased toward mu
    ts = []
    for _ in range(60):
        t = rng.normal(mu, sigma)
        if 0 < t < trial_duration:
            ts.append(t)
    ts = np.sort(ts)
    metrics = analyze_peak_trial_binned(ts.tolist(), trial_duration=trial_duration, bin_width=2.0)

    # Re-bin for the figure
    edges = np.arange(0, trial_duration + 2, 2)
    hist, _ = np.histogram(ts, bins=edges)
    rates = hist / 2.0

    fig, ax = _new_fig()
    ax.plot(edges[:-1] + 1.0, rates, color="black", linewidth=1.0)
    ax.axvline(metrics.peak_time, color="black", linestyle="--", linewidth=1.0,
               label=f"peak = {metrics.peak_time:.1f}s")
    ax.axvspan(metrics.start_time, metrics.stop_time, color="black", alpha=0.08,
               label=f"spread = {metrics.spread:.1f}s")
    ax.set_xlabel("Time in trial (s)")
    ax.set_ylabel("Response rate (per s)")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "peak_procedure")


# ---------------------------------------------------------------------------
# 13. Generalization gradient (Gaussian)
# ---------------------------------------------------------------------------


def fig_generalization() -> None:
    rng = np.random.default_rng(12)
    r0_true, sigma_true = 48.0, 120.0
    distances_signed = np.array([-300, -200, -120, -60, 0, 60, 120, 200, 300])
    responses = r0_true * np.exp(-distances_signed**2 / (2 * sigma_true**2)) + rng.normal(0, 1.5, len(distances_signed))
    abs_d = np.abs(distances_signed)
    params = fit_gaussian_gradient(abs_d.tolist(), responses.tolist())

    fig, ax = _new_fig()
    dfit = np.linspace(-350, 350, 200)
    rfit = params.peak_response * np.exp(-dfit**2 / (2 * params.sigma**2))
    ax.plot(dfit, rfit, color="black", linewidth=1.0, label=f"σ={params.sigma:.0f}")
    ax.plot(distances_signed, responses, "o", markersize=4, markerfacecolor="white", color="black")
    ax.axvline(0, color="black", linestyle=":", linewidth=0.75)
    ax.set_xlabel("Distance from S+ (Hz)")
    ax.set_ylabel("Responses per minute")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "generalization_gradient")


# ---------------------------------------------------------------------------
# 14. Behavioral momentum
# ---------------------------------------------------------------------------


def fig_momentum() -> None:
    rng = np.random.default_rng(13)
    m_true = 14.0
    baseline = 60.0
    xs = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
    ratios = 10 ** (-xs / m_true) * np.exp(rng.normal(0, 0.02, len(xs)))
    disrupted = (ratios * baseline).tolist()
    params = estimate_resistance_to_change(
        baseline_rate=baseline,
        disrupted_rates=disrupted,
        disruptor_magnitudes=xs.tolist(),
    )

    fig, ax = _new_fig()
    xfit = np.linspace(0, 2.2, 100)
    yfit = 10 ** (-xfit / params.m)
    ax.plot(xfit, yfit, color="black", linewidth=1.0, label=f"m={params.m:.1f}")
    ax.plot(xs, ratios, "o", markersize=4, markerfacecolor="white", color="black")
    ax.set_xlabel("Disruptor magnitude x")
    ax.set_ylabel("$B_x / B_o$")
    ax.set_yscale("log")
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    _save(fig, "behavioral_momentum")


# ---------------------------------------------------------------------------
# 15. Operant variability (U-value)
# ---------------------------------------------------------------------------


def fig_variability() -> None:
    rng = np.random.default_rng(14)
    # Simulate an L4-sequence study with A=2 alternatives (2^4 = 16 possibilities)
    sequences = [tuple(rng.integers(0, 2, size=4).tolist()) for _ in range(120)]
    metrics = compute_u_value(sequences, alphabet_size=2, sequence_length=4)

    # Distribution of sequence frequencies
    from collections import Counter

    counts = Counter(sequences)
    heights = sorted(counts.values(), reverse=True)
    labels = list(range(1, len(heights) + 1))

    fig, ax = _new_fig()
    ax.bar(labels, heights, color="#555555", edgecolor="black")
    ax.text(0.95, 0.95, f"U = {metrics.u_value:.2f}\nH = {metrics.entropy_bits:.2f} bits",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_xlabel("Sequence rank")
    ax.set_ylabel("Frequency")
    _save(fig, "variability_u")


# ---------------------------------------------------------------------------
# 16. Survival (Kaplan-Meier)
# ---------------------------------------------------------------------------


def fig_survival() -> None:
    rng = np.random.default_rng(15)
    n = 30
    true_times = rng.exponential(scale=90, size=n)
    # Censor the tail 20% that survive past session end
    session_end = 180.0
    observed = np.minimum(true_times, session_end)
    events = (true_times <= session_end).astype(int)

    curve = kaplan_meier(observed.tolist(), events.tolist())

    fig, ax = _new_fig()
    ts = [0.0] + list(curve.times)
    ss = [1.0] + list(curve.survival)
    ax.step(ts, ss, where="post", color="black", linewidth=1.0)
    if curve.median_survival_time is not None:
        ax.axvline(curve.median_survival_time, color="black", linestyle="--", linewidth=1.0,
                   label=f"median = {curve.median_survival_time:.1f}s")
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.75)
        ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("S(t)")
    ax.set_ylim(0, 1.05)
    _save(fig, "survival")


# ---------------------------------------------------------------------------
# 17. Time-allocation matching
# ---------------------------------------------------------------------------


def fig_time_matching() -> None:
    rng = np.random.default_rng(20)
    log_r = np.linspace(-1.0, 1.0, 6)
    log_t = 0.79 * log_r + rng.normal(0, 0.04, 6) + 0.05
    fig, ax = _new_fig()
    xs = np.linspace(-1.2, 1.2, 50)
    params = estimate_time_matching_parameters(
        (10**log_t).tolist(), (10**log_r).tolist()
    )
    ax.plot(xs, params.sensitivity * xs + params.bias, color="black",
            linewidth=1.0, label=f"a={params.sensitivity:.2f}")
    ax.plot(log_r, log_t, "o", color="black", markersize=4, markerfacecolor="white")
    ax.plot(xs, xs, color="black", linewidth=0.5, linestyle=":", label="strict")
    ax.set_xlabel("log(R1 / R2)")
    ax.set_ylabel("log(T1 / T2)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    _save(fig, "time_matching")


# ---------------------------------------------------------------------------
# 18. Concurrent-chains prediction
# ---------------------------------------------------------------------------


def fig_concurrent_chains() -> None:
    pred = predict_concurrent_chains_choice(
        left_amount=1.0, left_delay=2.0,
        right_amount=1.0, right_delay=10.0,
        left_responses=78, right_responses=22, k=1.0,
    )
    fig, ax = _new_fig(height_ratio=0.9)
    pos = np.arange(2)
    ax.bar(pos - 0.18, [pred.observed_proportion_left, 1 - pred.observed_proportion_left],
           width=0.35, color="#333333", edgecolor="black", label="observed")
    ax.bar(pos + 0.18, [pred.predicted_proportion_left, 1 - pred.predicted_proportion_left],
           width=0.35, color="#aaaaaa", edgecolor="black", label="Mazur k=1")
    ax.set_xticks(pos)
    ax.set_xticklabels(["Left (D=2)", "Right (D=10)"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "concurrent_chains")


# ---------------------------------------------------------------------------
# 19. Tracking index (sign- vs goal-tracker distribution)
# ---------------------------------------------------------------------------


def fig_tracking_index() -> None:
    rng = np.random.default_rng(21)
    indices = []
    # Mixture of sign-trackers, intermediates, goal-trackers
    for _ in range(40):
        p_cs = float(np.clip(rng.beta(8, 2), 0.01, 0.99))
        p_food = 1.0 - p_cs + rng.normal(0, 0.05)
        p_food = float(np.clip(p_food, 0.0, 1.0))
        contacts_cs = int(40 * p_cs + rng.integers(0, 5))
        contacts_food = int(40 * p_food + rng.integers(0, 5))
        lat_cs = max(0.5, 8.0 * (1 - p_cs))
        lat_food = max(0.5, 8.0 * (1 - p_food))
        ti = compute_tracking_index(
            contacts_cs=contacts_cs, contacts_food=contacts_food,
            latency_cs=lat_cs, latency_food=lat_food,
            probability_cs=p_cs, probability_food=p_food,
            cs_duration=8.0,
        )
        indices.append(ti.index)
    indices = np.array(indices)
    fig, ax = _new_fig()
    ax.hist(indices, bins=15, color="#777777", edgecolor="black")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="ST cutoff")
    ax.axvline(-0.5, color="black", linestyle=":", linewidth=1.0, label="GT cutoff")
    ax.set_xlabel("PCA tracking index")
    ax.set_ylabel("# subjects")
    ax.set_xlim(-1.05, 1.05)
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    _save(fig, "tracking_index")


# ---------------------------------------------------------------------------
# 20. Resurgence (3-phase rate)
# ---------------------------------------------------------------------------


def fig_resurgence() -> None:
    fig, ax = _new_fig()
    bins = np.arange(15)
    rng = np.random.default_rng(22)
    target = np.concatenate([
        np.full(5, 40.0) + rng.normal(0, 3, 5),  # phase 1
        np.linspace(35, 3, 5) + rng.normal(0, 1, 5),  # phase 2 suppression
        np.array([15, 12, 9, 6, 4]) + rng.normal(0, 1, 5),  # phase 3 resurgence
    ])
    alt = np.concatenate([
        np.zeros(5),
        np.linspace(5, 38, 5),
        np.array([30, 22, 15, 10, 7]),
    ])
    ax.plot(bins, target, "o-", color="black", markersize=4,
            markerfacecolor="white", linewidth=1.0, label="target")
    ax.plot(bins, alt, "s--", color="#777777", markersize=4,
            markerfacecolor="white", linewidth=1.0, label="alternative")
    for x in (4.5, 9.5):
        ax.axvline(x, color="black", linestyle=":", linewidth=0.75)
    for x, txt in [(2, "Phase 1"), (7, "Phase 2"), (12, "Phase 3")]:
        ax.text(x, 45, txt, ha="center", fontsize=7)
    ax.set_xlabel("Session bin")
    ax.set_ylabel("Rate (per min)")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "resurgence")


# ---------------------------------------------------------------------------
# 21. Renewal (3-phase bar comparison)
# ---------------------------------------------------------------------------


def fig_renewal() -> None:
    fig, ax = _new_fig(height_ratio=0.9)
    rates = [40.0, 4.0, 26.0]
    labels = ["A train", "B extinction", "A test\n(renewed)"]
    bars = ax.bar(np.arange(3), rates, color=["#333333", "#bbbbbb", "#777777"],
                  edgecolor="black")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.1f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Response rate (per min)")
    _save(fig, "renewal")


# ---------------------------------------------------------------------------
# 22. Observing response (S+ vs S- production rates)
# ---------------------------------------------------------------------------


def fig_observing() -> None:
    fig, ax = _new_fig(height_ratio=0.9)
    rates = [12.0, 8.0, 4.0]
    labels = ["Total observe", "S+ production", "S- production"]
    ax.bar(np.arange(3), rates, color=["#333333", "#777777", "#bbbbbb"],
           edgecolor="black")
    for i, r in enumerate(rates):
        ax.text(i, r + 0.3, f"{r:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Rate (per min)")
    _save(fig, "observing")


# ---------------------------------------------------------------------------
# 23. Exponential gradient (Shepard)
# ---------------------------------------------------------------------------


def fig_exponential_gradient() -> None:
    rng = np.random.default_rng(23)
    r0_true, lam_true = 50.0, 80.0
    distances = np.array([0, 30, 60, 120, 200, 320])
    responses = np.clip(
        r0_true * np.exp(-distances / lam_true) + rng.normal(0, 1.0, len(distances)),
        0.0, None,
    )
    params = fit_exponential_gradient(distances.tolist(), responses.tolist())
    fig, ax = _new_fig()
    dfit = np.linspace(0, 350, 200)
    rfit = params.peak_response * np.exp(-dfit / params.length_scale)
    ax.plot(dfit, rfit, color="black", linewidth=1.0, label=f"λ={params.length_scale:.0f}")
    ax.plot(distances, responses, "o", markersize=4, markerfacecolor="white", color="black")
    ax.set_xlabel("Distance from S+ (Hz)")
    ax.set_ylabel("Responses per minute")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "exponential_gradient")


# ---------------------------------------------------------------------------
# 24. Discrimination ratio (S+ vs S- bars)
# ---------------------------------------------------------------------------


def fig_discrimination_ratio() -> None:
    s_plus, s_minus = 170, 30
    dr = s_plus / (s_plus + s_minus)
    fig, ax = _new_fig(height_ratio=0.9)
    ax.bar(np.arange(2), [s_plus, s_minus], color=["#333333", "#aaaaaa"], edgecolor="black")
    for i, c in enumerate([s_plus, s_minus]):
        ax.text(i, c + 4, f"n={c}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["S+", "S−"])
    ax.set_ylabel("Responses")
    ax.text(0.95, 0.95, f"DR = {dr:.2f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontweight="bold")
    _save(fig, "discrimination_ratio")


# ---------------------------------------------------------------------------
# 25. Peak shift gradient
# ---------------------------------------------------------------------------


def fig_peak_shift() -> None:
    s_plus, s_minus = 550.0, 540.0
    rng = np.random.default_rng(24)
    stims = np.array([520, 530, 540, 550, 560, 570, 580])
    # Empirical peak shifted to ~560 (away from S-)
    peak_emp = 560.0
    rs = 50.0 * np.exp(-((stims - peak_emp) ** 2) / (2 * 12.0**2)) + rng.normal(0, 1.0, len(stims))
    result = detect_peak_shift(
        stimulus_values=stims.tolist(), response_rates=rs.tolist(),
        s_plus=s_plus, s_minus=s_minus,
    )
    fig, ax = _new_fig()
    xs = np.linspace(515, 585, 200)
    ys = result.fitted_peak_response * np.exp(-((xs - result.fitted_peak_value) ** 2) / (2 * result.fitted_sigma**2))
    ax.plot(xs, ys, color="black", linewidth=1.0)
    ax.plot(stims, rs, "o", markersize=4, markerfacecolor="white", color="black")
    ax.axvline(s_plus, color="black", linestyle="--", linewidth=1.0, label=f"S+ = {s_plus:.0f}")
    ax.axvline(s_minus, color="black", linestyle=":", linewidth=1.0, label=f"S- = {s_minus:.0f}")
    ax.axvline(result.fitted_peak_value, color="black", linestyle="-.", linewidth=1.0,
               label=f"peak = {result.fitted_peak_value:.0f}")
    ax.set_xlabel("Stimulus dimension")
    ax.set_ylabel("Responses per minute")
    ax.legend(loc="upper right", frameon=False, fontsize=6)
    _save(fig, "peak_shift")


# ---------------------------------------------------------------------------
# 26. ROC curve (signal detection)
# ---------------------------------------------------------------------------


def fig_signal_detection_roc() -> None:
    # (false_alarm_rate, hit_rate) operating points sweeping criterion.
    operating_points = [
        (0.01, 0.05),
        (0.10, 0.30),
        (0.30, 0.60),
        (0.55, 0.85),
        (0.90, 0.99),
    ]
    roc = compute_roc(operating_points)
    fig, ax = _new_fig()
    ax.plot([0, 1], [0, 1], "k:", linewidth=0.75)
    ax.plot(roc.fa_rates, roc.hit_rates, "o-", color="black", markersize=4,
            markerfacecolor="white", linewidth=1.0)
    ax.set_xlabel("False-alarm rate")
    ax.set_ylabel("Hit rate")
    ax.text(0.95, 0.05, f"AUC = {roc.auc:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "signal_detection_roc")


# ---------------------------------------------------------------------------
# 27. Within-session dynamics (inverted-U)
# ---------------------------------------------------------------------------


def fig_within_session() -> None:
    rng = np.random.default_rng(25)
    bins = np.arange(8)
    rates = np.array([0.5, 1.4, 2.1, 2.4, 2.0, 1.5, 1.0, 0.7]) + rng.normal(0, 0.08, 8)
    fig, ax = _new_fig()
    ax.plot(bins + 0.5, rates, "o-", color="black", markersize=4,
            markerfacecolor="white", linewidth=1.0)
    peak = int(np.argmax(rates))
    ax.axvline(peak + 0.5, color="black", linestyle="--", linewidth=1.0,
               label=f"peak bin={peak}")
    ax.set_xlabel("Session bin (8-bin division)")
    ax.set_ylabel("Response rate (per min)")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "within_session")


# ---------------------------------------------------------------------------
# 28. Unit-price elasticity (log-log line)
# ---------------------------------------------------------------------------


def fig_unit_price() -> None:
    rng = np.random.default_rng(26)
    log_p = np.linspace(0, 2.5, 8)
    elasticity_true = -1.18
    log_q = elasticity_true * log_p + 4.0 + rng.normal(0, 0.08, 8)
    fig, ax = _new_fig()
    xs = np.linspace(-0.2, 2.7, 50)
    ax.plot(xs, elasticity_true * xs + 4.0, color="black", linewidth=1.0,
            label=f"α={elasticity_true:.2f}")
    ax.plot(log_p, log_q, "o", markersize=4, markerfacecolor="white", color="black")
    ax.set_xlabel("log unit price")
    ax.set_ylabel("log consumption Q")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "unit_price")


# ---------------------------------------------------------------------------
# 29. Indifference point (adjusting trajectory)
# ---------------------------------------------------------------------------


def fig_indifference() -> None:
    rng = np.random.default_rng(27)
    target = 50.0
    trial = []
    cur = 100.0
    for t in range(28):
        cur = cur + (target - cur) * 0.4 + rng.normal(0, 0.7)
        trial.append(cur)
    point = estimate_indifference_point(adjusting_amounts=trial, stable_window=6)
    fig, ax = _new_fig()
    ax.plot(np.arange(len(trial)), trial, "o-", color="black", markersize=3,
            markerfacecolor="white", linewidth=1.0)
    ax.axhline(point.indifference_value, color="black", linestyle="--",
               linewidth=1.0, label=f"indifference = {point.indifference_value:.1f}")
    ax.axvspan(len(trial) - 6, len(trial) - 1, color="black", alpha=0.08, label="stability window")
    ax.set_xlabel("Adjusting trial")
    ax.set_ylabel("Adjusting amount")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "indifference_point")


# ---------------------------------------------------------------------------
# 30. Lag-N variability
# ---------------------------------------------------------------------------


def fig_lag_variability() -> None:
    rng = np.random.default_rng(28)
    sequences = [tuple(rng.integers(0, 2, size=4).tolist()) for _ in range(120)]
    metrics = compute_u_value(sequences, alphabet_size=2, sequence_length=4)
    # Pass rate by lag
    from collections import deque
    lags = list(range(1, 9))
    pass_rates = []
    for L in lags:
        passes = 0
        recent: deque = deque(maxlen=L)
        for seq in sequences:
            ok = all(seq != r for r in recent) if recent else True
            if ok:
                passes += 1
            recent.append(seq)
        pass_rates.append(passes / len(sequences))
    fig, ax = _new_fig()
    ax.plot(lags, pass_rates, "o-", color="black", markersize=4,
            markerfacecolor="white", linewidth=1.0)
    ax.text(0.95, 0.95, f"U = {metrics.u_value:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8)
    ax.set_xlabel("Lag N")
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0, 1.05)
    _save(fig, "lag_variability")


# ---------------------------------------------------------------------------
# 31. Log-survivor curve with bilinear knot
# ---------------------------------------------------------------------------


def fig_log_survivor() -> None:
    rng = np.random.default_rng(29)
    short = rng.lognormal(np.log(0.3), 0.4, size=400)
    long = rng.lognormal(np.log(8.0), 0.6, size=80)
    irts = np.sort(np.concatenate([short, long]))
    curve = compute_log_survivor(irts.tolist())
    fit = fit_bilinear_log_survivor(irts.tolist())
    fig, ax = _new_fig()
    ax.plot(curve.times, curve.log_survival, color="black", linewidth=1.0)
    if fit is not None:
        ts = np.array(curve.times)
        knot = fit.knot_time_sec
        left_mask = ts <= knot
        right_mask = ts >= knot
        ax.plot(ts[left_mask], fit.intercept_fast + fit.slope_fast * ts[left_mask],
                color="black", linestyle="--", linewidth=1.0, label="fast")
        ax.plot(ts[right_mask], fit.intercept_slow + fit.slope_slow * ts[right_mask],
                color="black", linestyle=":", linewidth=1.0, label="slow")
        ax.axvline(knot, color="black", linewidth=0.5, linestyle="-.")
    ax.set_xlabel("IRT (s)")
    ax.set_ylabel("log S(t)")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "log_survivor")


# ---------------------------------------------------------------------------
# 32. Rescorla-Wagner trace
# ---------------------------------------------------------------------------


def fig_rescorla_wagner() -> None:
    us = [True] * 20 + [False] * 20 + [True] * 20
    trace = simulate_rescorla_wagner(us, alpha_beta=0.25, lambda_=0.8)
    fit = fit_rescorla_wagner(trace, us)
    fig, ax = _new_fig()
    trials = np.arange(1, len(us) + 1)
    ax.plot(trials, trace, "o-", color="black", markersize=3,
            markerfacecolor="white", linewidth=1.0, label="V_t")
    for x in (20.5, 40.5):
        ax.axvline(x, color="black", linestyle=":", linewidth=0.75)
    ax.text(10, 0.85, "acquire", ha="center", fontsize=7)
    ax.text(30, 0.85, "extinction", ha="center", fontsize=7)
    ax.text(50, 0.85, "reacquire", ha="center", fontsize=7)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Associative strength V")
    ax.text(0.02, 0.98,
            f"α·β={fit.alpha_beta:.2f}, λ={fit.lambda_:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    _save(fig, "rescorla_wagner")


# ---------------------------------------------------------------------------
# 33. TD(0) value backup
# ---------------------------------------------------------------------------


def fig_td_learning() -> None:
    states = [0, 1, 2]
    rewards = [0.0, 1.0]
    last = None
    history = []
    for _ in range(60):
        last = simulate_td_zero(
            states, rewards, alpha=0.3, gamma=0.9, n_states=3,
            terminal_state=2, initial_values=last.values if last else None,
        )
        history.append(last.values)
    history = np.array(history)
    fig, ax = _new_fig()
    ep = np.arange(1, history.shape[0] + 1)
    ax.plot(ep, history[:, 0], color="black", linewidth=1.0, label="V(s0)")
    ax.plot(ep, history[:, 1], color="black", linestyle="--", linewidth=1.0, label="V(s1)")
    ax.axhline(0.9, color="black", linestyle=":", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value estimate")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    _save(fig, "td_learning")


# ---------------------------------------------------------------------------
# 34. Scalar timing (response distributions over criteria)
# ---------------------------------------------------------------------------


def fig_scalar_timing() -> None:
    rng = np.random.default_rng(30)
    fig, ax = _new_fig()
    cv = 0.25
    for criterion, color in [(15.0, "#bbbbbb"), (30.0, "#777777"), (60.0, "#222222")]:
        sd = cv * criterion
        samples = rng.normal(criterion, sd, size=500).clip(min=1e-3)
        # Normalised response density
        bins = np.linspace(0, 100, 60)
        h, e = np.histogram(samples, bins=bins, density=True)
        ax.plot(0.5 * (e[:-1] + e[1:]) / criterion, h * criterion, color=color, linewidth=1.0,
                label=f"FI {criterion:.0f}s")
    ax.set_xlabel("Time / criterion")
    ax.set_ylabel("Normalised density")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "scalar_timing")


# ---------------------------------------------------------------------------
# 35. Killeen MPR (B vs R)
# ---------------------------------------------------------------------------


def fig_killeen_law() -> None:
    rng = np.random.default_rng(31)
    k_true, a_true, delta_true = 2.0, 0.5, 0.02
    rs = np.array([0.2, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0])
    bs = (k_true * a_true * rs) / (a_true * rs * delta_true + 1.0)
    bs = bs * rng.normal(1.0, 0.04, len(bs))
    params = fit_killeen_law(rs.tolist(), bs.tolist(), delta=delta_true)
    fig, ax = _new_fig()
    rfit = np.logspace(np.log10(0.1), np.log10(200), 200)
    bfit = (params.k * params.a * rfit) / (params.a * rfit * params.delta + 1.0)
    ax.semilogx(rfit, bfit, color="black", linewidth=1.0,
                label=f"k={params.k:.2f}, a={params.a:.2f}")
    ax.semilogx(rs, bs, "o", markersize=4, markerfacecolor="white", color="black")
    ax.axhline(params.k / params.delta, color="black", linestyle=":", linewidth=0.75,
               label=f"k/δ={params.k / params.delta:.0f}")
    ax.set_xlabel("Reinforcement rate R")
    ax.set_ylabel("Response rate B")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    _save(fig, "killeen_law")


# ---------------------------------------------------------------------------
# 36. Stay-switch (changeover bouts)
# ---------------------------------------------------------------------------


def fig_stay_switch() -> None:
    rng = np.random.default_rng(32)
    sides = []
    times = []
    t = 0.0
    side = "L"
    for _ in range(100):
        # Stay bout 3-8 responses
        bout_len = int(rng.integers(3, 8))
        for _ in range(bout_len):
            t += rng.exponential(0.5)
            sides.append(side)
            times.append(t)
        side = "R" if side == "L" else "L"
    metrics = analyze_stay_switch(sides, times)
    fig, ax = _new_fig()
    cum = np.arange(1, len(sides) + 1)
    sides_arr = np.array(sides)
    ax.step(times, cum, where="post", color="black", linewidth=0.6)
    for i, s in enumerate(sides):
        if i > 0 and sides[i] != sides[i - 1]:
            ax.axvline(times[i], color="black", linewidth=0.4, linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative responses (both sides)")
    if metrics is not None:
        ax.text(0.02, 0.98,
                f"changeovers = {metrics.n_changeovers}\nmean stay = {metrics.mean_stay_responses:.1f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7)
    _save(fig, "stay_switch")


# ---------------------------------------------------------------------------
# 37. Extinction burst
# ---------------------------------------------------------------------------


def fig_extinction_burst() -> None:
    baseline_rate_per_s = 0.5
    burst_rate = 1.0
    fig, ax = _new_fig()
    ts = np.arange(0, 360, 5)
    rates = np.where(
        ts < 120, baseline_rate_per_s,
        np.where(ts < 180, burst_rate, baseline_rate_per_s * np.exp(-(ts - 180) / 60))
    )
    ax.plot(ts, rates * 60.0, color="black", linewidth=1.0)
    ax.axvline(120, color="black", linestyle="--", linewidth=1.0, label="extinction onset")
    ax.axvspan(120, 180, color="black", alpha=0.08, label="burst window")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response rate (per min)")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "extinction_burst")


# ---------------------------------------------------------------------------
# 38. Schneider two-state segmentation
# ---------------------------------------------------------------------------


def fig_schneider() -> None:
    rng = np.random.default_rng(33)
    # FI 60 s. Pause then run.
    pause = 25.0
    response_times = np.sort(np.concatenate([
        np.array([]),
        np.linspace(pause + 1, 58, 32) + rng.normal(0, 0.4, 32),
    ])).tolist()
    result = schneider_two_state(response_times,
                                 interval_start=0.0, interval_end=60.0)
    fig, ax = _new_fig()
    cum = np.arange(1, len(response_times) + 1)
    ax.step(response_times, cum, where="post", color="black", linewidth=1.0)
    if result is not None:
        ax.axvline(result.break_time, color="black", linestyle="--", linewidth=1.0,
                   label=f"break = {result.break_time:.1f}s")
        ax.text(0.02, 0.98,
                f"low = {result.low_rate:.2f} r/s\nhigh = {result.high_rate:.2f} r/s",
                transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.set_xlabel("Time in interval (s)")
    ax.set_ylabel("Cumulative responses")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    _save(fig, "schneider_segment")


# ---------------------------------------------------------------------------
# 39. Psychometric function
# ---------------------------------------------------------------------------


def fig_psychometric() -> None:
    rng = np.random.default_rng(34)
    mu_true, sigma_true = 0.0, 1.5
    stimuli = np.linspace(-5, 5, 11)
    p_true = 1 / (1 + np.exp(-(stimuli - mu_true) / sigma_true))
    n_total = np.full(len(stimuli), 50)
    n_correct = rng.binomial(n_total, p_true)
    fit = fit_psychometric(stimuli.tolist(), n_correct.tolist(), n_total.tolist(), link="logit")
    fig, ax = _new_fig()
    xs = np.linspace(-6, 6, 200)
    ys = 1 / (1 + np.exp(-(xs - fit.mu) / fit.sigma))
    ax.plot(xs, ys, color="black", linewidth=1.0, label=f"μ={fit.mu:.2f}, σ={fit.sigma:.2f}")
    ax.plot(stimuli, n_correct / n_total, "o", markersize=4, markerfacecolor="white",
            color="black")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.5)
    ax.axvline(fit.mu, color="black", linestyle="--", linewidth=0.75)
    ax.set_xlabel("Stimulus magnitude")
    ax.set_ylabel("P(yes)")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    _save(fig, "psychometric")


# ---------------------------------------------------------------------------
# 40. EZ-DDM (Pc / RT visualization)
# ---------------------------------------------------------------------------


def fig_ddm() -> None:
    fig, ax = _new_fig()
    # Sweep drift; show how Pc and Ter shift
    a = 0.12
    ter = 0.30
    s = 0.1
    drifts = np.linspace(0.05, 0.40, 30)
    pcs = []
    mrts = []
    for v in drifts:
        y = np.exp(-v * a / s**2)
        pc = 1.0 / (1.0 + y)
        mdt = (a / (2 * v)) * (1 - y) / (1 + y)
        pcs.append(pc)
        mrts.append(mdt + ter)
    ax2 = ax.twinx()
    ax.plot(drifts, pcs, color="black", linewidth=1.0, label="P(correct)")
    ax2.plot(drifts, mrts, color="#777777", linestyle="--", linewidth=1.0, label="mean RT")
    ax.set_xlabel("Drift rate v")
    ax.set_ylabel("P(correct)", color="black")
    ax2.set_ylabel("Mean RT (s)", color="#444444")
    ax.set_ylim(0.5, 1.0)
    # Recover one point
    pc_obs = 0.84
    mrt_obs = 0.55
    vrt_obs = 0.04
    fit = fit_ez_diffusion(proportion_correct=pc_obs,
                           mean_rt_correct=mrt_obs, var_rt_correct=vrt_obs)
    ax.text(0.02, 0.95,
            f"recovered: v={fit.v:.2f}, a={fit.a:.2f}\nTer={fit.ter:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    _save(fig, "ez_diffusion")


# ---------------------------------------------------------------------------
# 41. IRT autocorrelation
# ---------------------------------------------------------------------------


def fig_irt_autocorrelation() -> None:
    rng = np.random.default_rng(35)
    n = 400
    eps = rng.standard_normal(n)
    # AR(1) phi=0.6
    irts = [1.0]
    for e in eps[1:]:
        irts.append(0.6 * irts[-1] + e)
    irts = np.array(irts) + 5.0
    acf = compute_irt_autocorrelation(irts.tolist(), max_lag=20)
    fig, ax = _new_fig()
    ax.bar(acf.lags, acf.values, color="#555555", edgecolor="black", width=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation ρ(k)")
    _save(fig, "irt_autocorrelation")


# ---------------------------------------------------------------------------
# 42. IRT periodogram
# ---------------------------------------------------------------------------


def fig_irt_spectrum() -> None:
    fs = 100.0
    n = 512
    t = np.arange(n) / fs
    rng = np.random.default_rng(36)
    series = np.sin(2 * np.pi * 5.0 * t) + 0.3 * rng.standard_normal(n)
    spec = compute_irt_spectrum(series.tolist(), sampling_rate=fs)
    fig, ax = _new_fig()
    ax.semilogy(spec.frequencies, spec.power, color="black", linewidth=1.0)
    ax.axvline(spec.peak_frequency, color="black", linestyle="--", linewidth=1.0,
               label=f"peak={spec.peak_frequency:.2f} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "irt_spectrum")


# ---------------------------------------------------------------------------
# 43. N-gram entropy / Markov order
# ---------------------------------------------------------------------------


def fig_ngram_entropy() -> None:
    rng = np.random.default_rng(37)
    seq = []
    state = "L"
    for _ in range(2000):
        if state == "L":
            state = "L" if rng.random() < 0.85 else "R"
        else:
            state = "R" if rng.random() < 0.85 else "L"
        seq.append(state)
    orders = list(range(1, 5))
    entropies = [compute_block_entropy(seq, order=n).entropy_bits for n in orders]
    fig, ax = _new_fig()
    ax.plot(orders, entropies, "o-", color="black", markersize=4,
            markerfacecolor="white", linewidth=1.0)
    ax.set_xlabel("Block length n")
    ax.set_ylabel("H_n (bits)")
    ax.set_xticks(orders)
    _save(fig, "ngram_entropy")


# ---------------------------------------------------------------------------
# 44. Cross-elasticity matrix (heatmap)
# ---------------------------------------------------------------------------


def fig_cross_elasticity_matrix() -> None:
    rng = np.random.default_rng(38)
    prices = {
        "A": [1.0, 2.0, 4.0, 8.0, 16.0],
        "B": [1.0, 2.0, 4.0, 8.0, 16.0],
        "C": [4.0, 4.0, 4.0, 4.0, 4.0],
    }
    cons = {
        "A": [100 / (p ** 0.5) for p in prices["A"]],
        "B": [80 * pa for pa in prices["A"]],   # substitute for A (positive cross)
        "C": [50 / (pc ** 0.7) * (1 + 0.3 * pa) for pa, pc in zip(prices["A"], prices["C"], strict=False)],
    }
    matrix = compute_cross_elasticity_matrix(cons, prices)
    arr = np.array(matrix.matrix)
    fig, ax = _new_fig(height_ratio=1.0)
    im = ax.imshow(arr, cmap="Greys", aspect="auto")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(arr[i, j]) > 0.5 else "black")
    ax.set_xticks(range(len(matrix.goods)))
    ax.set_xticklabels([f"P({g})" for g in matrix.goods])
    ax.set_yticks(range(len(matrix.goods)))
    ax.set_yticklabels([f"Q({g})" for g in matrix.goods])
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, "cross_elasticity_matrix")


# ---------------------------------------------------------------------------
# 45. Hierarchical GML (subjects vs group)
# ---------------------------------------------------------------------------


def fig_hierarchical_gml() -> None:
    rng = np.random.default_rng(39)
    mu_a, mu_b, tau_a, tau_b, sigma = 0.85, 0.0, 0.10, 0.08, 0.05
    x_grid = np.linspace(np.log(0.1), np.log(10.0), 8)
    xs_dict = {}
    ys_dict = {}
    fig, ax = _new_fig()
    xs_plot = np.linspace(-2.5, 2.5, 50)
    for s in range(8):
        a_i = rng.normal(mu_a, tau_a)
        b_i = rng.normal(mu_b, tau_b)
        y = a_i * x_grid + b_i + rng.normal(0, sigma, x_grid.size)
        xs_dict[f"S{s}"] = x_grid.tolist()
        ys_dict[f"S{s}"] = y.tolist()
        ax.plot(x_grid, y, "o-", color="#aaaaaa", markersize=3,
                markerfacecolor="white", linewidth=0.6)
    fit = fit_hierarchical_gml(ys_dict, xs_dict)
    ax.plot(xs_plot, fit.mu_sensitivity * xs_plot + fit.mu_bias,
            color="black", linewidth=1.5,
            label=f"group: a={fit.mu_sensitivity:.2f}±{fit.sd_sensitivity:.2f}")
    lo, hi = fit.sensitivity_ci
    ax.text(0.02, 0.98, f"95% CI a: [{lo:.2f}, {hi:.2f}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.set_xlabel("log(R1/R2)")
    ax.set_ylabel("log(B1/B2)")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    _save(fig, "hierarchical_gml")


# ---------------------------------------------------------------------------
# 46. HMM segmentation
# ---------------------------------------------------------------------------


def fig_hmm() -> None:
    rng = np.random.default_rng(40)
    obs = np.concatenate([
        rng.normal(1.0, 0.4, 80),
        rng.normal(5.0, 0.4, 80),
        rng.normal(1.0, 0.4, 80),
    ]).tolist()
    fit = fit_gaussian_hmm(obs, n_states=2)
    fig, ax = _new_fig()
    t = np.arange(len(obs))
    ax.plot(t, obs, color="#888888", linewidth=0.6)
    # Color shaded by inferred state
    state_seq = np.array(fit.state_sequence)
    for s in fit.segments:
        color = "#dddddd" if s.state == 0 else "#888888"
        ax.axvspan(s.start_index, s.end_index, color=color, alpha=0.3)
    for m in fit.means:
        ax.axhline(m, color="black", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Observation")
    ax.text(0.02, 0.98,
            f"means: {', '.join(f'{m:.2f}' for m in fit.means)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    _save(fig, "hmm_segment")


# ---------------------------------------------------------------------------
# 47. IRT-coded cumulative record (visualizer demo)
# ---------------------------------------------------------------------------


def fig_irt_coded_cumulative() -> None:
    rng = np.random.default_rng(41)
    times = []
    t = 0.0
    while t < 200:
        # DRL 5: most IRTs > 5 (long), some short bursts
        irt = rng.choice([2.0, 6.0], p=[0.3, 0.7])
        t += irt + rng.normal(0, 0.3)
        times.append(t)
    threshold = 5.0
    cum_t = []
    cum_y = []
    short_x, short_y = [], []
    long_x, long_y = [], []
    last = None
    for i, rt in enumerate(times):
        cum_t.append(rt)
        cum_y.append(i + 1)
        if last is not None:
            if rt - last >= threshold:
                long_x.append(rt)
                long_y.append(i + 1)
            else:
                short_x.append(rt)
                short_y.append(i + 1)
        last = rt
    fig, ax = _new_fig()
    ax.step(cum_t, cum_y, where="post", color="black", linewidth=0.6, alpha=0.5)
    ax.scatter(short_x, short_y, marker="o", color="#aaaaaa", s=12, label="IRT < 5s")
    ax.scatter(long_x, long_y, marker="o", color="black", s=12, label="IRT ≥ 5s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative responses")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    _save(fig, "irt_coded_cumulative_record")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


FIGURES = {
    "cumulative_record": fig_cumulative_record,
    "post_reinforcement_pause": fig_prp,
    "irt_distribution": fig_irt_bouts,
    "run_length": fig_run_length,
    "latency": fig_latency,
    "matching_law": fig_matching_law,
    "single_alternative_matching": fig_single_alternative_matching,
    "choice_allocation": fig_choice_allocation,
    "breakpoint": fig_breakpoint,
    "demand_curve": fig_demand_curve,
    "delay_discounting": fig_discounting,
    "peak_procedure": fig_peak_procedure,
    "generalization_gradient": fig_generalization,
    "behavioral_momentum": fig_momentum,
    "variability_u": fig_variability,
    "survival": fig_survival,
    # Newly added (this update)
    "time_matching": fig_time_matching,
    "concurrent_chains": fig_concurrent_chains,
    "tracking_index": fig_tracking_index,
    "resurgence": fig_resurgence,
    "renewal": fig_renewal,
    "observing": fig_observing,
    "exponential_gradient": fig_exponential_gradient,
    "discrimination_ratio": fig_discrimination_ratio,
    "peak_shift": fig_peak_shift,
    "signal_detection_roc": fig_signal_detection_roc,
    "within_session": fig_within_session,
    "unit_price": fig_unit_price,
    "indifference_point": fig_indifference,
    "lag_variability": fig_lag_variability,
    "log_survivor": fig_log_survivor,
    "rescorla_wagner": fig_rescorla_wagner,
    "td_learning": fig_td_learning,
    "scalar_timing": fig_scalar_timing,
    "killeen_law": fig_killeen_law,
    "stay_switch": fig_stay_switch,
    "extinction_burst": fig_extinction_burst,
    "schneider_segment": fig_schneider,
    "psychometric": fig_psychometric,
    "ez_diffusion": fig_ddm,
    "irt_autocorrelation": fig_irt_autocorrelation,
    "irt_spectrum": fig_irt_spectrum,
    "ngram_entropy": fig_ngram_entropy,
    "cross_elasticity_matrix": fig_cross_elasticity_matrix,
    "hierarchical_gml": fig_hierarchical_gml,
    "hmm_segment": fig_hmm,
    "irt_coded_cumulative_record": fig_irt_coded_cumulative,
}


def main() -> None:
    print(f"Rendering {len(FIGURES)} figures with theme '{THEME.name}' to {OUT}")
    for name, fn in FIGURES.items():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
