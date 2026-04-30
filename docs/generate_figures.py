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
    compute_choice_allocation,
    compute_u_value,
    detect_breakpoint,
    estimate_matching_parameters,
    estimate_resistance_to_change,
    fit_exponential_demand,
    fit_exponential_gradient,
    fit_gaussian_gradient,
    fit_hyperbolic_discounting,
    fit_single_alternative_matching,
    kaplan_meier,
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
