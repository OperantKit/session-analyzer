# 応答率・局所パターン

[← カタログ目次](index.md) · [English](../../en/analyses/rate-and-pattern.md)

セッション内の応答が *いつ* 起こり *どう局所的にクラスタ化される* かを
要約する可視化と統計群。

---

### `cumulative_record`

- **テーマ** — Skinner 流の正典プロット。時間に対する累積応答を描き、強化子呈示の位置に目盛を入れる。任意の点における傾きが局所応答率で、pause / break-and-run が視覚的に一目瞭然。
- **式** — `y(t) = Σ_{i: t_i ≤ t} 1(response_i)`
- **モジュール** — `session_analyzer.visualizer.cumulative_record.plot_cumulative_record`
- **出力** — matplotlib の `Axes`。y = 累積応答数、x = 時間 (s)、曲線上の各強化子位置に短い縦線。
- **引用** — Ferster, C. B., & Skinner, B. F. (1957). *Schedules of reinforcement*. Appleton-Century-Crofts.

![累積記録](../../assets/figures/cumulative_record.png)

### `post_reinforcement_pause` — `PRPMetrics`

- **テーマ** — 間隔・比率スケジュールでは強化子呈示直後に応答が一時停止する。その持続は ratio size と共にスケールし（Felton & Lyon, 1966）、スケジュール制御の古典的指標。
- **式** — `PRP_i = t(SR_i 後の最初の応答) - t(SR_i)`
- **出力** — `PRPMetrics(mean=4.12, median=3.80, std=1.55, min=0.90, max=9.40, n=42)`（秒）
- **引用** — Felton, M., & Lyon, D. O. (1966). The post-reinforcement pause. *JEAB*, 9(2), 131-134. <https://doi.org/10.1901/jeab.1966.9-131>

![PRP 分布と平均線](../../assets/figures/post_reinforcement_pause.png)

関連: `compute_quarter_life(responses, deadline)` は bin 内で最終応答数の
25% に達した時刻を返す（Herrnstein & Morse, 1957 の FI scalloping 指標）。

### `irt_distribution` — `BoutMetrics` によるバウト分解

- **テーマ** — VI / DRL の操作応答は Poisson ではない。IRT は **within-bout**（短く速い）と **between-bout**（長い deliberation）の混合。Shull et al. (2001) は 2 つの対数正規分布の EM 混合で取り出した。
- **式** — `p(IRT) = π · lognormal(μ_w, σ_w) + (1-π) · lognormal(μ_b, σ_b)`
- **出力** — `BoutMetrics(bout_initiation_rate=2.3, within_bout_rate=3.1, mean_bout_length=6.4, mean_inter_bout_interval_sec=26.1, within_bout_irt_sec=0.31, within_bout_log_sigma=0.48, inter_bout_log_sigma=0.72, pi_within=0.84, n_irts=512, converged=True)`
- **引用** — Shull, R. L., Gaynor, S. T., & Grimes, J. A. (2001). Response rate viewed as engagement bouts. *JEAB*, 75(3), 247-274. <https://doi.org/10.1901/jeab.2001.75-247>

![log-IRT のヒストグラム: within-bout / between-bout の二峰](../../assets/figures/irt_distribution.png)

### `log_survivor` — `LogSurvivorCurve` / `LogSurvivorBilinearFit`

- **テーマ** — IRT の経験的サバイバル関数を log 軸でプロットした図（Shull, 1991）。直線なら単一 Poisson 過程と整合、二段折線（bilinear）ならバウト構造の正典シグネチャ。bilinear fit は屈折点（knot）と 2 本の傾きを返す（急 = within-bout、緩 = between-bout）。
- **式** — `S(t) = P(IRT > t)`; bilinear fit は最適 knot index で `Σ (log S(t) − [a₁ t + b₁])² + Σ (log S(t) − [a₂ t + b₂])²` を最小化。
- **出力** — `LogSurvivorBilinearFit(knot_time_sec=0.62, slope_fast=-3.1, slope_slow=-0.18, intercept_fast=-0.04, intercept_slow=-1.2, sse=0.07, n_irts=400)`
- **引用** — Shull, R. L. (1991). Mathematical description of operant behavior. In I. H. Iversen & K. A. Lattal (Eds.), *Experimental analysis of behavior* (Vol. 2, pp. 243-282). Elsevier.

### `run_length` — `RunMetrics`

- **テーマ** — FR スケジュールは「pause-then-run」サイクルを作る。PRP の後に完了までの高速バースト。この分解により、介入が pause / run / 両方のどこを遅らせているかが見える。
- **出力** — `RunMetrics(n_runs=48, mean_run_length=20.0, mean_run_duration=1.82, mean_run_rate=10.99, mean_pause_duration=3.41, pause_to_run_ratio=1.87)`
- **引用** — Schlinger, H. D., Derenne, A., & Baron, A. (2008). What 50 years of research tell us about pausing under ratio schedules. *The Behavior Analyst*, 31(1), 39-60. <https://doi.org/10.1007/BF03392160>

![FR pause-then-run の累積記録抜粋](../../assets/figures/run_length.png)

### `latency` — `LatencyMetrics`

- **テーマ** — 刺激呈示（または試行開始）から最初の応答までの時間。短い潜時は刺激制御の強さ、長い潜時は消去・飽和・弁別不良を示唆。
- **出力** — `LatencyMetrics(mean=2.05, median=1.78, std=0.81, min=0.42, max=6.10, n=96)`（秒）
- **引用** — 標準的記述子。例: Blough, D. S. (1956). *JEAB*, 3(1), 3-13. <https://doi.org/10.1901/jeab.1960.3-3>

![潜時ヒストグラム + 中央値マーカ](../../assets/figures/latency.png)
