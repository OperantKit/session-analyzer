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

![knot 付き bilinear log-survivor](../../assets/figures/log_survivor.png)

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

### IRT 色分け累積記録 — `plot_irt_coded_cumulative_record`

- **テーマ** — 累積応答トレースの各点を、直前の IRT が閾値（例: DRL 基準）以上か未満かで色分けする。失敗（短 IRT）と成功（長 IRT）が一目瞭然。
- **モジュール** — `session_analyzer.visualizer.cumulative_diff.plot_irt_coded_cumulative_record`
- **引用** — DRL/DRO 標準的可視化（Sidman, 1956; Reynolds, 1968）。

![wrap した累積トレース上に IRT < 5s と IRT ≥ 5s の二色 scatter](../../assets/figures/irt_coded_cumulative_record.png)

### Schneider の 2 状態セグメンテーション — `SchneiderTwoStateResult`

- **テーマ** — Schneider (1969) は FI 累積記録を 2 つの状態（休止 → 走行）でモデル化した。原典の数値手続きに従い、強化間間隔 1 つの累積記録に、breakpoint までの原点を通る直線と、breakpoint でそれに接続する直線の 2 本を最小二乗で当てる。breakpoint を間隔の格子上で動かし、二乗誤差の和が最小の点を採る。2 本の傾き（反応率）は負にならないよう制約し、状態 1 に反応が 1 つも無くてもよい。
- **出力** — `SchneiderTwoStateResult(break_time=25.5, pause_duration=25.5, run_duration=34.5, low_rate=0.02, high_rate=0.97, slope_difference=0.95, n_low=0, n_high=32)`
- **引用** — Schneider, B. A. (1969). A two-state analysis of fixed-interval responding in the pigeon. *JEAB*, 12(5), 677-687. <https://doi.org/10.1901/jeab.1969.12-677>

![最小二乗の breakpoint で分割した累積記録](../../assets/figures/schneider_segment.png)

### IRT 自己相関 — `IRTAutocorrelation`

- **テーマ** — IRT 系列の biased 自己相関 `ρ(k) = E[(x_t − μ)(x_{t+k} − μ)] / σ²`。正の `ρ(1)` は前 IRT の影響残存（バウト等）、負の `ρ(1)` は交替を示す。長ラグ構造で周期性も検出できる。
- **出力** — `IRTAutocorrelation(lags=(0, 1, 2, ..., 20), values=(1.0, 0.58, 0.32, ..., 0.02), n=400)`
- **引用** — Wei, W. W. S. (2006). *Time series analysis* (2nd ed.). Pearson.

![AR(1) 様 IRT 系列の ACF（lag-1 が正）](../../assets/figures/irt_autocorrelation.png)

### IRT スペクトル — `IRTSpectrum`

- **テーマ** — 平均中心化した IRT 系列の片側ピリオドグラム。schedule-induced rhythms（タイミング・交替等）が特徴的周波数のピークとして現れる。
- **出力** — `IRTSpectrum(frequencies=(0.0, 0.2, 0.4, ..., 50.0), power=(0.001, 0.002, ..., 0.05, ...), peak_frequency=5.00, n=512, sampling_rate=100.0)`
- **引用** — Wei, W. W. S. (2006). *Time series analysis* (2nd ed.). Pearson.

![5 Hz サイン波 + ノイズのピリオドグラム、ピーク注釈付き](../../assets/figures/irt_spectrum.png)

### Gaussian HMM セグメンテーション — `GaussianHMMFit`

- **テーマ** — Schneider の 2 状態モデルを `K` 状態 Gaussian HMM（Baum-Welch で訓練、Viterbi で復号）に一般化。固定 pause/run スケジュール以外の non-stationary 状態区分検出に適する。
- **出力** — `GaussianHMMFit(pi=(...), transition_matrix=((...),(...)), means=(1.0, 5.0), sigmas=(0.4, 0.4), log_likelihood=-220.0, state_sequence=(0, 0, ..., 1, 1, ..., 0), segments=(...), n=240, n_iterations=18, converged=True)`
- **引用** — Rabiner, L. R. (1989). A tutorial on hidden Markov models. *Proceedings of the IEEE*, 77(2), 257-286. <https://doi.org/10.1109/5.18626>

![Viterbi で状態シェードした観測系列](../../assets/figures/hmm_segment.png)
