# 持続性・消去

[← カタログ目次](index.md) · [English](../../en/analyses/persistence-and-extinction.md)

応答が disruptor にどれだけ耐えるか、消去でどう弱まるか、文脈変化
（renewal）や代替随伴性の撤去（resurgence）でどう回復するか。

---

### `behavioral_momentum` — `MomentumParameters`

- **テーマ** — Nevin (1992) は変化抵抗を行動質量として定式化：`log(B_x / B_o) = -x / m`。`x` は disruptor（pre-feeding、消去時間、代替強化）、`m` は成分固有の質量。`m` が大きいほど持続的。
- **式** — `log10(B_x / B_o) = -x / m`
- **出力** — `MomentumParameters(m=14.2, r_squared=0.91, n=5)`
- **引用** — Nevin, J. A. (1992). An integrative model for the study of behavioral momentum. *JEAB*, 57(3), 301-316. <https://doi.org/10.1901/jeab.1992.57-301>

![相対率の log-linear 減衰](../../assets/figures/behavioral_momentum.png)

### `survival` — `SurvivalCurve`（Kaplan-Meier）

- **テーマ** — 消去下での行動「生存」の非パラメトリック推定：時刻 `t` までに最終応答を出していない確率。右打ち切り観測も扱う。
- **式** — `S(t) = Π_{t_i ≤ t} (1 - d_i / n_i)`
- **出力** — `SurvivalCurve(times=(12.3, 45.8, ...), survival=(0.95, 0.80, ...), at_risk=(20, 16, ...), events=(1, 3, ...), n=20, n_events=18, median_survival_time=89.4)`
- **引用** — Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *JASA*, 53(282), 457-481. <https://doi.org/10.1080/01621459.1958.10501452>

![Kaplan-Meier 階段曲線](../../assets/figures/survival.png)

### Resurgence — `ResurgenceMetrics`

- **テーマ** — 3 相手続き（Epstein, 1985; Shahan & Sweeney, 2011）：Phase 1 でターゲット応答を強化、Phase 2 で代替の DRA により抑制、Phase 3 で両方消去。Phase 3 でターゲットが一過的に **再出現** する。magnitude（test rate − suppression rate）と ratio（test rate / baseline rate）、代替自身のダイナミクスも報告。
- **出力** — `ResurgenceMetrics(baseline_target_rate=42.7, suppression_target_rate=2.7, test_target_rate=15.0, resurgence_magnitude=12.3, resurgence_ratio=0.35, alt_baseline_rate=41.5, alt_test_rate=30.0, resurged=True, n_phase1=5, n_phase2=5, n_phase3=5)`
- **引用** — Epstein, R. (1985). Extinction-induced resurgence. *Psychological Record*, 35(2), 143-153. / Shahan, T. A., & Sweeney, M. M. (2011). A model of resurgence based on behavioral momentum theory. *JEAB*, 95(1), 91-108. <https://doi.org/10.1901/jeab.2011.95-91>

### Renewal（ABA / ABC / AAB）— `RenewalMetrics`

- **テーマ** — Renewal は消去文脈と異なる文脈でテストすると消去応答が回復する現象（Bouton, 2004）。3 つの正典設計：**ABA**（A で訓練、B で消去、A でテスト）、**ABC**（A 訓練、B 消去、C テスト）、**AAB**（A 訓練、A 消去、B テスト）。3 件の `PhaseSummary` を入力に文脈ラベルから設計を分類し magnitude を報告。
- **式** — `magnitude = test_rate − extinction_rate`; `ratio = test_rate / train_rate`
- **出力** — `RenewalMetrics(design="ABA", contexts=("A", "B", "A"), train_rate=1.0, extinction_terminal_rate=0.1, test_rate=0.5, renewal_magnitude=0.4, renewal_ratio=0.5, renewed=True)`
- **引用** — Bouton, M. E. (2004). Context and behavioral processes in extinction. *Learning & Memory*, 11(5), 485-494. <https://doi.org/10.1101/lm.78804> / Bouton, M. E., & Bolles, R. C. (1979). Contextual control of the extinction of conditioned fear. *Learning and Motivation*, 10(4), 445-466. <https://doi.org/10.1016/0023-9690(79)90057-2>

> 記録メモ：``session-recorder`` は OKL v1 wire format で
> ``phase_enter`` / ``phase_exit`` レコードを emit し、オプションの
> 文脈座標（``name`` / ``time`` / ``location`` / ``cue``）を保持する。
> 下流側で `PhaseSummary` のリストに変換して本 analyzer に与える。
