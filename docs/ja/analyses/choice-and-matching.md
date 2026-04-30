# 選択・マッチング

[← カタログ目次](index.md) · [English](../../en/analyses/choice-and-matching.md)

同時利用可能な 2 つ以上の選択肢間で応答（または時間）がどう配分される
かを扱う分析群。

---

### `matching_law` — `MatchingParameters`（一般化マッチング法則）

- **テーマ** — 並立スケジュールでは応答配分が強化配分に比例する。一般化マッチング法則は厳密マッチングを感度 `a` と bias `log b` で緩める。実証的に `a < 1`（undermatching）が最頻。
- **式** — `log(B1/B2) = a · log(R1/R2) + log b`（Baum, 1974）
- **出力** — `MatchingParameters(sensitivity=0.84, bias=0.02, r_squared=0.97, n=6)`
- **引用** — Baum, W. M. (1974). On two types of deviation from the matching law. *JEAB*, 22(1), 231-242. <https://doi.org/10.1901/jeab.1974.22-231>

![一般化マッチング法則の log-log fit](../../assets/figures/matching_law.png)

### 単一選択肢マッチング — `SingleAlternativeMatchingParameters`

- **テーマ** — Herrnstein (1970) は単一 VI 上でも応答率 `B` が強化率 `R` の双曲線関数であることを示した。`k` は漸近応答率、`R_e` は外因性強化率。薬理研究では `k` を有効性指標として用いる。
- **式** — `B = k · R / (R + R_e)`
- **出力** — `SingleAlternativeMatchingParameters(k=82.3, r_e=24.1, r_squared=0.93, n=8)`
- **引用** — Herrnstein, R. J. (1970). On the law of effect. *JEAB*, 13(2), 243-266. <https://doi.org/10.1901/jeab.1970.13-243>

![Herrnstein 双曲線 fit](../../assets/figures/single_alternative_matching.png)

### 時間配分マッチング — `TimeMatchingParameters`

- **テーマ** — マッチング法則は応答数だけでなく **滞在時間** にも適用できる。応答トポグラフィーが選択肢ごとに異なる場合や応答が連続的な場合、時間配分のほうが選好の感度が高い。
- **式** — `log(T1 / T2) = a · log(R1 / R2) + log b`（Baum & Rachlin, 1969）
- **出力** — `TimeMatchingParameters(sensitivity=0.79, bias=0.05, r_squared=0.95, n=6)`
- **引用** — Baum, W. M., & Rachlin, H. C. (1969). Choice as time allocation. *JEAB*, 12(6), 861-874. <https://doi.org/10.1901/jeab.1969.12-861>

### `choice_allocation` — `ChoiceAllocation` / `ExclusiveChoiceResult`

- **テーマ** — operandum ごとの素の応答比率。マッチング法則をフィットする前のベースライン。配分が片側に潰れる「排他的選択」の検出にも使える（訓練不足・選好飽和の典型）。
- **出力** — `ChoiceAllocation(proportions=(0.62, 0.38), counts=(248, 152), total=400)`、`ExclusiveChoiceResult(is_exclusive=False, dominant_operandum=None, dominant_proportion=None, threshold=0.95)`
- **引用** — Catania, A. C. (1966). Concurrent performances. *JEAB*, 9(3), 253-263. <https://doi.org/10.1901/jeab.1966.9-253>

![2-operandum 配分の棒グラフ](../../assets/figures/choice_allocation.png)

### Concurrent-chains 予測 — `ConcurrentChainsPrediction`

- **テーマ** — 並立連鎖スケジュールでは terminal-link の価値が双曲割引され、initial-link 効果と加算される。Mazur (2001) は terminal-link の遅延・強化子量を選好予測に変換する単一 `k` を与える。
- **式** — `V = A / (1 + k · D)`; predicted `P(left) = V_L / (V_L + V_R)`
- **出力** — `ConcurrentChainsPrediction(v_left=0.12, v_right=0.07, predicted_proportion_left=0.63, observed_proportion_left=0.58, deviation=-0.05, k=0.05)`
- **引用** — Mazur, J. E. (2001). Hyperbolic value addition and general models of animal choice. *Psychological Review*, 108(1), 96-112. <https://doi.org/10.1037/0033-295X.108.1.96>
