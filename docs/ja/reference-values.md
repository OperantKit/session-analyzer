# 公表値との照合

[English](../en/reference-values.md)

`session_analyzer.analytics` の各分析を、出典とどう照合しているかの一覧です。analytics の 46 本のうち、公表値と一致 6 本、独立な参照実装と一致 5 本、原典の式と一致 1 本、公表値はあるが本文を取れていない 7 本、公表値の無い 27 本です。後ろの 34 本は閉形式・既知のパラメータの回復・手計算で確かめています。

各行は `@pytest.mark.reference(module=..., verdict=..., source=...)` を付けたテストです。`tests/test_reference_value_coverage.py` は、印の付いたテストを持たない analytics の module があるとき、`properties`・`not-retrieved` の module が理由付きの許可リストに無いとき、この表が印と食い違うときに失敗します。

判定:

- `published`: 原典、または参照パッケージの公表出力に載る値と一致。
- `reference-implementation`: 同じデータを独立なパッケージにかけた結果と一致。
- `definition`: 原典の式と一致（原典に数値例が無い）。
- `not-retrieved`: 原典に値は載るが、本文を取れていない（スキャン PDF・有料）。それまでは性質で確かめる。
- `properties`: 公表値が無い。閉形式・既知のパラメータの回復・手計算で確かめる。

| module | 判定 | 出典 | 参照実装 | テストが確かめること |
|---|---|---|---|---|
| `ddm` | `published` | Wagenmakers, van der Maas & Grasman (2007) | EZ-diffusion (R, appendix) | 付録の出力 v 0.09993853・a 0.1399702・Ter 0.30003 |
| `demand` | `published` | Hursh & Silberberg (2008); Kaplan et al. (2019) | beezdemand (R) | beezdemand README の出力: Q0 10.158665（`fit_space="log10"`）、Pmaxd 8.453799、Omaxd 22.63159 |
| `pr_schedule` | `published` | Richardson & Roberts (1996) | — | 比率列の 1〜10 段が公表の列（二次資料）と一致、11〜20 段は未照合 |
| `signal_detection` | `published` | Stanislaw & Todorov (1999) | — | 図 1: d′ 2.00・c −0.50・ln β −1.00・A′ .89・B″ −0.55 |
| `single_case` | `published` | Parker, Vannest & Davis (2011) | SingleCaseES (R), scan (R) | NAP・PND・Tau-U が SingleCaseES / scan の公表出力と一致 |
| `visual_analysis` | `published` | Kratochwill et al. (2010) | scan (R) | 傾向が scan の `trend()` 出力（scan-Book 8 章）と一致 |
| `discounting` | `reference-implementation` | Myerson & Green (1995) | discountingtools (R) | discountingtools の生成式で作ったデータから k を回復 |
| `hierarchical_gml` | `reference-implementation` | Gelman & Hill (2007) | statsmodels `MixedLM` | 推定値・対数尤度・固定効果の標準誤差が statsmodels（最尤法）と一致 |
| `hmm_segment` | `reference-implementation` | Rabiner (1989) | hmmlearn `GaussianHMM` | 推定値・対数尤度・Viterbi 経路が hmmlearn と一致 |
| `psychometric` | `reference-implementation` | Wichmann & Hill (2001) | statsmodels binomial GLM | μ・σ・対数尤度が statsmodels と一致（γ = 0・lapse = 0） |
| `survival` | `reference-implementation` | Kaplan & Meier (1958) | lifelines `KaplanMeierFitter` | Freireich の 6-MP 群で S(t)・中央値 23 が lifelines と一致 |
| `tracking` | `definition` | Meyer et al. (2012) | — | Table 1 の式と ±0.5 のしきい値（数値例なし） |
| `generalization` | `not-retrieved` | Guttman & Kalish (1956) | — | 有料で未取得。独立なプロファイル最小二乗と一致 |
| `killeen_law` | `not-retrieved` | Killeen (1994) | — | 当てはめ値の表は有料で未取得。β・γ の識別性 |
| `matching` | `not-retrieved` | Baum (1974) | `numpy.polyfit` | Table 1 の本文未取得。当てはめは最小二乗と一致、単一観測は手計算 |
| `momentum` | `not-retrieved` | Nevin (1992) | `numpy.linalg.lstsq` | 公表値は本文未取得。既知の m を回復 |
| `peak` | `not-retrieved` | Church, Meck & Gibbon (1994) | — | 有料で未取得。既知の低-高-低の試行を回復 |
| `schneider_segment` | `not-retrieved` | Schneider (1969) | — | breakpoint の表は本文未取得。正確な 2 直線の記録から回復 |
| `time_matching` | `not-retrieved` | Baum & Rachlin (1969) | `numpy.polyfit` | データの表は本文未取得。最小二乗・`matching` と一致 |
| `bout` | `properties` | Shull, Gaynor & Grimes (2001) | — | 既知の 2 成分混合を回復（原著の対数生存法とは推定法が違う） |
| `breakpoint` | `properties` | Richardson & Roberts (1996) | — | 公表の比率列の上で最後に完了した比率（閾値ちょうどを含む） |
| `choice` | `properties` | Herrnstein (1961) | — | 比率としきい値の手計算 |
| `concurrent_chains` | `properties` | Mazur (2001) | — | V = A/(1 + kD) の閉形式 |
| `cross_elasticity_matrix` | `properties` | Hursh & Roma (2013) | `numpy.polyfit` | 偏弾力性を回復、`unit_price` と一致 |
| `extinction_burst` | `properties` | Lerman & Iwata (1995) | — | しきい値と窓の境界の手計算 |
| `indifference` | `properties` | Mazur (1987) | — | 安定窓の手計算、滴定の収束 |
| `irt_spectrum` | `properties` | Wei (2006) | `numpy.correlate`, `scipy.signal.periodogram` | NumPy / SciPy の標準関数と一致 |
| `lag_variability` | `properties` | Page & Neuringer (1985) | — | Lag-N の合格率の手計算 |
| `latency` | `properties` | none cited | Python `statistics` | 対応付けと打ち切りの手計算、要約は `statistics` と一致 |
| `laws_of_covariance` | `properties` | Podlesnik & Baum (2024) | — | 閉形式（点予測の式は本 module 独自） |
| `log_survivor` | `properties` | Shull (1991) | — | Poisson の IRT で傾き −rate の直線 |
| `matching_induction` | `properties` | Baum & Grace (2020) | — | 誘導項 = tanh(ln r / 2)、パラメータを回復（式は本 module 独自） |
| `ngram_entropy` | `properties` | Towse & Neil (1998) | `scipy.stats.entropy` | de Bruijn 列で H_n = n ビット |
| `observing` | `properties` | Dinsmoor (1983) | — | 率と比の手計算 |
| `prp` | `properties` | Fry, Kelleher & Cook (1960) | — | 累積が t^p のとき quarter-life = 0.25^(1/p) |
| `renewal` | `properties` | Bouton (2004) | — | 文脈の設計の全組と比の手計算 |
| `rescorla_wagner` | `properties` | Rescorla & Wagner (1972) | — | V_n = λ(1 − (1 − αβ)^n) |
| `resurgence` | `properties` | Epstein (1985) | — | 窓つきの平均・差・比の手計算 |
| `run_length` | `properties` | Schlinger, Derenne & Baron (2008) | — | 休止とランの手計算 |
| `scalar_timing` | `properties` | Gibbon (1977) | — | 変動係数の手計算 |
| `stay_switch` | `properties` | Stubbs & Pliskoff (1969) | — | 切替と COD 遵守の手計算 |
| `stimulus_control` | `properties` | Hanson (1959) | — | 抑制比とピークシフトの手計算 |
| `td_learning` | `properties` | Sutton (1988) | — | 割引収益（閉形式）に収束 |
| `unit_price` | `properties` | Hursh (1980) | `numpy.polyfit` | 単位価格の等価性、弾力性の回復、最小二乗と一致 |
| `variability` | `properties` | Page & Neuringer (1985) | `scipy.stats.entropy` | U = 1 / 0 / 2/3 の閉形式 |
| `within_session` | `properties` | McSweeney, Roll & Cannon (1994) | — | 既知の逆 U を回復 |

各出典の書誌は[分析カタログ](analyses/index.md)と各 module の docstring にあります。
