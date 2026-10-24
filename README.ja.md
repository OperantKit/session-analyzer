# session-analyzer

:gb: [English README](README.md)

session-analyzer は、セッションのデータから累積記録を描き、行動分析学の指標を計算します。強化後休止、一般化マッチング法則、需要曲線、遅延割引、単一事例デザインの効果量などです。オペラント実験のセッションを Python で分析する研究者・学生のためのパッケージです。[OperantKit](https://github.com/OperantKit) の中では、[session-recorder](https://github.com/OperantKit/session-recorder) が書くセッションログを読みます。関数は時刻や値の単純なリストも受け取ります。

- **公刊されたモデルを、テストされた 1 か所に。** 46 の分析モジュールがあり、それぞれがモデルの出典の論文を示しています。
- **入力を検査します。** 時刻が並んでいない、長さが合わない、データ点が少なすぎる、といった場合は、意味のない当てはめを返さずにエラーを出します。
- **累積記録を慣例どおりに描きます。** リセットのあるステップ式のペンと強化の印に加え、DRL / DRO 用に反応間時間（IRT）で符号化した版もあります。

AI コーディングエージェント向けの案内は [AGENTS.md](AGENTS.md) にあります。

<p align="center"><img src="docs/assets/figures/cumulative_record.png" width="360" alt="累積記録：階段状の線と、強化ごとの短い斜めの印"></p>

## インストール

Python 3.11 以上が必要です。session-analyzer はまだ PyPI に登録されていないため、`pip install session-analyzer` は今のところ使えません。session-analyzer と、それが必要とする OperantKit のパッケージを依存の解決をせずに GitHub から入れ、使う 3 つの外部ライブラリを足してください。

```bash
G=git+https://github.com/OperantKit
pip install --no-deps $G/contingency-dsl-py $G/contingency-py $G/experiment-core \
  $G/session-recorder $G/session-analyzer
pip install numpy scipy matplotlib
```

`pip check` はこれらの開発版どうしのバージョンの食い違いを報告しますが、以下の例の動作には影響しません。

## 例

session-recorder で短い FR 5 のセッションを書き、強化後休止（PRP）を測り、累積記録を描き、一般化マッチング法則を当てはめます。

```python
import math
from pathlib import Path

from experiment_core import ReinforcerStartEvent, ResponseEvent
from session_recorder import OklSink, iter_records
from session_analyzer import plot_cumulative_record
from session_analyzer.analytics import analyze_prps, compute_prps, estimate_matching_parameters

# 短い FR 5 の記録：休止、0.5 秒間隔の 5 反応、強化子。
sink = OklSink("fr5.txt")
sink.write_header(session_name="fr5", clock_type="VirtualClock", session_start=0.0)
t, n = 0.0, 0
for pause in (4.0, 6.0, 8.0):
    t += pause
    for k in range(5):
        t += 0.5 if k else 0.0
        n += 1
        sink.emit(ResponseEvent(id=n, timestamp=t))
    sink.emit(ReinforcerStartEvent(id=n // 5, timestamp=t))

records = list(iter_records("fr5.txt"))
responses = [r.timestamp for r in records if r.type == "response"]
reinforcers = [r.timestamp for r in records if r.type == "reinforcer_start"]
print(compute_prps(reinforcers, responses))  # [6.0, 8.0]
print(analyze_prps(compute_prps(reinforcers, responses)).mean)  # 7.0

plot_cumulative_record(Path("fr5.txt"), Path("fr5.png"))

# 並立 VI VI の 4 条件に一般化マッチング法則を当てはめる
# （a = 0.8、log b = 0.1 から作った説明用の比）:
r_ratios = [0.25, 0.5, 2.0, 4.0]
b_ratios = [math.exp(0.8 * math.log(r) + 0.1) for r in r_ratios]
fit = estimate_matching_parameters(b_ratios, r_ratios)
print(round(fit.sensitivity, 3), round(fit.bias, 3), round(fit.r_squared, 3))  # 0.8 0.1 1.0
```

1 回目と 2 回目の強化子の後の休止は 6 秒と 8 秒です。一般化マッチング法則の当てはめ（Baum, 1974）は、比を作ったときの感度（a = 0.8）と偏好（log b = 0.1）を復元します。実際の研究では、ログは [session-runner](https://github.com/OperantKit/session-runner) や [operantkit-cli](https://github.com/OperantKit/operantkit-cli) で走らせたセッションから得られ、マッチングの比はそうしたログを複数集めて求めます。

## 中身

- **分析**（`session_analyzer.analytics`）：反応率と局所的なパターン（強化後休止、バウト、反応間時間の対数生存、FR のラン）、選択とマッチング（一般化マッチング法則、時間配分、切替遅延の遵守、並立連鎖）、価値と経済（累進比率の breakpoint、需要、遅延割引、単価）、計時（ピーク手続き、スカラー計時）、刺激性制御（般化勾配、信号検出）、持続と消去（行動モメンタム、リサージェンス、リニューアル、消去バースト）、パブロフ型の指標（サイントラッキングとゴールトラッキング、Rescorla-Wagner）、変動性、単一事例デザイン。[すべてのモジュールとその出典](docs/ja/modules.md)と、主な分析の[式と出力例](docs/ja/analyses/index.md)があります。
- **図**（`session_analyzer.visualizer`）：`plot_cumulative_record`、`plot_irt_coded_cumulative_record`、`render_single_case_plot`。
- **分析の提案**（`session_analyzer.suggester`）：[contingency-dsl](https://github.com/OperantKit/contingency-dsl) の式を与えると、そのデザインで計算する価値のある分析を返します（たとえば `Conc(VI, VI)` ならマッチング法則）。[session-visualizer](https://github.com/OperantKit/session-visualizer) と [operantkit-frontend](https://github.com/OperantKit/operantkit-frontend) にとって、実験デザインから分析への正規の対応づけです。対応表・HTTP の取り決め・TypeScript の型は [DSL にもとづく分析の提案](docs/ja/suggestions.md) にあります。

### セッションログを読む

ログは session-recorder が書く OKL v1 のファイルです（形式の仕様は [OperantKitLog](https://github.com/OperantKit/OperantKitLog) にあります）。session-analyzer は `session_recorder.iter_records` を通して読み、各レコードの `.timestamp`（セッション開始からの秒）・`.type`・`.args` を使います。正典の出来事の種類のうち使うのは `response` と `reinforcer_start` です。フェイズが必要な分析（たとえばリニューアル）は `phase_enter` / `phase_exit` も読みます。それ以外の種類は無視します。

### コマンドライン

```
session-analyzer plot session.txt --output cumulative_record.png
session-analyzer diff-plot session.txt --irt-threshold 5.0 -o drl.png
session-analyzer latency session.txt --reference-type reinforcer_start
session-analyzer breakpoint session.txt --pause-threshold 300
session-analyzer discounting delays_values.csv --amount 100 --model hyperbolic
session-analyzer demand prices_consumption.csv --k 3.0
session-analyzer single-case cases.csv --plot cases.png
```

分析のサブコマンドは JSON を出力します。CSV の入力には、`delay,value`（`discounting`）、`price,consumption`（`demand`）、または `phase,value` と省略可能な `case`（`single-case`、行はセッションの順）の列が必要です。オプションは `session-analyzer <サブコマンド> --help` で確認できます。

### 単一事例デザイン

反転デザイン・多層ベースラインデザイン・条件交替デザインでは、セッションごとに 1 つの値を、そのフェイズのラベルとともに与えます。`visual_analysis` は視覚的な判断のよりどころとなる量を返します。フェイズごとに水準（平均、中央値）・傾向（最小二乗の傾き。セッション番号は各フェイズで 0 から数えます）・変動を、隣り合うフェイズの組ごとに水準と傾向の変化・即時性（後のフェイズの最初の 3 点から前のフェイズの最後の 3 点を引いたもの）・重なりを返します。定義は What Works Clearinghouse の単一事例デザインの基準（Kratochwill et al., 2010）に従います。`nonoverlap` は NAP、PND、Tau、Tau-U を加えます。

```python
from session_analyzer.analytics import SingleCaseSeries, nonoverlap, visual_analysis
from session_analyzer.visualizer import render_single_case_plot

case = SingleCaseSeries(
    values=(54, 53, 56, 58, 52, 61, 62, 71, 66, 64, 78, 70, 74, 82, 77),
    phases=("A",) * 5 + ("B",) * 10,
    name="Case 1",
)
va = visual_analysis(case)
print([(p.label, round(p.mean, 1), round(p.slope, 2)) for p in va.phases])
# [('A', 54.6, 0.1), ('B', 70.5, 1.97)]
print(round(va.contrasts[0].immediacy, 2), va.contrasts[0].overlap)  # 9.33 0.0
print(round(nonoverlap(case).nap, 3))  # 1.0
render_single_case_plot(case, "case1.png")
```

フェイズは連続した並びとして数えるため、`A, A, B, B, A, A, B, B` とラベルを付けた反転デザインは 4 つのフェイズと 3 つの比較になります。条件交替デザインでは `by="label"` を渡してください。複数の系列は、セッションの軸を共有する積み重ねたパネルとして描かれます（多層ベースライン）。次の図は [`docs/examples/single_case_multiple_baseline.csv`](docs/examples/single_case_multiple_baseline.csv) の合成データから描いたものです。

![フェイズ変更線、水準（点線）、傾向（破線）を引いた多層ベースラインのグラフ](docs/assets/figures/single_case.png)

これらは視覚的な判断の補助です。効果が示されたかどうかの判断は分析者に委ねます。

## 公表値との照合

analytics の 46 本のうち、公表値と一致 6 本、独立な参照実装と一致 5 本、原典の式と一致 1 本、公表値はあるが本文を取れていない 7 本、公表値の無い 27 本です。後ろの 34 本は閉形式・既知のパラメータの回復・手計算で確かめています。[module ごとの照合のしかた](docs/ja/reference-values.md)。出典に照らしたテストを持たない module があると `tests/test_reference_value_coverage.py` が失敗します。

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


## 制限

- 公刊された参照値と照合してテストしているのは、単一事例デザインの非重複の指標（`single_case`）だけです。他の分析は出典を示していますが、原著論文が報告した数値とはまだ照合していません。使う分析は、報告する前に原著と照らし合わせてください。
- ログは OKL v1 である必要があります。他のシステムのデータは先に変換してください（session-recorder は、プロトコルのあるプログラムについて MED-PC のデータファイルを変換できます）。

## 開発

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
```

この README と `README.md` の `python` のコードブロックはすべて、`tests/test_readme_examples.py` が実行します。

## 参考文献

- Baum, W. M. (1974). On two types of deviation from the matching law: bias and undermatching. *Journal of the Experimental Analysis of Behavior*, 22(1), 231-242. https://doi.org/10.1901/jeab.1974.22-231
- Kratochwill, T. R., Hitchcock, J. H., Horner, R. H., Levin, J. R., Odom, S. L., Rindskopf, D. M., & Shadish, W. R. (2010). *Single-case designs technical documentation*. What Works Clearinghouse. https://ies.ed.gov/ncee/wwc/Docs/ReferenceResources/wwc_scd.pdf

各分析の文献は [docs/ja/modules.md](docs/ja/modules.md) にあります。

## ライセンス

MIT。[LICENSE](LICENSE) を参照してください。
