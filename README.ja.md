# session-analyzer

:gb: [English README](README.md)

OperantKit セッションログのための累積記録可視化、統計分析、行動モデル・フィッティング。

OKL v1 セッションログ（通常は `session-recorder` が生成）を読み込み、以下を生成する:

- 強化 tick マーク付きの累積反応記録
- DRL / DRO 可視化のための IRT 色分け累積記録
- バウト分解された反応率指標 (Shull, Gaynor, & Grimes, 2001)
- 一般化マッチング法則 (GML) のパラメータ・フィット (Baum, 1974)
- 単一選択双曲フィット (Herrnstein, 1970)
- 並立スケジュールにおける選択配分分析
- 反応潜時の要約統計量
- Progressive Ratio ブレイクポイント検出 (Richardson & Roberts, 1996)。ブレイクポイントは最後に完了した比率。MED-PC のプログラムによっては進行中の比率をブレイクポイント変数に入れるので、その値とは 1 段ずれる。`extract_step_ratios(..., timeout_sec=...)` は強化後タイムアウト中の反応を数えない。
- 遅延割引モデルのフィット (Mazur, 1987; Rachlin, 2006)
- 指数型需要曲線フィットと P_max / O_max (Hursh & Silberberg, 2008)
- 行動慣性・変化への抵抗の推定 (Nevin, 1992)
- Peak procedure 分析 — ビン平滑化と 3 状態 MLE (Roberts, 1981; Church, Meck, & Gibbon, 1994)
- 刺激般化勾配 — ガウス型と指数型 (Guttman & Kalish, 1956; Shepard, 1987)
- 強化後ポーズ (PRP) 要約と quarter-life 指標 (Felton & Lyon, 1966; Herrnstein & Morse, 1957)
- FR ラン長分解 (Schlinger, Derenne, & Baron, 2008)
- Kaplan-Meier 生存曲線・消去曲線 (Kaplan & Meier, 1958)
- 並立連鎖スケジュールの双曲減衰予測 (Mazur, 2001)
- パヴロフ型条件づけ接近の tracking index (Flagel et al., 2007; Meyer et al., 2012)
- オペラント変動性 U-value (Page & Neuringer, 1985) と Lag-N pass rate (Neuringer, 2002)
- Log-survivor IRT 分析と bilinear バウトフィット (Shull, 1991)
- 信号検出理論 (d', c, β, A', B'') と ROC / AUC (Green & Swets, 1966; Hautus, 1995)
- Resurgence (3 相手続き; Shahan & Sweeney, 2011)
- Renewal 分類 (ABA / ABC / AAB; Bouton, 2004)
- 時間配分マッチング (Baum & Rachlin, 1969)
- セッション内応答率ダイナミクス (McSweeney, Roll, & Cannon, 1994)
- Discrimination ratio と peak shift (Hanson, 1959; Annau & Kamin, 1961)
- Adjusting 無差異点 (Mazur, 1987)
- Unit-price と cross-price 弾力性 (Hursh, 1980; Hursh & Roma, 2013)
- 観察反応と条件性強化子の強度 (Wyckoff, 1952; Dinsmoor, 1983)
- Rescorla-Wagner 連合強度シミュレーション・フィッティング (Rescorla & Wagner, 1972)
- 表形式 TD(0) 強化学習トレース (Sutton, 1988; Schultz, Dayan, & Montague, 1997)
- スカラー時間要約とスカラー特性検定 (Gibbon, 1977)
- Killeen の定量的効果の法則 / MPR (Killeen, 1994)
- 並立スケジュールの stay-switch 分解と COD 遵守率 (Stubbs & Pliskoff, 1969)
- 消去バースト検出 (Lerman & Iwata, 1995)
- Schneider の 2 状態 break-and-run セグメンテーション (Schneider, 1969)
- 心理測定関数フィッティング（logit / probit; Wichmann & Hill, 2001）
- EZ 拡散モデルパラメータ復元 (Wagenmakers, van der Maas, & Grasman, 2007)
- IRT 自己相関とピリオドグラム・スペクトル
- N-gram ブロック / 条件付き entropy と Markov 次数推定 (Hopson, Burt, & Hopson, 2002)
- 多財クロス価格弾力性行列
- 階層経験ベイズ Generalized Matching Law フィット (Gelman & Hill, 2007)
- ガウス Hidden Markov Model セグメンテーション (Rabiner, 1989)

## インストール

```bash
pip install session-analyzer
```

開発用:

```
mise install                                # Python 3.11
mise exec -- python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pytest
```

## 30 秒で動く例

[session-recorder](https://github.com/OperantKit/session-recorder) で短い FR 5 のセッションを書き、読み戻し、強化後休止を測り、累積記録を描き、一般化対応法則を当てはめる:

```python
import math
from pathlib import Path

from experiment_core import ReinforcerStartEvent, ResponseEvent
from session_recorder import OklSink, iter_records
from session_analyzer import plot_cumulative_record
from session_analyzer.analytics import analyze_prps, compute_prps, estimate_matching_parameters

# 短い FR 5 の記録: 休止、0.5 秒間隔の 5 反応、強化子。
sink = OklSink("fr5.okl")
sink.write_header(session_name="fr5", clock_type="VirtualClock", session_start=0.0)
t, n = 0.0, 0
for pause in (4.0, 6.0, 8.0):
    t += pause
    for k in range(5):
        t += 0.5 if k else 0.0
        n += 1
        sink.emit(ResponseEvent(id=n, timestamp=t))
    sink.emit(ReinforcerStartEvent(id=n // 5, timestamp=t))

records = list(iter_records("fr5.okl"))
responses = [r.timestamp for r in records if r.type == "response"]
reinforcers = [r.timestamp for r in records if r.type == "reinforcer_start"]
print(compute_prps(reinforcers, responses))  # [6.0, 8.0]  強化後休止
print(analyze_prps(compute_prps(reinforcers, responses)).mean)  # 7.0

plot_cumulative_record(Path("fr5.okl"), Path("fr5.png"))  # 強化子の印つきの累積記録

# 並立 VI VI の 4 条件に一般化対応法則を当てはめる
# （a = 0.8、log b = 0.1 から作った説明用の比）:
r_ratios = [0.25, 0.5, 2.0, 4.0]
b_ratios = [math.exp(0.8 * math.log(r) + 0.1) for r in r_ratios]
fit = estimate_matching_parameters(b_ratios, r_ratios)
print(round(fit.sensitivity, 3), round(fit.bias, 3), round(fit.r_squared, 3))  # 0.8 0.1 1.0
```

`fr5.png` は OKL v1 ファイルから読んだ累積記録（階段状のペンと強化子の印）である。実際の研究では、記録は [session-runner](https://github.com/OperantKit/session-runner) や [operantkit-cli](https://github.com/OperantKit/operantkit-cli) で動かしたセッションから得られ、対応法則の比は複数の記録から得る。同じ分析はコマンドラインからも使える（下記 CLI）。

## 自作の 50 行の分析スクリプトでは足りない理由

短いスクリプトは細部を誤りやすい。本パッケージはそれらを 1 か所でテストして固めている:

- **公刊されたモデルを式と出典つきで.** 上に挙げた 43 の分析（分析モジュール 44 個）。一般化対応法則 (Baum, 1974) や Herrnstein の双曲線から、需要曲線・遅延割引・ピーク法の最尤推定・信号検出・Rescorla-Wagner の当てはめまで。レイテンシの要約を除くすべての分析モジュールは、モジュールの docstring に出典を書いている。
- **入力の検査で明示的に失敗する.** 例えば `compute_prps` は時刻が昇順でなければ例外を出し、`estimate_matching_parameters` は長さの不一致や正の比の組が 2 つ未満のとき、空の当てはめを返さず例外を出す。
- **セッション記録をそのまま読む.** OKL v1 の解析は session-recorder（`iter_records`）に任せるので、分析は記録されたのと同じイベントを見る。復活 (renewal) の分析が使うフェーズの区切りも含む。
- **慣習どおりの累積記録.** リセットつきの階段状のペン、一定角度の強化子の印、イベントペン。DRL / DRO 向けに IRT で色分けした版もある。
- **実験計画に応じた提案.** `session_analyzer.suggester` は contingency-dsl の AST から、その計画で計算する価値のある分析を返す（例: `Conc(VI, VI)` なら対応法則）。
- **テスト済み.** 488 件のテスト（`pytest`）と、この README の例。

## 分析カタログ

各分析のテーマ・式・出力例・正典引用の一覧は
[`docs/ja/analyses/index.md`](docs/ja/analyses/index.md)
（[English](docs/en/analyses/index.md)）を参照。
カタログは 8 つのカテゴリファイルに分割されている
（rate-and-pattern / choice-and-matching / value-and-economics /
temporal / stimulus-control / persistence-and-extinction /
pavlovian / variability）。

## 公開 API

```python
from session_analyzer import plot_cumulative_record
from session_analyzer.visualizer import plot_irt_coded_cumulative_record
from session_analyzer.analytics import (
    analyze_bouts, BoutMetrics,
    compute_log_survivor, fit_bilinear_log_survivor,
    LogSurvivorCurve, LogSurvivorBilinearFit,
    estimate_matching_parameters, MatchingParameters,
    fit_single_alternative_matching, SingleAlternativeMatchingParameters,
    estimate_time_matching_parameters, TimeMatchingParameters,
    compute_choice_allocation, classify_choice_mode,
    compute_latencies, analyze_latencies, LatencyMetrics,
    detect_breakpoint, BreakpointResult,
    fit_hyperbolic_discounting, fit_hyperboloid_discounting,
    fit_exponential_demand, DemandParameters,
    fit_unit_price_elasticity, compute_cross_price_elasticity,
    compute_unit_prices, UnitPriceElasticity, CrossPriceElasticity,
    estimate_indifference_point, IndifferencePoint,
    estimate_resistance_to_change, MomentumParameters,
    analyze_peak_trial_binned, analyze_peak_trial_mle,
    PeakTrialBinnedMetrics, PeakTrialMLEMetrics,
    analyze_within_session, WithinSessionDynamics,
    fit_gaussian_gradient, fit_exponential_gradient,
    GaussianGradientParameters, ExponentialGradientParameters,
    compute_discrimination_ratio, detect_peak_shift,
    DiscriminationRatio, PeakShiftResult,
    compute_sdt, compute_roc, SignalDetectionMetrics, ROCCurve,
    compute_prps, analyze_prps, compute_quarter_life, PRPMetrics,
    analyze_fr_runs, RunMetrics,
    kaplan_meier, SurvivalCurve,
    analyze_resurgence, ResurgenceMetrics,
    analyze_renewal, classify_renewal_design, PhaseSummary, RenewalMetrics,
    predict_concurrent_chains_choice, ConcurrentChainsPrediction,
    compute_tracking_index, TrackingIndex, TrackingType,
    compute_observing_metrics, compute_conditioned_reinforcement_strength,
    ObservingMetrics, ConditionedReinforcementStrength,
    compute_u_value, VariabilityMetrics,
    analyze_lag_variability, compute_lag_pass_rate, LagVariabilityMetrics,
)
```

## CLI

```
session-analyzer plot path/to/session.txt --output cumulative_record.png
session-analyzer diff-plot path/to/session.txt --irt-threshold 5.0 -o drl.png
session-analyzer latency path/to/session.txt --reference-type reinforcer_start
session-analyzer breakpoint path/to/session.txt --pause-threshold 300
session-analyzer discounting delays_values.csv --amount 100 --model hyperbolic
session-analyzer demand prices_consumption.csv --k 3.0
```

分析サブコマンドは JSON を標準出力に出力する。CSV 入力は `discounting`
が `delay,value`、`demand` が `price,consumption` のカラムを要求する。

## DSL 駆動の分析提案（`operantkit-frontend` 連携用）

`session_analyzer.suggester` は、`contingency-dsl` の解決済み AST を、
ダッシュボードやレポートが表示すべき分析の集合にマップする。
**実験デザイン → どの統計を計算する価値があるか** の正規マッピングが
このモジュールに一元化されており、`session-visualizer`、
`operantkit-frontend`、レポート生成器がすべて同じ源泉を参照する。

モジュール自体は**純 stdlib**。`from session_analyzer.suggester import
suggest_from_ast` は呼び出し時点で `scipy` や `matplotlib` の import を
必要としない（`session-analyzer` 本体はそれらを宣言しているが）。

### Python

```python
from session_analyzer.suggester import suggest_from_ast, suggestions_to_json

ast = {
    "type": "Program",
    "param_decls": [],
    "bindings": [],
    "schedule": {
        "type": "Compound",
        "combinator": "Conc",
        "components": [
            {"type": "Atomic", "dist": "V", "domain": "I", "value": 30},
            {"type": "Atomic", "dist": "V", "domain": "I", "value": 60},
        ],
    },
}
suggestions_to_json(suggest_from_ast(ast))
# [
#   {"name": "cumulative_record", "reason": "...", "tier": "light"},
#   {"name": "matching_law",      "reason": "...", "tier": "light"},
# ]
```

`Program` 全体でも `ScheduleExpr` 部分木でも受け付ける。frontend 側から
phase-sequence の 1 フェーズだけ渡して what-if チェックをすることも可能。

### HTTP 契約

`session-analyzer` がインストールされていれば、`session-visualizer` が
`POST /suggest` として suggester を公開する:

```
POST /suggest
Content-Type: application/json

<contingency-dsl の解決済み AST — Program または ScheduleExpr 部分木>
```

レスポンス:

```
{"suggestions": [{"name": "...", "reason": "...", "tier": "light" | "heavy"}, ...]}
```

不正入力時は HTTP 200 + `{"suggestions": [], "error": "..."}` を返し、
frontend 側で graceful な空状態を描画できる。

### TypeScript 型定義

手書きの TS ミラーを [`frontend-types/suggestions.ts`](frontend-types/suggestions.ts)
に同梱。コード生成なしで `operantkit-frontend` から直接 import できる:

```ts
import type { AnalysisSuggestion, SuggestResponse } from
  "session-analyzer/frontend-types/suggestions";

const { suggestions }: SuggestResponse = await fetch(`${VIZ}/suggest`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(ast),
}).then(r => r.json());
```

### マッピング

| DSL ノード | 推薦パネル | tier |
|---|---|---|
| *（全セッション共通）* | `cumulative_record` | light |
| `Atomic dist=V`（VR/VI） | `irt_distribution` | light |
| `Atomic domain=I`（FI/VI） | `post_reinforcement_pause` | light |
| `Compound combinator=Conc` | `matching_law` | light |
| `Compound combinator=Conc` + `PUNISH` | `punishment_sensitivity` | light |
| `Compound combinator=Mult/Mix/Chain/Tand` | `component_cumulative_records` | light |
| `Modifier modifier=PR` | `breakpoint`, `demand_curve` | light / heavy |
| `Modifier modifier=DRL` | `irt_distribution` | light |
| `Modifier modifier=DRH` | `rate_histogram` | light |
| `AversiveSchedule kind=Sidman/DiscrimAv` | `avoidance_rate` | light |
| `AversiveSchedule kind=Escape` | `escape_latency` | light |

文脈依存の抑制: `Mult/Mix/Chain/Tand` の中に入れ子になった `Conc` は
`matching_law` を**返さない** — matching law は共通時間軸を前提とし、
弁別刺激で分離されたコンポーネント間には適用できないため。
入れ子スケジュールは再帰的に walk し、重複名は集約する。

### tier の意味

- **light** — in-process の定期 tick で走らせてよい（移動窓、log-log OLS）。
- **heavy** — 本パッケージの非線形フィット群に委譲（需要曲線の非線形
  最小二乗、EM によるバウト分解、ブートストラップ CI 等）。

## 依存関係

`session-analyzer` は `session-recorder` に依存する（`pip install session-analyzer`
で一緒に入る）。OKL v1 の解析器は session-recorder が持つ。`contingency-py`
や `session-runner` には依存しない。セッション側との結合は OKL v1 ログの
スキーマ（後述の「ログフォーマット」）である。

## ログフォーマット

`session-analyzer` は `session-recorder` が生成する OKL v1
（OperantKitLog v1）— `.txt` 拡張子の UTF-8 プレーンテキスト —
を入力とする。完全な wire 仕様（magic 行、codebook ヘッダ、TAB 区切り
本文）は `session_recorder.format` に定義されており、本パッケージは
`session_recorder.iter_records` 経由で読み込む。各レコードに
`.timestamp`（float、セッション開始からの秒数、単調）、`.type`（str）、
`.args`（dict）があれば充分。

正典イベント型のうち `session-analyzer` が消費するのは `"response"` と
`"reinforcer_start"`。フェーズ境界（`"phase_enter"` / `"phase_exit"`）は
それを必要とする分析（例: renewal）が読み込む。未知の型は無視される。

## 参考文献

- Baum, W. M. (1974). On two types of deviation from the matching law: bias and undermatching. *Journal of the Experimental Analysis of Behavior*, 22(1), 231-242. https://doi.org/10.1901/jeab.1974.22-231
- Church, R. M., Meck, W. H., & Gibbon, J. (1994). Application of scalar timing theory to individual trials. *Journal of Experimental Psychology: Animal Behavior Processes*, 20(2), 135-155. https://doi.org/10.1037/0097-7403.20.2.135
- Felton, M., & Lyon, D. O. (1966). The post-reinforcement pause. *Journal of the Experimental Analysis of Behavior*, 9(2), 131-134. https://doi.org/10.1901/jeab.1966.9-131
- Flagel, S. B., Watson, S. J., Robinson, T. E., & Akil, H. (2007). Individual differences in the propensity to approach signals vs goals promote different adaptations in the dopamine system of rats. *Psychopharmacology*, 191(3), 599-607. https://doi.org/10.1007/s00213-006-0535-8
- Guttman, N., & Kalish, H. I. (1956). Discriminability and stimulus generalization. *Journal of Experimental Psychology*, 51(1), 79-88. https://doi.org/10.1037/h0046219
- Herrnstein, R. J. (1970). On the law of effect. *Journal of the Experimental Analysis of Behavior*, 13(2), 243-266. https://doi.org/10.1901/jeab.1970.13-243
- Herrnstein, R. J., & Morse, W. H. (1957). A conjunctive schedule of reinforcement. *Journal of the Experimental Analysis of Behavior*, 1(1), 15-24. https://doi.org/10.1901/jeab.1958.1-15
- Hursh, S. R., & Silberberg, A. (2008). Economic demand and essential value. *Psychological Review*, 115(1), 186-198. https://doi.org/10.1037/0033-295X.115.1.186
- Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*, 53(282), 457-481. https://doi.org/10.1080/01621459.1958.10501452
- Mazur, J. E. (1987). An adjusting procedure for studying delayed reinforcement. In M. L. Commons, J. E. Mazur, J. A. Nevin, & H. Rachlin (Eds.), *Quantitative analyses of behavior, Vol. 5* (pp. 55-73). Lawrence Erlbaum Associates.
- Mazur, J. E. (2001). Hyperbolic value addition and general models of animal choice. *Psychological Review*, 108(1), 96-112. https://doi.org/10.1037/0033-295X.108.1.96
- Meyer, P. J., Lovic, V., Saunders, B. T., Yager, L. M., Flagel, S. B., Morrow, J. D., & Robinson, T. E. (2012). Quantifying individual variation in the propensity to attribute incentive salience to reward cues. *PLoS ONE*, 7(6), e38987. https://doi.org/10.1371/journal.pone.0038987
- Nevin, J. A. (1992). An integrative model for the study of behavioral momentum. *Journal of the Experimental Analysis of Behavior*, 57(3), 301-316. https://doi.org/10.1901/jeab.1992.57-301
- Page, S., & Neuringer, A. (1985). Variability is an operant. *Journal of Experimental Psychology: Animal Behavior Processes*, 11(3), 429-452. https://doi.org/10.1037/0097-7403.11.3.429
- Rachlin, H. (2006). Notes on discounting. *Journal of the Experimental Analysis of Behavior*, 85(3), 425-435. https://doi.org/10.1901/jeab.2006.85-05
- Richardson, N. R., & Roberts, D. C. S. (1996). Progressive ratio schedules in drug self-administration studies in rats: a method to evaluate reinforcing efficacy. *Journal of Neuroscience Methods*, 66(1), 1-11. https://doi.org/10.1016/0165-0270(95)00153-0
- Roberts, S. (1981). Isolation of an internal clock. *Journal of Experimental Psychology: Animal Behavior Processes*, 7(3), 242-268. https://doi.org/10.1037/0097-7403.7.3.242
- Schlinger, H. D., Derenne, A., & Baron, A. (2008). What 50 years of research tell us about pausing under ratio schedules of reinforcement. *The Behavior Analyst*, 31(1), 39-60. https://doi.org/10.1007/BF03392160
- Shepard, R. N. (1987). Toward a universal law of generalization for psychological science. *Science*, 237(4820), 1317-1323. https://doi.org/10.1126/science.3629243
- Shull, R. L., Gaynor, S. T., & Grimes, J. A. (2001). Response rate viewed as engagement bouts: effects of relative reinforcement and schedule type. *Journal of the Experimental Analysis of Behavior*, 75(3), 247-274. https://doi.org/10.1901/jeab.2001.75-247
