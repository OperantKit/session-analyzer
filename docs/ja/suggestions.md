# DSL にもとづく分析の提案

[English](../en/suggestions.md)

`session_analyzer.suggester` は、`contingency-dsl` の解決済み AST を、ダッシュボードやレポートが表示すべき分析の集合に対応づけます。**実験デザインから、どの統計を計算する価値があるか**への正規の対応づけです。`session-visualizer`、`operantkit-frontend`、レポートの生成プログラムは、どれも同じこの源泉から表示するパネルの一覧を導きます。

このモジュールは **Python の標準ライブラリだけ**で動きます。`from session_analyzer.suggester import suggest_from_ast` は、呼び出しの時点で `scipy` や `matplotlib` を import できる必要がありません（`session-analyzer` 本体はそれらを依存として宣言しています）。

## Python

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
[(s["name"], s["tier"]) for s in suggestions_to_json(suggest_from_ast(ast))]
# [('cumulative_record', 'light'), ('matching_law', 'light'),
#  ('post_reinforcement_pause', 'light'), ('irt_distribution', 'light')]
```

各提案には `reason`（理由の文）も付きます。`Program` 全体でも `ScheduleExpr` の部分木でも受け付けます。フロントエンドは、Program を組み直さずに、フェイズの並びのうち 1 つのフェイズだけを渡して「もしこうなら」の確認ができます。

## HTTP の取り決め

`session-analyzer` が入っていれば、`session-visualizer` が提案機能を `POST /suggest` として公開します。

```
POST /suggest
Content-Type: application/json

<contingency-dsl の解決済み AST — Program または ScheduleExpr の部分木>
```

応答:

```
{"suggestions": [{"name": "...", "reason": "...", "tier": "light" | "heavy"}, ...]}
```

入力が不正なときは、HTTP 200 と `{"suggestions": [], "error": "..."}` を返します。フロントエンドはこれで空の状態を穏当に表示できます。

## TypeScript の型

手書きの TypeScript の写しを [`frontend-types/suggestions.ts`](../../frontend-types/suggestions.ts) に同梱しています。コード生成の手順なしで、`operantkit-frontend` から import できます。

```ts
import type { AnalysisSuggestion, SuggestResponse } from
  "session-analyzer/frontend-types/suggestions";

const { suggestions }: SuggestResponse = await fetch(`${VIZ}/suggest`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(ast),
}).then(r => r.json());
```

## 対応表

| DSL のノード | 提案するパネル | tier |
|---|---|---|
| *（すべてのセッション）* | `cumulative_record` | light |
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

文脈による抑制: `Mult/Mix/Chain/Tand` の中に入れ子になった `Conc` では、`matching_law` を**提案しません**。マッチング法則は 1 つの共通の時間軸を前提とし、弁別刺激で分けられたコンポーネントを前提としないためです。入れ子のスケジュールは再帰的にたどり、重複する名前はまとめます。

## tier の意味

- **light** — プロセス内の定期的な更新で走らせてよいもの（移動窓、両対数の最小二乗）。
- **heavy** — このパッケージの非線形の当てはめに回すもの（需要曲線の非線形最小二乗、EM によるバウトの分解、ブートストラップの信頼区間）。
