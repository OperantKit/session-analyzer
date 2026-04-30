# 分析カタログ

`session_analyzer.analytics` に実装されている全分析のリファレンス。
各項目は **テーマ**（何を測っているか）、必要に応じて **式**、
パッケージが返すのと全く同じ形の **出力例**、そして **正典引用** から成る。

分析は 8 つのカテゴリファイルに分割されている。各ファイル内で参照する
`suggester.name` 列は [`session_analyzer.suggester`](../../../src/session_analyzer/suggester.py)
が返し、`operantkit-frontend` が消費する安定識別子。

同梱の図は `jeab-bw` テーマ（JEAB / JABA 系モノクロ、3.3 × 2.5 in、
300 dpi Helvetica）で [`docs/generate_figures.py`](../../generate_figures.py)
が描画する。再生成は `.venv/bin/python docs/generate_figures.py`。
データは合成のパラメータ回復カリカチュアであり実験データではない。

[English](../../en/analyses/index.md)

---

## カテゴリ

1. [応答率・局所パターン](rate-and-pattern.md) — 累積記録、PRP、
   IRT バウト分解、FR run length、潜時、log-survivor。
2. [選択・マッチング](choice-and-matching.md) — 一般化マッチング法則、
   単一選択肢双曲線、選択配分、concurrent-chains 予測、
   時間配分マッチング。
3. [価値・行動経済学](value-and-economics.md) — Progressive-Ratio
   breakpoint、指数需要曲線、遅延割引、unit-price / cross-price 弾力性、
   adjusting 無差異点。
4. [時間制御](temporal.md) — peak procedure（binned + MLE）、
   セッション内ダイナミクス。
5. [刺激性制御](stimulus-control.md) — 般化勾配（Gaussian / 指数）、
   discrimination ratio、peak shift、信号検出理論（d', c, β, ROC）。
6. [持続性・消去](persistence-and-extinction.md) — 行動モーメンタム、
   Kaplan-Meier 生存、resurgence（3 相）、renewal（ABA / ABC / AAB）。
7. [Pavlovian](pavlovian.md) — tracking index（sign- / goal-tracking）、
   観察反応、条件性強化子の強度。
8. [変動性](variability.md) — U-value、Lag-N variability。

---

## ティアリング

`AnalysisSuggestion.tier` は CPU 予算であって認識的地位ではない：

- **light** — 計算コストが O(ms)：閉形要約、log-log OLS、Kaplan-Meier、
  binned peak。`session-visualizer` の周期 tick (10 s / 1 min) で安全。
- **heavy** — 非線形最小二乗（demand、discounting、generalization）、
  EM 混合（bout）、3 状態 MLE（peak）。末尾で数分。
  `session-analyzer` 単発実行に回す。

---

## 関連

- [`suggester.py`](../../../src/session_analyzer/suggester.py) — 
  `contingency-dsl` AST → 上記分析のサブセット への写像。
- 最上位 [`README.md`](../../../README.md) の "Public API" セクションで
  正確な import パスと引数シグネチャを参照。
