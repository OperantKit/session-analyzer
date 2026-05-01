# 時間制御

[← カタログ目次](index.md) · [English](../../en/analyses/temporal.md)

応答時刻の分析：応答がスケジュールの時間構造をどう追従しているか、
セッション内でどう推移するか。

---

### `peak_procedure_binned` — `PeakTrialBinnedMetrics`

- **テーマ** — peak trial（強化子なし試行）では応答率が上昇し、期待時刻付近でピークを打ち、再び下降する — 内部時計のシグネチャ。binned rate 分析で peak time / peak rate / spread を得る。
- **出力** — `PeakTrialBinnedMetrics(peak_time=30.2, peak_rate=1.24, start_time=18.7, stop_time=42.8, spread=24.1, middle_time=30.75, n_responses=58, bin_width=1.0)`
- **引用** — Roberts, S. (1981). Isolation of an internal clock. *J. Exp. Psychol.: ABP*, 7(3), 242-268. <https://doi.org/10.1037/0097-7403.7.3.242>

![peak-trial レートカーブ + peak / spread マーカ](../../assets/figures/peak_procedure.png)

### `peak_procedure_mle` — `PeakTrialMLEMetrics`（3 状態 MLE、**heavy tier**）

- **テーマ** — Church, Meck, & Gibbon (1994) は peak trial を 3 つの Poisson 状態（低・高・低）でモデル化し、2 つの switch time `(s1, s2)` を MLE 推定する。grand-average ではなく試行レベルの遷移時刻が得られる。
- **出力** — `PeakTrialMLEMetrics(start_time=20.1, stop_time=41.5, high_rate=1.82, low_rate=0.21, middle_time=30.8, spread=21.4, n_responses=58, log_likelihood=-132.7)`
- **引用** — Church, R. M., Meck, W. H., & Gibbon, J. (1994). Application of scalar timing theory to individual trials. *J. Exp. Psychol.: ABP*, 20(2), 135-155. <https://doi.org/10.1037/0097-7403.20.2.135>

### セッション内ダイナミクス — `WithinSessionDynamics`

- **テーマ** — セッション内応答率は典型的に逆 U 字を辿る（McSweeney, Roll, & Cannon, 1994）：短い warm-up 上昇、ピーク、飽和による下降。bin 軌道と rise / fall 傾きでこの軌道を要約し、動機減衰と純粋な rate 制御を分離できる。
- **出力** — `WithinSessionDynamics(rates=[0.5, 1.4, 2.1, 2.4, 2.0, 1.5, 1.0, 0.7], peak_bin=3, peak_rate=2.4, rise_slope=0.62, fall_slope=-0.34, bilinear_r_squared=0.96, n_bins=8, n_responses=240)`
- **引用** — McSweeney, F. K., Roll, J. M., & Cannon, C. B. (1994). The generality of within-session patterns of responding. *Animal Learning & Behavior*, 22(3), 252-266. <https://doi.org/10.3758/BF03209833>

![逆 U 字の bin 軌道と peak マーカ](../../assets/figures/within_session.png)

### スカラー時間 — `ScalarTimingResult` / `ScalarPropertyTest`

- **テーマ** — Scalar Expectancy Theory（Gibbon, 1977）は時間弁別課題の応答分布が **scalar property** を満たすことを予測する：標準偏差が平均と比例し、変動係数 (CV) が criterion を超えて一定になる。
- **式** — `CV = σ / μ`; scalar 性は `std(CVs) / mean(CVs) <= tolerance` のとき成立。
- **出力** — `ScalarTimingResult(mean=30.0, std=7.5, coefficient_of_variation=0.25, n=500)`、`ScalarPropertyTest(criteria=(15.0, 30.0, 60.0), cvs=(0.25, 0.25, 0.25), mean_cv=0.25, cv_of_cvs=0.005, is_scalar=True, tolerance=0.20)`
- **引用** — Gibbon, J. (1977). Scalar expectancy theory and Weber's law in animal timing. *Psychological Review*, 84(3), 279-325. <https://doi.org/10.1037/0033-295X.84.3.279>

![criterion で正規化すると分布が重なる（scalar 性）](../../assets/figures/scalar_timing.png)

### Drift-diffusion (EZ-DDM) — `EZDiffusionParameters`

- **テーマ** — 拡散モデルの 3 主パラメータを精度・RT モーメントから閉形式で復元する Wagenmakers et al. (2007) の EZ 近似。フル尤度ベースの DDM が過剰なときに `(Pc, MRT, VRT)` の組から `(v, a, Ter)` を返す。
- **出力** — `EZDiffusionParameters(v=0.20, a=0.12, ter=0.30, s=0.1)`
- **引用** — Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P. (2007). An EZ-diffusion model for response time and accuracy. *Psychonomic Bulletin & Review*, 14(1), 3-22. <https://doi.org/10.3758/BF03194023>

![drift に対する Pc と平均 RT、EZ 復元点の注釈付き](../../assets/figures/ez_diffusion.png)

### TD(0) 学習 — `TDLearningTrace`

- **テーマ** — 表形式 TD(0) 価値学習（Sutton, 1988）。固定された state-reward 軌跡に対して `simulate_td_zero` が `δ_t = r + γ V(s_{t+1}) − V(s_t)`、`V(s_t) ← V(s_t) + α δ_t` を 1 パス実行する。Schultz, Dayan, & Montague (1997) のドーパミン予測誤差説の基盤として使える。
- **出力** — `TDLearningTrace(values=(1.0, 1.0, 0.0), errors=(0.0, 1.0), alpha=1.0, gamma=1.0, n_steps=2)`
- **引用** — Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*, 3(1), 9-44. <https://doi.org/10.1007/BF00115009>

![エピソードを通じた V(s_0) と V(s_1) のバックアップ](../../assets/figures/td_learning.png)
