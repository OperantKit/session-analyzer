# モジュールとその出典

[English](../en/modules.md)

`session_analyzer.analytics` のすべてのモジュール（46 個）と、`session_analyzer.visualizer` の 2 つの累積記録の図を、それぞれが引く出典とともに挙げます。主な分析の式と出力例は [分析カタログ](analyses/index.md) を参照してください。

## 図（`session_analyzer.visualizer`）

| 関数 | 描くもの |
|---|---|
| `plot_cumulative_record` | 強化の印を付けた累積反応記録 |
| `plot_irt_coded_cumulative_record` | DRL / DRO 用の、反応間時間（IRT）で符号化した累積記録 |

## 分析（`session_analyzer.analytics`）

| モジュール | 分析 | 出典 |
|---|---|---|
| `bout` | バウトに分解した反応率 | Shull, Gaynor, & Grimes (2001) |
| `log_survivor` | 反応間時間の対数生存分析と、折れ線によるバウトの当てはめ | Shull (1991) |
| `irt_spectrum` | 反応間時間の自己相関とピリオドグラム | |
| `latency` | 反応潜時の要約 | |
| `prp` | 強化後休止（PRP）の要約と quarter-life 指標 | Felton & Lyon (1966); Herrnstein & Morse (1957) |
| `run_length` | FR のラン長の分解 | Schlinger, Derenne, & Baron (2008) |
| `schneider_segment` | 休止と反応連続の 2 状態への分割 | Schneider (1969) |
| `within_session` | セッション内の反応率の推移 | McSweeney, Roll, & Cannon (1994) |
| `matching` | 一般化マッチング法則の当てはめ、単一選択肢の双曲線 | Baum (1974); Herrnstein (1970) |
| `hierarchical_gml` | 階層的な経験ベイズによる一般化マッチング法則の当てはめ | Gelman & Hill (2007) |
| `matching_induction` | マッチングと誘導（induction）による当てはめ | Baum & Grace (2020) |
| `time_matching` | 時間配分のマッチング | Baum & Rachlin (1969) |
| `choice` | 並立スケジュールでの選択の配分 | |
| `stay_switch` | 並立スケジュールでの滞在と切替の分解、切替遅延（COD）の遵守 | Stubbs & Pliskoff (1969) |
| `concurrent_chains` | 並立連鎖スケジュールの双曲割引による予測 | Mazur (2001) |
| `killeen_law` | Killeen の強化の数理原理 | Killeen (1994) |
| `breakpoint` | 累進比率（PR）の breakpoint の検出 | Richardson & Roberts (1996) |
| `pr_schedule` | 累進比率スケジュールの分類 | Hodos (1961); Richardson & Roberts (1996) |
| `demand` | P_max / O_max を含む指数型需要曲線の当てはめ | Hursh & Silberberg (2008) |
| `unit_price` | 単価弾力性と交差価格弾力性 | Hursh (1980); Hursh & Roma (2013) |
| `cross_elasticity_matrix` | 複数の財の交差弾力性の行列 | Hursh & Roma (2013) |
| `discounting` | 遅延割引の当てはめ | Mazur (1987); Rachlin (2006) |
| `indifference` | 調整手続きによる無差別点 | Mazur (1987) |
| `peak` | ピーク手続きの分析（ビン集計と 3 状態の最尤推定） | Roberts (1981); Church, Meck, & Gibbon (1994) |
| `scalar_timing` | スカラー計時の要約とスカラー性の検定 | Gibbon (1977) |
| `generalization` | 刺激般化勾配（ガウス型と指数型） | Guttman & Kalish (1956); Shepard (1987) |
| `stimulus_control` | 弁別比とピークシフト | Hanson (1959); Annau & Kamin (1961) |
| `signal_detection` | 信号検出理論（d'、c、β、A'、B''）と ROC / AUC | Green & Swets (1966); Hautus (1995) |
| `psychometric` | 心理測定関数の当てはめ（ロジット / プロビット） | Wichmann & Hill (2001) |
| `ddm` | EZ 拡散モデルのパラメータ推定 | Wagenmakers, van der Maas, & Grasman (2007) |
| `momentum` | 変化への抵抗 / 行動モメンタム | Nevin (1992) |
| `survival` | Kaplan-Meier の生存曲線 / 消去曲線 | Kaplan & Meier (1958) |
| `extinction_burst` | 消去バーストの検出 | Lerman & Iwata (1995) |
| `resurgence` | リサージェンス（標的反応と代替反応の 3 フェイズの手続き） | Epstein (1985); Shahan & Sweeney (2011) |
| `laws_of_covariance` | 共変の法則によるリサージェンスの予測 | Podlesnik & Baum (2024) |
| `renewal` | リニューアルの分類（ABA / ABC / AAB） | Bouton (2004) |
| `tracking` | パブロフ型の接近反応のトラッキング指標（サイントラッキングとゴールトラッキング） | Flagel et al. (2007); Meyer et al. (2012) |
| `observing` | 観察反応と条件性強化の強さ | Wyckoff (1952); Dinsmoor (1983) |
| `rescorla_wagner` | Rescorla-Wagner モデルの連合強度のシミュレーションと当てはめ | Rescorla & Wagner (1972) |
| `td_learning` | 表形式の TD(0) 強化学習の推移 | Sutton (1988); Schultz, Dayan, & Montague (1997) |
| `variability` | オペラント変動性の U 値 | Page & Neuringer (1985) |
| `lag_variability` | Lag-N の達成率 | Neuringer (2002) |
| `ngram_entropy` | N-gram のブロック / 条件付きエントロピーとマルコフ次数の推定 | Towse & Neil (1998) |
| `hmm_segment` | ガウス型の隠れマルコフモデルによる分割 | Rabiner (1989) |
| `single_case` | 単一事例デザインの非重複の効果量：NAP、PND、Tau、Tau-U | Parker & Vannest (2009); Scruggs, Mastropieri, & Casto (1987); Parker, Vannest, Davis, & Sauber (2011) |
| `visual_analysis` | 単一事例デザインの視覚的分析の補助：水準・傾向・変動・即時性・重なり | Kratochwill et al. (2010) |

この表には、各分析のもとになった主な公刊モデルを挙げています。各モジュールの docstring には、ここに挙げていない文献も含む完全な文献リストがあります（たとえば `irt_spectrum`、`choice`、累積記録の図）。

## 参考文献

- Annau, Z., & Kamin, L. J. (1961). The conditioned emotional response as a function of intensity of the US. *Journal of Comparative and Physiological Psychology*, 54(4), 428-432. https://doi.org/10.1037/h0042199
- Baum, W. M. (1974). On two types of deviation from the matching law: bias and undermatching. *Journal of the Experimental Analysis of Behavior*, 22(1), 231-242. https://doi.org/10.1901/jeab.1974.22-231
- Baum, W. M., & Grace, R. C. (2020). Matching theory and induction explain operant performance. *Journal of the Experimental Analysis of Behavior*, 113(2), 390-418. https://doi.org/10.1002/jeab.583
- Baum, W. M., & Rachlin, H. C. (1969). Choice as time allocation. *Journal of the Experimental Analysis of Behavior*, 12(6), 861-874. https://doi.org/10.1901/jeab.1969.12-861
- Bouton, M. E. (2004). Context and behavioral processes in extinction. *Learning & Memory*, 11(5), 485-494. https://doi.org/10.1101/lm.78804
- Church, R. M., Meck, W. H., & Gibbon, J. (1994). Application of scalar timing theory to individual trials. *Journal of Experimental Psychology: Animal Behavior Processes*, 20(2), 135-155. https://doi.org/10.1037/0097-7403.20.2.135
- Dinsmoor, J. A. (1983). Observing and conditioned reinforcement. *Behavioral and Brain Sciences*, 6(4), 693-704. https://doi.org/10.1017/S0140525X00017969
- Epstein, R. (1985). Extinction-induced resurgence: Preliminary investigations and possible applications. *The Psychological Record*, 35(2), 143-153. https://doi.org/10.1007/BF03394918
- Felton, M., & Lyon, D. O. (1966). The post-reinforcement pause. *Journal of the Experimental Analysis of Behavior*, 9(2), 131-134. https://doi.org/10.1901/jeab.1966.9-131
- Flagel, S. B., Watson, S. J., Robinson, T. E., & Akil, H. (2007). Individual differences in the propensity to approach signals vs goals promote different adaptations in the dopamine system of rats. *Psychopharmacology*, 191(3), 599-607. https://doi.org/10.1007/s00213-006-0535-8
- Gelman, A., & Hill, J. (2007). *Data analysis using regression and multilevel/hierarchical models*. Cambridge University Press.
- Gibbon, J. (1977). Scalar expectancy theory and Weber's law in animal timing. *Psychological Review*, 84(3), 279-325. https://doi.org/10.1037/0033-295X.84.3.279
- Green, D. M., & Swets, J. A. (1966). *Signal detection theory and psychophysics*. Wiley.
- Guttman, N., & Kalish, H. I. (1956). Discriminability and stimulus generalization. *Journal of Experimental Psychology*, 51(1), 79-88. https://doi.org/10.1037/h0046219
- Hanson, H. M. (1959). Effects of discrimination training on stimulus generalization. *Journal of Experimental Psychology*, 58(5), 321-334. https://doi.org/10.1037/h0042606
- Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on estimated values of d'. *Behavior Research Methods, Instruments, & Computers*, 27(1), 46-51. https://doi.org/10.3758/BF03203619
- Herrnstein, R. J. (1970). On the law of effect. *Journal of the Experimental Analysis of Behavior*, 13(2), 243-266. https://doi.org/10.1901/jeab.1970.13-243
- Herrnstein, R. J., & Morse, W. H. (1957). A conjunctive schedule of reinforcement. *Journal of the Experimental Analysis of Behavior*, 1(1), 15-24. https://doi.org/10.1901/jeab.1958.1-15
- Hodos, W. (1961). Progressive ratio as a measure of reward strength. *Science*, 134(3483), 943-944. https://doi.org/10.1126/science.134.3483.943
- Hursh, S. R. (1980). Economic concepts for the analysis of behavior. *Journal of the Experimental Analysis of Behavior*, 34(2), 219-238. https://doi.org/10.1901/jeab.1980.34-219
- Hursh, S. R., & Roma, P. G. (2013). Behavioral economics and empirical public policy. *Journal of the Experimental Analysis of Behavior*, 99(1), 98-124. https://doi.org/10.1002/jeab.7
- Hursh, S. R., & Silberberg, A. (2008). Economic demand and essential value. *Psychological Review*, 115(1), 186-198. https://doi.org/10.1037/0033-295X.115.1.186
- Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*, 53(282), 457-481. https://doi.org/10.1080/01621459.1958.10501452
- Killeen, P. R. (1994). Mathematical principles of reinforcement. *Behavioral and Brain Sciences*, 17(1), 105-135. https://doi.org/10.1017/S0140525X00033628
- Kratochwill, T. R., Hitchcock, J. H., Horner, R. H., Levin, J. R., Odom, S. L., Rindskopf, D. M., & Shadish, W. R. (2010). *Single-case designs technical documentation*. What Works Clearinghouse. https://ies.ed.gov/ncee/wwc/Docs/ReferenceResources/wwc_scd.pdf
- Lerman, D. C., & Iwata, B. A. (1995). Prevalence of the extinction burst and its attenuation during treatment. *Journal of Applied Behavior Analysis*, 28(1), 93-94. https://doi.org/10.1901/jaba.1995.28-93
- Mazur, J. E. (1987). An adjusting procedure for studying delayed reinforcement. In M. L. Commons, J. E. Mazur, J. A. Nevin, & H. Rachlin (Eds.), *Quantitative analyses of behavior, Vol. 5: The effect of delay and of intervening events on reinforcement value* (pp. 55-73). Lawrence Erlbaum Associates.
- Mazur, J. E. (2001). Hyperbolic value addition and general models of animal choice. *Psychological Review*, 108(1), 96-112. https://doi.org/10.1037/0033-295X.108.1.96
- McSweeney, F. K., Roll, J. M., & Cannon, C. B. (1994). The generality of within-session patterns of responding. *Animal Learning & Behavior*, 22(3), 252-266. https://doi.org/10.3758/BF03209833
- Meyer, P. J., Lovic, V., Saunders, B. T., Yager, L. M., Flagel, S. B., Morrow, J. D., & Robinson, T. E. (2012). Quantifying individual variation in the propensity to attribute incentive salience to reward cues. *PLoS ONE*, 7(6), e38987. https://doi.org/10.1371/journal.pone.0038987
- Neuringer, A. (2002). Operant variability: Evidence, functions, and theory. *Psychonomic Bulletin & Review*, 9(4), 672-705. https://doi.org/10.3758/BF03196324
- Nevin, J. A. (1992). An integrative model for the study of behavioral momentum. *Journal of the Experimental Analysis of Behavior*, 57(3), 301-316. https://doi.org/10.1901/jeab.1992.57-301
- Page, S., & Neuringer, A. (1985). Variability is an operant. *Journal of Experimental Psychology: Animal Behavior Processes*, 11(3), 429-452. https://doi.org/10.1037/0097-7403.11.3.429
- Parker, R. I., & Vannest, K. J. (2009). An improved effect size for single-case research: Nonoverlap of all pairs. *Behavior Therapy*, 40(4), 357-367. https://doi.org/10.1016/j.beth.2008.10.006
- Parker, R. I., Vannest, K. J., Davis, J. L., & Sauber, S. B. (2011). Combining nonoverlap and trend for single-case research: Tau-U. *Behavior Therapy*, 42(2), 284-299. https://doi.org/10.1016/j.beth.2010.08.006
- Podlesnik, C. A., & Baum, W. M. (2024). Understanding resurgence and other emergent activity with the laws of allocation, induction, and covariance. *Journal of the Experimental Analysis of Behavior*, 122(3), 375-391. https://doi.org/10.1002/jeab.4212
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286. https://doi.org/10.1109/5.18626
- Rachlin, H. (2006). Notes on discounting. *Journal of the Experimental Analysis of Behavior*, 85(3), 425-435. https://doi.org/10.1901/jeab.2006.85-05
- Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. In A. H. Black & W. F. Prokasy (Eds.), *Classical conditioning II: Current research and theory* (pp. 64-99). Appleton-Century-Crofts.
- Richardson, N. R., & Roberts, D. C. S. (1996). Progressive ratio schedules in drug self-administration studies in rats: a method to evaluate reinforcing efficacy. *Journal of Neuroscience Methods*, 66(1), 1-11. https://doi.org/10.1016/0165-0270(95)00153-0
- Roberts, S. (1981). Isolation of an internal clock. *Journal of Experimental Psychology: Animal Behavior Processes*, 7(3), 242-268. https://doi.org/10.1037/0097-7403.7.3.242
- Schlinger, H. D., Derenne, A., & Baron, A. (2008). What 50 years of research tell us about pausing under ratio schedules of reinforcement. *The Behavior Analyst*, 31(1), 39-60. https://doi.org/10.1007/BF03392160
- Schneider, B. A. (1969). A two-state analysis of fixed-interval responding in the pigeon. *Journal of the Experimental Analysis of Behavior*, 12(5), 677-687. https://doi.org/10.1901/jeab.1969.12-677
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*, 275(5306), 1593-1599. https://doi.org/10.1126/science.275.5306.1593
- Scruggs, T. E., Mastropieri, M. A., & Casto, G. (1987). The quantitative synthesis of single-subject research: Methodology and validation. *Remedial and Special Education*, 8(2), 24-33. https://doi.org/10.1177/074193258700800206
- Shahan, T. A., & Sweeney, M. M. (2011). A model of resurgence based on behavioral momentum theory. *Journal of the Experimental Analysis of Behavior*, 95(1), 91-108. https://doi.org/10.1901/jeab.2011.95-91
- Shepard, R. N. (1987). Toward a universal law of generalization for psychological science. *Science*, 237(4820), 1317-1323. https://doi.org/10.1126/science.3629243
- Shull, R. L. (1991). Mathematical description of operant behavior: An introduction. In I. H. Iversen & K. A. Lattal (Eds.), *Experimental analysis of behavior* (Vol. 2, pp. 243-282). Elsevier.
- Shull, R. L., Gaynor, S. T., & Grimes, J. A. (2001). Response rate viewed as engagement bouts: effects of relative reinforcement and schedule type. *Journal of the Experimental Analysis of Behavior*, 75(3), 247-274. https://doi.org/10.1901/jeab.2001.75-247
- Stubbs, D. A., & Pliskoff, S. S. (1969). Concurrent responding with fixed relative rate of reinforcement. *Journal of the Experimental Analysis of Behavior*, 12(6), 887-895. https://doi.org/10.1901/jeab.1969.12-887
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*, 3(1), 9-44. https://doi.org/10.1007/BF00115009
- Towse, J. N., & Neil, D. (1998). Analyzing human random generation behavior: A review of methods used and a computer program for describing performance. *Behavior Research Methods, Instruments, & Computers*, 30(4), 583-591. https://doi.org/10.3758/BF03209475
- Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P. (2007). An EZ-diffusion model for response time and accuracy. *Psychonomic Bulletin & Review*, 14(1), 3-22. https://doi.org/10.3758/BF03194023
- Wichmann, F. A., & Hill, N. J. (2001). The psychometric function: I. Fitting, sampling, and goodness of fit. *Perception & Psychophysics*, 63(8), 1293-1313. https://doi.org/10.3758/BF03194544
- Wyckoff, L. B. (1952). The role of observing responses in discrimination learning. Part I. *Psychological Review*, 59(6), 431-442. https://doi.org/10.1037/h0053932
