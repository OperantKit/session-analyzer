# Choice & matching

[← Catalog index](index.md) · [日本語版](../../ja/analyses/choice-and-matching.md)

How responding (or time) is allocated across two or more
simultaneously-available alternatives.

---

### `matching_law` — `MatchingParameters` (Generalized Matching Law)

- **Theme** — On concurrent schedules, subjects allocate responding in proportion to reinforcement. The Generalized Matching Law relaxes strict matching with a sensitivity `a` and bias `log b`. Undermatching (`a < 1`) is the most common empirical finding.
- **Equation** — `log(B1/B2) = a · log(R1/R2) + log b` (Baum, 1974)
- **Output** — `MatchingParameters(sensitivity=0.84, bias=0.02, r_squared=0.97, n=6)`
- **Reference** — Baum, W. M. (1974). On two types of deviation from the matching law. *JEAB*, 22(1), 231-242. <https://doi.org/10.1901/jeab.1974.22-231>

![Generalized matching law log-log fit](../../assets/figures/matching_law.png)

### Single-alternative matching — `SingleAlternativeMatchingParameters`

- **Theme** — Herrnstein (1970) showed that even on a single VI, response rate `B` is a hyperbolic function of reinforcement rate `R`, with `k` the asymptotic rate and `R_e` the extraneous reinforcement rate. Pharmacological studies often report `k` as the efficacy measure.
- **Equation** — `B = k · R / (R + R_e)`
- **Output** — `SingleAlternativeMatchingParameters(k=82.3, r_e=24.1, r_squared=0.93, n=8)`
- **Reference** — Herrnstein, R. J. (1970). On the law of effect. *JEAB*, 13(2), 243-266. <https://doi.org/10.1901/jeab.1970.13-243>

![Herrnstein hyperbola fit](../../assets/figures/single_alternative_matching.png)

### Time-allocation matching — `TimeMatchingParameters`

- **Theme** — The matching law applies to the *time* spent on each alternative as well as to response counts. Time allocation is sometimes a more sensitive measure of preference than response allocation, especially when response topographies differ across alternatives or responses are not discrete.
- **Equation** — `log(T1 / T2) = a · log(R1 / R2) + log b` (Baum & Rachlin, 1969)
- **Output** — `TimeMatchingParameters(sensitivity=0.79, bias=0.05, r_squared=0.95, n=6)`
- **Reference** — Baum, W. M., & Rachlin, H. C. (1969). Choice as time allocation. *JEAB*, 12(6), 861-874. <https://doi.org/10.1901/jeab.1969.12-861>

### `choice_allocation` — `ChoiceAllocation` / `ExclusiveChoiceResult`

- **Theme** — Raw per-operandum response proportions. Useful as the baseline view before fitting a matching-law equation; also surfaces "exclusive choice" when allocation collapses onto one alternative (a common failure mode of undertraining or saturated preference).
- **Output** — `ChoiceAllocation(proportions=(0.62, 0.38), counts=(248, 152), total=400)` and `ExclusiveChoiceResult(is_exclusive=False, dominant_operandum=None, dominant_proportion=None, threshold=0.95)`
- **Reference** — Catania, A. C. (1966). Concurrent performances. *JEAB*, 9(3), 253-263. <https://doi.org/10.1901/jeab.1966.9-253>

![Two-operandum allocation bar chart](../../assets/figures/choice_allocation.png)

### Concurrent-chains prediction — `ConcurrentChainsPrediction`

- **Theme** — On concurrent-chains schedules, terminal-link value is discounted hyperbolically and summed with initial-link effects. Mazur (2001) gives a single `k` that converts terminal-link delays and reinforcer magnitudes into preference predictions.
- **Equation** — `V = A / (1 + k · D)`; predicted `P(left) = V_L / (V_L + V_R)`
- **Output** — `ConcurrentChainsPrediction(v_left=0.12, v_right=0.07, predicted_proportion_left=0.63, observed_proportion_left=0.58, deviation=-0.05, k=0.05)`
- **Reference** — Mazur, J. E. (2001). Hyperbolic value addition and general models of animal choice. *Psychological Review*, 108(1), 96-112. <https://doi.org/10.1037/0033-295X.108.1.96>
