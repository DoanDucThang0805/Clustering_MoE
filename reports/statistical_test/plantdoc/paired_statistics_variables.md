# Description of Variables in the Seed-Wise Paired Statistical Analysis

**Suggested caption:** Description of variables in the seed-wise paired statistical analysis on PlantDoc.

| Variable | Symbol | Description / Calculation |
|---|---:|---|
| `dataset` | – | The benchmark dataset evaluated (PlantDoc in this analysis). |
| `metric` | $m$ | The performance metric evaluated (Accuracy or Macro-F1). |
| `model_A` | $A$ | The reference architecture: Cluster-MoE G4 cosine top-2. |
| `model_B` | $B$ | The baseline or alternative architecture compared against Model A. |
| `n_seeds` | $N$ | The number of paired random seeds ($N=5$; seeds 42–46). |
| `mean_A` | $\bar{m}^{(A)}$ | The empirical mean of the metric for Model A across $N$ seeds. |
| `std_A` | $s^{(A)}$ | The sample standard deviation for Model A (denominator $N-1$). |
| `mean_B` | $\bar{m}^{(B)}$ | The empirical mean of the metric for Model B across $N$ seeds. |
| `std_B` | $s^{(B)}$ | The sample standard deviation for Model B (denominator $N-1$). |
| `mean_delta` | $\bar{\Delta}$ | The average paired difference: $\bar{\Delta}=N^{-1}\sum_{i=1}^{N}\left(m_i^{(A)}-m_i^{(B)}\right)$. |
| `paired_t_p` | $p_t$ | The raw two-sided paired $t$-test p-value for $H_0:\mathbb{E}[\Delta]=0$. |
| `wilcoxon_p` | $p_W$ | The raw two-sided Wilcoxon signed-rank p-value for paired differences. |
| `holm_p` | $p_{\mathrm{Holm}}$ | The Holm–Bonferroni-adjusted paired $t$-test p-value across all six tests. |
| `bh_p` | $p_{\mathrm{BH}}$ | The Benjamini–Hochberg-adjusted paired $t$-test p-value across all six tests. |
| `conclusion` | – | Significance at $\alpha=0.05$ based on `holm_p`; direction follows `mean_delta`. |

## Evaluation of Results

Cluster-MoE G4 cosine top-2 achieved the highest mean performance among all
four configurations, with an accuracy of 0.8028 and a Macro-F1 score of
0.7669. All six estimated paired differences were positive, indicating that
the cosine-routing configuration performed better on average than each
comparison model across the five matched seeds.

| Comparison (Model A vs. Model B) | Metric | Mean A | Mean B | Mean difference | Holm-adjusted p | BH-adjusted p | Holm conclusion |
|---|---|---:|---:|---:|---:|---:|---|
| Cosine Cluster-MoE vs. MobileNetV3-Small | Accuracy | 0.8028 | 0.7656 | +0.0372 | 0.2707 | 0.1353 | Not significant |
| Cosine Cluster-MoE vs. MobileNetV3-Small | Macro-F1 | 0.7669 | 0.7307 | +0.0362 | 0.3025 | 0.1791 | Not significant |
| Cosine Cluster-MoE vs. MobileNetV3-Small-MoE | Accuracy | 0.8028 | 0.7839 | +0.0189 | 0.3025 | 0.1512 | Not significant |
| Cosine Cluster-MoE vs. MobileNetV3-Small-MoE | Macro-F1 | 0.7669 | 0.7498 | +0.0171 | 0.3025 | 0.2964 | Not significant |
| Cosine Cluster-MoE vs. Euclidean Cluster-MoE | Accuracy | 0.8028 | 0.7382 | +0.0646 | 0.0748 | 0.0417 | Not significant |
| Cosine Cluster-MoE vs. Euclidean Cluster-MoE | Macro-F1 | 0.7669 | 0.6953 | +0.0716 | 0.0748 | 0.0417 | Not significant |

The largest improvements were observed against the Euclidean-routing
Cluster-MoE: 6.46 percentage points in accuracy and 7.16 percentage points in
Macro-F1. The corresponding unadjusted paired $t$-test p-values were 0.0125
and 0.0139, and both comparisons remained significant after
Benjamini–Hochberg adjustment ($p_{\mathrm{BH}}=0.0417$). However, they did not
remain significant under the more conservative Holm–Bonferroni family-wise
error correction ($p_{\mathrm{Holm}}=0.0748$). Therefore, these results provide
suggestive evidence that cosine routing is preferable to Euclidean routing,
but they do not meet the confirmatory significance criterion adopted in this
analysis.

The improvements over MobileNetV3-Small were 3.72 percentage points in
accuracy and 3.62 percentage points in Macro-F1. The gains over the
context-aware MobileNetV3-Small-MoE baseline were smaller, at 1.89 and 1.71
percentage points, respectively. None of these comparisons was statistically
significant before or after multiple-comparison correction.

Because only five paired seeds were available, the tests have limited
statistical power. In particular, for $N=5$, the smallest attainable
two-sided Wilcoxon signed-rank p-value without zero differences is 0.0625.
Consequently, a Wilcoxon result below 0.05 is impossible at this sample size.
The non-significant findings should therefore not be interpreted as evidence
of equal performance; additional independent training seeds are needed for a
more conclusive assessment.

### Paper-Ready Interpretation

> Across five matched random seeds, Cluster-MoE with cosine routing achieved
> the highest mean accuracy (0.8028) and Macro-F1 (0.7669). Its largest gains
> were obtained over the Euclidean-routing variant, improving accuracy and
> Macro-F1 by 6.46 and 7.16 percentage points, respectively. These differences
> were significant using the unadjusted paired t-test and remained significant
> under Benjamini–Hochberg false-discovery-rate control, but not under the
> prespecified Holm–Bonferroni family-wise correction. Improvements over
> MobileNetV3-Small and the context-aware MoE baseline were positive but not
> statistically significant. Thus, the results favor cosine routing
> descriptively, while further runs are required to establish confirmatory
> statistical significance.
