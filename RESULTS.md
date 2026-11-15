# What does the ECI add to, or lose from, its benchmarks?

Data: Epoch AI `benchmark_data.zip` (CC-BY; Epoch AI, https://epoch.ai/benchmarks), retrieved 2026-11-15; `processed_data_for_eci.csv` is the table the ECI fit uses. Plan: `PLAN.md`. Main script: `scripts/spec_pca.py`; outputs in `outputs/spec_logit/`; log in `outputs/spec_logit_log.txt`.

**Sample:** 191 models x 29 benchmarks, 37.5% of cells observed (models need >= 4 scores). Scores are logit-transformed (raw scores were not re-run). Excluded: 2 superseded benchmarks, benchmarks with < 25 models, and 9 older benchmarks (MMLU, GSM8K, HellaSwag, etc.) that share almost no models with modern ones. The official ECI uses those too, so some gap between ECI and anything computed here is expected.

## Specifications
| | transform | what it removes |
|---|---|---|
| A | subtract benchmark mean | benchmark difficulty (ordinary PCA; model quality remains) |
| B | subtract model mean, **no** column centering | model quality (benchmark difficulty remains) |
| C | residual of two-way fixed effects (model + benchmark) | both, assuming every benchmark has slope 1 on ability |
| CJ | fixed effects and low-rank part fitted jointly | robustness check on C |
| D | residual after rank-1 fit with benchmark means | both, **allowing each benchmark its own slope** (added after C's diagnostics) |

In complete data, B with column centering would equal C. The specs differ here only because the data are sparse, which is why B is left uncentered. Low-rank fits use iterative SVD on observed cells. Columns are scaled by RMS after the base is removed.

## Q1. How is ECI *not* the first PC?
- ECI vs rank-1 first PC (spec A): **r = 0.968** [0.96, 0.98], R^2 = 0.94, Spearman 0.96. Versus the two-way-FE model effect: r = 0.990. Adding a quadratic term gains nothing, so the relationship is linear.
- It is not sparse-data noise: r is 0.957 / 0.957 / 0.972 for low / medium / high-coverage models.
- So ~6% of ECI variance is not in PC1. What it correlates with (ECI residual after PC1):
  - mean easiness of the benchmarks the model took: **r = -0.52** [-0.62, -0.41]. PC1 gives inflated scores to models tested mostly on easy benchmarks, and ECI corrects for difficulty.
  - release date: +0.42; number of benchmarks taken: +0.26; closed weights: -0.17.
  - by benchmark (partial correlation of ECI residual with that benchmark's score residual after PC1, `q1_benchmark_partial_corr.csv`): highest for METR Time Horizons (0.78), OTIS AIME (0.77), ARC-AGI-1/2 (0.76), MATH level 5 (0.72), GPQA (0.70), Terminal Bench (0.63), FrontierMath T1-3 (0.57). Negative for GeoBench (-0.42), Mystery Game Puzzles (-0.28), SimpleQA Verified (-0.27). Relative to PC1, ECI leans on math, science reasoning and ARC-AGI, and less on geography, factual recall and puzzle benchmarks. Caveat: benchmarks with many scores (GPQA, AIME, n~170) mechanically weigh more in ECI, so this mixes "weights" and "sample size".

## Q2. Interpretable components, and how ECI relates to them
Second dimension: held-out prediction error (5-fold CV, `cv_mse.csv`) is best at about two dimensions in total: A at k=2 (0.70 vs 1.12 at k=1), C at k=1 (0.69 vs 0.87 at k=0). Beyond that it overfits. So there is roughly one reliable dimension after general ability. Further components are weak (bootstrap stability below).

Components (loadings: `loadings_*.csv`, `fig_loadings.png`):
- **A PC2** (bootstrap stability 0.85): a difficulty/era axis. Easier or older benchmarks (Aider polyglot, Fiction.LiveBench, MATH level 5, METR) vs newest hardest ones (WeirdML v3, Furniture Assembly, MirrorCode, EBR-bench). Correlated 0.72 with model release date. Not a skill.
- **D PC1** (stability 0.62): long-horizon agentic work (MirrorCode, APEX-Agents, WeirdML v3, Furniture Assembly) vs math and abstract reasoning (FrontierMath T1-3, ARC-AGI-1/2, chess puzzles).
- **D PC3** (stability 0.71): GeoBench, MATH level 5, VPCT, OTIS AIME vs SimpleQA, SimpleBench, Mystery Game Puzzles. I don't have a confident label.

ECI after removing its relation to PC1 (`q2_eci_vs_components.csv`, `fig_eci_vs_components.png`). Since D is built orthogonal to PC1, these say what ECI deviates toward:
- **D PC1: r = -0.48** [-0.58, -0.36], and -0.48 with release date partialled out. Models relatively stronger at long-horizon agentic work than at math/abstract reasoning get a lower ECI than PC1 implies. ECI relatively favours math/abstract reasoning over long-horizon agentic work.
- **D PC3: r = +0.60** [+0.51, +0.69], +0.49 after the date control.
- B PC3 points the same way as D PC1 (-0.30, -0.44 after date). The other components are near zero or lose significance after the date control.

## Why C is not trusted
- The FE-residual eigenvalue test finds three components above a shuffled null (7.4, 4.6, 3.1 vs 3.1, 2.8, 2.5). But they are contaminated by benchmarks having different slopes on ability (0.27 to 2.03). C PC3's loadings correlate -0.71 with benchmark slope and its scores correlate -0.66 with model quality.
- D allows per-benchmark slopes, and those correlations go to about 0.
- C and CJ loadings agree only weakly (cosine 0.2 to 0.4).

## Caveats
- D's components are only moderately stable under model bootstrap (cosine 0.6 to 0.7; 10th percentile as low as 0.3). Treat the labels as hypotheses.
- Confidence intervals treat models as independent and ignore uncertainty in the components. Models from the same family are correlated, so true uncertainty is wider.
- A mechanical caveat: correlating ECI with "ECI after FE" is partly artifactual (a spike inflates the model's fixed effect). I therefore base the conclusions on ECI after PC1, and report FE versions only in the CSV.
- In-sample fit (`insample_r2.csv`): ECI per-benchmark linear 87% of within-benchmark variance, FE additive 76%, EM rank-1 PCA 69%, rank-2 89%. This is not a fair "information lost" measure: ECI also used the other benchmarks and these same cells, and the PCA objective is weighted differently. A held-out comparison was not done, because ECI was fit on the held-out cells.
- Labels like "agentic" or "math" are my reading of loadings.

## Earlier analysis (superseded)
`scripts/run_pca.py` and `scripts/residual_pca.py` (outputs in `outputs/`, not `outputs/spec_logit/`) are the first pass. The "residual after ECI" analysis there used the official ECI as the control, so it can't say what ECI adds, and it was nonlinear and not equivalent to PCs 2+. Its main useful finding survives: PC2 of benchmark-demeaned data is largely difficulty/era.

## Re-run
`python -I scripts/spec_pca.py data/extracted outputs/spec_logit logit` (about 4 minutes).
