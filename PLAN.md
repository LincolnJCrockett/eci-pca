# ECI benchmark PCA: plan

**Goal:** see how the benchmarks behind the Epoch Capabilities Index (ECI) differ in the aspects of AI capability they measure, using PCA.

## Data
- Source: https://epoch.ai/data/benchmark_data.zip (CC-BY, Epoch AI), the same data the `epoch-research/eci-public` pipeline uses.
- Use `epoch_capabilities_index/processed_data_for_eci.csv`: one row per (model, benchmark) with a score in [0, 1]. These are exactly the observations the ECI fit uses.
  - 256 models x 57 benchmarks, ~20% filled.
- `benchmark_metadata.csv` and `eci_scores.csv` / `edi_scores.csv` give metadata, official ECI, and benchmark difficulty.

## Challenges
1. **Sparsity.** Most models are evaluated on only a few benchmarks, so ordinary PCA is undefined.
2. **Difficulty confound.** Scores depend on model capability and benchmark difficulty, so PC1 will mostly be "general capability" (ECI-like).
3. **Superseded benchmarks.** Several benchmarks have versions that overlap (e.g. FrontierMath variants).

## Steps
1. Load data and drop benchmarks superseded by another benchmark in the set.
2. Keep benchmarks with >= 25 scored models, so correlations are estimable.
3. Standardize each benchmark column (z-score over observed entries).
4. Two PCA estimators, to check robustness:
   - **A. Pairwise-correlation PCA**: pairwise-complete correlation matrix (min 12 shared models), project to nearest PSD, eigendecompose.
   - **B. Iterative-SVD (EM) PCA** on the masked matrix, with the number of components k chosen by held-out-entry cross-validation.
5. Validate: correlate PC1 scores with official ECI; compare loadings from A and B.
6. Interpret: PC1 = general capability. PC2/PC3 = contrasts between benchmark families (e.g. knowledge vs. math vs. agentic/coding). Also repeat after removing PC1 and after a logit transform of scores, as a sensitivity check.
7. Outputs in `outputs/`: scree plot, loadings table and heatmap, PC1-vs-PC2 benchmark map, summary `RESULTS.md`.

## Caveats to report
- Heavy missingness means loadings for sparsely covered benchmarks are noisy.
- Model-set composition changes over time. Old benchmarks are mostly scored on old models, and the new ones on new models, so some structure is era, not skill.
