# PCA of the benchmarks behind the Epoch Capabilities Index

Data: Epoch AI `benchmark_data.zip` (CC-BY; Epoch AI, https://epoch.ai/benchmarks), retrieved 2026-11-15. I used `processed_data_for_eci.csv`, the exact model x benchmark table the ECI fit uses (256 models, 57 benchmarks, 2,879 scores, ~20% of cells filled). Plan: `PLAN.md`. Code: `scripts/`. Outputs: `outputs/`.

## What I did
1. Dropped 2 benchmarks that Epoch marks as superseded (older FrontierMath variants), and benchmarks with < 25 scored models, leaving 38 (29 after step 2).
2. **Data problem found:** 9 older benchmarks (MMLU, GSM8K, HellaSwag, TriviaQA, BBH, PIQA, Winogrande, ARC AI2, OpenBookQA) share a median of ~1 model with all other benchmarks. They were evaluated on 2023-era models, and modern benchmarks on modern ones. Their correlations with the rest can't be estimated, so they are excluded from the main PCA and analysed separately. My first run filled their missing correlations with 0, which produced fake zero loadings. That was caught and fixed.
3. PCA on the remaining 29 benchmarks with two estimators for sparse data: (A) PCA of the pairwise-complete correlation matrix, and (B) iterative-SVD/EM PCA, with k chosen by held-out-cell CV. Loadings agree (cosine 0.94 for both PC1 and PC2), and a logit transform changes nothing (cosine > 0.99).
4. Second analysis: remove general capability first (see below).

## Findings

**1. PC1 is general capability (50% of variance).** Every benchmark loads positively (0.09 to 0.25). Model PC1 scores correlate r = 0.80 with Epoch's official ECI. The highest loadings are LMCA, DTBench, ARC-AGI / ARC-AGI-2, SimpleBench, FrontierMath Tiers 1-3, and OTIS AIME. The lowest are Balrog, Fiction.LiveBench, GeoBench, MATH level 5, METR Time Horizons and VPCT, which are more idiosyncratic or saturated.

**2. PC2 (16%) is mostly benchmark *difficulty*, not a skill type.** Its loadings correlate r = -0.81 with Epoch's benchmark difficulty (EDI). One end is the newest and hardest benchmarks (Furniture Assembly, EBR-bench, ProofBench, WeirdML v3, MirrorCode, FrontierMath Tier 4). The other end is easier, older, saturating ones (MATH level 5, Aider polyglot, METR Time Horizons, Fiction.LiveBench, GPQA). This is the usual artifact of S-shaped score curves under linear PCA. CV supports only 2 components, so I don't interpret PC3+ of the raw PCA.

**3. After removing general capability, there is some skill structure, but it is weak.** I fit a logistic curve of score against official ECI for each benchmark (R^2 of 0.5 to 0.98) and ran PCA on the residuals. Five components sit above a shuffled null, but only modestly (e.g. 4.07 vs 3.10), so treat the groupings below as suggestive:
- **R1:** competition math and puzzles (FrontierMath, OTIS AIME, VPCT, chess) versus agentic/software work and abstraction (GSO-Bench, METR Time Horizons, ARC-AGI-2, SimpleBench).
- **R2:** knowledge, puzzle and "common-sense" tasks (SimpleQA, Mystery Game Puzzles, SimpleBench, DTBench) versus long-horizon engineering and proof (GSO-Bench, ProofBench, MirrorCode, Furniture Assembly).
- **R3:** ARC-AGI and math (ARC-AGI 1/2, AIME, MATH level 5) versus applied coding/tooling and open-ended tasks (Aider polyglot, LMCA, WeirdML, ProofBench, Fiction.LiveBench).

Overall: the benchmarks are dominated by one general factor, and most of the rest of the apparent structure is difficulty.

**4. Legacy benchmarks:** among the 9 older benchmarks plus the others in that era, PC1 is 67% of variance with fairly uniform loadings. It is essentially one "general language/knowledge ability" factor (`outputs/legacy_pc1_loadings.csv`).

## Caveats
- Sparse data: 90 of 406 benchmark pairs have < 12 shared models, so those correlations are filled with 0 in the pairwise PCA (EM PCA doesn't need this, and agrees). Loadings for benchmarks with ~25-30 models (VPCT, WeirdML v3, GSO-Bench, Balrog) are noisy.
- Era confound: some structure reflects which models were tested on which benchmarks.
- The residual analysis uses official ECI, which was itself fit from these same scores. Residuals are shrunk toward zero, and so any residual structure is, if anything, understated.
- Correlation is not a statement about what a benchmark "measures". Labels like "math" or "agentic" are my reading of the loadings.

## Files
- `outputs/loadings_pairwise.csv`, `loadings_em.csv`, `residual_loadings.csv`, `legacy_pc1_loadings.csv`, `model_scores_em.csv`
- Plots: `scree_cv.png`, `loadings_heatmap.png`, `benchmark_map.png`, `residual_pca.png`
- Logs: `run_log.txt`, `interpret_log.txt`, `residual_log.txt`
- Re-run: `python -I scripts/run_pca.py data/extracted outputs`, then `interpret.py` and `residual_pca.py` with the same arguments.
