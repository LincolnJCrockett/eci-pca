# How much do the ECI benchmarks share, and what does the ECI miss?

Data: Epoch AI `benchmark_data.zip` (CC-BY; Epoch AI, https://epoch.ai/benchmarks), retrieved 2026-11-15; `processed_data_for_eci.csv` is the table the ECI fit uses.
**Sample:** 191 models x 29 benchmarks, 2,078 scores (37.5% of cells). Logit scores unless noted. Excluded: 2 superseded benchmarks, benchmarks with < 25 models, and 9 older benchmarks (MMLU, GSM8K, HellaSwag, ...) that share almost no models with modern ones; models need >= 4 scores. The official ECI also uses those extra benchmarks.
Code: `scripts/variance_and_eci_resid.py`, `scripts/eci_vs_components_v2.py`, shared estimators in `scripts/common.py`. Outputs: `outputs/variance/`; logs: `outputs/variance_log.txt`, `outputs/eci_vs_components_v2_log.txt`.

## Correction to my previous write-up
Earlier I centered each benchmark on its observed mean (the standard PCA step, "A"). That is biased here, because the models tested on a benchmark are not representative: hard benchmarks are mostly run on strong models. Re-estimating benchmark intercepts jointly with the factor ("AJ") fixes this, and it changes several earlier conclusions:
- PC1 vs ECI: r = **0.995** (was 0.968). Only **~1%** of ECI variance is outside PC1 (was ~6%).
- The earlier "ECI corrects for benchmark difficulty" finding (r = -0.52 with easiness of benchmarks taken) was an artifact; it is -0.09 now.
- "About two reliable dimensions" was wrong: with joint intercepts a second component does *not* improve held-out prediction.
- The earlier spec-D result ("ECI disfavors agentic work, r = -0.48") is retracted. With the corrected version it is +0.10, not significant.

## 1. How much of the benchmark data does PC1 explain?
"Variation in the data" = variation across models within each benchmark (benchmark means, i.e. difficulty, are excluded).

| | in-sample, benchmarks weighted equally | in-sample, pooled logit | held-out R^2* |
|---|---|---|---|
| Classical PCA (observed benchmark means), PC1 | 66% | 69% | 60% |
| ... first 2 / 3 / 4 PCs | 87% / 91% / 94% | 89% / 92% / 94% | 75% (k=2, best) |
| **Joint-intercept PC1 (AJ)** | **84%** | **87%** | **82%** (best) |
| ... first 2 / 3 / 4 PCs | 89% / 92% / 94% | 90% / 93% / 94% | 79% / 74% / 73% |
| Official ECI as the single score (per-benchmark OLS) | 83% | 87% | n/a (ECI saw these cells) |
| Two-way FE additive (no slopes) | 63% | 76% | 69% |

*Held-out R^2 is relative to the benchmark-mean model, from 5-fold CV repeated twice. In the correlation-matrix view, PC1 is 51% of total variance, PC2 15%, PC3 6.5% (pairwise-overlap estimate). The classical numbers understate PC1 for the selection reason above.

**Answer:** PC1, properly estimated, accounts for roughly **85% of within-benchmark variation** (82% out of sample). More components fit better in-sample but predict worse out of sample.

## 2. What is not explained by the ECI?
- Overall: ECI explains ~83-87% of within-benchmark variation, so **~13-17%** is left. It is essentially the same as PC1 (r = 0.995, `fig_variance_explained.png`).
- It is very uneven across benchmarks (`per_benchmark_variance_explained.csv`, `fig_eci_explained_by_benchmark.png`; share of variance across models explained by ECI, raw-score logistic fit):
  - **>= 94%:** ARC-AGI (98%), METR Time Horizons (96%), OTIS AIME, FrontierMath T1-3, LMCA, WeirdML v3, GSO-Bench, ARC-AGI-2 (~95%), GPQA, MATH level 5 (94%).
  - **Poorly captured:** GeoBench (19%), Furniture Assembly (46%), MirrorCode (50%), Fiction.LiveBench (57%), SimpleQA Verified (62%), APEX-Agents (74%), Chess Puzzles (76%), FrontierCode (77%). VPCT shows 4% under the logistic fit, but 55% under the logit-linear fit, so I suspect a failed curve fit, not a true value.
  - The poorly captured benchmarks are geolocation, long-context fiction, factual recall, physical/spatial tasks and long-horizon software tasks.

## 3. Is there structure in what the ECI misses? (your residual specification, "E")
For each benchmark I fit score = floor + (1-floor) x sigmoid(a (ECI - c)) on observed models, took residuals, standardized columns, and ran low-rank PCA on them (raw scores).
- **Statistically, yes but weakly.** Five eigenvalues of the residual correlation matrix exceed a shuffled null (4.05, 3.71, 2.95, 2.74, 2.11 vs 3.02, 2.68, 2.42, 2.24, 2.05).
- **Predictively, no.** Adding components to the ECI-only prediction makes held-out error worse (MSE 0.0090 to 0.0113 at k=1; the ECI baseline is optimistic because ECI was fit on these cells). In-sample the first three components "explain" 22/45/59% of the residual, but that is overfitting: models have a median of 9 scores.
- **Stability is modest** (bootstrap best-match cosine 0.62, 0.68, 0.73). E2 and E3 agree with the corresponding components of the joint-intercept PCA residual (DJ2, DJ3; cosine 0.75, 0.82). E1 does not (0.30), so I trust it least.

Components (loadings in `E_loadings.csv`, `fig_E_loadings.png`):
- **E1** (+) Furniture Assembly, SimpleQA, GeoBench, Chess vs (-) FrontierCode, WeirdML v3, APEX-Agents, MirrorCode, ARC-AGI: factual/spatial vs agentic software work. Correlated with ECI (0.41) and date (0.35), so it is partly a trend where the curve doesn't fit.
- **E2** (+) GSO-Bench, ProofBench, MirrorCode, WeirdML, Terminal Bench, Aider vs (-) VPCT, SimpleBench, ARC-AGI-2, Mystery Game Puzzles, Fiction.LiveBench, Chess: software engineering and formal proof vs perceptual/puzzle tasks.
- **E3** (+) FrontierMath T1-3 and T4, MATH level 5, OTIS AIME vs (-) SimpleBench, MirrorCode, LMCA, Balrog, Aider: competition math vs agentic/common-sense tasks.

Bottom line: after the shared factor, the leftover variation is mostly benchmark-specific and noisy; a modest "math vs. agentic/common-sense" axis (E3 / DJ3) is the clearest, and the others are tentative.

## 4. What does ECI over- or under-value relative to PC1?
ECI minus its linear fit on PC1 is only ~1% of ECI's variance (`eci_vs_components_v2_log.txt`). It correlates with:
- release date: +0.21 [0.07, 0.34]. It does not correlate with the number of benchmarks taken (-0.01) or benchmark easiness (-0.09).
- The component "competition math vs. agentic/common-sense": **E3 r = +0.38 [0.25, 0.50]**, **DJ3 r = +0.37 [0.24, 0.48]**, essentially unchanged after partialling out release date (0.40, 0.35). Models strong in competition math relative to their other scores get slightly higher ECI than PC1 implies.
- By benchmark (`q1_partial_corr_AJ.csv`): ECI residual is positively related to METR Time Horizons (0.71), Furniture Assembly (0.66), MATH level 5 (0.61), WeirdML v3 (0.57), OTIS AIME (0.52), FrontierMath T4 (0.48) and negatively to Fiction.LiveBench (-0.63), GeoBench (-0.62), VPCT (-0.58), SimpleQA (-0.56), Mystery Game Puzzles (-0.53), Balrog (-0.44).
- Reading: ECI leans slightly toward competition math and METR, and slightly away from long-context, geolocation, vision, factual-recall and puzzle benchmarks. The size is small: ECI and PC1 are almost the same quantity.

## 5. Alternative loss functions (see METHODOLOGY.md section 8)
One score per model, with a free intercept and slope per benchmark, chosen under different losses and compared with the official ECI (`scripts/loss_comparison.py`, `outputs/loss_comparison/`).

| Loss | Mean R^2 | Worst R^2 | Corr. with ECI | SD of score change (pts) | Max change (pts) | Held-out mean R^2 | Held-out worst R^2 |
|---|---|---|---|---|---|---|---|
| Official ECI | 0.80 | 0.23 (GeoBench) | 1.000 | 0 | 0 | n/a | n/a |
| MSE, pooled cells | 0.80 | 0.25 | 0.997 | 1.1 | 5.3 | **0.74** | **0.17** |
| MSE, benchmark average | 0.82 | 0.28 | 0.990 | 2.0 | 6.7 | 0.70 | 0.14 |
| Absolute-value (MAE), benchmark average | 0.81 | 0.23 | 0.992 | 1.8 | 8.2 | 0.65 | -0.27 (Aider polyglot) |
| Worst benchmark (p = 8) | 0.77 | 0.67 | 0.945 | 4.6 | 19.6 | 0.61 | -0.01 |
| Worst benchmark (p = 64, about minimax) | 0.74 | 0.71 | 0.928 | 5.3 | 22.1 | 0.58 | -0.03 (MirrorCode) |

- The mean ECI 95% half-width is 2.5 points, so MSE and MAE solutions reproduce ECI within its own uncertainty. Pooled MSE is closest and has the best held-out fit, consistent with ECI being roughly a least-squares fit.
- The worst-benchmark loss raises the worst R^2 from 0.23 to 0.71 (GeoBench) by flattening fit across benchmarks, and boosts models that do well on GeoBench, MirrorCode, Fiction.LiveBench, VPCT and Mystery Game Puzzles at the expense of METR, ARC-AGI, FrontierMath T4 and AIME. But it is worse than the average losses on held-out cells, both on average and in the worst case. The worst-fit benchmarks have only 25-44 models, and the models that move most have few scores.
- **Why these may not beat the current ECI** is spelled out in METHODOLOGY.md section 11: this is an approximation of ECI (logit-linear, 29-benchmark subset, no priors or anchors); there is no external ground truth for "better"; minimax is fragile in sparse data; and the average-loss differences are within ECI's uncertainty.

## Figures
One figure per claim in `outputs/figures/` (`fig01` to `fig16`); the claim-to-figure map is METHODOLOGY.md section 9.

## Caveats
- Confidence intervals treat models as independent and ignore uncertainty in the components. Models from the same family are correlated, so true uncertainty is wider.
- "In-sample" figures are optimistic for higher-rank fits. Held-out R^2 is the more reliable comparison, but it is itself noisy given the sparsity.
- Component labels are my reading of the loadings.
- All of this is on the 29-benchmark subset. ECI also uses the 9 older benchmarks and others with < 25 models; some of the ~1% gap between ECI and PC1 may come from those.
- Logit transform; the raw-score version of the PCA analyses was not run.
- Spec E uses raw scores and the others use logits.

## Earlier analyses (partly superseded)
- `scripts/spec_pca.py`, `outputs/spec_logit/`: demeaned-by-benchmark (A), by-model (B), two-way FE (C), joint FE (CJ) and slope-adjusted (D). Conclusions that depended on A's fixed-mean centering (A PC2 "difficulty axis", D's ECI relationships, "two reliable dimensions") should be treated as superseded by the above. C's leading components reflect benchmark-slope differences, not skills.
- `scripts/run_pca.py`, `scripts/residual_pca.py`: first pass.

## Re-run
```
python -I scripts/variance_and_eci_resid.py data/extracted outputs/variance
python -I scripts/eci_vs_components_v2.py data/extracted outputs/variance
```
