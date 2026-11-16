# Methodology

This document explains, step by step, how every number in the README is computed. Each step says what question it answers, what it does in plain words, the equation, and which script does it.

## Contents
1. [The four steps at a glance](#1-the-four-steps-at-a-glance)
2. [Data, notation, and the one approach used throughout](#2-data-notation-and-the-one-approach-used-throughout)
3. [Step 1: what the ECI explains, benchmark by benchmark](#3-step-1-what-the-eci-explains-benchmark-by-benchmark)
4. [Step 2: principal components, and how close the first is to the ECI](#4-step-2-principal-components-and-how-close-the-first-is-to-the-eci)
5. [Step 3: remove what the ECI explains, and look at what is left](#5-step-3-remove-what-the-eci-explains-and-look-at-what-is-left)
6. [Step 4: how much the index cares about the worst-fitted benchmark](#6-step-4-how-much-the-index-cares-about-the-worst-fitted-benchmark)
7. [Figures and the claims they support](#7-figures-and-the-claims-they-support)
8. [Running it](#8-running-it)
9. [Caveats](#9-caveats)

---

## 1. The four steps at a glance

The Epoch Capabilities Index (ECI) boils many benchmark scores down to one number per model. The analysis asks four questions about that aggregation.

| Step | Question | Script | Output folder |
|---|---|---|---|
| 1 | Aggregating many benchmarks into one number must lose something. How much, and where? | `01_pc1_vs_eci.py`, `02_eci_residual.py` | `results/01_pc1_vs_eci/`, `results/02_eci_residual/` |
| 2 | If we run PCA on the benchmarks, is the first principal component (PC1) essentially the ECI? How much does each component explain? | `01_pc1_vs_eci.py` | `results/01_pc1_vs_eci/` |
| 3 | If we remove everything the ECI explains, what is left, and does it have shared structure? | `02_eci_residual.py` | `results/02_eci_residual/` |
| 4 | If the index cared less, or more, about the benchmark it fits worst, would the scores change? | `03_loss_functions.py` | `results/03_loss_functions/` |

Shared code (data loading, PCA with missing data, the fit measure, the p-norm solver) is in `scripts/common.py`. `scripts/04_make_figures.py` draws the figures from the saved results.

---

## 2. Data, notation, and the one approach used throughout

**Source.** Epoch AI's `benchmark_data.zip` (CC-BY 4.0), downloaded by `scripts/download_data.py`. We use `epoch_capabilities_index/processed_data_for_eci.csv`, which is the exact model-by-benchmark table Epoch's ECI is fit on, plus `eci_scores.csv` (the official ECI and its 90% interval; `eci_bootstraps.json` gives `ci_level: 0.9`) and `benchmark_metadata.csv`.

**The matrix.** Each model $m$ has been evaluated on only some benchmarks $b$. Write $y_{mb}\in[0,1]$ for the raw score when it exists. The set of observed (model, benchmark) cells is $\Omega$; benchmark $b$ was scored on $n_b$ models; there are $N=|\Omega|$ scores in total.

**Which benchmarks and models are kept** (`common.load`):

1. Drop benchmarks that Epoch marks as superseded by another benchmark in the data (2 FrontierMath variants).
2. Keep benchmarks scored on at least 25 models.
3. Drop 9 older benchmarks (MMLU, GSM8K, HellaSwag, TriviaQA, BBH, PIQA, Winogrande, ARC AI2, OpenBookQA). They share a median of about one model with the other benchmarks (they were run on 2023-era models), so their relationship with the rest cannot be estimated. Formally: let $O_{bc}$ be the number of models scored on both $b$ and $c$; drop $b$ if $\operatorname{median}_{c\ne b}O_{bc}<7$.
4. Keep models with at least 4 scores and benchmarks with at least 15 models (repeated until stable).

Result: **191 models × 29 benchmarks, 2,078 scores (37.5% of cells).** The official ECI also uses the excluded benchmarks and models (`fig01`).

**Logit scores**, used everywhere:

$$x_{mb}=\log\frac{\tilde y_{mb}}{1-\tilde y_{mb}},\qquad \tilde y_{mb}=\operatorname{clip}(y_{mb},\,0.01,\,0.99).$$

Why: the ECI assumes $y_{mb}=\sigma\big(d_b(\theta_m-c_b)\big)$, an S-curve in the model's capability $\theta_m$, with benchmark difficulty $c_b$ and discriminability $d_b$ (Epoch's `eci-public` README). Taking logits turns that into a straight line, $x_{mb}=d_b(\theta_m-c_b)$, so a benchmark is a line in capability with its own slope and intercept.

**The one approach.** Every step uses the same scale, model, weighting and fit measure.

| | Choice |
|---|---|
| Scale | logit scores |
| Model | one score per model (or $k$ scores, for $k$ components) and, per benchmark, an intercept and slope estimated together with the scores |
| Weighting | every observed score counts once, after each benchmark is scaled to unit spread, so a benchmark's weight is proportional to $n_b$. This is what PCA with missing data does. |
| Fit measure | the share of each benchmark's variation (logit scores) explained by a straight line in the score(s) (§3) |

**Notation used below**

| Symbol | Meaning |
|---|---|
| $x_{mb}$ | logit score of model $m$ on benchmark $b$ (only for $(m,b)\in\Omega$) |
| $z_m$ | the official ECI of model $m$, standardized (mean 0, SD 1) |
| $\sigma_b$ | SD of benchmark $b$'s logit scores over the $n_b$ models scored on it |
| $R^2_b$ | share of benchmark $b$'s variation across models explained by a score |

---

## 3. Step 1: what the ECI explains, benchmark by benchmark

**In words.** If the ECI were a complete summary, every benchmark would be a perfect S-curve in the ECI, which on the logit scale is a straight line. For each benchmark, fit the best straight line in the ECI and see how much of the benchmark's variation across models it captures. What it misses is information the aggregation lost.

**Equation.** For each benchmark $b$, least squares on the models scored on it (`common.logit_r2`):

$$x_{mb}=a_b+g_b\,z_m+e_{mb},\qquad R^2_b=1-\frac{\sum_{m\in\Omega_b}e_{mb}^2}{\sum_{m\in\Omega_b}(x_{mb}-\bar x_b)^2}.$$

**Overall.** Every score counts once after scaling each benchmark to unit spread (§2):

$$R^2=1-\frac1N\sum_{(m,b)\in\Omega}\frac{e_{mb}^2}{\sigma_b^2}=\frac{\sum_b n_b R^2_b}{\sum_b n_b},$$

the average of the $R^2_b$ weighted by the number of models on each benchmark. For the ECI it is 83.4%. (The unweighted average of the 29 $R^2_b$ is 80.0%.)

*Clipping.* On the logit scale, 0% and 100% are $\mp\infty$ and must be clipped (to 1% and 99% here). A model scoring 0% on a benchmark that is too hard for it then gets a residual whose size depends on where the clip is set. With clipping at 0.1%, 1% and 3%, the $R^2$ of Mystery Game Puzzles (18% of scores at 0%) is 0.51, 0.61 and 0.67; measured on raw scores with an S-curve fit it would be 0.89 (Furniture Assembly, 26% clipped: 0.79 here, 0.91 on raw scores). Benchmarks with no floor or ceiling scores (GeoBench, MirrorCode, Fiction.LiveBench, SimpleQA) get the same $R^2$ either way. Logit scores are used anyway so that every step, including the PCA, uses one scale.


**Scripts.** `01_pc1_vs_eci.py` (`per_benchmark_explained.csv`, `eci_variance_explained.csv`) and `02_eci_residual.py` (the same per-benchmark table with an "unexplained" column, `eci_explained_by_benchmark.csv`).

---

## 4. Step 2: principal components, and how close the first is to the ECI

### 4.1 PCA when most of the table is empty
Ordinary PCA needs a complete matrix; here 62.5% of cells are missing. We use **iterative SVD** on the observed cells. Subtract a "base" matrix $B$ (the benchmark intercepts, §4.2), divide each benchmark by its observed SD $\sigma_b$, and find the rank-$k$ matrix that best fits the observed cells:

$$\min_{B,U,V}\ \sum_{(m,b)\in\Omega}\Big(\frac{x_{mb}-B_{mb}}{\sigma_b}-\sum_{j=1}^{k}U_{mj}V_{bj}\Big)^2 .$$

Algorithm: fill each missing cell with the current low-rank estimate, re-estimate the intercepts, take the SVD of the filled matrix, keep the top $k$ components, and repeat until nothing changes (largest change $<10^{-6}$). The scores $U$ give one number per model per component; the loadings $V$ say how much each benchmark contributes. (`common.fit_lowrank`, `base="joint"`)

Every observed score counts once in this sum, so benchmarks with more models count more. This is the weighting used throughout (§2).

### 4.2 Two ways of centering: conventional (A) and joint intercepts (AJ)
All results use joint intercepts. Conventional centering appears only in `eci_vs_pc1.csv` and `fig04`, to show why.
- **Conventional.** Subtract each benchmark's observed mean: $B_{mb}=\bar x_b$.
- **Joint intercepts.** Treat each benchmark's intercept as unknown and estimate it *together with* the components: at each iteration, set $B_{mb}$ equal to the mean over observed cells of $x-\text{(current low-rank part)}$.

*Why joint intercepts.* Suppose the truth is $x_{mb}=\beta_b+g_b\theta_m$. The observed mean is

$$\bar x_b=\beta_b+g_b\,\bar\theta_{\Omega_b},$$

shifted by the average ability $\bar\theta_{\Omega_b}$ of the models that happened to be tested. Hard benchmarks are mostly run on strong models, so $\bar x_b$ overstates the benchmark's typical score. Conventional PCA bakes that error in; joint intercepts remove it. `fig04` shows the bias: across benchmarks, (observed mean − model-implied mean over all models) correlates $r=0.90$ with how strong the tested models were. With conventional centering PC1 correlates 0.968 with the ECI, against 0.995 with joint intercepts.

### 4.3 Variation explained by each component
For a rank-$k$ fit with model scores $u_{m1},\dots,u_{mk}$, regress each benchmark's logit scores on them (`common.logit_r2`),

$$x_{mb}=a_b+\textstyle\sum_{j=1}^k g_{bj}\,u_{mj}+e_{mb},$$

and compute the overall $R^2$ of §3, written $R^2(k)$. This is the share of variation the rank-$k$ PCA itself captures, since at convergence its intercepts and loadings are these least-squares fits. Component $k$ explains $R^2(k)-R^2(k-1)$, with $R^2(0)=0$ (`variance_by_component.csv`, `fig03`): 84.5%, 5.4%, 2.7%, 2.4%, 1.5%.

### 4.4 Comparing PC1 with the ECI
The Pearson correlation $r$ between each model's PC1 score (rank-1 fit) and its official ECI, with a 95% interval from the Fisher transform,

$$\Big[\tanh\!\big(\operatorname{arctanh}r-\tfrac{1.96}{\sqrt{n-3}}\big),\ \tanh\!\big(\operatorname{arctanh}r+\tfrac{1.96}{\sqrt{n-3}}\big)\Big],$$

and the Spearman rank correlation.

**Script.** `01_pc1_vs_eci.py`. Outputs: `variance_by_component.csv`, `eci_variance_explained.csv`, `eci_vs_pc1.csv`, `pc1_scores.csv`, `selection_bias.csv`, `per_benchmark_explained.csv`.

---

## 5. Step 3: remove what the ECI explains, and look at what is left

**In words.** Take away from every score the part a straight line in the ECI explains. What remains, the residual, is the variation the ECI does not capture. A benchmark with a large residual is one the ECI undervalues. Then ask whether the residuals have shared structure across benchmarks, using the same PCA.

**Equations.** On logit scores, for each benchmark, least squares on the ECI (`common.ols_on_eci`):

$$e_{mb}=x_{mb}-(a_b+g_b z_m).$$

We run the PCA of §4.1 with base $B_{mb}=a_b+g_b z_m$, so it decomposes the $e_{mb}$, with each benchmark scaled by the RMS $s_b$ of its residuals. The share of residual variation captured by $k$ components is

$$S(k)=1-\frac1N\sum_\Omega\Big(\frac{e_{mb}-\hat e^{(k)}_{mb}}{s_b}\Big)^2,$$

and component $k$ captures $S(k)-S(k-1)$ (`residual_variance_by_component.csv`). This alone is optimistic: with about nine scores per model, a few components will always fit some of the observed cells.

**Is the structure real? Shuffled null.** Shuffle each benchmark's residuals among the models scored on it, which destroys any structure shared across benchmarks while keeping each benchmark's spread and the missing-data pattern. Then run the same PCA and compute the same shares (100 shuffles). A component is above chance if its share exceeds the 95th percentile of the shuffled shares. Three are: 23.7%, 17.7% and 16.2%, against 17.4%, 15.4% and 14.3% (`fig06`).

**Bootstrap stability.** Resample models with replacement (60 times), refit, and for each full-sample loading vector $v_k$ compute its best match $\max_j|\langle v_k,v^*_j\rangle|$ among the bootstrap components (cosine similarity). Report the mean (0.62–0.72) and 10th percentile. A value near 1 means the component is reproducible.

**Reading the components.** Loadings are signed and the sign of a component is arbitrary (we flip each so its largest loading is positive). Labels like "agentic" are interpretation, not output.

**Script.** `02_eci_residual.py`. Outputs: `eci_explained_by_benchmark.csv`, `residual_loadings.csv`, `residual_scores.csv`, `residual_variance_by_component.csv`, `residual_stability.csv`.

**Comparison with "Claudiness"** (README note 2; `05_claudiness.py`, outputs in `results/05_claudiness/`). G. Burnham's Gradient Update (Epoch AI, 20 November 2025) publishes the seven largest weights of the second principal component of Epoch's benchmark data. The script compares these with the E1 loadings on the benchmarks both include, and computes the cosine between E1's loadings and those of the second component of the ordinary PCA (§4). It reports the E1 percentile of the ten models Burnham names as most and least Claude-like, and of every Claude Opus model. It also gives the share of E1's variance explained by developer: the ratio of between-developer to total variance, over the 10 developers with at least 5 models, with a permutation p-value from 2,000 shuffles of the developer labels. Finally it gives the share explained by Anthropic against all other developers.

---

## 6. Step 4: how much the index cares about the worst-fitted benchmark

**In words.** Suppose someone built the one-number index to care more about the benchmark it fits worst. We build a one-number score per model under a family of losses that differ only in that, and measure how the scores move.

**Model.** One score $\theta_m$ per model, and a straight line per benchmark (logit scores) with its own intercept and slope. Because these are free, the best a benchmark can do is $R_b^2=r_b(\theta)^2$, the squared correlation of the score with the benchmark over the models scored on it. Define the unexplained share $u_b=1-r_b|r_b|$ (equal to $1-R^2_b$ when the slope is positive; using $r|r|$ stops a negative-slope fit from looking good).

**Loss: a power mean over benchmarks** (`common.pnorm_obj`), weighted by the number of models on each benchmark, $w_b=n_b/N$:

$$F_p(\theta)=\Big(\sum_b w_b\,u_b^{\,p}\Big)^{1/p}.$$

| p | Loss | What it values |
|---|---|---|
| 1 | weighted arithmetic mean of $u_b$ | every score counts once. This equals the rank-1 PCA objective of §4.1, so the solution is PC1 (the script checks: correlation 1.0000) |
| ∞ | $\max_b u_b$ | only the worst-fitted benchmark matters (solved as $p=64$) |

**Optimization.** L-BFGS-B with analytic gradients. $p=1$ starts from the standardized official ECI. $p=\infty$ is reached by warm starts through $p=2,4,\dots,64$. Scores are standardized after each solve, because the losses do not change under shifting and rescaling $\theta$. The intermediate $p$ values trace the trade-off curve in `fig08` (`pnorm_path.csv`).

**Comparison with the official ECI.**
- *Fit:* $R^2_b$ as in §3, with $\theta$ in place of the ECI (mean = the overall $R^2$ of §3, worst, and which benchmark is worst). This is the same $R^2_b$ the losses are built from.
- *Rank:* $|\operatorname{rank}(\theta_m)-\operatorname{rank}(\text{ECI}_m)|$ for each model (mean shown as % of the 191 models).
- *Points:* the score change in ECI points after mapping $\theta$ onto the ECI scale by matching mean and SD, $\widehat{\text{ECI}}_m=\overline{\text{ECI}}+\mathrm{sd}(\text{ECI})\,\theta_m$. We compare against the typical half-width of the ECI's 90% interval, 2.5 points (the two anchor models have no interval and are skipped). The matching is a convention (§9).

**Scores over release date** (the counterpart of Epoch's ECI-versus-date chart). Each model's release date is the `date` column of `eci_scores.csv`. A score's value in ECI points is the official ECI plus its change from the previous paragraph. The *frontier* is the set of models that scored higher than every model released before them (a running maximum by release date); its *slope* is the least-squares line through those record-setting points, in points per year. We also report (a) the share of record-setting models in common with the official ECI's frontier (intersection over union), (b) the correlation of each model's score change with its release date, and (c) as a robustness check, the frontier slope using only models with at least 8 scores, because models with few scores are the ones whose scores move most.

*Two-segment fit.* Epoch estimated a speed-up in the ECI frontier around 8 April 2024 (`common.BREAK_DATE`). For each score we fit, by least squares through its record-setting points (`common.hinge_fit`),

$$s=a+\beta_1 t+\beta_2\max(t-t_0,0),$$

with $t$ in years and the break $t_0$ fixed at that date, not estimated. The slope is $\beta_1$ before the break and $\beta_1+\beta_2$ after it. As a reference we fit the same line to the official ECI frontier over every model in `eci_scores.csv`, not just the 191 analysed.

**Script.** `03_loss_functions.py`. Outputs: `loss_summary.csv`, `pnorm_path.csv`, `loss_benchmark_r2.csv`, `loss_score_change_points.csv`, `loss_over_time.csv`, `loss_frontier_models.csv`.

---

## 7. Figures and the claims they support

All figures are in `results/figures/`, drawn by `04_make_figures.py` from the saved CSVs. There is no `fig11` (it showed held-out fit, which is no longer part of the analysis).

| Figure | Claim | Step |
|---|---|---|
| `fig01_data_sparsity` | Only ~20% of cells in Epoch's full table (256 models × 55 benchmarks) are observed, against 37.5% in the analysed subset; older benchmarks barely overlap with newer ones | data |
| `fig02_pc1_vs_eci` | The ECI is nearly the same as PC1 (r = 0.995) | 2 |
| `fig03_variance_explained` | PC1 explains 84.5% of benchmark variation (the ECI 83.4%); the second component adds 5.4%, each later one less | 2 |
| `fig04_selection_bias` | Observed benchmark means are biased by who was tested, which is why intercepts are estimated jointly | 2 |
| `fig05_eci_by_benchmark` | The ECI explains most benchmarks well and a few poorly | 1, 3 |
| `fig06_residual_structure` | Three components of what the ECI leaves out capture more than shuffled data | 3 |
| `fig07_residual_components` | The three leading axes of what the ECI does not explain (tentative) | 3 |
| `fig08_loss_tradeoff` | Caring more about the worst benchmark lifts it, at a cost to average fit | 4 |
| `fig09_loss_per_benchmark` | Fitting only the worst benchmark flattens fit across benchmarks | 4 |
| `fig10_loss_score_changes` | p = 1 (the first PC) moves scores within the ECI's own uncertainty; p = ∞ moves some by 20+ points | 4 |
| `fig12_scores_over_time` | Each score against release date, with its frontier (Epoch-style chart) | 4 |
| `fig13_frontier_over_time` | Frontier under each p with a two-segment fit broken at Apr 2024: post-break slopes agree (+15.8 to +17.3 points per year); too few earlier models to estimate the pre-break slope | 4 |

---

## 8. Running it

```
pip install pandas numpy scipy matplotlib
python -I scripts/download_data.py data
python -I scripts/01_pc1_vs_eci.py       data/extracted results/01_pc1_vs_eci
python -I scripts/02_eci_residual.py     data/extracted results/02_eci_residual
python -I scripts/03_loss_functions.py   data/extracted results/03_loss_functions
python -I scripts/04_make_figures.py     data/extracted results
python -I scripts/05_claudiness.py       data/extracted results
```

Each script prints its main numbers (the logs saved in the results folders are these printouts). Step 2's shuffled null takes several minutes; the others take seconds.

---

## 9. Caveats

### 9.1 Data
- **Sparse and non-random.** 62.5% of cells are missing and strong models are tested on hard benchmarks. Joint intercepts fix the intercept bias but not every kind of selection.
- **A subset.** Conclusions concern 29 benchmarks and 191 models. The ECI also uses the 9 older benchmarks and benchmarks with fewer than 25 models, so some gap between the ECI and PC1 may come from those.
- **Models are not independent.** Models from the same family have correlated scores; the confidence intervals here ignore that and are too narrow.

### 9.2 Steps 1–3
- **Range restriction.** $R^2_b$ compares the residual spread with the spread of the models tested on $b$, so a benchmark run on a narrow band of models gets a low $R^2_b$ even if its line in the ECI is steep and tight. The ECI SD of the tested models is 4.8 for MirrorCode, 5.7 for Fiction.LiveBench and 7.7 for VPCT, against 11.6 for GeoBench, 9.8 for SimpleQA and 13.8 for all 191 models. Across benchmarks, $R^2_b$ and that spread have Spearman correlation 0.40. The Thorndike Case II correction, $R^2_c=g_b^2S^2/(g_b^2S^2+\sigma_e^2)$ with $S$ the ECI SD over all models, gives MirrorCode 0.89, Fiction.LiveBench 0.89, VPCT 0.80, SimpleQA 0.77 and GeoBench 0.29 (overall 90% instead of 83%). It assumes the line extends beyond the tested range, so it is not used in the main results.
- **Noise.** Noise in a small benchmark also lowers $R^2_b$, so part of a low $R^2_b$ may be measurement noise rather than an ability the ECI undervalues.
- **Weighting.** Every score counts once, so benchmarks with many models (GPQA 170, AIME 166) count about 7× as much as the smallest (VPCT 25) in the overall $R^2$, in PCA and in the losses. The per-benchmark $R^2_b$, and so the list of poorly explained benchmarks, does not depend on this.
- **Residual structure is modest.** Three components are above the shuffled null, and their bootstrap matches are 0.6–0.7. Treat the component labels as hypotheses.
- **In-sample only.** No held-out prediction is reported. The shuffled null is what guards against reading structure into noise.
- **Clipping.** Everything is on clipped logit scores, so benchmarks with many 0% or 100% scores look less well explained partly because of where the clip is set (§3). This mainly affects Mystery Game Puzzles and Furniture Assembly.

### 9.3 Why the p-norm scores may not be better than the current ECI
1. **A different model.** Epoch's ECI fits a sigmoid with a Beta likelihood and priors (Bayesian by default), on all benchmarks and models, and fixes its scale by anchoring two models: Claude 3.5 Sonnet = 130 and GPT-5 = 150 (`anchors` in `eci_bootstraps.json`). The anchors have no interval, so every ECI interval is relative to them. Its published intervals are 90%. My one-dimensional logit-linear model on a 29-benchmark subset, with no priors or anchors, approximates it. Every comparison is between my approximations, not Epoch's estimator.
2. **No external ground truth for "better".** Each p is best at its own criterion by construction. Choosing among them is a value judgement about what an index is for.
3. **p = ∞ is fragile in sparse data.** The binding benchmarks are small (GeoBench 44 models, MirrorCode 39, VPCT 25) with about nine scores per model. With 191 free scores and a few dozen observations on those benchmarks, the optimizer can bend scores to fit them. The ten models that move most have somewhat fewer scores than average (median 7 against 9), though across all models the link between score changes and number of scores is weak (Spearman −0.08).
4. **Instability as a design property.** Under p = ∞, adding or removing one noisy benchmark can reshuffle scores, because the objective is driven by whichever benchmark currently fits worst.
5. **Scale and interpretation.** Mapping $\theta$ to "ECI points" by matching mean and SD ignores the ECI's anchoring, so point changes are indicative. The ECI's interpretability (benchmark difficulty on the same scale as model capability) is not reproduced.
6. **p = 1 sits inside the ECI's own uncertainty.** It moves scores by about 1.3 points (SD) against a 2.5-point half-width of the 90% interval. Only p = ∞ moves scores well outside it. I did not bootstrap uncertainty for the p-norm scores.
7. **Local optima.** The p = ∞ objective is non-convex; it is reached by warm starts from the ECI through increasing p, so other optima were not explored.
8. **Frontier comparisons are fragile to a few models.** The flatter frontier under p = ∞ (+13.6 vs +16.3 points per year) comes from early models with few scores that move by 15–22 points; restricted to models with at least 8 scores the slopes agree (17.6–17.9). Slopes are also in "points" defined by the mean-and-SD mapping above, so they are indicative. With a break fixed at April 2024, each score has only 1–3 record-setting models before the break (4 for the official ECI over all of Epoch's models), so slopes before the break are not estimable here; only slopes after the break are compared.
