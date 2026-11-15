# Methodology

This document explains, step by step, how every number in the README is computed. Each step says what question it answers, what it does in plain words, the equation, and which script does it.

## Contents
1. [The four steps at a glance](#1-the-four-steps-at-a-glance)
2. [Data and notation](#2-data-and-notation)
3. [Step 1: what the ECI explains, benchmark by benchmark](#3-step-1-what-the-eci-explains-benchmark-by-benchmark)
4. [Step 2: the first principal component, and how close it is to the ECI](#4-step-2-the-first-principal-component-and-how-close-it-is-to-the-eci)
5. [Step 3: remove what the ECI explains, and look at what is left](#5-step-3-remove-what-the-eci-explains-and-look-at-what-is-left)
6. [Step 4: change the loss function](#6-step-4-change-the-loss-function)
7. [How "held-out" numbers are computed](#7-how-held-out-numbers-are-computed)
8. [Figures and the claims they support](#8-figures-and-the-claims-they-support)
9. [Running it](#9-running-it)
10. [Caveats](#10-caveats)

---

## 1. The four steps at a glance

The Epoch Capabilities Index (ECI) boils many benchmark scores down to one number per model. The analysis asks four questions about that aggregation.

| Step | Question | Script | Output folder |
|---|---|---|---|
| 1 | Aggregating many benchmarks into one number must lose something. How much, and where? | `01_pc1_vs_eci.py`, `02_eci_residual.py` | `results/01_pc1_vs_eci/`, `results/02_eci_residual/` |
| 2 | If we run PCA on the benchmarks, is the first principal component (PC1) essentially the ECI? | `01_pc1_vs_eci.py` | `results/01_pc1_vs_eci/` |
| 3 | If we remove everything the ECI explains, what is left? Which benchmarks does the ECI undervalue? | `02_eci_residual.py` | `results/02_eci_residual/` |
| 4 | If we valued fitting each benchmark differently (a different loss function), would the scores change? | `03_loss_functions.py` | `results/03_loss_functions/` |

Shared code (data loading, PCA with missing data, loss solvers) is in `scripts/common.py`. `scripts/04_make_figures.py` draws the figures from the saved results.

---

## 2. Data and notation

**Source.** Epoch AI's `benchmark_data.zip` (CC-BY 4.0), downloaded by `scripts/download_data.py`. We use `epoch_capabilities_index/processed_data_for_eci.csv`, which is the exact model-by-benchmark table Epoch's ECI is fit on, plus `eci_scores.csv` (the official ECI and its 95% interval) and `benchmark_metadata.csv`.

**The matrix.** Each model $m$ has been evaluated on only some benchmarks $b$. Write $y_{mb}\in[0,1]$ for the score when it exists. The set of observed (model, benchmark) cells is $\Omega$; benchmark $b$ was scored on $n_b$ models; there are $N=|\Omega|$ scores in total.

**Which benchmarks and models are kept** (`common.load`):

1. Drop benchmarks that Epoch marks as superseded by another benchmark in the data (2 FrontierMath variants).
2. Keep benchmarks scored on at least 25 models.
3. Drop 9 older benchmarks (MMLU, GSM8K, HellaSwag, TriviaQA, BBH, PIQA, Winogrande, ARC AI2, OpenBookQA). They share a median of about one model with the other benchmarks (they were run on 2023-era models), so their relationship with the rest cannot be estimated. Formally: let $O_{bc}$ be the number of models scored on both $b$ and $c$; drop $b$ if $\operatorname{median}_{c\ne b}O_{bc}<7$.
4. Keep models with at least 4 scores and benchmarks with at least 15 models (repeated until stable).

Result: **191 models × 29 benchmarks, 2,078 scores (37.5% of cells).** The official ECI also uses the excluded benchmarks and models (`fig01`).

**Logit scale.** All analysis is on logit scores,

$$x_{mb}=\log\frac{\tilde y_{mb}}{1-\tilde y_{mb}},\qquad \tilde y_{mb}=\operatorname{clip}(y_{mb},\,0.01,\,0.99).$$

Why: the ECI assumes $y_{mb}=\sigma\big(d_b(\theta_m-c_b)\big)$, an S-curve in the model's capability $\theta_m$, with benchmark difficulty $c_b$ and discriminability $d_b$ (Epoch's `eci-public` README). Taking logits turns that into a straight line, $x_{mb}=d_b(\theta_m-c_b)$, so a benchmark is a line in capability with its own slope and intercept.

**Notation used below**

| Symbol | Meaning |
|---|---|
| $x_{mb}$ | logit score of model $m$ on benchmark $b$ (only for $(m,b)\in\Omega$) |
| $z_m$ | the official ECI of model $m$, standardized (mean 0, SD 1) |
| $\bar x_b$, $\sigma_b$ | mean and SD of benchmark $b$'s scores over the $n_b$ models scored on it |
| $R^2_b$ | share of benchmark $b$'s variation across models explained by a score |

---

## 3. Step 1: what the ECI explains, benchmark by benchmark

**In words.** If the ECI were a complete summary, every benchmark would be a perfect straight line in the ECI. For each benchmark, find the best straight line in the ECI and see how much of the benchmark's variation it captures. What it misses is information the aggregation lost.

**Equation.** For each benchmark $b$, least squares on the models scored on it:

$$x_{mb}=a_b+g_b\,z_m+e_{mb},\qquad R^2_b=1-\frac{\sum_{m\in\Omega_b}e_{mb}^2}{\sum_{m\in\Omega_b}(x_{mb}-\bar x_b)^2}.$$

Overall figures weight benchmarks in two ways:

$$R^2_{\text{equal}}=1-\frac1N\sum_{(m,b)\in\Omega}\frac{e_{mb}^2}{\sigma_b^2},\qquad
R^2_{\text{pooled}}=1-\frac{\sum_\Omega e_{mb}^2}{\sum_\Omega (x_{mb}-\bar x_b)^2}.$$

"Equal" divides each benchmark by its own spread, so no benchmark dominates because its scores are spread out. "Pooled" works in raw logit units, so large, high-spread benchmarks count more.

**Scripts.** `01_pc1_vs_eci.py` (overall and per-benchmark $R^2$, `per_benchmark_explained.csv`) and `02_eci_residual.py` (the same per-benchmark table with an "unexplained" column, `eci_explained_by_benchmark.csv`).

---

## 4. Step 2: the first principal component, and how close it is to the ECI

### 4.1 PCA when most of the table is empty
Ordinary PCA needs a complete matrix; here 62.5% of cells are missing. We use **iterative SVD** on the observed cells. Choose a "base" matrix $B$ to subtract (§4.2), divide each benchmark by its root-mean-square remainder $s_b$, and find the rank-$k$ matrix that best fits the observed cells:

$$\min_{U,V}\ \sum_{(m,b)\in\Omega}\Big(\frac{x_{mb}-B_{mb}}{s_b}-\sum_{j=1}^{k}U_{mj}V_{bj}\Big)^2 .$$

Algorithm: fill each missing cell with the current low-rank estimate, take the SVD of the filled matrix, keep the top $k$ components, and repeat until nothing changes (largest change $<10^{-6}$). The scores $U$ give one number per model per component; the loadings $V$ say how much each benchmark contributes. (`common.fit_lowrank`)

### 4.2 Two ways of centering: conventional (A) and joint intercepts (AJ)
- **A (conventional PCA).** Subtract each benchmark's observed mean: $B_{mb}=\bar x_b$.
- **AJ (joint intercepts).** Treat each benchmark's intercept as unknown and estimate it *together with* the factor: at each iteration, set $B_{mb}$ equal to the mean over observed cells of $x-\text{(current low-rank part)}$.

*Why AJ.* Suppose the truth is $x_{mb}=\beta_b+g_b\theta_m$. The observed mean is

$$\bar x_b=\beta_b+g_b\,\bar\theta_{\Omega_b},$$

shifted by the average ability $\bar\theta_{\Omega_b}$ of the models that happened to be tested. Hard benchmarks are mostly run on strong models, so $\bar x_b$ overstates the benchmark's typical score. Conventional PCA bakes that error in. AJ removes it. `fig04` shows the bias: across benchmarks, (observed mean − model-implied mean over all models) correlates $r=0.90$ with how strong the tested models were.

### 4.3 Variance explained
With $P$ a matrix of predictions, the in-sample $R^2_{\text{equal}}$ and $R^2_{\text{pooled}}$ of §3 are computed for: rank-1, 2, 3 PCA under A and under AJ, and the official ECI as a single score (the per-benchmark lines of §3).

The held-out version is described in §7. We report it as

$$R^2_{\text{held-out}}(k)=1-\frac{\mathrm{MSE}_k}{\mathrm{MSE}_{\text{benchmark means}}}.$$

### 4.4 Comparing PC1 with the ECI
The Pearson correlation $r$ between each model's PC1 score (rank-1 fit) and its official ECI, with a 95% interval from the Fisher transform,

$$\Big[\tanh\!\big(\operatorname{arctanh}r-\tfrac{1.96}{\sqrt{n-3}}\big),\ \tanh\!\big(\operatorname{arctanh}r+\tfrac{1.96}{\sqrt{n-3}}\big)\Big],$$

and the Spearman rank correlation. For a one-variable comparison, $r^2$ is the share of the ECI's variance that PC1 reproduces.

**Script.** `01_pc1_vs_eci.py`. Outputs: `variance_explained_in_sample.csv`, `variance_explained_held_out.csv`, `eci_vs_pc1.csv`, `pc1_scores.csv`, `selection_bias.csv`.

---

## 5. Step 3: remove what the ECI explains, and look at what is left

**In words.** Take away from every score the part a straight line in the ECI explains. What remains, the residual, is the variation the ECI does not capture. A benchmark with a large residual is one the ECI undervalues (or one that measures something other than what the ECI measures; see §10). Then ask whether the residuals have shared structure across benchmarks, using PCA.

**Equations.** The residual is the $e_{mb}$ from §3:

$$e_{mb}=x_{mb}-(a_b+g_b z_m).$$

We run the low-rank PCA of §4.1 with base $B_{mb}=a_b+g_b z_m$, so it decomposes the $e_{mb}$. The share of residual variation that $k$ components capture in-sample is

$$1-\frac1N\sum_\Omega\Big(\frac{e_{mb}-\hat e^{(k)}_{mb}}{s_b}\Big)^2,$$

where $s_b$ is the RMS of benchmark $b$'s residuals. This is optimistic: with about nine scores per model, a few components can always fit the observed cells. So we test whether the structure is real in three ways.

1. **Shuffled null.** Compute the eigenvalues of the benchmark-by-benchmark correlation matrix of the residuals (pairwise-complete correlations over shared models, at least 12; missing pairs set to 0; matrix made positive semi-definite by clipping eigenvalues at $10^{-6}$ and renormalizing). Compare with the same eigenvalues after shuffling each benchmark's residuals among the models scored on it (100 shuffles; we use the 95th percentile). Real structure should sit above the shuffled curve.
2. **Held-out prediction.** Predict held-out cells with the ECI line alone, then with $k=1,\dots,6$ residual components added; compare held-out squared error (§7). If the structure is real it should lower the error.
3. **Bootstrap stability.** Resample models with replacement (60 times), refit, and for each full-sample loading vector $v_k$ compute its best match $\max_j|\langle v_k,v^*_j\rangle|$ among the bootstrap components (cosine similarity). Report the mean and 10th percentile. A value near 1 means the component is reproducible.

**Reading the components.** Loadings are signed and the sign of a component is arbitrary (we flip each so its largest loading is positive). Labels like "agentic" are interpretation, not output.

**Script.** `02_eci_residual.py`. Outputs: `eci_explained_by_benchmark.csv`, `residual_loadings.csv`, `residual_scores.csv`, `residual_eigenvalues.csv`, `residual_held_out.csv`, `residual_stability.csv`.

---

## 6. Step 4: change the loss function

**In words.** Suppose someone built the one-number index differently: for example, caring most about the benchmark the index fits worst. We build a one-number score per model under several losses, starting from the official ECI, and measure how the scores move.

**Model.** One score $\theta_m$ per model, and a straight line per benchmark with its own intercept $a_b$ and slope $g_b$. With normalized scores $y_{mb}=(x_{mb}-\bar x_b)/\sigma_b$, the errors are

$$e_{mb}=y_{mb}-a_b-g_b\,\theta_m .$$

**Losses** (`common.cell_loss_solve`, `common.worst_benchmark_path`):

| Loss | What is minimized | What it values |
|---|---|---|
| MSE, cells | $\dfrac1N\sum_b\sigma_b^2\sum_{m\in\Omega_b}e_{mb}^2$ | squared error in raw logit units; large, high-spread benchmarks count more |
| MSE, benchmark average | $\dfrac1B\sum_b\dfrac1{n_b}\sum_{m\in\Omega_b}e_{mb}^2$ | squared error, every benchmark counts equally |
| MAE (absolute value), benchmark average | $\dfrac1B\sum_b\dfrac1{n_b}\sum_{m\in\Omega_b}|e_{mb}|$ | like the above but less sensitive to large errors; $|e|$ is smoothed as $\sqrt{e^2+\delta^2}-\delta$ with $\delta=0.1,0.03,0.01$ in turn |
| Worst benchmark | $\max_b\,(1-R^2_b)$ | only the worst-fitted benchmark matters |

**The worst-benchmark loss.** Because $a_b$ and $g_b$ are free, the best a benchmark can do is $R_b^2=r_b(\theta)^2$, the squared correlation of the score with the benchmark over the models scored on it. Define $u_b=1-r_b|r_b|$ (equal to $1-R^2_b$ when the slope is positive; using $r|r|$ stops a negative-slope fit from looking good), and minimize

$$F_p(\theta)=\Big(\frac1B\sum_b u_b^{\,p}\Big)^{1/p}.$$

$p=1$ is the benchmark-averaged MSE exactly (the script checks this: the two solutions have correlation 1.0000). As $p$ grows, $F_p$ approaches the worst benchmark's $u_b$; we use $p=8$ and $p=64$ (the latter stands in for $p=\infty$). Increasing $p$ gradually gives a trade-off curve between average fit and worst-case fit (`fig08`).

**Optimization.** L-BFGS-B with analytic gradients. Every solution starts from the standardized official ECI. The $p$ values are solved in order, each starting from the previous solution. Scores are standardized after each solve, because the losses do not change under shifting and rescaling $\theta$.

**Comparison with the official ECI.**
- *Fit:* $R^2_b=r_b^2$ per benchmark (mean, worst, and which benchmark is worst).
- *Rank:* $|\operatorname{rank}(\theta_m)-\operatorname{rank}(\text{ECI}_m)|$ for each model (mean shown as % of the 191 models).
- *Points:* the score change in ECI points after mapping $\theta$ onto the ECI scale by matching mean and SD, $\widehat{\text{ECI}}_m=\overline{\text{ECI}}+\mathrm{sd}(\text{ECI})\,\theta_m$. We compare against the typical ECI 95% half-width, 2.5 points. The matching is a convention (§10).

**Scores over release date** (the counterpart of Epoch's ECI-versus-date chart). Each model's release date is the `date` column of `eci_scores.csv`. A measure's score in ECI points is the official ECI plus its change from the previous paragraph. The *frontier* of a measure is the set of models that scored higher than every model released before them (a running maximum by release date); its *slope* is the least-squares line through those record-setting points, in points per year. We also report (a) the share of record-setting models in common with the official ECI's frontier (intersection over union), (b) the correlation of each model's score change with its release date, and (c) as a robustness check, the frontier slope using only models with at least 8 scores, because models with few scores are the ones whose scores move most.

**Script.** `03_loss_functions.py`. Outputs: `loss_summary.csv`, `worst_benchmark_path.csv`, `loss_benchmark_r2.csv`, `loss_score_change_points.csv`, `loss_held_out.csv`, `loss_over_time.csv`, `loss_frontier_models.csv`.

---

## 7. How "held-out" numbers are computed

"In-sample" fit always improves with more flexibility. Held-out fit asks whether the model predicts scores it did not see. There are two variants.

**Variant 1: held-out cells for PCA models (steps 2 and 3; `common.cv_mse`).** Randomly split the observed cells into 5 folds (repeated twice). For each fold, blank those cells, refit the model on the rest (including its centering or ECI lines), and predict the blanked cells. A blanked cell is only scored if its model keeps at least 2 scores and its benchmark at least 5 scores. Report the mean squared error $\mathrm{MSE}_k$ for each number of components $k$. The official ECI cannot be scored this way, because it was fit on those cells; where an ECI line is used as a base, the line is refit on training cells, but the ECI values themselves were estimated using the held-out scores. That is optimistic for the ECI.

**Variant 2: refit and predict for the loss functions (step 4).** Split observed cells into 5 folds. For each fold, rebuild each one-dimensional score on the training cells only (starting from the ECI), then fit an ordinary least squares line per benchmark of score on $\theta$ on the training cells and predict the held-out cells. For each benchmark with at least 8 test cells, held-out

$$R^2_b=1-\frac{\sum_{\text{test}}(x-\hat x)^2}{\sum_{\text{test}}(x-\bar x_{b,\text{train}})^2}.$$

Report the mean and worst across benchmarks, plus mean absolute error in units of the benchmark's training SD. The same OLS line is used for all losses, so the comparison is on the score, not on how the line was fit. This favors scores trained with squared error somewhat.

---

## 8. Figures and the claims they support

All figures are in `results/figures/`, drawn by `04_make_figures.py` from the saved CSVs.

| Figure | Claim | Step |
|---|---|---|
| `fig01_data_sparsity` | Only ~20% of cells are observed; older benchmarks barely overlap with newer ones | data |
| `fig02_pc1_vs_eci` | The ECI is nearly the same as PC1 (r = 0.995 with joint intercepts, 0.968 with conventional centering) | 2 |
| `fig03_variance_explained` | One joint-intercept component explains ~84% of benchmark variation in-sample; out of sample it is the best choice | 1, 2 |
| `fig04_selection_bias` | Observed benchmark means are biased by who was tested, which is why joint intercepts are used | 2 |
| `fig05_eci_by_benchmark` | The ECI explains most benchmarks well and a few poorly | 1, 3 |
| `fig06_residual_structure` | The structure left after removing the ECI is above chance but does not improve prediction and is only moderately stable | 3 |
| `fig07_residual_components` | The three leading axes of what the ECI does not explain (tentative) | 3 |
| `fig08_loss_tradeoff` | Fitting the worst benchmark better costs average fit | 4 |
| `fig09_loss_per_benchmark` | Worst-benchmark loss flattens fit across benchmarks | 4 |
| `fig10_loss_score_changes` | MSE moves scores within the ECI's own uncertainty; worst-benchmark loss moves some by 20+ points | 4 |
| `fig11_loss_held_out` | Out of sample the worst-benchmark loss is worse on average and in the worst case | 4 |
| `fig12_scores_over_time` | Each measure against release date, with its frontier (Epoch-style chart) | 4 |
| `fig13_frontier_over_time` | Average losses track the official frontier; the worst-benchmark loss lifts early records, but only through models with few scores | 4 |

---

## 9. Running it

```
pip install pandas numpy scipy matplotlib
python -I scripts/download_data.py data
python -I scripts/01_pc1_vs_eci.py       data/extracted results/01_pc1_vs_eci
python -I scripts/02_eci_residual.py     data/extracted results/02_eci_residual
python -I scripts/03_loss_functions.py   data/extracted results/03_loss_functions
python -I scripts/04_make_figures.py     data/extracted results
```

Each script prints its main numbers (the logs saved in the results folders are these printouts). Together they take a few minutes. Tested from a fresh download: step 1 reproduces the saved results exactly.

---

## 10. Caveats

### 10.1 Data
- **Sparse and non-random.** 62.5% of cells are missing and strong models are tested on hard benchmarks. Joint intercepts fix the intercept bias but not every kind of selection.
- **A subset.** Conclusions concern 29 benchmarks and 191 models. The ECI also uses the 9 older benchmarks and benchmarks with fewer than 25 models, so some gap between the ECI and PC1 may come from those.
- **Models are not independent.** Models from the same family have correlated scores; the confidence intervals here ignore that and are too narrow.

### 10.2 Steps 1–3
- **"Unexplained" is not "undervalued".** A benchmark with a low $R^2_b$ may be measuring something other than general capability (for example geolocation), in which case the ECI is not at fault. Equally, noise in a small benchmark lowers $R^2_b$. The unexplained share is a candidate list, not a verdict.
- **Residual structure is weak.** It is above the shuffled null but does not improve held-out prediction, and the bootstrap matches are only 0.6–0.7. An earlier version of this analysis fit the ECI curve on raw (not logit) scores; its components matched the current ones only moderately (cosine 0.4–0.6), though the per-benchmark ranking of how much the ECI explains was similar (Spearman 0.90; VPCT, Furniture Assembly and Mystery Game Puzzles moved most). Treat the component labels as hypotheses.
- **In-sample numbers are optimistic** for higher-rank fits. Held-out numbers are the reliable ones but are noisy under sparsity.
- **Logit only.** A raw-score version of steps 1–2 was not run.

### 10.3 Why the alternative loss functions may not be better than the current ECI
1. **A different model.** Epoch's ECI fits a sigmoid with a Beta likelihood and priors (Bayesian by default), on all benchmarks and models, and flags some benchmarks as anchors (`is_anchor` in `edi_scores.csv`; I did not check how they fix the scale). My one-dimensional logit-linear model on a 29-benchmark subset, with no priors or anchors, approximates it. Every comparison is between my approximations, not Epoch's estimator.
2. **No external ground truth for "better".** Each loss is best at its own criterion by construction. Choosing among them is a value judgement about what an index is for. Held-out fit is the only neutral yardstick, and there the worst-benchmark loss is worse.
3. **The worst-benchmark loss is fragile in sparse data.** The binding benchmarks are small (GeoBench 44 models, MirrorCode 39, VPCT 25) with about nine scores per model. With 191 free scores and a few dozen observations on those benchmarks, the optimizer can bend scores to fit them, and the models that move most are those with few scores.
4. **Instability as a design property.** Under a worst-benchmark loss, adding or removing one noisy benchmark can reshuffle scores, because the objective is driven by whichever benchmark currently fits worst.
5. **A poorly fitted benchmark may not be a defect** (see 10.2). Forcing the index to fit GeoBench as well as AIME changes what the index means.
6. **Weighting is a choice.** MSE over cells weights large benchmarks; MSE per benchmark weights all equally. They give scores that correlate 0.99 but differ in held-out fit (mean $R^2$ 0.74 vs 0.70). I did not tune this.
7. **MAE does not help here.** In principle it is more robust to outlying cells, but its scores move slightly more than MSE's and its held-out worst case is the weakest of the average-type losses. The held-out OLS line favors squared-error scores somewhat.
8. **Scale and interpretation.** Mapping $\theta$ to "ECI points" by matching mean and SD ignores the ECI's anchoring, so point changes are indicative. The ECI's interpretability (benchmark difficulty on the same scale as model capability) is not reproduced.
9. **Average-type losses sit inside the ECI's own uncertainty.** MSE moves scores by about 1 point (SD) against a 2.5-point half-width. Only the worst-benchmark loss moves scores well outside it. I did not bootstrap uncertainty for the alternative-loss scores.
10. **Local optima.** Worst-benchmark and MAE objectives are non-smooth or non-convex; all solutions start at the ECI, so other optima were not explored.
11. **Frontier comparisons are fragile to a few models.** The flatter frontier under the worst-benchmark loss (+13.6 vs +16.3 points per year) comes from early models with 6–9 scores that move by 10–22 points; restricted to models with at least 8 scores the slopes agree (17.7–18.3). Slopes are also in "points" defined by the mean-and-SD mapping above, so they are indicative.
12. **Evaluation scope.** The held-out comparison excludes the official ECI (it saw those cells), covers benchmarks with at least 8 test cells, and uses a single 5-fold split.
