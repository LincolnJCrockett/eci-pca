# Methodology

Everything in this repository, in order, with the equations used and the file that implements each step. Findings are in `RESULTS.md`; the plan is in `PLAN.md`. All commands assume the repository root; scripts are run as `python -I scripts/<name>.py <args>`.

## 0. Map of files

| Step | Script | Main outputs |
|---|---|---|
| Download data | `scripts/download_data.py` | `data/raw/benchmark_data.zip`, `data/extracted/` |
| Exploration (shape, sparsity, overlap) | `scripts/explore.py`, `scripts/diag.py` | console only |
| First-pass PCA (z-scored raw scores) | `scripts/run_pca.py`, `scripts/interpret.py` | `outputs/*.csv`, `outputs/run_log.txt`, `outputs/interpret_log.txt` |
| First-pass "ECI removed" residual PCA (superseded) | `scripts/residual_pca.py` | `outputs/residual_*` |
| Demeaning specifications A, B, C, CJ, D | `scripts/spec_pca.py` | `outputs/spec_logit/`, `outputs/spec_logit_log.txt` |
| Shared estimators | `scripts/common.py` | (library) |
| Variance explained; ECI-removed residual PCA (E) | `scripts/variance_and_eci_resid.py` | `outputs/variance/` |
| ECI vs components (corrected) | `scripts/eci_vs_components_v2.py` | `outputs/variance/q1_partial_corr_AJ.csv`, `q2_components_v2.csv` |
| Alternative loss functions | `scripts/loss_comparison.py` | `outputs/loss_comparison/` |
| Figures | `scripts/make_figures.py` | `outputs/figures/` |

## 1. Data

Source: Epoch AI `benchmark_data.zip` (CC-BY 4.0), downloaded by `download_data.py`. The analysis uses `epoch_capabilities_index/processed_data_for_eci.csv`, the table Epoch's ECI pipeline fits: one row per (model $m$, benchmark $b$) with score $y_{mb}\in[0,1]$. Also used: `benchmark_metadata.csv` (random-baseline floor $f_b$, `superseded_by`), `epoch_capabilities_index/eci_scores.csv` (official ECI, 95% interval, release date, weights accessibility) and `edi_scores.csv` (benchmark difficulty, in `interpret.py` only).

Let $\Omega$ be the set of observed (model, benchmark) cells, $\Omega_b$ the models observed on benchmark $b$, $n_b=|\Omega_b|$, and $N=|\Omega|$.

**Sample construction** (`common.load`; `run_pca.py` differs slightly, see below):
1. Drop benchmarks whose `superseded_by` benchmark is also in the data (2 FrontierMath variants).
2. Keep benchmarks with $n_b\ge 25$.
3. *Legacy exclusion.* With overlap matrix $O_{bc}=\sum_m \mathbf 1[(m,b)\in\Omega]\,\mathbf 1[(m,c)\in\Omega]$ and $\operatorname{med}_b=\operatorname{median}_{c\ne b}O_{bc}$, drop benchmarks with $\operatorname{med}_b<7$. This removes 9 older benchmarks (MMLU, GSM8K, HellaSwag, TriviaQA, BBH, PIQA, Winogrande, ARC AI2, OpenBookQA) that share a median of about 1 model with other benchmarks (their pairwise correlations are not estimable; see `outputs/run_log.txt` and `fig01`). They are analysed separately in `run_pca.py` only.
4. Iterate (5 passes): keep models with $\ge 4$ observed scores and benchmarks with $\ge 15$ models.

Result: 191 models × 29 benchmarks, $N=2078$ (37.5% of cells). `run_pca.py` (first pass) keeps models with $\ge 2$ scores, giving 198 models × 29 benchmarks.

**Transform.** Main analyses use logits, $x_{mb}=\log\frac{\tilde y_{mb}}{1-\tilde y_{mb}}$ with $\tilde y=\operatorname{clip}(y,0.01,0.99)$. `run_pca.py` used raw $y$ first and logit as a sensitivity check (loading cosine > 0.99). Spec E uses raw $y$ (§6).

## 2. Low-rank PCA with missing data (`common.fit`)

Plain PCA needs a complete matrix. For a base matrix $B$ (which spec determines), scale $s_b=\sqrt{\frac1{n_b}\sum_{m\in\Omega_b}(x_{mb}-B_{mb})^2}$ (RMS of the residual), and rank $k$, solve

$$\min_{U,V}\ \sum_{(m,b)\in\Omega}\Big(\tfrac{x_{mb}-B_{mb}}{s_b}-\textstyle\sum_{j=1}^k U_{mj}V_{bj}\Big)^2 .$$

Algorithm (iterative SVD / EM): set missing cells of $Z=(X-B)/s$ to the current rank-$k$ reconstruction $R$; take the SVD of the filled matrix; set $R\leftarrow$ rank-$k$ truncation; repeat to convergence ($\max|\Delta R|<10^{-6}$). Scores are $US$ (columns of $U$ times singular values), loadings are the columns of $V$, and the prediction is $\hat x=B+sR$. Components are sign-flipped for readability only (the largest-magnitude loading is made positive).

## 3. Specifications (what is removed before the low-rank step)

All implemented as `base()`/`fit()` in `common.py`; A–D were first run in `spec_pca.py`.

| Spec | Base $B_{mb}$ | Removes |
|---|---|---|
| **A** | $\bar x_b=\frac1{n_b}\sum_{m\in\Omega_b}x_{mb}$ | benchmark difficulty; standard PCA |
| **B** | $\bar x_m=\frac1{|\Omega_m|}\sum_{b}x_{mb}$, *no* column centering afterward | model quality |
| **C** | $\hat\alpha_m+\hat\beta_b$ from two-way fixed effects, below | both, assuming every benchmark has slope 1 |
| **CJ** | $\alpha_m+\beta_b$ re-estimated jointly with the low-rank part | robustness for C |
| **AJ** | $\beta_b$ (benchmark intercept) re-estimated jointly with the low-rank part | selection bias in A |
| **D** | rank-1 fit of A | model quality × benchmark slope, plus mean |
| **DJ** | rank-1 fit of AJ | as D, with unbiased intercepts |
| **E** | per-benchmark logistic fit on official ECI (§6) | what the ECI explains |

*Two-way fixed effects* (`fe()`): $x_{mb}=\alpha_m+\beta_b+\varepsilon_{mb}$ on observed cells, solved by alternating means $\alpha_m\leftarrow\overline{(x_{mb}-\beta_b)}_b$, $\beta_b\leftarrow\overline{(x_{mb}-\alpha_m)}_m$ until the change is $<10^{-9}$ (unbalanced panel).

*Why joint intercepts (AJ).* If $x_{mb}=\beta_b+g_b\theta_m+\epsilon$, then the observed mean is $\bar x_b=\beta_b+g_b\bar\theta_{\Omega_b}+\bar\epsilon$, i.e. it is shifted by the mean ability of the models that happened to be tested. Hard benchmarks are mostly run on strong models, so centering on $\bar x_b$ is biased; AJ estimates $\beta_b$ and the factor together. `fig02` shows the bias (observed mean minus model-implied mean vs mean ECI of tested models, $r=0.90$). With AJ: PC1–ECI correlation is 0.995 vs 0.968 with A.

*Remark (B vs C).* With a complete matrix, row-demeaning plus column-centering equals the two-way residual; the specs differ here only because the data are sparse, which is why B is not column-centered.

## 4. Variance explained (`variance_and_eci_resid.py` §1)

For a prediction matrix $P$, with $\bar x_b$ the observed benchmark mean and $\sigma_b$ its within-benchmark SD:

$$R^2_{\text{pool}}=1-\frac{\sum_\Omega(x-P)^2}{\sum_\Omega(x-\bar x_b)^2},\qquad R^2_{\text{eq}}=1-\frac1N\sum_\Omega\Big(\frac{x-P}{\sigma_b}\Big)^2 .$$

$R^2_{\text{eq}}$ weights benchmarks by their number of scores but standardizes each by its spread. In-sample $P$: rank-$k$ fits of A and AJ, the additive FE model, and the official ECI as a one-dimensional score through per-benchmark OLS, $\hat x_{mb}=a_b+g_b z_m$, where $z$ is standardized ECI (also a quadratic version).

*Held-out* (`common.cv`): 5 folds × 2 repeats over observed cells; cell $(m,b)$ is eligible for testing only if model $m$ retains $\ge 2$ and benchmark $b$ retains $\ge5$ training scores. Held-out $R^2_k = 1-\mathrm{MSE}_k/\mathrm{MSE}_{\text{A},k=0}$ (relative to the benchmark-mean model). The official ECI is not scored held-out, because it was fit on those cells. (The first pass, `run_pca.py`, used its own `cv_k`: 5 folds, one repeat, $k\le6$, on z-scored raw scores, so its CV numbers are not comparable with these.)

*Classical share.* `pairwise_eigs`: pairwise-complete correlations $r_{bc}$ over models with both scores (min 12 shared; missing set to 0; diagonal 1), made positive semi-definite by clipping eigenvalues at $10^{-6}$ and rescaling to unit diagonal, then eigendecomposed; PC share $=\lambda_1/\sum\lambda$.

## 5. ECI versus the components (`variance_and_eci_resid.py` §2, `eci_vs_components_v2.py`)

- Correlation with Fisher interval: $r$, $\;\operatorname{CI}=\tanh\big(\operatorname{arctanh}r\pm1.96/\sqrt{n-3}\big)$; Spearman reported alongside. Coverage terciles split models by number of scores.
- *ECI residual after PC1*: $\rho_m=\mathrm{ECI}_m-(\hat a+\hat b\,\mathrm{PC1}_m)$ (linear; the quadratic adds nothing). Its variance share is $\operatorname{Var}\rho/\operatorname{Var}\mathrm{ECI}\approx1\%$ for AJ.
- *What it correlates with*: number of benchmarks, mean easiness of the benchmarks taken (mean of $\hat\beta_b$ over observed $b$), release date.
- *Per-benchmark partial correlation*: $\operatorname{corr}\big(x_{\cdot b}-\widehat{x_{\cdot b}}(\mathrm{PC1}),\ \rho\big)$ over $\Omega_b$ with $n_b\ge15$ (`q1_partial_corr_AJ.csv`).
- *Component scores vs ECI*: correlations of component scores with ECI, with $\rho$, and with date-adjusted versions (both residualized on a linear date trend); `q2_components_v2.csv`.

## 6. Spec E: remove what the official ECI explains (`variance_and_eci_resid.py` §3)

For each benchmark, on raw scores, fit by nonlinear least squares over $(a_b,c_b)$:

$$\hat y_{mb}=f_b+(1-f_b)\,\sigma\big(a_b(z_m-c_b)\big),\qquad \sigma(t)=\tfrac1{1+e^{-t}},$$

with $f_b$ the benchmark's random-baseline floor, $z$ standardized ECI, bounds $a_b\in[10^{-3},50]$, $c_b\in[\min z-4,\max z+4]$ (fallback to OLS if the fit fails). Residuals $e_{mb}=y_{mb}-\hat y_{mb}$ go into the §2 low-rank step with base $B=\hat y$. Reported: per-benchmark $R^2_b=1-\sum e^2/\sum(y-\bar y_b)^2$, the share of residual variance explained by $k=1,2,3$ components, and held-out gain $1-\mathrm{MSE}_k/\mathrm{MSE}_0$ with the logistic base refit on training cells in each fold (`E_cv.csv`; optimistic because ECI saw the held-out cells). The first-pass version (`residual_pca.py`) did the same in raw ECI units ($a\in[10^{-4},2]$, $c\pm80$) and is superseded.

## 7. Is the structure real? Null model and stability

*Permutation null* (parallel analysis). The eigenvalues of the residual correlation matrix are compared with a null built by permuting each column's observed residuals among the models observed on it (100 draws; 95th percentile). For spec C (`spec_pca.py`) the null also refits the fixed effects after permuting, so it includes fixed-effect-induced negative correlation; the E null does not refit the per-benchmark logistic (small effect, 2 parameters per column). Outputs: `E_eigs.csv`.

*Bootstrap stability.* Resample models with replacement (60 draws), refit, and compute the best-match $\max_j|\langle v_k,v_j^{*}\rangle|$ for each full-sample loading vector $v_k$; report mean and 10th percentile (`E_stability.csv`; for A, C, D in `spec_logit_log.txt`). A cosine near 1 means the component is reproduced.

*Slope diagnostic* (`spec_pca.py` §7). $\gamma_b$ = OLS slope of benchmark $b$ on the fitted model effect $\hat\alpha_m$; slopes range 0.27–2.03 (`benchmark_slopes.csv`, `fig15`). The correlation of C's loadings with $\gamma_b$ ($-0.71$ for PC3) shows C's components are slope heterogeneity, which motivated D, DJ and AJ.

## 8. Alternative loss functions for a one-dimensional score (`loss_comparison.py`, `common.py`)

Question: if the score $\theta_m$ were chosen under a different loss, how far would it move from the official ECI? Model: $x_{mb}\approx a_b+g_b\theta_m$ (one factor, free intercept and slope per benchmark, logit scale). With normalized scores $y_{mb}=(x_{mb}-\bar x_b)/\sigma_b$, errors are $e_{mb}=y_{mb}-a_b-g_b\theta_m$.

| Loss | Objective (`joint_solve` / `pnorm_solve`) |
|---|---|
| MSE, cells | $\frac1N\sum_b\sigma_b^2\sum_{m\in\Omega_b}e_{mb}^2$ (squared error in raw logit units; large or high-variance benchmarks count more) |
| MSE, benchmark average | $\frac1B\sum_b\frac1{n_b}\sum_{m}e_{mb}^2$ (benchmarks weighted equally) |
| MAE (absolute value), benchmark average | $\frac1B\sum_b\frac1{n_b}\sum_m\rho_\delta(e_{mb})$, $\rho_\delta(e)=\sqrt{e^2+\delta^2}-\delta\to|e|$, with $\delta=0.1,0.03,0.01$ as a warm-started sequence |
| Worst benchmark | $\max_b u_b$, $u_b=1-R_b^2$ |

Because $a_b,g_b$ are free, the optimal $\sum_m e^2/\sum(y-\bar y)^2$ for a benchmark is $1-r_b(\theta)^2$ with $r_b$ the correlation of $\theta$ with the benchmark over $\Omega_b$. So the average-over-benchmarks MSE is $\frac1B\sum_b u_b$, and the worst-benchmark loss is the $p\to\infty$ limit of

$$F_p(\theta)=\Big(\frac1B\sum_b u_b^{\,p}\Big)^{1/p},\qquad u_b=1-r_b|r_b| .$$

($r|r|$ instead of $r^2$ keeps a negative-slope fit from being rewarded.) $p=1$ reproduces the benchmark-averaged MSE solution exactly (a sanity check in the output), and $p=64$ stands in for $\infty$. Minimization: L-BFGS-B with analytic gradients (for $r_b$: $\partial r_b/\partial\theta_m = \frac{\tilde y_m}{\|\tilde\theta\|\|\tilde y\|}-r_b\frac{\tilde\theta_m}{\|\tilde\theta\|^2}$, tilde = centered over $\Omega_b$), warm-started along $p=1,2,4,\dots,64$, initialized at standardized ECI, $\theta$ standardized after each solve (loss is invariant to affine changes of $\theta$).

*Comparison metrics.* In-sample per-benchmark $R^2_b=r_b^2$ (mean, minimum, and which benchmark is worst); correlation with ECI; rank shift $|\operatorname{rank}(\theta_m)-\operatorname{rank}(\mathrm{ECI}_m)|$; and score change in ECI points after the affine map $\widehat{\mathrm{ECI}}_m=\overline{\mathrm{ECI}}+\mathrm{sd}(\mathrm{ECI})\,\theta_m$ (this mapping is a convention; see caveats), compared with the mean ECI 95% half-width (2.5 points).

*Held-out protocol.* 5 folds over observed cells (eligibility: model keeps $\ge 2$, benchmark keeps $\ge 8$ training scores). Scores are refit on the training cells (initialized at standardized ECI), then for each benchmark an OLS line $x\sim\theta$ is fit on training cells and used to predict the test cells. Reported: held-out $R^2_b=1-\sum_{test}(x-\hat x)^2/\sum_{test}(x-\bar x_{b,\text{train}})^2$ (mean, minimum, 10th percentile over benchmarks with $\ge 8$ test cells) and mean absolute error in training-SD units.

## 9. Figures and the claims they support (`make_figures.py`, `outputs/figures/`)

| Figure | Claim |
|---|---|
| `fig01_data_structure` | The data are sparse (~20% filled) with an older block that barely overlaps newer benchmarks |
| `fig02_selection_bias` | Observed benchmark means are biased by who was tested, so conventional PCA understates PC1 |
| `fig03_variance_explained` | One properly estimated component explains ~84% in-sample, ~82% held-out; more components don't help out of sample |
| `fig04_eci_vs_pc1` | ECI ≈ PC1 (r = 0.995 with AJ vs 0.968 with fixed means) |
| `fig05_eci_by_benchmark` | ECI explains most benchmarks well and a few (GeoBench, MirrorCode, Furniture Assembly, Fiction.LiveBench, SimpleQA) poorly |
| `fig06_residual_structure` | Some residual structure beats a null, but it does not improve held-out prediction and is only moderately stable |
| `fig07_residual_components` | The residual axes (E1–E3) and their loadings |
| `fig08_eci_tilt_by_benchmark` | Relative to PC1, ECI leans toward math and METR, away from long-context, geolocation, vision, factual recall |
| `fig09_eci_residual_vs_E3` | The one robust tilt: ECI's leftover correlates with the competition-math axis |
| `fig10_loss_tradeoff` | Fitting the worst benchmark better costs average fit |
| `fig11_loss_per_benchmark` | Worst-benchmark loss flattens per-benchmark fit |
| `fig12_score_changes` | MSE moves scores within ECI's interval; worst-benchmark loss moves some by 20+ points |
| `fig13_loss_heldout` | Out of sample the worst-benchmark loss is worse on average *and* in the worst case |
| `fig14_who_moves` | The largest rank shifts are among models with few scores |
| `fig15_benchmark_slopes` | Benchmarks rise at very different rates with model quality (why additive FE fails) |
| `fig16_solution_correlations` | MSE, MAE and ECI give nearly the same scores; only worst-benchmark loss departs |

Older figures from earlier passes remain in `outputs/` (`scree_cv.png`, `loadings_heatmap.png`, `benchmark_map.png`, `residual_pca.png`, `outputs/spec_logit/fig_*.png`, `outputs/variance/fig_*.png`).

## 10. Reproduction order

```
python -I scripts/download_data.py data
python -I scripts/run_pca.py data/extracted outputs
python -I scripts/interpret.py data/extracted outputs
python -I scripts/residual_pca.py data/extracted outputs
python -I scripts/spec_pca.py data/extracted outputs/spec_logit logit
python -I scripts/variance_and_eci_resid.py data/extracted outputs/variance
python -I scripts/eci_vs_components_v2.py data/extracted outputs/variance
python -I scripts/loss_comparison.py data/extracted outputs/loss_comparison
python -I scripts/make_figures.py data/extracted outputs outputs/figures
```
(`explore.py <processed_csv> <metadata_csv>` and `diag.py <extracted_dir>` are exploratory. Console outputs were saved to the `outputs/*_log.txt` files by redirecting stdout.)

## 11. Caveats

### Why the alternative losses may not be better than the current ECI
1. **Different model.** Epoch's ECI fits $\text{performance}=\sigma(\text{discriminability}\times(\text{capability}-\text{difficulty}))$ with a Beta likelihood and priors (default Bayesian fit; the least-squares fit is deprecated), on all benchmarks and models, with some benchmarks flagged as anchors (`is_anchor` in `edi_scores.csv`; I have not checked exactly how they fix the scale) (Epoch's `eci-public` README). My one-dimensional model is logit-linear, on a 29-benchmark, 191-model subset, with no priors or anchors. Every comparison here is between *my* approximations, not Epoch's actual estimator.
2. **No external ground truth for "better".** Each loss is best at its own criterion by construction: ECI-like least squares maximizes average fit; minimax maximizes worst fit. Choosing among them is a value judgement about what a capability index is for, not an empirical finding. Held-out fit is the only neutral yardstick, and there the worst-benchmark loss is worse.
3. **Minimax is fragile in sparse data.** It focuses on the benchmarks with the lowest $R^2$, which here are small or idiosyncratic (GeoBench $n_b=44$, MirrorCode 39, VPCT 25) with ~9 scores per model. With 191 free scores and a few dozen observations on the binding benchmarks, the optimizer can bend scores to fit them (held-out worst-case $R^2\approx0$ vs $0.14$ for MSE). Models with few scores move most (`fig14`).
4. **Instability as a design property.** Under minimax, adding or removing one noisy benchmark can reshuffle scores, because the objective is driven by whichever benchmark currently fits worst. The average-loss index is more stable. It also gives an incentive to include benchmarks that merely correlate poorly.
5. **A poorly fit benchmark may not be a defect.** If GeoBench tests something other than general capability, a low $R^2$ is informative, and forcing the index to fit it as well as AIME changes what the index means.
6. **Weighting is a choice.** MSE over cells weights large benchmarks; MSE per benchmark weights all equally; the two differ little in scores (corr 0.99) but differ in held-out fit (0.74 vs 0.70 mean). I did not tune this.
7. **MAE does not help here.** Absolute-value loss is more robust to outlying cells in principle, but its scores shift slightly more than MSE and its held-out worst-case is the weakest of the three average losses (worst benchmark R² $-0.27$, Aider polyglot). Held-out error is measured with OLS-calibrated lines, which favors least-squares-trained scores somewhat.
8. **Scale and interpretation.** Mapping $\theta$ to "ECI points" by matching ECI's mean and SD is a convention. It ignores ECI's anchors, so the point changes are indicative. ECI's interpretability (benchmark difficulty on the same scale as model capability) is not reproduced by my alternative scores.
9. **Differences are mostly within ECI's own uncertainty for the average losses.** MSE (cells) moves scores by about 1 point (SD) against a 2.5-point half-width; only the worst-benchmark loss produces changes well outside it. I did not bootstrap uncertainty for the alternative-loss scores.
10. **Local optima.** Minimax and MAE are non-convex or non-smooth; solutions start at ECI and use warm-started smoothing, so other optima were not explored.
11. **Evaluation set.** The held-out comparison excludes the official ECI (it saw those cells), covers benchmarks with $\ge8$ test cells, and uses 5 folds once.

### General caveats for the whole analysis
- **Sparsity and selection.** Only 37.5% of cells are observed and missingness is not random (strong models are tested on hard benchmarks). Methods that ignore this (fixed-mean PCA) are biased; AJ addresses the intercept but not all selection.
- **Subset.** Conclusions concern 29 benchmarks and 191 models; ECI uses more (including the 9 older benchmarks and benchmarks with $<25$ models). Some of the ~1% gap between ECI and PC1 may come from those.
- **In-sample vs held-out.** Higher-rank fits always look better in-sample; held-out numbers are noisy under sparsity.
- **Uncertainty.** Confidence intervals treat models as independent and ignore estimation uncertainty in the components; model families are correlated, so the true uncertainty is wider.
- **Component labels** ("agentic", "math", …) are my interpretation of loadings.
- **Scale.** Main analyses are logit; spec E and the first pass use raw scores; the raw-score versions of the later analyses were not run.
- **Superseded results.** `spec_pca.py`, `run_pca.py`, `residual_pca.py` produced conclusions that depended on fixed-mean centering (A) or on the ECI as control; `RESULTS.md` lists which were retracted.
