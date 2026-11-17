"""Points 1-2: how much of each benchmark does the ECI explain, how much does each principal component explain, and how close is PC1 to the ECI?

Usage: python -I scripts/01_pc1_vs_eci.py <extracted_dir> <out_dir>
PCA: benchmark intercepts estimated together with the components, each benchmark scaled by its observed SD (common.fit_lowrank, "joint").
Fit is measured on logit scores: a straight line per benchmark in the score(s) (common.logit_r2).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *

ex, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
D = load(ex); Xv, Mv, obs, eci, bench = D.Xv, D.Mv, D.obs, D.eci, D.bench; n, m = Xv.shape; N = int(obs.sum())
print(f"{n} models x {m} benchmarks, {N} scores ({obs.mean():.1%} of cells)")

# ---- variation explained by each principal component (logit scores) ----
K = 5; tot = {0: 0.0}
for k in range(1, K + 1): tot[k] = logit_r2(fit_lowrank(Xv, k, "joint")["scores"], Xv)[1]
eci_r2 = logit_r2(eci, Xv)[1]
V = pd.DataFrame({"explained_by_this_component": {k: tot[k] - tot[k - 1] for k in range(1, K + 1)}, "cumulative": {k: tot[k] for k in range(1, K + 1)}}); V.index.name = "component"
print(f"\nShare of within-benchmark variation of logit scores explained (each score counts once; official ECI as one score: {eci_r2:.3f}):"); print(V.round(3).to_string())
V.round(4).to_csv(f"{out}/variance_by_component.csv"); pd.Series({"official_ECI_single_score": eci_r2}).round(4).to_csv(f"{out}/eci_variance_explained.csv", header=["R2"])

# ---- ECI vs PC1 ----
rows = []; pc1 = {}
for lab, base in (("joint intercepts", "joint"), ("conventional centering (for comparison)", "fixed")):
    s = fit_lowrank(Xv, 1, base)["scores"][:, 0]; s = s if np.corrcoef(s, eci)[0, 1] > 0 else -s; pc1[lab] = s
    r, lo, hi, rho = corr(s, eci); rows.append((lab, r, lo, hi, rho, r ** 2))
C = pd.DataFrame(rows, columns=["spec", "pearson_r", "ci_low", "ci_high", "spearman", "r_squared"]).set_index("spec")
print("\nECI vs first principal component:"); print(C.round(3).to_string()); C.round(4).to_csv(f"{out}/eci_vs_pc1.csv")
pd.DataFrame({"eci": eci, "pc1": pc1["joint intercepts"]}, index=D.M.index).to_csv(f"{out}/pc1_scores.csv")

# ---- why joint intercepts: observed benchmark means are biased by who was tested ----
cm = np.where(obs, Xv, 0).sum(0) / obs.sum(0); z = (eci - eci.mean()) / eci.std(); implied = fit_lowrank(Xv, 1, "joint")["pred"].mean(0)
S = pd.DataFrame({"observed_mean": cm, "model_implied_mean_all_models": implied, "mean_std_ECI_of_tested_models": [z[obs[:, j]].mean() for j in range(m)]}, index=bench)
S["bias"] = S.observed_mean - S.model_implied_mean_all_models
print(f"\ncorrelation of bias with strength of tested models: {np.corrcoef(S.bias, S.mean_std_ECI_of_tested_models)[0, 1]:.2f}")
S.round(3).to_csv(f"{out}/selection_bias.csv")

# ---- per benchmark: the noise level Epoch's Bayesian ECI assigns ----
# Epoch's model: score ~ Beta with mean mu = sigmoid(slope_b (ECI_m - EDI_b)) and sd 2 sigma_b sqrt(mu (1 - mu)).
# Epoch publishes ECI, EDI (difficulty) and slope but not sigma_b, so sigma_b is recovered as the RMS of the standardized residual
# (y - mu) / (2 sqrt(mu (1 - mu))) over all of Epoch's scores on b (best score per model, clipped to [0.001, 0.999] as Epoch does).
dd = D.raw; Eall = pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv").set_index("Model").eci
ed = pd.read_csv(f"{ex}/epoch_capabilities_index/edi_scores.csv").set_index("benchmark_name")
dd = dd[dd.model.isin(Eall.index) & dd.benchmark.isin(ed.index)].groupby(["model", "benchmark"]).performance.max().reset_index()
yb = dd.performance.clip(.001, .999); mu = 1 / (1 + np.exp(-ed.estimated_slope_scaled.reindex(dd.benchmark).values * (Eall.reindex(dd.model).values - ed.edi.reindex(dd.benchmark).values)))
sigma_b = ((yb - mu) / (2 * np.sqrt(mu * (1 - mu)))).groupby(dd.benchmark).apply(lambda v: np.sqrt((v ** 2).mean()))
print(f"\ncheck vs Epoch's post: a predicted 50% typically lands within 50 +/- {100*sigma_b['METR Time Horizons']:.0f}% on METR (Epoch: 46-54%), 50 +/- {100*sigma_b['VPCT']:.0f}% on VPCT (Epoch: 32-68%)")

# ---- per benchmark: share of variation across models explained by the ECI and by PC1 ----
B = pd.DataFrame({"n_models": obs.sum(0), "R2_ECI": logit_r2(eci, Xv)[0], "R2_PC1": logit_r2(pc1["joint intercepts"], Xv)[0],
                  "epoch_noise_sigma_b": sigma_b.reindex(bench).values}, index=bench).sort_values("R2_ECI", ascending=False)
B["noise_rank_of_29"] = B.epoch_noise_sigma_b.rank(ascending=False).astype(int)
print(f"\nSpearman(R2_ECI, Epoch noise sigma_b) across the 29 benchmarks: {B.R2_ECI.corr(B.epoch_noise_sigma_b, method='spearman'):.2f}")
print("\nShare of each benchmark's variation across models explained (logit scores):"); print(B.round(2).to_string()); B.round(4).to_csv(f"{out}/per_benchmark_explained.csv")
