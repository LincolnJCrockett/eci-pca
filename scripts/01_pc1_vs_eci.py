"""Points 1-2: how much of the benchmark data does the first principal component explain, and how close is it to the ECI?

Usage: python -I scripts/01_pc1_vs_eci.py <extracted_dir> <out_dir>
Compares two ways of centering before PCA:
  A  = conventional (subtract each benchmark's observed mean)
  AJ = benchmark intercepts estimated together with the factor (removes selection bias from who was tested)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *

ex, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
D = load(ex); Xv, obs, eci, bench = D.Xv, D.obs, D.eci, D.bench; n, m = Xv.shape; N = int(obs.sum())
print(f"{n} models x {m} benchmarks, {N} scores ({obs.mean():.1%} of cells)")

# ---- variance explained (within-benchmark variation of logit scores) ----
cm = np.where(obs, Xv, 0).sum(0) / obs.sum(0)                                       # observed benchmark means
sd = np.sqrt(np.where(obs, (Xv - cm) ** 2, 0).sum(0) / obs.sum(0))
r2_eq = lambda P: 1 - (((Xv - P) / sd) ** 2)[obs].sum() / N                         # each benchmark standardized
r2_pool = lambda P: 1 - ((Xv - P) ** 2)[obs].sum() / ((Xv - cm) ** 2)[obs].sum()   # raw logit units
rows = []
for k in (1, 2, 3):
    P = fit_lowrank(Xv, k, "fixed")["pred"]; rows.append((f"PCA rank {k}, conventional (A)", r2_eq(P), r2_pool(P)))
for k in (1, 2, 3):
    P = fit_lowrank(Xv, k, "joint")["pred"]; rows.append((f"PCA rank {k}, joint intercepts (AJ)", r2_eq(P), r2_pool(P)))
P = ols_on_eci(Xv, eci); rows.append(("official ECI as the single score", r2_eq(P), r2_pool(P)))
V = pd.DataFrame(rows, columns=["model", "R2_in_sample_equal_weight", "R2_in_sample_pooled"]).set_index("model")

ks = [0, 1, 2, 3, 4]                                                                  # held-out R2 relative to the benchmark-mean model
cvA = cv_mse(Xv, lambda Xt, k: fit_lowrank(Xt, k, "fixed")["pred"], ks); cvAJ = cv_mse(Xv, lambda Xt, k: fit_lowrank(Xt, k, "joint")["pred"], ks)
H = pd.DataFrame({"A": {k: 1 - cvA[k] / cvA[0] for k in ks}, "AJ": {k: 1 - cvAJ[k] / cvA[0] for k in ks}}); H.index.name = "k"
print("\nIn-sample share of within-benchmark variance explained:"); print(V.round(3).to_string())
print("\nHeld-out R2 vs the benchmark-mean model (5-fold CV x 2):"); print(H.round(3).to_string())
V.round(4).to_csv(f"{out}/variance_explained_in_sample.csv"); H.round(4).to_csv(f"{out}/variance_explained_held_out.csv")

# ---- ECI vs PC1 ----
rows = []; pc1 = {}
for lab, base in (("A", "fixed"), ("AJ", "joint")):
    s = fit_lowrank(Xv, 1, base)["scores"][:, 0]; s = s if np.corrcoef(s, eci)[0, 1] > 0 else -s; pc1[lab] = s
    r, lo, hi, rho = corr(s, eci); rows.append((lab, r, lo, hi, rho, r ** 2))
C = pd.DataFrame(rows, columns=["spec", "pearson_r", "ci_low", "ci_high", "spearman", "r_squared"]).set_index("spec")
print("\nECI vs first principal component:"); print(C.round(3).to_string()); C.round(4).to_csv(f"{out}/eci_vs_pc1.csv")
pd.DataFrame({"eci": eci, "pc1_A": pc1["A"], "pc1_AJ": pc1["AJ"]}, index=D.M.index).to_csv(f"{out}/pc1_scores.csv")

# ---- why AJ: observed benchmark means are biased by who was tested ----
z = (eci - eci.mean()) / eci.std(); implied = fit_lowrank(Xv, 1, "joint")["pred"].mean(0)
S = pd.DataFrame({"observed_mean": cm, "model_implied_mean_all_models": implied, "mean_std_ECI_of_tested_models": [z[obs[:, j]].mean() for j in range(m)]}, index=bench)
S["bias"] = S.observed_mean - S.model_implied_mean_all_models
print(f"\ncorrelation of bias with strength of tested models: {np.corrcoef(S.bias, S.mean_std_ECI_of_tested_models)[0, 1]:.2f}")
S.round(3).to_csv(f"{out}/selection_bias.csv")

# ---- per benchmark: share of variation across models explained by ECI vs by rank-1 PCA (AJ) ----
sst = np.where(obs, (Xv - cm) ** 2, 0).sum(0); r2 = lambda P: 1 - np.where(obs, (Xv - P) ** 2, 0).sum(0) / sst
B = pd.DataFrame({"n_models": obs.sum(0), "R2_ECI": r2(ols_on_eci(Xv, eci)), "R2_PCA_rank1_AJ": r2(fit_lowrank(Xv, 1, "joint")["pred"])}, index=bench).sort_values("R2_ECI", ascending=False)
print("\nShare of each benchmark's variation across models explained:"); print(B.round(2).to_string()); B.round(4).to_csv(f"{out}/per_benchmark_explained.csv")
