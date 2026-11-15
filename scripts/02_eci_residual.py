"""Point 3: remove the variation the ECI explains (spec E). What is left, and is it structured?

Usage: python -I scripts/02_eci_residual.py <extracted_dir> <out_dir>
For every benchmark, regress the logit score on the official ECI (free intercept and slope). Residuals are what ECI does not explain.
Then run low-rank PCA on the residuals, and test whether the structure is real (shuffled null, held-out prediction, bootstrap stability).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *

ex, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
D = load(ex); Xv, obs, eci, bench = D.Xv, D.obs, D.eci, D.bench; n, m = Xv.shape; N = int(obs.sum())
P = ols_on_eci(Xv, eci); R0 = np.where(obs, Xv - P, np.nan)                         # what ECI does not explain

# ---- how much does ECI explain, benchmark by benchmark ----
cm = np.where(obs, Xv, 0).sum(0) / obs.sum(0); sst = np.where(obs, (Xv - cm) ** 2, 0).sum(0); sse = np.where(obs, (Xv - P) ** 2, 0).sum(0)
sd = np.sqrt(sst / obs.sum(0))
B = pd.DataFrame({"n_models": obs.sum(0), "explained_by_ECI": 1 - sse / sst}, index=bench); B["unexplained"] = 1 - B.explained_by_ECI; B = B.sort_values("unexplained", ascending=False)
print(f"ECI explains {1 - (sse / sd ** 2).sum() / N:.1%} (benchmarks weighted equally) / {1 - sse.sum() / sst.sum():.1%} (pooled) of within-benchmark variation")
print("\nLeast explained benchmarks first:"); print(B.round(2).to_string()); B.round(4).to_csv(f"{out}/eci_explained_by_benchmark.csv")

# ---- low-rank structure in the residuals ----
f = fit_lowrank(Xv, 3, base=P); L, S = f["load"].copy(), f["scores"].copy()
for j in range(3):
    sg = 1 if L[np.abs(L[:, j]).argmax(), j] > 0 else -1; L[:, j] *= sg; S[:, j] *= sg    # sign: largest loading positive
Dres = np.where(obs, Xv - P, 0); sres = np.sqrt((Dres ** 2).sum(0) / obs.sum(0))
share = {k: 1 - (((Dres - fit_lowrank(Xv, k, base=P)["R"]) / sres) ** 2)[obs].sum() / N for k in (1, 2, 3)}
print("\nIn-sample share of the residual variation captured by k components (optimistic):", {k: round(v, 3) for k, v in share.items()})
pd.DataFrame(L, index=bench, columns=["E1", "E2", "E3"]).assign(n_models=obs.sum(0)).round(3).to_csv(f"{out}/residual_loadings.csv")
pd.DataFrame(S, index=D.M.index, columns=["E1", "E2", "E3"]).round(4).to_csv(f"{out}/residual_scores.csv")
for j in range(3):
    o = np.argsort(L[:, j]); print(f"E{j+1} +: " + ", ".join(f"{bench[i]} {L[i,j]:.2f}" for i in o[::-1][:6]) + "\n    -: " + ", ".join(f"{bench[i]} {L[i,j]:.2f}" for i in o[:6]))

# ---- is the structure real? (1) eigenvalues vs shuffled null ----
w = pairwise_eigs(R0); rng = np.random.default_rng(1); nulls = []
for _ in range(100):
    Rp = R0.copy()
    for j in range(m): o = obs[:, j]; Rp[o, j] = rng.permutation(R0[o, j])         # shuffle residuals within each benchmark
    nulls.append(pairwise_eigs(Rp))
n95 = np.percentile(nulls, 95, axis=0)
pd.DataFrame({"eigenvalue": w[:10], "null_95th_pct": n95[:10]}, index=range(1, 11)).round(4).to_csv(f"{out}/residual_eigenvalues.csv")
print("\nEigenvalues of the residual correlation matrix vs shuffled null:", [(round(a, 2), round(b, 2)) for a, b in zip(w[:5], n95[:5])])

# ---- (2) does adding components improve held-out prediction? ----
ks = list(range(0, 7)); cv = cv_mse(Xv, lambda Xt, k: fit_lowrank(Xt, k, base=ols_on_eci(Xt, eci))["pred"], ks)
CV = pd.DataFrame({"held_out_MSE": cv, "change_vs_ECI_only": {k: 1 - cv[k] / cv[0] for k in ks}}); CV.index.name = "k"
print("\nHeld-out MSE with k residual components added to the ECI prediction:"); print(CV.round(4).to_string()); CV.round(5).to_csv(f"{out}/residual_held_out.csv")

# ---- (3) bootstrap stability of the loadings ----
rngb = np.random.default_rng(2); st = []
for _ in range(60):
    ib = rngb.choice(n, n); Xb = Xv[ib]
    if np.isnan(Xb).all(0).any(): continue
    Lb = fit_lowrank(Xb, 3, base=ols_on_eci(Xb, eci[ib]))["load"]; st.append(np.abs(L.T @ Lb).max(1))
ST = pd.DataFrame({"mean_best_match_cosine": np.mean(st, 0), "p10_best_match_cosine": np.percentile(st, 10, 0)}, index=["E1", "E2", "E3"])
print("\nBootstrap stability:"); print(ST.round(2).to_string()); ST.round(4).to_csv(f"{out}/residual_stability.csv")
