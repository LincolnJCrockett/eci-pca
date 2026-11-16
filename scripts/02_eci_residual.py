"""Point 3: remove the variation the ECI explains. What is left, and is it structured?

Usage: python -I scripts/02_eci_residual.py <extracted_dir> <out_dir>
For every benchmark, regress the logit score on the official ECI (free intercept and slope). Residuals are what the ECI does not explain.
Then run the same PCA as point 2 on the residuals, and compare the share of residual variation each component captures with a
shuffled null (residuals shuffled among the models within each benchmark, then the same PCA).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *

ex, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
D = load(ex); Xv, Mv, obs, eci, bench = D.Xv, D.Mv, D.obs, D.eci, D.bench; n, m = Xv.shape; N = int(obs.sum())
P = ols_on_eci(Xv, eci); R0 = np.where(obs, Xv - P, np.nan)                         # what the ECI does not explain (logit)

# ---- how much does the ECI explain, benchmark by benchmark (logit scores) ----
r2b, r2_all = logit_r2(eci, Xv)
B = pd.DataFrame({"n_models": obs.sum(0), "explained_by_ECI": r2b}, index=bench); B["unexplained"] = 1 - B.explained_by_ECI; B = B.sort_values("unexplained", ascending=False)
print(f"ECI explains {r2_all:.1%} of within-benchmark variation of logit scores (each score counts once)")
print("\nLeast explained benchmarks first:"); print(B.round(2).to_string()); B.round(4).to_csv(f"{out}/eci_explained_by_benchmark.csv")

# ---- PCA of the residuals ----
f = fit_lowrank(Xv, 3, base=P); L, S = f["load"].copy(), f["scores"].copy()
for j in range(3):
    sg = 1 if L[np.abs(L[:, j]).argmax(), j] > 0 else -1; L[:, j] *= sg; S[:, j] *= sg    # sign: largest loading positive
pd.DataFrame(L, index=bench, columns=["E1", "E2", "E3"]).assign(n_models=obs.sum(0)).round(3).to_csv(f"{out}/residual_loadings.csv")
pd.DataFrame(S, index=D.M.index, columns=["E1", "E2", "E3"]).round(4).to_csv(f"{out}/residual_scores.csv")
for j in range(3):
    o = np.argsort(L[:, j]); print(f"E{j+1} +: " + ", ".join(f"{bench[i]} {L[i,j]:.2f}" for i in o[::-1][:6]) + "\n    -: " + ", ".join(f"{bench[i]} {L[i,j]:.2f}" for i in o[:6]))


# ---- share of residual variation captured by each component, vs shuffled null ----
K = 8
def shares(Xr):
    """share of the residual variation (each benchmark scaled to unit RMS) captured by component k = total(k) - total(k-1)"""
    Dr = np.where(obs, Xr - P, 0); s = np.sqrt((Dr ** 2).sum(0) / obs.sum(0)); tot = [0.0]
    for k in range(1, K + 1): tot.append(1 - (((Dr - fit_lowrank(Xr, k, base=P)["R"]) / s) ** 2)[obs].sum() / N)
    return np.diff(tot)
real = shares(Xv); rng = np.random.default_rng(1); nulls = []
for _ in range(100):
    Rp = R0.copy()
    for j in range(m): o = obs[:, j]; Rp[o, j] = rng.permutation(R0[o, j])         # shuffle residuals within each benchmark
    nulls.append(shares(np.where(obs, P + Rp, np.nan)))
SH = pd.DataFrame({"share_of_residual_variation": real, "null_95th_pct": np.percentile(nulls, 95, axis=0), "null_mean": np.mean(nulls, 0)}, index=range(1, K + 1)); SH.index.name = "component"
print("\nShare of residual variation captured by each component vs shuffled null:"); print(SH.round(3).to_string()); SH.round(4).to_csv(f"{out}/residual_variance_by_component.csv")

# ---- bootstrap stability of the loadings ----
rngb = np.random.default_rng(2); st = []
for _ in range(60):
    ib = rngb.choice(n, n); Xb = Xv[ib]
    if np.isnan(Xb).all(0).any(): continue
    Lb = fit_lowrank(Xb, 3, base=ols_on_eci(Xb, eci[ib]))["load"]; st.append(np.abs(L.T @ Lb).max(1))
ST = pd.DataFrame({"mean_best_match_cosine": np.mean(st, 0), "p10_best_match_cosine": np.percentile(st, 10, 0)}, index=["E1", "E2", "E3"])
print("\nBootstrap stability:"); print(ST.round(2).to_string()); ST.round(4).to_csv(f"{out}/residual_stability.csv")
