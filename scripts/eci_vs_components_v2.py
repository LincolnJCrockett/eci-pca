"""Redo the 'what does ECI over/under-value' analysis with PC1 estimated with joint intercepts (AJ), and with ECI-residual components (E).
Usage: python -I scripts/eci_vs_components_v2.py <extracted_dir> <out_dir>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *

ex, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
Dd = load(ex); M, Xv, Mv, obs, eci, date, floor = Dd.M, Dd.Xv, Dd.Mv, Dd.obs, Dd.eci, Dd.date, Dd.floor
n, m = Xv.shape; bench = list(M.columns)

f1 = fit(Xv, "AJ", 1); pc1 = f1["scores"][:, 0]; pc1 = pc1 if np.corrcoef(pc1, eci)[0, 1] > 0 else -pc1
b = np.polyfit(pc1, eci, 1); rs = eci - np.polyval(b, pc1)
print(f"ECI vs AJ rank-1 PC1: r={np.corrcoef(pc1, eci)[0, 1]:.3f}; ECI variance not in PC1: {rs.var() / eci.var():.3%}")
a_fe, b_fe = fe(np.where(obs, Xv, 0), obs); nobs = obs.sum(1)
easy = np.array([b_fe[obs[i]].mean() for i in range(n)])
print("ECI residual (after AJ PC1) correlates with:")
for nm, v in (("# benchmarks evaluated", nobs.astype(float)), ("mean easiness of benchmarks taken", easy), ("release date (yrs)", date)):
    r = corr(v, rs); print(f"   {nm:36s} r={r[0]:+.2f} [{r[2]:+.2f},{r[3]:+.2f}]")
pr = []
for j, bn in enumerate(bench):
    o = obs[:, j]
    if o.sum() < 15: continue
    sres = Xv[o, j] - np.polyval(np.polyfit(pc1[o], Xv[o, j], 1), pc1[o]); pr.append((bn, np.corrcoef(sres, rs[o])[0, 1], int(o.sum())))
pr = pd.DataFrame(pr, columns=["benchmark", "partial_r", "n"]).sort_values("partial_r"); pr.round(3).to_csv(f"{out}/q1_partial_corr_AJ.csv", index=False)
print("benchmarks whose residual (given PC1) correlates with the ECI residual:"); print("  highest:", ", ".join(f"{r.benchmark} {r.partial_r:.2f}" for r in pr.tail(6).itertuples()))
print("  lowest: ", ", ".join(f"{r.benchmark} {r.partial_r:.2f}" for r in pr.head(6).itertuples()))

Pe = logistic_on_eci(Mv, eci, floor); fE = fit(Mv, "E", 3, base_ov=Pe); fDJ = fit(Xv, "DJ", 3, base_ov=fit(Xv, "AJ", 1)["pred"])
dres = lambda v: v - np.polyval(np.polyfit(date, v, 1), date)
rows = []
for lab, f_ in (("E (ECI removed)", fE), ("DJ (PC1 removed)", fDJ)):
    S = f_["scores"].copy(); L = f_["load"].copy()
    for j in range(3):
        sg = 1 if L[np.abs(L[:, j]).argmax(), j] > 0 else -1; S[:, j] *= sg
    for j in range(3):
        r = corr(S[:, j], eci); r2 = corr(S[:, j], rs); r3 = corr(dres(S[:, j]), dres(rs)); rd = corr(S[:, j], date)
        rows.append((lab, f"comp{j+1}", r[0], r[2], r[3], r2[0], r2[2], r2[3], r3[0], r3[2], r3[3], rd[0]))
q = pd.DataFrame(rows, columns=["spec", "comp", "r_ECI", "lo", "hi", "r_ECIresid", "lo2", "hi2", "r_ECIresid_dateadj", "lo3", "hi3", "r_date"]); q.round(3).to_csv(f"{out}/q2_components_v2.csv", index=False)
print("\ncorrelation of model scores on each component with ECI, with ECI-after-PC1, and with ECI-after-PC1 net of release date"); print(q.round(2).to_string(index=False))
