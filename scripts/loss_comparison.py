"""Compare one-dimensional model scores obtained under different loss functions with the official ECI.
Losses: MSE over cells, MSE averaged over benchmarks, absolute-value loss (MAE) averaged over benchmarks,
        worst-benchmark loss (p-norm of 1 - R2_b, p -> infinity; p = 1 is shown as a sanity check).
Usage: python -I scripts/loss_comparison.py <extracted_dir> <out_dir>
Supersedes scripts/minimax_eci.py (same p-path, now in common.py).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *

ex, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
Dd = load(ex); Xv, M, eci = Dd.Xv, Dd.M, Dd.eci; n, m = Xv.shape; bench = list(M.columns); obs = ~np.isnan(Xv)
Ecsv = pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv").set_index("Model").reindex(M.index)
halfw = float(((Ecsv.eci_ci_high - Ecsv.eci_ci_low) / 2).mean())
z = (eci - eci.mean()) / eci.std(); data = prep(Xv)
print(f"sample {n} models x {m} benchmarks; mean ECI 95% half-width = {halfw:.1f} points")

PS = [1, 2, 4, 8, 16, 32, 64]
path = pnorm_solve(Xv, PS, z)
sol = {"official ECI": z, "MSE (cells)": joint_solve(Xv, "mse_cells", z), "MSE (benchmark avg)": joint_solve(Xv, "mse_bench", z),
       "MAE (benchmark avg)": joint_solve(Xv, "mae_bench", z), "worst-benchmark p=1": path[1], "worst-benchmark p=8": path[8], "worst-benchmark p=64": path[64]}
rk0 = pd.Series(eci).rank().to_numpy(); rows = []; prof = {}
for name, th in sol.items():
    th = th if np.corrcoef(th, eci)[0, 1] > 0 else -th; sol[name] = th; r2 = profile(th, data); prof[name] = r2
    rk1 = pd.Series(th).rank().to_numpy(); d = (eci.mean() + eci.std() * th) - eci
    rows.append(dict(solution=name, mean_R2=r2.mean(), min_R2=r2.min(), worst_benchmark=bench[int(r2.argmin())], corr_ECI=np.corrcoef(th, eci)[0, 1],
                     mean_abs_rank_shift_pct=100 * np.mean(np.abs(rk1 - rk0)) / n, max_rank_shift=int(np.abs(rk1 - rk0).max()), sd_change_pts=d.std(), max_abs_change_pts=np.abs(d).max()))
T = pd.DataFrame(rows).set_index("solution"); print("\nIN-SAMPLE (R2_b = squared correlation of the score with benchmark b over observed models)"); print(T.round(3).to_string())
T.round(4).to_csv(f"{out}/loss_summary.csv"); pd.DataFrame(prof, index=bench).round(3).to_csv(f"{out}/loss_benchmark_R2.csv")
C = pd.DataFrame({k: v for k, v in sol.items()}, index=M.index).corr(); print("\ncorrelation between solutions:"); print(C.round(3).to_string()); C.round(4).to_csv(f"{out}/loss_solution_correlations.csv")
pd.DataFrame(sol, index=M.index).round(4).to_csv(f"{out}/loss_scores_standardized.csv")

# biggest movers, worst-benchmark solution
th = sol["worst-benchmark p=64"]; d = pd.Series((eci.mean() + eci.std() * th) - eci, index=M.index)
print("\nlargest score changes (ECI points) under worst-benchmark p=64:"); print(pd.concat([d.nlargest(4), d.nsmallest(4)]).round(1).to_string())

# held-out comparison
rng = np.random.default_rng(0); idxs = np.argwhere(obs); parts_ = np.array_split(idxs[rng.permutation(len(idxs))], 5)
names = ["MSE (cells)", "MSE (benchmark avg)", "MAE (benchmark avg)", "worst-benchmark p=1", "worst-benchmark p=8", "worst-benchmark p=64"]
sse = {k: np.zeros(m) for k in names}; sae = {k: np.zeros(m) for k in names}; sst = np.zeros(m); cnt = np.zeros(m)
for part in parts_:
    Xt = Xv.copy(); Xt[part[:, 0], part[:, 1]] = np.nan
    ro = (~np.isnan(Xt)).sum(1); co = (~np.isnan(Xt)).sum(0); pt = part[(ro[part[:, 0]] >= 2) & (co[part[:, 1]] >= 8)]
    pp = pnorm_solve(Xt, [1, 8, 64], z)
    th_f = {"MSE (cells)": joint_solve(Xt, "mse_cells", z), "MSE (benchmark avg)": joint_solve(Xt, "mse_bench", z), "MAE (benchmark avg)": joint_solve(Xt, "mae_bench", z),
            "worst-benchmark p=1": pp[1], "worst-benchmark p=8": pp[8], "worst-benchmark p=64": pp[64]}
    mu = np.nanmean(Xt, 0); sdt = np.nanstd(Xt, 0)
    for k, t in th_f.items():
        for j in range(m):
            tj = pt[pt[:, 1] == j][:, 0]
            if not len(tj): continue
            o = ~np.isnan(Xt[:, j]); b = np.polyfit(t[o], Xt[o, j], 1); err = Xv[tj, j] - np.polyval(b, t[tj])
            sse[k][j] += (err ** 2).sum(); sae[k][j] += np.abs(err).sum() / sdt[j]
    for j in range(m):
        tj = pt[pt[:, 1] == j][:, 0]; sst[j] += ((Xv[tj, j] - mu[j]) ** 2).sum(); cnt[j] += len(tj)
ok = cnt >= 8; hr = []
print("\nHELD-OUT (5-fold; scores refit on 80% of cells; benchmark line fit by OLS on training cells; benchmarks with >= 8 test cells)")
for k in names:
    h = 1 - sse[k][ok] / sst[ok]; mae = (sae[k][ok] / cnt[ok]).mean()
    hr.append(dict(solution=k, mean_R2=h.mean(), min_R2=h.min(), worst_benchmark=np.array(bench)[ok][h.argmin()], p10_R2=np.percentile(h, 10), mean_MAE_in_sd_units=mae))
H = pd.DataFrame(hr).set_index("solution"); print(H.round(3).to_string()); H.round(4).to_csv(f"{out}/loss_heldout.csv")
