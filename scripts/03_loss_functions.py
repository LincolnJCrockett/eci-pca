"""Point 4: change the loss function used to build a one-dimensional score and see how the scores change.

Usage: python -I scripts/03_loss_functions.py <extracted_dir> <out_dir>
Each solution is one score per model with a free intercept and slope per benchmark (logit scale), initialised at the official ECI.
Losses: mean squared error over cells, mean squared error averaged over benchmarks, absolute-value error averaged over benchmarks,
and the worst benchmark (p-norm of 1 - R2_b with p -> infinity).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *

ex, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
D = load(ex); Xv, obs, eci, bench = D.Xv, D.obs, D.eci, D.bench; n, m = Xv.shape
z = (eci - eci.mean()) / eci.std(); data = prep(Xv); rk_eci = pd.Series(eci).rank().to_numpy()
print(f"{n} models x {m} benchmarks; mean ECI 95% half-width = {D.eci_halfwidth:.1f} points")

PS = [1, 2, 4, 8, 16, 32, 64]; path = worst_benchmark_path(Xv, PS, z)
sol = {"Official ECI": z, "MSE (cells)": cell_loss_solve(Xv, "mse_cells", z), "MSE (benchmark avg)": cell_loss_solve(Xv, "mse_bench", z),
       "MAE (benchmark avg)": cell_loss_solve(Xv, "mae_bench", z), "Worst benchmark (p=8)": path[8], "Worst benchmark (p=64)": path[64]}
print(f"sanity: worst-benchmark p=1 vs MSE (benchmark avg), correlation of scores = {np.corrcoef(path[1], sol['MSE (benchmark avg)'])[0, 1]:.4f}  (should be ~1)")


def summarize(name, th):
    th = th if np.corrcoef(th, eci)[0, 1] > 0 else -th; r2 = r2_by_benchmark(th, data)
    pts = (eci.mean() + eci.std() * th) - eci                                           # score change in ECI points (affine map to ECI's mean and SD)
    shift = np.abs(pd.Series(th).rank().to_numpy() - rk_eci)
    return th, r2, dict(solution=name, mean_R2=r2.mean(), worst_R2=r2.min(), worst_benchmark=bench[int(r2.argmin())], corr_with_ECI=np.corrcoef(th, eci)[0, 1],
                        sd_change_points=pts.std(), max_abs_change_points=np.abs(pts).max(), mean_rank_shift_pct=100 * shift.mean() / n, max_rank_shift=int(shift.max()))


rows, R2, SC = [], {}, {}
for k, th in sol.items():
    th, r2, row = summarize(k, th); rows.append(row); R2[k] = r2; SC[k] = (eci.mean() + eci.std() * th) - eci
T = pd.DataFrame(rows).set_index("solution"); print("\nIN-SAMPLE (R2_b = share of benchmark b's variation across models explained by the score)"); print(T.round(3).to_string())
T.round(4).to_csv(f"{out}/loss_summary.csv"); pd.DataFrame(R2, index=bench).round(4).to_csv(f"{out}/loss_benchmark_r2.csv"); pd.DataFrame(SC, index=D.M.index).round(3).to_csv(f"{out}/loss_score_change_points.csv")
pd.DataFrame([summarize(f"p={p}", path[p])[2] for p in PS]).set_index("solution").round(4).to_csv(f"{out}/worst_benchmark_path.csv")
print(f"largest changes (points), worst benchmark p=64:\n{pd.concat([pd.Series(SC['Worst benchmark (p=64)'], index=D.M.index).nlargest(3), pd.Series(SC['Worst benchmark (p=64)'], index=D.M.index).nsmallest(3)]).round(1).to_string()}")

# ---- scores over time: frontier (running maximum by release date) and date trend for each measure ----
yrs = ((D.dates - D.dates.min()).dt.days / 365.25).to_numpy(); order = np.argsort(D.dates.to_numpy(), kind="stable"); fr_rows = []; official_set = None
enough = obs.sum(1) >= 8                         # robustness: frontier among models with at least 8 scores (few-score models move most)
for k in sol:
    score = eci + SC[k]; best = -np.inf; rec = []
    for i in order:
        if score[i] > best: best = score[i]; rec.append(i)
    best8, rec8 = -np.inf, []
    for i in order:
        if enough[i] and score[i] > best8: best8 = score[i]; rec8.append(i)
    names_f = [D.M.index[i] for i in rec]; official_set = set(names_f) if k == "Official ECI" else official_set
    fr_rows.append(dict(solution=k, n_frontier_models=len(rec), jaccard_with_official_frontier=len(set(names_f) & official_set) / len(set(names_f) | official_set),
                        frontier_slope_points_per_year=np.polyfit(yrs[rec], score[rec], 1)[0], frontier_slope_models_8plus_scores=np.polyfit(yrs[rec8], score[rec8], 1)[0], all_models_slope_points_per_year=np.polyfit(yrs, score, 1)[0],
                        corr_of_change_with_release_date=np.corrcoef(yrs, SC[k])[0, 1] if k != "Official ECI" else np.nan, latest_frontier_model=names_f[-1], frontier_models="; ".join(names_f)))
FR = pd.DataFrame(fr_rows).set_index("solution"); FR.drop(columns="frontier_models").round(3).to_csv(f"{out}/loss_over_time.csv"); FR[["frontier_models"]].to_csv(f"{out}/loss_frontier_models.csv")
print("\nOVER TIME (frontier = models that set a new record at their release date)"); print(FR.drop(columns="frontier_models").round(3).to_string())

# ---- held-out: refit on 80% of cells, predict the rest with an OLS line per benchmark ----
rng = np.random.default_rng(0); idxs = np.argwhere(obs); folds = np.array_split(idxs[rng.permutation(len(idxs))], 5)
names = ["MSE (cells)", "MSE (benchmark avg)", "MAE (benchmark avg)", "Worst benchmark (p=8)", "Worst benchmark (p=64)"]
sse = {k: np.zeros(m) for k in names}; sae = {k: np.zeros(m) for k in names}; sst = np.zeros(m); cnt = np.zeros(m)
for part in folds:
    Xt = Xv.copy(); Xt[part[:, 0], part[:, 1]] = np.nan
    ro = (~np.isnan(Xt)).sum(1); co = (~np.isnan(Xt)).sum(0); pt = part[(ro[part[:, 0]] >= 2) & (co[part[:, 1]] >= 8)]
    pp = worst_benchmark_path(Xt, [1, 8, 64], z)
    fold_sol = {"MSE (cells)": cell_loss_solve(Xt, "mse_cells", z), "MSE (benchmark avg)": cell_loss_solve(Xt, "mse_bench", z), "MAE (benchmark avg)": cell_loss_solve(Xt, "mae_bench", z),
                "Worst benchmark (p=8)": pp[8], "Worst benchmark (p=64)": pp[64]}
    mu = np.nanmean(Xt, 0); sdt = np.nanstd(Xt, 0)
    for k, t in fold_sol.items():
        for j in range(m):
            tj = pt[pt[:, 1] == j][:, 0]
            if not len(tj): continue
            o = ~np.isnan(Xt[:, j]); err = Xv[tj, j] - np.polyval(np.polyfit(t[o], Xt[o, j], 1), t[tj]); sse[k][j] += (err ** 2).sum(); sae[k][j] += np.abs(err).sum() / sdt[j]
    for j in range(m):
        tj = pt[pt[:, 1] == j][:, 0]; sst[j] += ((Xv[tj, j] - mu[j]) ** 2).sum(); cnt[j] += len(tj)
ok = cnt >= 8; hr = []
for k in names:
    h = 1 - sse[k][ok] / sst[ok]
    hr.append(dict(solution=k, mean_R2=h.mean(), worst_R2=h.min(), worst_benchmark=np.array(bench)[ok][h.argmin()], mean_abs_error_in_sd_units=(sae[k][ok] / cnt[ok]).mean()))
H = pd.DataFrame(hr).set_index("solution"); print("\nHELD-OUT (benchmarks with >= 8 test cells)"); print(H.round(3).to_string()); H.round(4).to_csv(f"{out}/loss_held_out.csv")
