"""Point 4: how much do the scores change if the one-number index cares more, or less, about the worst-fitted benchmark?

Usage: python -I scripts/03_loss_functions.py <extracted_dir> <out_dir>
One score per model with a free intercept and slope per benchmark (logit scale). The loss is a weighted power mean over benchmarks of
u_b = 1 - R2_b (weights = number of models on each benchmark; common.pnorm_obj):
  p = 1   arithmetic mean  - every score counts once; this is exactly the first principal component of point 2
  p = inf the worst-fitted benchmark only (solved as p = 64, reached by warm starts through p = 2, 4, ..., 32)
Fit is measured on logit scores, like points 1-3.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *

ex, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
D = load(ex); Xv, Mv, obs, eci, bench = D.Xv, D.Mv, D.obs, D.eci, D.bench; n, m = Xv.shape
z = (eci - eci.mean()) / eci.std(); rk_eci = pd.Series(eci).rank().to_numpy()
print(f"{n} models x {m} benchmarks; mean ECI 90% half-width = {D.eci_halfwidth:.1f} points")

PS = [1, 2, 4, 8, 16, 32, 64]; path = pnorm_path(Xv, PS, z)
NAMES = {1: "p=1 (average = first PC)", 64: "p=inf (worst benchmark)"}
sol = {"Official ECI": z, **{NAMES[p]: path[p] for p in (1, 64)}}
pc1 = fit_lowrank(Xv, 1, "joint")["scores"][:, 0]
print(f"sanity: p=1 solution vs first principal component, correlation = {abs(np.corrcoef(path[1], pc1)[0, 1]):.4f}  (should be ~1)")


def summarize(name, th):
    th = th if np.corrcoef(th, eci)[0, 1] > 0 else -th; r2, r2_all = logit_r2(th, Xv)
    pts = (eci.mean() + eci.std() * th) - eci                                           # score change in ECI points (affine map to ECI's mean and SD)
    shift = np.abs(pd.Series(th).rank().to_numpy() - rk_eci)
    return th, r2, dict(solution=name, mean_R2=r2_all, worst_R2=r2.min(), worst_benchmark=bench[int(r2.argmin())], corr_with_ECI=np.corrcoef(th, eci)[0, 1],
                        sd_change_points=pts.std(), max_abs_change_points=np.abs(pts).max(), mean_rank_shift_pct=100 * shift.mean() / n, max_rank_shift=int(shift.max()))


rows, R2, SC = [], {}, {}
for k, th in sol.items():
    th, r2, row = summarize(k, th); rows.append(row); R2[k] = r2; SC[k] = (eci.mean() + eci.std() * th) - eci
T = pd.DataFrame(rows).set_index("solution")
print("\nR2_b = share of benchmark b's variation across models explained (logit scores); mean_R2 weights benchmarks by number of models"); print(T.round(3).to_string())
T.round(4).to_csv(f"{out}/loss_summary.csv"); pd.DataFrame(R2, index=bench).round(4).to_csv(f"{out}/loss_benchmark_r2.csv"); pd.DataFrame(SC, index=D.M.index).round(3).to_csv(f"{out}/loss_score_change_points.csv")
pd.DataFrame([summarize(f"p={p}", path[p])[2] for p in PS]).set_index("solution").round(4).to_csv(f"{out}/pnorm_path.csv")
w = NAMES[64]; print(f"largest changes (points), {w}:\n{pd.concat([pd.Series(SC[w], index=D.M.index).nlargest(3), pd.Series(SC[w], index=D.M.index).nsmallest(3)]).round(1).to_string()}")

# ---- scores over time: frontier (running maximum by release date), one line and a two-segment line broken at Apr 2024 ----
yrs = ((D.dates - D.dates.min()).dt.days / 365.25).to_numpy(); t_break = (BREAK_DATE - D.dates.min()).days / 365.25; fr_rows = []
enough = obs.sum(1) >= 8                         # robustness: frontier among models with at least 8 scores (few-score models move most)
official_set = None
for k in sol:
    score = eci + SC[k]; rec = frontier_idx(score, D.dates.to_numpy()); rec8 = frontier_idx(score, D.dates.to_numpy(), enough)
    names_f = [D.M.index[i] for i in rec]; official_set = set(names_f) if k == "Official ECI" else official_set
    pre, post, _ = hinge_fit(yrs[rec], score[rec], t_break); _, post8, _ = hinge_fit(yrs[rec8], score[rec8], t_break)
    fr_rows.append(dict(solution=k, n_frontier_models=len(rec), jaccard_with_official_frontier=len(set(names_f) & official_set) / len(set(names_f) | official_set),
                        frontier_slope_points_per_year=np.polyfit(yrs[rec], score[rec], 1)[0], frontier_slope_models_8plus_scores=np.polyfit(yrs[rec8], score[rec8], 1)[0],
                        n_frontier_before_break=int((yrs[rec] < t_break).sum()), slope_before_break=pre, slope_after_break=post, slope_after_break_8plus=post8,
                        corr_of_change_with_release_date=np.corrcoef(yrs, SC[k])[0, 1] if k != "Official ECI" else np.nan, latest_frontier_model=names_f[-1], frontier_models="; ".join(names_f)))
# reference: official ECI over ALL models Epoch scores, so the pre-break period is populated
EA = pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv").dropna(subset=["eci", "date"]); da = pd.to_datetime(EA.date); ya = ((da - D.dates.min()).dt.days / 365.25).to_numpy()
ra = frontier_idx(EA.eci.to_numpy(), da.to_numpy()); pa, qa, _ = hinge_fit(ya[ra], EA.eci.to_numpy()[ra], t_break)
fr_rows.append(dict(solution="Official ECI, all Epoch models (reference)", n_frontier_models=len(ra), frontier_slope_points_per_year=np.polyfit(ya[ra], EA.eci.to_numpy()[ra], 1)[0],
                    n_frontier_before_break=int((ya[ra] < t_break).sum()), slope_before_break=pa, slope_after_break=qa, latest_frontier_model=EA.Model.iloc[ra[-1]], frontier_models="; ".join(EA.Model.iloc[ra])))
FR = pd.DataFrame(fr_rows).set_index("solution"); FR.drop(columns="frontier_models").round(3).to_csv(f"{out}/loss_over_time.csv"); FR[["frontier_models"]].to_csv(f"{out}/loss_frontier_models.csv")
print("\nOVER TIME (frontier = models that set a new record at their release date)"); print(FR.drop(columns="frontier_models").round(3).to_string())

# ---- who gains under p = inf? mean score change and share of record-setting models, by developer ----
org = pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv").set_index("Model").Organization.reindex(D.M.index)
big = org.value_counts()[lambda c: c >= 5].index; rows = []
for k in sol:
    rec = set(FR.loc[k, "frontier_models"].split("; "))
    for o in big:
        mo = (org == o).to_numpy(); rows.append(dict(solution=k, developer=o, n_models=int(mo.sum()), mean_change_points=SC[k][mo].mean(),
                                                    frontier_models=sum(1 for mname in D.M.index[mo] if mname in rec), frontier_share=sum(1 for mname in D.M.index[mo] if mname in rec) / len(rec)))
DV = pd.DataFrame(rows); DV.round(3).to_csv(f"{out}/loss_by_developer.csv", index=False)
print("\nBY DEVELOPER (developers with >= 5 models)"); print(DV.pivot(index="developer", columns="solution", values="frontier_models").to_string())
print(DV[DV.solution == NAMES[64]].set_index("developer")[["n_models", "mean_change_points"]].sort_values("mean_change_points").round(2).to_string())
