"""Figures (one per claim). Reads the CSVs written by scripts 01-03.  fig11 (held-out fit) was removed.
Usage: python -I scripts/04_make_figures.py <extracted_dir> <results_dir>      -> <results_dir>/figures/
"""
import sys, os, textwrap, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import MaxNLocator
from common import *

ex, res = sys.argv[1], sys.argv[2]; fd = f"{res}/figures"; os.makedirs(fd, exist_ok=True)
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#6250d6"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#256abf"])
plt.rcParams.update({"axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
W = lambda s, w=60: textwrap.fill(s, w)
def save(fig, name):
    fig.tight_layout()
    for attempt in range(6):                      # retry: Windows occasionally locks the file for a moment (OSError 22)
        try: fig.savefig(f"{fd}/{name}.png", dpi=150); break
        except OSError:
            if attempt == 5: raise
            time.sleep(1)
    plt.close(fig); print("wrote", name)
R1 = lambda f: pd.read_csv(f"{res}/01_pc1_vs_eci/{f}", index_col=0); R2 = lambda f: pd.read_csv(f"{res}/02_eci_residual/{f}", index_col=0); R3 = lambda f: pd.read_csv(f"{res}/03_loss_functions/{f}", index_col=0)
D = load(ex); eci = D.eci

# ---- fig01: data sparsity ----
raw = D.raw; Mall = raw.pivot_table(index="model", columns="benchmark", values="performance"); mdate = pd.to_datetime(raw.groupby("model").date.first())
big = Mall.loc[:, Mall.notna().sum() >= 25]; O = big.notna().astype(int).T @ big.notna().astype(int); med = pd.Series({b: O.loc[b].drop(b).median() for b in O.index})
legacy = [b for b in med[med < 7].index if b not in D.bench]; others = [c for c in Mall.columns if c not in legacy and c not in D.bench]
cols = legacy + sorted(D.bench, key=lambda b: -Mall[b].notna().sum()) + sorted(others, key=lambda b: -Mall[b].notna().sum())
B = Mall.loc[mdate.sort_values().index, cols].notna().to_numpy(float); grp = np.r_[np.zeros(len(legacy)), np.ones(len(D.bench)), 2 * np.ones(len(others))]
fig, (a0, a1) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1, 14], "hspace": .02}, sharex=True)
a0.imshow(grp[None, :], aspect="auto", cmap=ListedColormap([ORANGE, BLUE, "#c3c2b7"])); a0.axis("off"); a0.grid(False)
a1.imshow(B, aspect="auto", cmap=ListedColormap([SURF, "#256abf"]), interpolation="nearest"); a1.grid(False); a1.set_xticks([]); a1.set_yticks([]); a1.set_ylabel("Models, oldest at top"); a1.set_xlabel("Benchmarks")
for x, t, c in [(len(legacy) / 2, "older, excluded", "white"), (len(legacy) + len(D.bench) / 2, "analysed (29)", "white"), (len(legacy) + len(D.bench) + len(others) / 2, "excluded (< 25 models)", INK)]: a0.text(x, 0, t, ha="center", va="center", fontsize=9, color=c)
fig.suptitle("Only ~20% of cells in Epoch's full model x benchmark table are observed, and older benchmarks barely overlap with newer ones", x=.01, ha="left", fontweight="bold", fontsize=11); save(fig, "fig01_data_sparsity")

# ---- fig02: ECI vs PC1 ----
pc = R1("pc1_scores.csv"); C = R1("eci_vs_pc1.csv"); fig, a = plt.subplots(figsize=(6, 4.8)); col = "pc1"
a.scatter(pc[col], pc.eci, color=BLUE, s=24, edgecolor="#555", linewidth=.3); xs = np.linspace(pc[col].min(), pc[col].max(), 50); a.plot(xs, np.polyval(np.polyfit(pc[col], pc.eci, 1), xs), color=INK, lw=1.1)
a.set(xlabel="First principal component score", ylabel="Official ECI", title=f"The ECI is nearly the first principal component: r = {C.pearson_r["joint intercepts"]:.3f}"); save(fig, "fig02_pc1_vs_eci")

# ---- fig03: variation explained by each principal component ----
Vc = R1("variance_by_component.csv"); e_ = R1("eci_variance_explained.csv").R2["official_ECI_single_score"]; fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.bar(Vc.index, Vc.explained_by_this_component, color=VIOLET, width=.6, edgecolor=SURF, linewidth=2)
for k, v, c in zip(Vc.index, Vc.explained_by_this_component, Vc.cumulative): ax.text(k, v + .012, f"{v:.1%}\n(total {c:.0%})", ha="center", va="bottom", fontsize=8)
ax.plot([.62, 1.38], [e_, e_], color=INK, ls="--", lw=1.2); ax.text(1.42, e_, f"official ECI as one score: {e_:.1%}", va="center", fontsize=8)
ax.set(xlabel="Principal component", ylabel="Share of benchmark variation explained", ylim=(0, 1), title=W(f"The first component explains {Vc.explained_by_this_component[1]:.0%} of benchmark variation; each later one adds under 5%", 60)); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); save(fig, "fig03_variance_explained")

# ---- fig04: why AJ (selection bias) ----
S = R1("selection_bias.csv"); fig, ax = plt.subplots(1, 2, figsize=(11, 4.8)); lo, hi = S[["observed_mean", "model_implied_mean_all_models"]].min().min() - .3, S[["observed_mean", "model_implied_mean_all_models"]].max().max() + .3
sc = ax[0].scatter(S.model_implied_mean_all_models, S.observed_mean, c=S.mean_std_ECI_of_tested_models, cmap=SEQ, s=45, edgecolor="#555", linewidth=.4); ax[0].plot([lo, hi], [lo, hi], color=MUTED, lw=1)
ax[0].set(xlabel="Model-implied mean over all 191 models (logit)", ylabel="Observed mean over models actually tested (logit)", title=W("Observed benchmark means are biased by who was tested", 45)); plt.colorbar(sc, ax=ax[0], label="mean ECI of tested models (std.)")
x, y = S.mean_std_ECI_of_tested_models, S.bias; ax[1].scatter(x, y, color=BLUE, s=40, edgecolor="#555", linewidth=.4); xs = np.linspace(x.min(), x.max(), 50); ax[1].plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=INK, lw=1.2); ax[1].axhline(0, color=MUTED, lw=.8)
ax[1].set(xlabel="Mean ECI of models tested on the benchmark (std.)", ylabel="Observed minus model-implied mean", title=W(f"Bias grows with how strong the tested models were (r = {np.corrcoef(x, y)[0, 1]:.2f})", 45)); save(fig, "fig04_selection_bias")

# ---- fig05: ECI explains most benchmarks, a few poorly ----
pb = R1("per_benchmark_explained.csv").sort_values("R2_ECI", ascending=False); fig, ax = plt.subplots(figsize=(8, 8.5)); y = np.arange(len(pb)); r = pb.R2_ECI.clip(0, 1)
ax.barh(y, r, color=BLUE, height=.6, edgecolor=SURF, linewidth=2)
for yi, v in zip(y, r): ax.text(v + .01, yi, f"{v:.2f}", va="center", fontsize=7, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels(pb.index, fontsize=8); ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(0, 1)
ax.set(xlabel="Proportion of benchmark variation explained by ECI", title=W("The ECI is highly correlated with most benchmarks, but explains much less of a few", 60)); save(fig, "fig05_eci_by_benchmark")

# ---- fig06: residual structure vs shuffled null ----
SH = R2("residual_variance_by_component.csv"); fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.bar(SH.index, SH.share_of_residual_variation, color=AQUA, width=.6, edgecolor=SURF, linewidth=2, label="ECI residuals")
ax.plot(SH.index, SH.null_95th_pct, "s--", color=MUTED, label="shuffled residuals (95th percentile)"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set(xlabel="Principal component of the residuals", ylabel="Share of residual variation captured", title=W(f"What the ECI leaves out has some structure: {int((SH.share_of_residual_variation > SH.null_95th_pct).sum())} components capture more than chance", 60)); ax.legend(frameon=False, fontsize=8); save(fig, "fig06_residual_structure")

# ---- fig07: residual components ----
EL = R2("residual_loadings.csv"); fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for a, c in zip(axs, ["E1", "E2", "E3"]):
    s_ = EL[c].sort_values(); sel = pd.concat([s_.head(6), s_.tail(6)]); a.barh(range(len(sel)), sel.values, color=[ORANGE if v < 0 else BLUE for v in sel.values], height=.6, edgecolor=SURF, linewidth=2)
    a.set_yticks(range(len(sel))); a.set_yticklabels(sel.index, fontsize=8); a.axvline(0, color=MUTED, lw=.8); a.set_title(f"Component {c}", fontsize=10); a.grid(axis="y", visible=False); a.set_xlabel("Loading")
fig.suptitle("What ECI does not explain: three tentative axes (six highest and lowest loadings each; signs are arbitrary)", x=.01, ha="left", fontweight="bold", fontsize=11); save(fig, "fig07_residual_components")

# ---- fig08-10: how much the index cares about the worst-fitted benchmark (p-norm loss) ----
T, P, R2b, SC = R3("loss_summary.csv"), R3("pnorm_path.csv"), R3("loss_benchmark_r2.csv"), R3("loss_score_change_points.csv")
L1, LI = "p=1 (average = first PC)", "p=inf (worst benchmark)"; PC = {"Official ECI": INK, L1: VIOLET, LI: ORANGE}
Pp = P; fig, ax = plt.subplots(figsize=(7.5, 5.5)); ax.plot(Pp.worst_R2, Pp.mean_R2, "-", color=MUTED, lw=1.5, zorder=1)
mid = Pp.drop(index=["p=1", "p=64"]); ax.scatter(mid.worst_R2, mid.mean_R2, color=MUTED, s=36, edgecolor="#555", linewidth=.4, zorder=2)
for nm in mid.index: ax.annotate(nm.replace("p=", "p = "), (mid.worst_R2[nm], mid.mean_R2[nm]), fontsize=8, xytext=(8, 0) if nm == "p=32" else (-8, -4), textcoords="offset points", ha="left" if nm == "p=32" else "right", va="center" if nm == "p=32" else "top")
for nm, c in PC.items():
    ax.scatter(T.worst_R2[nm], T.mean_R2[nm], color=c, s=70, edgecolor="#555", linewidth=.5, zorder=3); off = {"Official ECI": (-8, 6), L1: (8, 4), LI: (-10, -10)}[nm]
    ax.annotate(nm, (T.worst_R2[nm], T.mean_R2[nm]), fontsize=8, xytext=off, textcoords="offset points", ha="right" if off[0] < 0 else "left")
ax.set(xlabel="Worst benchmark R-squared", ylabel="Mean benchmark R-squared", xlim=(None, .73), title=W("Caring more about the worst benchmark lifts it, at a cost to average fit", 55)); save(fig, "fig08_loss_tradeoff")
o = R2b.sort_values("Official ECI"); fig, ax = plt.subplots(figsize=(8, 8)); y = np.arange(len(o)); ax.hlines(y, o["Official ECI"], o[LI], color=GRID, lw=2)
ax.scatter(o["Official ECI"], y, color=BLUE, s=36, label="official ECI", zorder=3); ax.scatter(o[LI], y, color=ORANGE, s=36, label=LI, zorder=3); ax.set_yticks(y); ax.set_yticklabels(o.index, fontsize=8); ax.grid(axis="y", visible=False)
ax.set(xlabel="R-squared", title=W("Fitting only the worst benchmark flattens the fit: GeoBench up, METR / ARC-AGI / AIME down", 60)); ax.legend(frameon=False, fontsize=9, loc="lower right"); save(fig, "fig09_loss_per_benchmark")
fig, ax = plt.subplots(figsize=(8, 5)); hw = D.eci_halfwidth; ax.axhspan(-hw, hw, color="#e8f0fb", label=f"typical ECI 90% interval (+/-{hw:.1f})")
for nm in (L1, LI): ax.scatter(eci, SC[nm], color=PC[nm], s=16, alpha=.85, label=nm)
for i in np.argsort(-np.abs(SC[LI].to_numpy()))[:4]: ax.annotate(SC.index[i], (eci[i], SC[LI].iloc[i]), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.axhline(0, color=MUTED, lw=.8); ax.set(xlabel="Official ECI", ylabel="Change in score vs official ECI (points)", title=W("p = 1 (the first PC) moves scores within the ECI's own uncertainty; fitting only the worst benchmark moves some by 20+", 60)); ax.legend(frameon=False, fontsize=8); save(fig, "fig10_loss_score_changes")

# ---- fig12-13: scores over release date (like Epoch's ECI chart), one panel per loss ----
import matplotlib.dates as mdates
OT = R3("loss_over_time.csv"); dates = D.dates.to_numpy(); order = np.argsort(dates, kind="stable")
COLS = PC
score = {k: eci + SC[k].to_numpy() for k in COLS}
def frontier(sc):
    best, rec = -np.inf, []
    for i in order:
        if sc[i] > best: best = sc[i]; rec.append(i)
    return np.array(rec)
fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
for a, k in zip(axs.ravel(), COLS):
    if k != "Official ECI": a.scatter(dates, eci, s=10, color="#c3c2b7", label="official ECI")
    a.scatter(dates, score[k], s=14, color=COLS[k] if k != "Official ECI" else BLUE, alpha=.85, label=k)
    rec = frontier(score[k]); a.step(dates[rec], score[k][rec], where="post", color=INK, lw=1.4, label="frontier"); a.scatter(dates[rec], score[k][rec], s=22, color=INK, zorder=3)
    a.set_title(f"{k}  |  frontier +{OT.frontier_slope_points_per_year[k]:.1f} pts/yr (+{OT.frontier_slope_models_8plus_scores[k]:.1f} if 8+ scores)", fontsize=9); a.xaxis.set_major_locator(mdates.YearLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if k != "Official ECI": a.legend(frameon=False, fontsize=7, loc="upper left")
axs[0].set_ylabel("Score (ECI points)")
fig.suptitle("Scores over release date under each p; black = frontier (models that set a record at release)", x=.01, ha="left", fontweight="bold", fontsize=11); save(fig, "fig12_scores_over_time")
# fig13: frontier with a two-segment line broken at Epoch's estimated date (Apr 2024), one panel per loss
yrs = ((D.dates - D.dates.min()).dt.days / 365.25).to_numpy(); tb = (BREAK_DATE - D.dates.min()).days / 365.25
EA = pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv").dropna(subset=["eci", "date"]); da = pd.to_datetime(EA.date).to_numpy(); ra = frontier_idx(EA.eci.to_numpy(), da)
ya = (pd.to_datetime(EA.date) - D.dates.min()).dt.days.to_numpy() / 365.25; ref = OT.loc["Official ECI, all Epoch models (reference)"]
to_date = lambda t: D.dates.min() + pd.to_timedelta(t * 365.25, unit="D")
fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), sharex=True, sharey=True)
for a, k in zip(axs.ravel(), COLS):
    if k == "Official ECI":
        a.step(da[ra], EA.eci.to_numpy()[ra], where="post", color="#c3c2b7", lw=1.4, label=f"all Epoch models (reference): {ref.slope_before_break:+.1f} -> {ref.slope_after_break:+.1f} pts/yr")
        _, _, cr = hinge_fit(ya[ra], EA.eci.to_numpy()[ra], tb); tt = np.linspace(ya[ra].min(), ya[ra].max(), 200); a.plot(to_date(tt), cr[0] + cr[1] * tt + cr[2] * np.clip(tt - tb, 0, None), color="#c3c2b7", lw=1, ls="--")
    rec = frontier_idx(score[k], dates); a.step(dates[rec], score[k][rec], where="post", color=COLS[k], lw=1.6); a.scatter(dates[rec], score[k][rec], s=20, color=COLS[k], zorder=3)
    _, _, c = hinge_fit(yrs[rec], score[k][rec], tb); tt = np.linspace(min(yrs[rec].min(), tb - .3), yrs[rec].max(), 200)
    a.plot(to_date(tt), c[0] + c[1] * tt + c[2] * np.clip(tt - tb, 0, None), color=INK, lw=1.3, ls="--", label=f"two-segment fit: {OT.slope_before_break[k]:+.1f} -> {OT.slope_after_break[k]:+.1f} pts/yr")
    a.axvline(BREAK_DATE, color=MUTED, lw=.8, ls=":")
    a.set_title(W(f"{k}: {int(OT.n_frontier_before_break[k])} frontier model(s) before the break; after it {OT.slope_after_break[k]:+.1f} pts/yr ({OT.slope_after_break_8plus[k]:+.1f} if 8+ scores)", 52), fontsize=9)
    a.legend(frameon=False, fontsize=7, loc="upper left"); a.xaxis.set_major_locator(mdates.YearLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axs[0].set_ylabel("Frontier score (ECI points)")
fig.suptitle(W("Frontier under each p with a break at Apr 2024 (Epoch's estimate). Slopes after the break agree; this sample has too few earlier models to estimate the slope before it", 110), x=.01, ha="left", fontweight="bold", fontsize=11); save(fig, "fig13_frontier_over_time")
print("done")
