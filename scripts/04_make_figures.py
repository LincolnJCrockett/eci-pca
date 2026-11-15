"""Figures (one per claim). Reads the CSVs written by scripts 01-03.
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
fig.suptitle("Only ~20% of model x benchmark cells are observed, and older benchmarks barely overlap with newer ones", x=.01, ha="left", fontweight="bold", fontsize=11); save(fig, "fig01_data_sparsity")

# ---- fig02: ECI vs PC1 ----
pc = R1("pc1_scores.csv"); C = R1("eci_vs_pc1.csv"); fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for a, col, ttl in ((ax[0], "pc1_A", "Conventional centering (A)"), (ax[1], "pc1_AJ", "Joint intercepts (AJ)")):
    a.scatter(pc[col], pc.eci, color=BLUE, s=24, edgecolor="#555", linewidth=.3); xs = np.linspace(pc[col].min(), pc[col].max(), 50); a.plot(xs, np.polyval(np.polyfit(pc[col], pc.eci, 1), xs), color=INK, lw=1.1)
    a.set(xlabel="First principal component score", title=f"{ttl}: r = {C.pearson_r[col[4:]]:.3f}")
ax[0].set_ylabel("Official ECI"); save(fig, "fig02_pc1_vs_eci")

# ---- fig03: variance explained ----
Vi, Vh = R1("variance_explained_in_sample.csv"), R1("variance_explained_held_out.csv"); ks = [1, 2, 3]; fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].plot(ks, [Vi.R2_in_sample_equal_weight[f"PCA rank {k}, conventional (A)"] for k in ks], "o-", color=BLUE, lw=2, label="A: conventional centering"); ax[0].plot(ks, [Vi.R2_in_sample_equal_weight[f"PCA rank {k}, joint intercepts (AJ)"] for k in ks], "o-", color=VIOLET, lw=2, label="AJ: joint intercepts")
e_ = Vi.R2_in_sample_equal_weight["official ECI as the single score"]; ax[0].axhline(e_, color=INK, ls="--", lw=1.2); ax[0].text(3, e_ - .015, "official ECI (one score)", ha="right", va="top", fontsize=9)
ax[0].set(xlabel="Number of components", ylabel="R-squared (in sample)", ylim=(.5, 1), title=W("One properly estimated component explains ~84% of benchmark variation", 45)); ax[0].xaxis.set_major_locator(MaxNLocator(integer=True)); ax[0].legend(frameon=False, fontsize=9, loc="lower right")
ax[1].plot(Vh.index, Vh.A, "o-", color=BLUE, lw=2, label="A"); ax[1].plot(Vh.index, Vh.AJ, "o-", color=VIOLET, lw=2, label="AJ"); ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].set(xlabel="Number of components", ylabel="R-squared (held-out cells)", title=W("Out of sample, one joint-intercept component is best", 45)); ax[1].legend(frameon=False, fontsize=9); save(fig, "fig03_variance_explained")

# ---- fig04: why AJ (selection bias) ----
S = R1("selection_bias.csv"); fig, ax = plt.subplots(1, 2, figsize=(11, 4.8)); lo, hi = S[["observed_mean", "model_implied_mean_all_models"]].min().min() - .3, S[["observed_mean", "model_implied_mean_all_models"]].max().max() + .3
sc = ax[0].scatter(S.model_implied_mean_all_models, S.observed_mean, c=S.mean_std_ECI_of_tested_models, cmap=SEQ, s=45, edgecolor="#555", linewidth=.4); ax[0].plot([lo, hi], [lo, hi], color=MUTED, lw=1)
ax[0].set(xlabel="Model-implied mean over all 191 models (logit)", ylabel="Observed mean over models actually tested (logit)", title=W("Observed benchmark means are biased by who was tested", 45)); plt.colorbar(sc, ax=ax[0], label="mean ECI of tested models (std.)")
x, y = S.mean_std_ECI_of_tested_models, S.bias; ax[1].scatter(x, y, color=BLUE, s=40, edgecolor="#555", linewidth=.4); xs = np.linspace(x.min(), x.max(), 50); ax[1].plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=INK, lw=1.2); ax[1].axhline(0, color=MUTED, lw=.8)
ax[1].set(xlabel="Mean ECI of models tested on the benchmark (std.)", ylabel="Observed minus model-implied mean", title=W(f"Bias grows with how strong the tested models were (r = {np.corrcoef(x, y)[0, 1]:.2f})", 45)); save(fig, "fig04_selection_bias")

# ---- fig05: ECI explains most benchmarks, a few poorly ----
pb = R1("per_benchmark_explained.csv").sort_values("R2_ECI", ascending=False); fig, ax = plt.subplots(figsize=(8, 8.5)); y = np.arange(len(pb)); r = pb.R2_ECI.clip(0, 1)
ax.barh(y, r, color=BLUE, height=.6, edgecolor=SURF, linewidth=2, label="explained by official ECI"); ax.barh(y, 1 - r, left=r, color="#f3c1ad", height=.6, edgecolor=SURF, linewidth=2, label="not explained")
ax.scatter(pb.R2_PCA_rank1_AJ, y, color=INK, s=22, zorder=3, label="explained by rank-1 PCA (AJ)"); ax.set_yticks(y); ax.set_yticklabels(pb.index, fontsize=8); ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(0, 1)
ax.set(xlabel="Share of the benchmark's variation across models", title=W("ECI explains most benchmarks well; short blue bars are benchmarks it undervalues (or that measure something else)", 60)); ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.07), ncol=3); save(fig, "fig05_eci_by_benchmark")

# ---- fig06: residual structure is weak ----
ee, ec, es = R2("residual_eigenvalues.csv"), R2("residual_held_out.csv"), R2("residual_stability.csv"); fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
ax[0].plot(ee.index, ee.eigenvalue, "o-", color=AQUA, lw=2, label="ECI residuals"); ax[0].plot(ee.index, ee.null_95th_pct, "s--", color=MUTED, label="shuffled null (95th pct)"); ax[0].set(xlabel="Component", ylabel="Eigenvalue", title=W("Some structure above chance...", 38)); ax[0].legend(frameon=False, fontsize=8); ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].bar(ec.index[1:], ec.change_vs_ECI_only[1:] * 100, color=ORANGE, edgecolor=SURF, linewidth=2); ax[1].axhline(0, color=MUTED, lw=.8); ax[1].set(xlabel="Components added to the ECI prediction", ylabel="Change in held-out error (%, negative = worse)", title=W("...but adding it hurts held-out prediction", 38)); ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[2].bar(es.index, es.mean_best_match_cosine, color=BLUE, width=.55, edgecolor=SURF, linewidth=2); ax[2].errorbar(es.index, es.mean_best_match_cosine, yerr=[es.mean_best_match_cosine - es.p10_best_match_cosine, np.zeros(len(es))], fmt="none", ecolor=INK, capsize=4)
ax[2].set(ylim=(0, 1), ylabel="Bootstrap stability (cosine; whisker = 10th pct)", title=W("...and components are only moderately stable", 38)); save(fig, "fig06_residual_structure")

# ---- fig07: residual components ----
EL = R2("residual_loadings.csv"); fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for a, c in zip(axs, ["E1", "E2", "E3"]):
    s_ = EL[c].sort_values(); sel = pd.concat([s_.head(6), s_.tail(6)]); a.barh(range(len(sel)), sel.values, color=[ORANGE if v < 0 else BLUE for v in sel.values], height=.6, edgecolor=SURF, linewidth=2)
    a.set_yticks(range(len(sel))); a.set_yticklabels(sel.index, fontsize=8); a.axvline(0, color=MUTED, lw=.8); a.set_title(f"Component {c}", fontsize=10); a.grid(axis="y", visible=False); a.set_xlabel("Loading")
fig.suptitle("What ECI does not explain: three tentative axes (six highest and lowest loadings each; signs are arbitrary)", x=.01, ha="left", fontweight="bold", fontsize=11); save(fig, "fig07_residual_components")

# ---- fig08-11: loss functions ----
T, P, HO, R2b, SC = R3("loss_summary.csv"), R3("worst_benchmark_path.csv"), R3("loss_held_out.csv"), R3("loss_benchmark_r2.csv"), R3("loss_score_change_points.csv")
fig, ax = plt.subplots(figsize=(7.5, 5.5)); ax.plot(P.worst_R2, P.mean_R2, "-", color=MUTED, lw=1.5, zorder=1)
for nm, c in [("Official ECI", INK), ("MSE (cells)", BLUE), ("MSE (benchmark avg)", AQUA), ("MAE (benchmark avg)", "#eda100"), ("Worst benchmark (p=8)", MAGENTA), ("Worst benchmark (p=64)", ORANGE)]:
    ax.scatter(T.worst_R2[nm], T.mean_R2[nm], color=c, s=70, edgecolor="#555", linewidth=.5, zorder=3); ax.annotate(nm, (T.worst_R2[nm], T.mean_R2[nm]), fontsize=8, xytext=(6, 4 if nm != "MSE (benchmark avg)" else -11), textcoords="offset points")
ax.set(xlabel="Worst benchmark R-squared", ylabel="Mean benchmark R-squared", title=W("Fitting the worst benchmark better costs average fit", 55)); save(fig, "fig08_loss_tradeoff")
o = R2b.sort_values("Official ECI"); fig, ax = plt.subplots(figsize=(8, 8)); y = np.arange(len(o)); ax.hlines(y, o["Official ECI"], o["Worst benchmark (p=64)"], color=GRID, lw=2)
ax.scatter(o["Official ECI"], y, color=BLUE, s=36, label="official ECI", zorder=3); ax.scatter(o["Worst benchmark (p=64)"], y, color=ORANGE, s=36, label="worst-benchmark loss (p=64)", zorder=3); ax.set_yticks(y); ax.set_yticklabels(o.index, fontsize=8); ax.grid(axis="y", visible=False)
ax.set(xlabel="R-squared", title=W("Worst-benchmark loss flattens the fit: GeoBench up, METR / ARC-AGI / AIME down", 60)); ax.legend(frameon=False, fontsize=9, loc="lower right"); save(fig, "fig09_loss_per_benchmark")
fig, ax = plt.subplots(figsize=(8, 5)); hw = D.eci_halfwidth; ax.axhspan(-hw, hw, color="#e8f0fb", label=f"typical ECI 95% interval (+/-{hw:.1f})")
ax.scatter(eci, SC["MSE (cells)"], color=BLUE, s=18, label="MSE (cells)"); ax.scatter(eci, SC["Worst benchmark (p=64)"], color=ORANGE, s=18, label="worst benchmark (p=64)")
for i in np.argsort(-np.abs(SC["Worst benchmark (p=64)"].to_numpy()))[:4]: ax.annotate(SC.index[i], (eci[i], SC["Worst benchmark (p=64)"].iloc[i]), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.axhline(0, color=MUTED, lw=.8); ax.set(xlabel="Official ECI", ylabel="Change in score vs official ECI (points)", title=W("MSE moves scores within ECI's own uncertainty; the worst-benchmark loss moves some by 20+", 60)); ax.legend(frameon=False, fontsize=8); save(fig, "fig10_loss_score_changes")
fig, ax = plt.subplots(figsize=(9, 4.8)); x = np.arange(len(HO)); ax.bar(x - .2, HO.mean_R2, .38, color=BLUE, edgecolor=SURF, linewidth=2, label="mean over benchmarks"); ax.bar(x + .2, HO.worst_R2, .38, color=ORANGE, edgecolor=SURF, linewidth=2, label="worst benchmark")
ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks(x); ax.set_xticklabels([textwrap.fill(s, 14) for s in HO.index], fontsize=8); ax.set(ylabel="Held-out R-squared", title=W("Out of sample, the worst-benchmark loss is worse on average and in the worst case", 70)); ax.legend(frameon=False, fontsize=9); save(fig, "fig11_loss_held_out")

# ---- fig12-13: scores over release date (like Epoch's ECI chart), one panel per loss ----
import matplotlib.dates as mdates
OT = R3("loss_over_time.csv"); dates = D.dates.to_numpy(); order = np.argsort(dates, kind="stable")
COLS = {"Official ECI": INK, "MSE (cells)": BLUE, "MSE (benchmark avg)": AQUA, "MAE (benchmark avg)": "#eda100", "Worst benchmark (p=8)": MAGENTA, "Worst benchmark (p=64)": ORANGE}
score = {k: eci + SC[k].to_numpy() for k in COLS}
def frontier(sc):
    best, rec = -np.inf, []
    for i in order:
        if sc[i] > best: best = sc[i]; rec.append(i)
    return np.array(rec)
fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), sharex=True, sharey=True)
for a, k in zip(axs.ravel(), COLS):
    if k != "Official ECI": a.scatter(dates, eci, s=10, color="#c3c2b7", label="official ECI")
    a.scatter(dates, score[k], s=14, color=COLS[k] if k != "Official ECI" else BLUE, alpha=.85, label=k)
    rec = frontier(score[k]); a.step(dates[rec], score[k][rec], where="post", color=INK, lw=1.4, label="frontier"); a.scatter(dates[rec], score[k][rec], s=22, color=INK, zorder=3)
    a.set_title(f"{k}  |  frontier +{OT.frontier_slope_points_per_year[k]:.1f} pts/yr (+{OT.frontier_slope_models_8plus_scores[k]:.1f} if 8+ scores)", fontsize=9); a.xaxis.set_major_locator(mdates.YearLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if k != "Official ECI": a.legend(frameon=False, fontsize=7, loc="upper left")
for a in axs[:, 0]: a.set_ylabel("Score (ECI points)")
fig.suptitle("Scores over release date under each loss; black = frontier (models that set a record at release)", x=.01, ha="left", fontweight="bold", fontsize=11); save(fig, "fig12_scores_over_time")
fig, ax = plt.subplots(figsize=(9.5, 5.5))
for k, c in COLS.items():
    rec = frontier(score[k]); ax.step(dates[rec], score[k][rec], where="post", color=c, lw=2.2 if k in ("Official ECI", "Worst benchmark (p=64)") else 1.5, label=f"{k} (+{OT.frontier_slope_points_per_year[k]:.1f} pts/yr; +{OT.frontier_slope_models_8plus_scores[k]:.1f} using models with 8+ scores)")
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.set(ylabel="Frontier score (ECI points)", title=W("Average losses track the official frontier; the worst-benchmark loss lifts early records, but only via models with few scores", 70)); ax.legend(frameon=False, fontsize=8, loc="upper left"); save(fig, "fig13_frontier_over_time")
print("done")
