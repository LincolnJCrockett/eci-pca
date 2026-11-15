"""One figure per claim. Reads the CSVs written by the analysis scripts.
Usage: python -I scripts/make_figures.py <extracted_dir> <outputs_dir> <figures_dir>
"""
import sys, os, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import MaxNLocator
from common import *

ex, op, fd = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(fd, exist_ok=True)
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
DIV = LinearSegmentedColormap.from_list("div", [ORANGE, "#f0efec", BLUE]); SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#256abf"])
plt.rcParams.update({"axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
W = lambda s, w=60: textwrap.fill(s, w)
def save(fig, name): fig.tight_layout(); fig.savefig(f"{fd}/{name}.png", dpi=150); plt.close(fig); print("wrote", name)

Dd = load(ex); Xv, M, eci, date, obs = Dd.Xv, Dd.M, Dd.eci, Dd.date, Dd.obs; n, m = Xv.shape; bench = list(M.columns); z = (eci - eci.mean()) / eci.std()
V = lambda f: pd.read_csv(f"{op}/variance/{f}", index_col=0); L = lambda f: pd.read_csv(f"{op}/loss_comparison/{f}", index_col=0)

# ---- 01 data structure ----
proc = pd.read_csv(f"{ex}/epoch_capabilities_index/processed_data_for_eci.csv"); meta = pd.read_csv(f"{ex}/benchmark_metadata.csv").set_index("benchmark")
Mall = proc.pivot_table(index="model", columns="benchmark", values="performance"); mdate = pd.to_datetime(proc.groupby("model").date.first())
big = Mall.loc[:, Mall.notna().sum() >= 25]; O = big.notna().astype(int).T @ big.notna().astype(int); med = pd.Series({b: O.loc[b].drop(b).median() for b in O.index})
legacy = [b for b in med[med < 7].index if b not in bench]; others = [c for c in Mall.columns if c not in legacy and c not in bench]
cols = legacy + sorted(bench, key=lambda b: -Mall[b].notna().sum()) + sorted(others, key=lambda b: -Mall[b].notna().sum()); rows_ = mdate.sort_values().index
B = Mall.loc[rows_, cols].notna().to_numpy(float); grp = np.r_[np.zeros(len(legacy)), np.ones(len(bench)), 2 * np.ones(len(others))]
fig, (a0, a1) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1, 14], "hspace": .02}, sharex=True)
a0.imshow(grp[None, :], aspect="auto", cmap=ListedColormap([ORANGE, BLUE, "#c3c2b7"])); a0.axis("off"); a0.grid(False)
a1.imshow(B, aspect="auto", cmap=ListedColormap([SURF, "#256abf"]), interpolation="nearest"); a1.grid(False); a1.set_xticks([]); a1.set_yticks([]); a1.set_ylabel("Models, oldest at top"); a1.set_xlabel("Benchmarks")
for x, t, c in [(len(legacy) / 2, "older, excluded", "white"), (len(legacy) + len(bench) / 2, "analysed (29)", "white"), (len(legacy) + len(bench) + len(others) / 2, "excluded (< 25 models)", INK)]: a0.text(x, 0, t, ha="center", va="center", fontsize=9, color=c)
fig.suptitle("Only ~20% of model x benchmark cells are observed, and older benchmarks barely overlap with newer ones", x=.01, ha="left", fontweight="bold", fontsize=11)
save(fig, "fig01_data_structure")

# ---- 02 selection bias in benchmark means ----
fA = fit(Xv, "AJ", 1); pm = fA["pred"].mean(0); cm = np.where(obs, Xv, 0).sum(0) / obs.sum(0); mt = np.array([z[obs[:, j]].mean() for j in range(m)])
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
sc = ax[0].scatter(pm, cm, c=mt, cmap=SEQ, s=45, edgecolor="#555", linewidth=.4); lo, hi = min(pm.min(), cm.min()) - .2, max(pm.max(), cm.max()) + .2; ax[0].plot([lo, hi], [lo, hi], color=MUTED, lw=1)
for j in np.argsort(-np.abs(cm - pm))[:4]: ax[0].annotate(bench[j], (pm[j], cm[j]), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax[0].set(xlabel="Model-implied mean over ALL 191 models (logit)", ylabel="Observed mean over models actually tested (logit)", title=W("Observed benchmark means are biased by who was tested", 45)); plt.colorbar(sc, ax=ax[0], label="mean ECI of tested models (std.)")
ax[1].scatter(mt, cm - pm, color=BLUE, s=40, edgecolor="#555", linewidth=.4); b_ = np.polyfit(mt, cm - pm, 1); xs = np.linspace(mt.min(), mt.max(), 50); ax[1].plot(xs, np.polyval(b_, xs), color=INK, lw=1.2)
ax[1].axhline(0, color=MUTED, lw=.8); ax[1].set(xlabel="Mean ECI of models tested on the benchmark (std.)", ylabel="Observed minus model-implied mean", title=W(f"Bias grows with how strong the tested models were (r = {np.corrcoef(mt, cm - pm)[0, 1]:.2f})", 45))
save(fig, "fig02_selection_bias")

# ---- 03 variance explained ----
ti, th = V("variance_explained_insample.csv"), V("variance_explained_heldout.csv"); ks = [1, 2, 3, 4]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ax[0].plot(ks, [ti.R2_in_sample_equal_weight[f"A rank-{k} (PCA, observed benchmark means)"] for k in ks], "o-", color=BLUE, lw=2, label="A: observed benchmark means"); ax[0].plot(ks, [ti.R2_in_sample_equal_weight[f"AJ rank-{k} (PCA, intercepts estimated jointly)"] for k in ks], "o-", color=VIOLET, lw=2, label="AJ: intercepts estimated jointly")
e_ = ti.R2_in_sample_equal_weight["official ECI, per-benchmark OLS linear"]; ax[0].axhline(e_, color=INK, ls="--", lw=1.2); ax[0].text(4, e_ - .015, "official ECI (one score)", ha="right", va="top", fontsize=9)
ax[0].set(xlabel="Number of components", ylabel="R-squared (in sample)", ylim=(.5, 1), title=W("One properly estimated component explains ~84% of benchmark variation", 45)); ax[0].xaxis.set_major_locator(MaxNLocator(integer=True)); ax[0].legend(frameon=False, fontsize=9, loc="lower right")
kk = th.index.to_numpy(); ax[1].plot(kk, th.iloc[:, 0], "o-", color=BLUE, lw=2, label="A"); ax[1].plot(kk, th.iloc[:, 1], "o-", color=VIOLET, lw=2, label="AJ"); ax[1].set(xlabel="Number of components", ylabel="R-squared (held-out cells)", title=W("Out of sample, one component is best", 45)); ax[1].legend(frameon=False, fontsize=9)
save(fig, "fig03_variance_explained")

# ---- 04 ECI vs PC1 ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for a, sp, ttl in ((ax[0], "A", "Fixed benchmark means"), (ax[1], "AJ", "Intercepts estimated jointly")):
    s_ = fit(Xv, sp, 1)["scores"][:, 0]; s_ = s_ if np.corrcoef(s_, eci)[0, 1] > 0 else -s_; r = np.corrcoef(s_, eci)[0, 1]
    a.scatter(s_, eci, c=obs.sum(1), cmap=SEQ, s=26, edgecolor="#555", linewidth=.3, vmin=0); xs = np.linspace(s_.min(), s_.max(), 50); a.plot(xs, np.polyval(np.polyfit(s_, eci, 1), xs), color=INK, lw=1.1)
    a.set(xlabel="First principal component score", title=f"{ttl}: r = {r:.3f}")
ax[0].set_ylabel("Official ECI"); save(fig, "fig04_eci_vs_pc1")

# ---- 05 per-benchmark explained by ECI vs by rank-1 PCA (logit) ----
pb = V("per_benchmark_variance_explained.csv").sort_values("R2_ECI_OLS_logit", ascending=False)
fig, ax = plt.subplots(figsize=(8, 8.5)); y = np.arange(len(pb))
ax.barh(y, pb.R2_ECI_OLS_logit.clip(0, 1), color=BLUE, height=.6, edgecolor=SURF, linewidth=2, label="explained by official ECI"); ax.barh(y, 1 - pb.R2_ECI_OLS_logit.clip(0, 1), left=pb.R2_ECI_OLS_logit.clip(0, 1), color="#f3c1ad", height=.6, edgecolor=SURF, linewidth=2, label="not explained")
ax.scatter(pb.R2_PCA_rank1_AJ_logit, y, color=INK, s=22, zorder=3, label="rank-1 PCA (AJ)"); ax.set_yticks(y); ax.set_yticklabels(pb.index, fontsize=8); ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(0, 1)
ax.set(xlabel="Share of variance across models (logit scores)", title=W("ECI explains most benchmarks well and a few poorly", 60)); ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.07), ncol=3)
save(fig, "fig05_eci_by_benchmark")

# ---- 06 residual structure: weak and does not generalise ----
ee, ec, es = V("E_eigs.csv"), V("E_cv.csv"), V("E_stability.csv")
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
ax[0].plot(ee.index, ee.eig_residual, "o-", color=AQUA, lw=2, label="ECI residuals"); ax[0].plot(ee.index, ee.null95, "s--", color=MUTED, label="shuffled null (95th pct)"); ax[0].set(xlabel="Component", ylabel="Eigenvalue", title=W("Some structure above chance...", 38)); ax[0].legend(frameon=False, fontsize=8); ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].bar(ec.index[1:], ec.gain_vs_ECI_only[1:] * 100, color=ORANGE, edgecolor=SURF, linewidth=2); ax[1].axhline(0, color=MUTED, lw=.8); ax[1].set(xlabel="Components added to ECI-only prediction", ylabel="Change in held-out error (%, negative = worse)", title=W("...but adding it hurts held-out prediction", 38)); ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[2].bar(es.index, es.mean_best_cos, color=BLUE, width=.55, edgecolor=SURF, linewidth=2); ax[2].errorbar(es.index, es.mean_best_cos, yerr=[es.mean_best_cos - es.p10_best_cos, np.zeros(len(es))], fmt="none", ecolor=INK, capsize=4); ax[2].set(ylim=(0, 1), ylabel="Bootstrap stability (cosine; whisker = 10th pct)", title=W("...and components are only moderately stable", 38))
save(fig, "fig06_residual_structure")

# ---- 07 residual components ----
EL = V("E_loadings.csv"); fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax_, c, lab in zip(axs, ["E1", "E2", "E3"], ["E1: factual / spatial  vs  agentic software", "E2: software eng. / proof  vs  perceptual / puzzle", "E3: competition math  vs  agentic / common sense"]):
    s_ = EL[c].sort_values(); sel = pd.concat([s_.head(6), s_.tail(6)]); ax_.barh(range(len(sel)), sel.values, color=[ORANGE if v < 0 else BLUE for v in sel.values], height=.6, edgecolor=SURF, linewidth=2)
    ax_.set_yticks(range(len(sel))); ax_.set_yticklabels(sel.index, fontsize=8); ax_.axvline(0, color=MUTED, lw=.8); ax_.set_title(W(lab, 34), fontsize=10); ax_.grid(axis="y", visible=False); ax_.set_xlabel("Loading")
save(fig, "fig07_residual_components")

# ---- 08/09 what ECI favours ----
pr = pd.read_csv(f"{op}/variance/q1_partial_corr_AJ.csv").sort_values("partial_r"); fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(range(len(pr)), pr.partial_r, color=[ORANGE if v < 0 else BLUE for v in pr.partial_r], height=.6, edgecolor=SURF, linewidth=2); ax.set_yticks(range(len(pr))); ax.set_yticklabels(pr.benchmark, fontsize=8); ax.axvline(0, color=MUTED, lw=.8); ax.grid(axis="y", visible=False)
ax.set(xlabel="Correlation of the benchmark's residual (given PC1) with ECI's residual (given PC1)", title=W("Relative to PC1, ECI leans toward math and METR, away from long-context, geo and vision", 60)); save(fig, "fig08_eci_tilt_by_benchmark")
f1 = fit(Xv, "AJ", 1); pc1 = f1["scores"][:, 0]; pc1 = pc1 if np.corrcoef(pc1, eci)[0, 1] > 0 else -pc1; rs = eci - np.polyval(np.polyfit(pc1, eci, 1), pc1); E3 = pd.read_csv(f"{op}/variance/E_scores.csv", index_col=0)["E3"].to_numpy()
fig, ax = plt.subplots(figsize=(6.5, 5)); ax.scatter(E3, rs, color=BLUE, s=24, edgecolor="#555", linewidth=.3); xs = np.linspace(E3.min(), E3.max(), 50); ax.plot(xs, np.polyval(np.polyfit(E3, rs, 1), xs), color=INK, lw=1.2); r3 = corr(E3, rs)
ax.axhline(0, color=MUTED, lw=.8); ax.set(xlabel="E3 score (competition math vs agentic / common sense)", ylabel="ECI not explained by PC1 (points)", title=W(f"The one robust tilt: r = {r3[0]:+.2f} [{r3[2]:+.2f}, {r3[3]:+.2f}]", 50)); save(fig, "fig09_eci_residual_vs_E3")

# ---- 10-16 loss comparison ----
S, HO, PRF, SC, CR = L("loss_summary.csv"), L("loss_heldout.csv"), L("loss_benchmark_R2.csv"), L("loss_scores_standardized.csv"), L("loss_solution_correlations.csv")
fig, ax = plt.subplots(figsize=(7.5, 5.5)); pth = S.loc[["worst-benchmark p=1", "worst-benchmark p=8", "worst-benchmark p=64"]]; ax.plot(pth.min_R2, pth.mean_R2, "-", color=MUTED, lw=1.5, zorder=1)
for nm, c in [("official ECI", INK), ("MSE (cells)", BLUE), ("MSE (benchmark avg)", AQUA), ("MAE (benchmark avg)", YELLOW), ("worst-benchmark p=8", MAGENTA), ("worst-benchmark p=64", ORANGE)]:
    ax.scatter(S.min_R2[nm], S.mean_R2[nm], color=c, s=70, edgecolor="#555", linewidth=.5, zorder=3); ax.annotate(nm, (S.min_R2[nm], S.mean_R2[nm]), fontsize=8, xytext=(6, 4 if nm != "MSE (benchmark avg)" else -11), textcoords="offset points")
ax.set(xlabel="Worst benchmark R-squared", ylabel="Mean benchmark R-squared", title=W("Fitting the worst benchmark better costs average fit", 55)); save(fig, "fig10_loss_tradeoff")
o = PRF.sort_values("official ECI"); fig, ax = plt.subplots(figsize=(8, 8)); y = np.arange(len(o)); ax.hlines(y, o["official ECI"], o["worst-benchmark p=64"], color=GRID, lw=2)
ax.scatter(o["official ECI"], y, color=BLUE, s=36, label="official ECI", zorder=3); ax.scatter(o["worst-benchmark p=64"], y, color=ORANGE, s=36, label="worst-benchmark loss (p=64)", zorder=3); ax.set_yticks(y); ax.set_yticklabels(o.index, fontsize=8); ax.grid(axis="y", visible=False)
ax.set(xlabel="R-squared", title=W("Worst-benchmark loss flattens fit: GeoBench up, METR / ARC-AGI / AIME down", 60)); ax.legend(frameon=False, fontsize=9, loc="lower right"); save(fig, "fig11_loss_per_benchmark")
fig, ax = plt.subplots(figsize=(8, 5)); dmm = (eci.mean() + eci.std() * SC["worst-benchmark p=64"].to_numpy()) - eci; dms = (eci.mean() + eci.std() * SC["MSE (cells)"].to_numpy()) - eci; hw = float(((pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv").set_index("Model").reindex(M.index).eval("(eci_ci_high - eci_ci_low) / 2")).mean()))
ax.axhspan(-hw, hw, color="#e8f0fb", label=f"typical ECI 95% interval (+/-{hw:.1f})"); ax.scatter(eci, dms, color=BLUE, s=18, label="MSE (cells)"); ax.scatter(eci, dmm, color=ORANGE, s=18, label="worst-benchmark p=64")
for i in np.argsort(-np.abs(dmm))[:4]: ax.annotate(M.index[i], (eci[i], dmm[i]), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.axhline(0, color=MUTED, lw=.8); ax.set(xlabel="Official ECI", ylabel="Change in score vs official ECI (points)", title=W("MSE moves scores within ECI's own uncertainty; worst-benchmark loss moves some by 20+", 60)); ax.legend(frameon=False, fontsize=8); save(fig, "fig12_score_changes")
fig, ax = plt.subplots(figsize=(9, 4.8)); x = np.arange(len(HO)); ax.bar(x - .2, HO.mean_R2, .38, color=BLUE, edgecolor=SURF, linewidth=2, label="mean over benchmarks"); ax.bar(x + .2, HO.min_R2, .38, color=ORANGE, edgecolor=SURF, linewidth=2, label="worst benchmark")
ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks(x); ax.set_xticklabels([textwrap.fill(s, 14) for s in HO.index], fontsize=8); ax.set(ylabel="Held-out R-squared", title=W("Out of sample, the worst-benchmark loss is worse on average AND in the worst case", 70)); ax.legend(frameon=False, fontsize=9); save(fig, "fig13_loss_heldout")
rk0 = pd.Series(eci).rank().to_numpy(); nob = obs.sum(1); fig, ax = plt.subplots(figsize=(8, 4.8)); rng = np.random.default_rng(0)
for nm, c, lab in (("MSE (cells)", BLUE, "MSE (cells)"), ("worst-benchmark p=64", ORANGE, "worst-benchmark p=64")):
    sh = np.abs(pd.Series(SC[nm].to_numpy()).rank().to_numpy() - rk0); ax.scatter(nob + rng.uniform(-.25, .25, n), sh, s=16, color=c, alpha=.8, label=lab)
ax.set(xlabel="Number of benchmarks the model was evaluated on", ylabel="Absolute rank shift vs official ECI (of 191)", title=W("The largest rank shifts are among models with few scores", 60)); ax.legend(frameon=False, fontsize=9); save(fig, "fig14_who_moves")
sl = pd.read_csv(f"{op}/spec_logit/benchmark_slopes.csv").sort_values("slope_on_FE"); fig, ax = plt.subplots(figsize=(7, 8)); ax.barh(range(len(sl)), sl.slope_on_FE, color=BLUE, height=.6, edgecolor=SURF, linewidth=2); ax.axvline(1, color=INK, ls="--", lw=1)
ax.set_yticks(range(len(sl))); ax.set_yticklabels(sl.benchmark, fontsize=8); ax.grid(axis="y", visible=False); ax.set(xlabel="Slope of benchmark (logit) on model quality (two-way FE); dashed = 1", title=W("Benchmarks rise at very different rates with model quality", 50)); save(fig, "fig15_benchmark_slopes")
fig, ax = plt.subplots(figsize=(7.5, 6.5)); im = ax.imshow(CR.values, cmap=SEQ, vmin=.9, vmax=1); ax.grid(False); ax.set_xticks(range(len(CR))); ax.set_xticklabels([textwrap.fill(s, 12) for s in CR.columns], fontsize=7, rotation=45, ha="right"); ax.set_yticks(range(len(CR))); ax.set_yticklabels(CR.index, fontsize=8)
for i in range(len(CR)):
    for j in range(len(CR)): ax.text(j, i, f"{CR.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK if CR.values[i, j] < .975 else "white")
ax.set_title(W("MSE, MAE and ECI give nearly the same scores; only worst-benchmark loss departs", 60)); plt.colorbar(im, shrink=.7, label="correlation of scores"); save(fig, "fig16_solution_correlations")
print("done")
