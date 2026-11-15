"""(1) How much of the benchmark data does PC1 explain?  (2) What is not explained by the official ECI, and does it have structure?
Usage: python -I scripts/variance_and_eci_resid.py <extracted_dir> <out_dir>

Spec E ("ECI removed"): for each benchmark fit score = floor + (1 - floor) * sigmoid(a (ECI - c)) on observed models,
take residuals, standardise columns, and run low-rank / PCA on the residuals.  Run on raw scores; the other pieces use logits.
"""
import sys, os, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from common import *

ex, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
DIV = LinearSegmentedColormap.from_list("div", [ORANGE, "#f0efec", BLUE])
plt.rcParams.update({"axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False})

Dd = load(ex)
M, Xv, Mv, obs, eci, date, floor = Dd.M, Dd.Xv, Dd.Mv, Dd.obs, Dd.eci, Dd.date, Dd.floor
n, m = Xv.shape; bench = list(M.columns); N = int(obs.sum()); KS = list(range(0, 7))
print(f"sample: {n} models x {m} benchmarks, {obs.mean():.3f} observed, {N} scores")

# ============ 1. variance in the data explained by PC1 ============
cm = np.where(obs, Xv, 0).sum(0) / obs.sum(0)
sd = np.sqrt(np.where(obs, (Xv - cm) ** 2, 0).sum(0) / obs.sum(0))
r2_pool = lambda P: 1 - ((Xv - P) ** 2)[obs].sum() / ((Xv - cm) ** 2)[obs].sum()
r2_std = lambda P: 1 - (((Xv - P) / sd) ** 2)[obs].sum() / N          # every benchmark gets equal weight
rows = [("benchmark means only", 0.0, 0.0)]
for k in (1, 2, 3, 4):
    P = fit(Xv, "A", k)["pred"]; rows.append((f"A rank-{k} (PCA, observed benchmark means)", r2_std(P), r2_pool(P)))
for k in (1, 2, 3, 4):
    P = fit(Xv, "AJ", k)["pred"]; rows.append((f"AJ rank-{k} (PCA, intercepts estimated jointly)", r2_std(P), r2_pool(P)))
P = fit(Xv, "C", 0)["pred"]; rows.append(("two-way FE additive (no slopes)", r2_std(P), r2_pool(P)))
for nm, q in (("linear", False), ("quadratic", True)):
    P = ols_on_eci(Xv, eci, q); rows.append((f"official ECI, per-benchmark OLS {nm}", r2_std(P), r2_pool(P)))
t1 = pd.DataFrame(rows, columns=["model", "R2_in_sample_equal_weight", "R2_in_sample_pooled_logit"]).set_index("model")
print("\nIN-SAMPLE share of within-benchmark variance explained (logit scores; observed cells)"); print(t1.round(3).to_string())
t1.round(4).to_csv(f"{out}/variance_explained_insample.csv")

cvA = cv(Xv, lambda Xt, k: fit(Xt, "A", k)["pred"], KS); cvAJ = cv(Xv, lambda Xt, k: fit(Xt, "AJ", k)["pred"], KS)
cvC = cv(Xv, lambda Xt, k: fit(Xt, "C", k)["pred"], [0, 1, 2])
base0 = cvA[0]
t2 = pd.DataFrame({"A (fixed means)": {k: 1 - cvA[k] / base0 for k in KS}, "AJ (joint intercepts)": {k: 1 - cvAJ[k] / base0 for k in KS}}); t2.index.name = "k"
print("\nHELD-OUT R2 vs the benchmark-mean model (5-fold CV x2, pooled logit MSE)"); print(t2.round(3).to_string())
print("two-way FE additive + k comps, held-out R2:", {k: round(1 - cvC[k] / base0, 3) for k in cvC})
t2.round(4).to_csv(f"{out}/variance_explained_heldout.csv")

Z = np.where(obs, (Xv - cm) / sd, np.nan); w = pairwise_eigs(Z)
print("\nclassical view: pairwise-correlation eigenvalues, share of total variance:", (w / w.sum())[:5].round(3))

# ============ 2. ECI vs PC1, with and without jointly estimated intercepts ============
nobs = obs.sum(1); terc = pd.qcut(nobs, 3, labels=False, duplicates="drop")
print("\nECI vs rank-1 PC1")
pcs = {}
for lab, sp in (("A  fixed means", "A"), ("AJ joint intercepts", "AJ")):
    s_ = fit(Xv, sp, 1)["scores"][:, 0]; s_ = s_ if np.corrcoef(s_, eci)[0, 1] > 0 else -s_; pcs[sp] = s_
    r = corr(s_, eci); print(f"  {lab:22s} r={r[0]:.3f} [{r[2]:.3f},{r[3]:.3f}] spearman={r[1]:.3f} | by coverage tercile:",
                            [round(corr(s_, eci, terc == t)[0], 3) for t in np.unique(terc)])

# ============ 3. what the ECI does not explain ============
cmr = cm_raw = np.where(obs, Mv, 0).sum(0) / obs.sum(0)
Pe = logistic_on_eci(Mv, eci, floor); R0 = np.where(obs, Mv - Pe, np.nan)
sst_c = np.where(obs, (Mv - cm_raw) ** 2, 0).sum(0); sse_c = np.where(obs, (Mv - Pe) ** 2, 0).sum(0)
r2_c = 1 - sse_c / sst_c
Pl = ols_on_eci(Xv, eci); sst_l = np.where(obs, (Xv - cm) ** 2, 0).sum(0); r2_l = 1 - np.where(obs, (Xv - Pl) ** 2, 0).sum(0) / sst_l
r2_a1 = 1 - np.where(obs, (Xv - fit(Xv, "A", 1)["pred"]) ** 2, 0).sum(0) / sst_l
r2_aj1 = 1 - np.where(obs, (Xv - fit(Xv, "AJ", 1)["pred"]) ** 2, 0).sum(0) / sst_l
pb = pd.DataFrame({"n_models": obs.sum(0), "R2_ECI_logistic_raw": r2_c, "R2_ECI_OLS_logit": r2_l, "R2_PCA_rank1_A_logit": r2_a1, "R2_PCA_rank1_AJ_logit": r2_aj1}, index=bench).sort_values("R2_ECI_logistic_raw")
pb.round(3).to_csv(f"{out}/per_benchmark_variance_explained.csv")
print("\nShare of each benchmark's variance across models explained by the official ECI (logistic, raw scores) and by rank-1 PCA (logit):")
print(pb.round(2).to_string())
print(f"pooled raw-score R2 of ECI logistic: {1 - sse_c.sum() / sst_c.sum():.3f};  equal-weight mean: {r2_c.mean():.3f}")

cache = cached(lambda Xt: logistic_on_eci(Xt, eci, floor))
cvE = cv(Mv, lambda Xt, k: fit(Xt, "E", k, base_ov=cache(Xt))["pred"], KS)
cvR = cv(Mv, lambda Xt, k: fit(Xt, "A", 0)["pred"], [0])[0]
print(f"\nSpec E held-out MSE (raw scores): benchmark means {cvR:.4f}; ECI-logistic only {cvE[0]:.4f} (held-out R2 {1 - cvE[0] / cvR:.3f}, optimistic: ECI saw these cells)")
print("  + k components of the ECI-residual: MSE", {k: round(v, 4) for k, v in cvE.items()}, "\n  share of the ECI-residual variance predictable out of sample:", {k: round(1 - cvE[k] / cvE[0], 3) for k in KS})
pd.DataFrame({"MSE": cvE, "gain_vs_ECI_only": {k: 1 - cvE[k] / cvE[0] for k in KS}}).round(5).to_csv(f"{out}/E_cv.csv")

fE = fit(Mv, "E", 3, base_ov=Pe); LE, SE = fE["load"].copy(), fE["scores"].copy()
for j in range(3):
    sg = 1 if LE[np.abs(LE[:, j]).argmax(), j] > 0 else -1; LE[:, j] *= sg; SE[:, j] *= sg
Dres = np.where(obs, Mv - Pe, 0); sres = np.sqrt((Dres ** 2).sum(0) / obs.sum(0))
print("in-sample share of ECI-residual variance (equal weight) explained by k components:",
      {k: round(1 - (((Dres - fit(Mv, "E", k, base_ov=Pe)["R"]) / sres) ** 2)[obs].sum() / N, 3) for k in (1, 2, 3)})
pd.DataFrame(LE, index=bench, columns=["E1", "E2", "E3"]).assign(n=obs.sum(0)).round(3).to_csv(f"{out}/E_loadings.csv")
pd.DataFrame(SE, index=M.index, columns=["E1", "E2", "E3"]).to_csv(f"{out}/E_scores.csv")
print("\nComponents of the ECI-residual (loadings; sign set so the largest is +)")
for j in range(3):
    o = np.argsort(LE[:, j]); print(f"  E{j+1} +: " + ", ".join(f"{bench[i]} {LE[i,j]:.2f}" for i in o[::-1][:6]) + "\n      -: " + ", ".join(f"{bench[i]} {LE[i,j]:.2f}" for i in o[:6]))
print("  corr of E scores with release date:", [round(corr(SE[:, j], date)[0], 2) for j in range(3)], "| with ECI:", [round(corr(SE[:, j], eci)[0], 2) for j in range(3)])

# eigenvalue test with permutation null
wr = pairwise_eigs(R0); rng = np.random.default_rng(1); nulls = []
for _ in range(100):
    Rp = R0.copy()
    for j in range(m): o = obs[:, j]; Rp[o, j] = rng.permutation(R0[o, j])
    nulls.append(pairwise_eigs(Rp))
n95 = np.percentile(nulls, 95, axis=0)
print("\nECI-residual correlation matrix, eigenvalues vs shuffled-null 95th pct:")
for i in range(6): print(f"  eig{i+1}: {wr[i]:.2f}  null {n95[i]:.2f} {'ABOVE' if wr[i] > n95[i] else ''}")

# stability and comparison with the slope-adjusted PCA residual (D / DJ)
rngb = np.random.default_rng(2); st = []
for _ in range(60):
    ib = rngb.choice(n, n); Xb = Mv[ib]
    if np.isnan(Xb).all(0).any(): continue
    Lb = fit(Xb, "E", 3, base_ov=logistic_on_eci(Xb, eci[ib], floor))["load"]; st.append(np.abs(LE.T @ Lb).max(1))
print("bootstrap stability of E1..E3 (mean best-match |cosine|):", np.round(np.mean(st, 0), 2), " 10th pct:", np.round(np.percentile(st, 10, 0), 2))
fD = fit(Xv, "D", 3, base_ov=fit(Xv, "A", 1)["pred"]); fDJ = fit(Xv, "DJ", 3, base_ov=fit(Xv, "AJ", 1)["pred"])
for nm, f_ in (("D (PCA rank-1 with fixed means removed)", fD), ("DJ (PCA rank-1 with joint intercepts removed)", fDJ)):
    print(f"|cosine| of E1..E3 loadings (rows) with {nm} components (cols):\n", np.round(np.abs(LE.T @ f_["load"]), 2))
    print("  |corr| of model scores:\n", np.round(np.abs(np.corrcoef(SE.T, f_["scores"].T)[:3, 3:]), 2))
pd.DataFrame(fDJ["load"], index=bench, columns=["DJ1", "DJ2", "DJ3"]).round(3).to_csv(f"{out}/DJ_loadings.csv")

# ============ figures ============
# F1: per-benchmark explained vs unexplained by ECI
o = pb.sort_values("R2_ECI_logistic_raw", ascending=False)
fig, ax = plt.subplots(figsize=(8, 9)); y = np.arange(len(o)); ex_ = o["R2_ECI_logistic_raw"].clip(0, 1).values
ax.barh(y, ex_, color=BLUE, height=.6, edgecolor=SURF, linewidth=2, label="explained by ECI"); ax.barh(y, 1 - ex_, left=ex_, color="#f3c1ad", height=.6, edgecolor=SURF, linewidth=2, label="not explained by ECI")
ax.set_yticks(y); ax.set_yticklabels(o.index, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
ax.set_xlabel("Share of variance across models (raw scores)"); ax.set_title("How much of each benchmark does the official ECI explain?"); ax.legend(frameon=False, fontsize=9, loc="lower left")
plt.tight_layout(); plt.savefig(f"{out}/fig_eci_explained_by_benchmark.png", dpi=150); plt.close()
# F2: variance explained vs k
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
ks_ = [1, 2, 3, 4]
axs[0].plot(ks_, [t1.R2_in_sample_equal_weight[f"A rank-{k} (PCA, observed benchmark means)"] for k in ks_], "o-", color=BLUE, lw=2, label="A: observed benchmark means")
axs[0].plot(ks_, [t1.R2_in_sample_equal_weight[f"AJ rank-{k} (PCA, intercepts estimated jointly)"] for k in ks_], "o-", color=VIOLET, lw=2, label="AJ: intercepts estimated jointly")
e_ = t1.R2_in_sample_equal_weight["official ECI, per-benchmark OLS linear"]; axs[0].axhline(e_, color=INK, lw=1.2, ls="--"); axs[0].text(4, e_, "official ECI (1-D)", ha="right", va="bottom", fontsize=9)
axs[0].set(xlabel="Number of components", ylabel="R-squared, in sample", title="Share of data variance explained", ylim=(0, 1)); axs[0].legend(frameon=False, fontsize=9, loc="lower right")
axs[1].plot(KS, [1 - cvA[k] / base0 for k in KS], "o-", color=BLUE, lw=2, label="A"); axs[1].plot(KS, [1 - cvAJ[k] / base0 for k in KS], "o-", color=VIOLET, lw=2, label="AJ")
axs[1].set(xlabel="Number of components", ylabel="R-squared, held-out cells", title="Same, out of sample"); axs[1].legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig(f"{out}/fig_variance_explained.png", dpi=150); plt.close()
# F3: eigenvalues
fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(range(1, 11), wr[:10], "o-", color=AQUA, lw=2, label="ECI residuals"); ax.plot(range(1, 11), n95[:10], "s--", color=MUTED, label="shuffled null, 95th pct")
ax.set(xlabel="Component", ylabel="Eigenvalue", title="Structure in what the ECI does not explain?"); ax.legend(frameon=False); plt.tight_layout(); plt.savefig(f"{out}/fig_E_eigs.png", dpi=150); plt.close()
# F4: loadings
oo = np.argsort(r2_c); fig, ax = plt.subplots(figsize=(5, 9)); ax.grid(False)
im = ax.imshow(LE[oo], cmap=DIV, vmin=-.6, vmax=.6, aspect="auto"); ax.set_yticks(range(m)); ax.set_yticklabels(np.array(bench)[oo], fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(["E1", "E2", "E3"]); ax.set_title("Components of ECI residual" + chr(10) + "(top: least explained by ECI)", fontsize=10); plt.colorbar(im, shrink=.4)
plt.tight_layout(); plt.savefig(f"{out}/fig_E_loadings.png", dpi=150); plt.close()
print("\nfigures written to", out)
