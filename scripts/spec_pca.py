"""PCA under different demeaning specifications, and how the official ECI relates to the components.

Usage: python -I scripts/spec_pca.py <extracted_dir> <out_dir> [logit|raw]

Specifications (X = logit score by default; NaN = not evaluated):
  A  benchmark-demeaned   X - benchmark mean                       (standard PCA; keeps model quality)
  B  model-demeaned       X - model mean (NO column centering)     (removes model quality, keeps benchmark difficulty)
  C  two-way FE residual  X - (model effect + benchmark effect)    (removes both; two-step)
  CJ joint                model + benchmark effects and low-rank fit jointly (robustness for C)
  D  slope-adjusted       residual after the rank-1 (benchmark-demeaned) fit: model quality x benchmark slope + benchmark mean removed
Columns are scaled by their RMS after the base is removed. Low-rank fits use iterative SVD (EM) on observed cells.
"""
import sys, os
import textwrap
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ex, out = sys.argv[1], sys.argv[2]
TR = sys.argv[3] if len(sys.argv) > 3 else "logit"
os.makedirs(out, exist_ok=True)
MIN_B, MIN_OBS_MODEL, LEG = 25, 4, 7
SPECS = ["A", "B", "C", "CJ", "D"]
NAMES = {"A": "A: benchmark-demeaned", "B": "B: model-demeaned (uncentered)", "C": "C: two-way FE residual", "CJ": "CJ: joint FE + low-rank", "D": "D: FE + benchmark slopes (rank-1 removed)"}
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "CJ": "#6250d6", "D": "#e87ba4"}
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])
plt.rcParams.update({"axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False})

# ---------- data ----------
d = pd.read_csv(f"{ex}/epoch_capabilities_index/processed_data_for_eci.csv")
meta = pd.read_csv(f"{ex}/benchmark_metadata.csv").set_index("benchmark")
E = pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv").set_index("Model")
d = d[~d.benchmark.isin(meta[meta.superseded_by.isin(set(d.benchmark))].index)]
M = d.pivot_table(index="model", columns="benchmark", values="performance")
M = M.loc[:, M.notna().sum() >= MIN_B]
O = M.notna().astype(int).T @ M.notna().astype(int)
med = pd.Series({b: O.loc[b].drop(b).median() for b in O.index})
M = M.drop(columns=med[med < LEG].index)          # legacy benchmarks: no overlap with modern ones
for _ in range(5):
    M = M.loc[M.notna().sum(axis=1) >= MIN_OBS_MODEL]
    M = M.loc[:, M.notna().sum() >= 15]
Mc = M.clip(.01, .99)
X = np.log(Mc / (1 - Mc)) if TR == "logit" else M
Xv = X.to_numpy(float); obs = ~np.isnan(Xv); n, m = Xv.shape
eci = E["eci"].reindex(M.index).to_numpy(float)
date = pd.to_datetime(E["date"].reindex(M.index)).astype("int64").to_numpy() / 86400e9 / 365.25
closed = (E["Accessibility group"].reindex(M.index) == "Closed weights").to_numpy(float)
nobs = obs.sum(1)
print(f"[{TR}] matrix {n} models x {m} benchmarks, density {obs.mean():.3f}, obs/model median {np.median(nobs):.0f}")


# ---------- estimators ----------
def fe(Xo, ob, iters=500):
    a = np.zeros(Xo.shape[0]); b = np.zeros(Xo.shape[1]); rc = ob.sum(1).clip(1); cc = ob.sum(0).clip(1)
    for _ in range(iters):
        a2 = np.where(ob, Xo - b[None, :], 0).sum(1) / rc
        b2 = np.where(ob, Xo - a2[:, None], 0).sum(0) / cc
        done = max(abs(a2 - a).max(), abs(b2 - b).max()) < 1e-9
        a, b = a2, b2
        if done: break
    return a, b


def base(Xo, ob, spec):
    r, c = Xo.shape
    if spec == "A": return np.tile((Xo * ob).sum(0) / ob.sum(0).clip(1), (r, 1))
    if spec == "B": return np.tile(((Xo * ob).sum(1) / ob.sum(1).clip(1))[:, None], (1, c))
    a, b = fe(Xo, ob); return a[:, None] + b[None, :]


def fit(Xin, spec, k, iters=400, tol=1e-6, base_ov=None):
    ob = ~np.isnan(Xin); Xo = np.where(ob, Xin, 0.0); R = np.zeros_like(Xo)
    joint = spec == "CJ"; Bf = base_ov if base_ov is not None else (None if joint else base(Xo, ob, spec))
    iters = 150 if joint else iters
    def parts(R):
        B = base(np.where(ob, Xo - R, 0.0), ob, "C") if joint else Bf
        D = np.where(ob, Xo - B, 0.0)
        s = np.sqrt((D ** 2).sum(0) / ob.sum(0).clip(1)); s[s < 1e-8] = 1
        return B, s, np.where(ob, D, R) / s
    for _ in range(iters if k > 0 else 0):
        B, s, Ez = parts(R)
        U, sv, Vt = np.linalg.svd(Ez, full_matrices=False)
        Rn = (U[:, :k] * sv[:k]) @ Vt[:k] * s
        dl = np.abs(Rn - R).max(); R = Rn
        if dl < tol: break
    B, s, Ez = parts(R)
    res = dict(B=B, R=R, s=s, pred=B + R)
    if k > 0:
        U, sv, Vt = np.linalg.svd(Ez, full_matrices=False)
        res.update(scores=U[:, :k] * sv[:k], load=Vt[:k].T, sv=sv[:k])
    return res


def cv(Xin, pred_fn, ks, folds=5, reps=2):
    rng = np.random.default_rng(0); idx = np.argwhere(~np.isnan(Xin)); sse = {k: 0.0 for k in ks}; cnt = 0
    for _ in range(reps):
        for part in np.array_split(idx[rng.permutation(len(idx))], folds):
            Xt = Xin.copy(); Xt[part[:, 0], part[:, 1]] = np.nan
            ro = (~np.isnan(Xt)).sum(1); co = (~np.isnan(Xt)).sum(0)
            pt = part[(ro[part[:, 0]] >= 2) & (co[part[:, 1]] >= 5)]
            for k in ks:
                P = pred_fn(Xt, k); sse[k] += ((P[pt[:, 0], pt[:, 1]] - Xin[pt[:, 0], pt[:, 1]]) ** 2).sum()
            cnt += len(pt)
    return {k: sse[k] / cnt for k in ks}


def eci_pred(Xt, k, quad=False):
    z = (eci - eci.mean()) / eci.std(); P = np.zeros_like(Xt)
    Dm = np.c_[np.ones(n), z, z ** 2] if quad else np.c_[np.ones(n), z]
    for j in range(Xt.shape[1]):
        o = ~np.isnan(Xt[:, j]); beta = np.linalg.lstsq(Dm[o], Xt[o, j], rcond=None)[0]; P[:, j] = Dm @ beta
    return P


def corr(a, b, mask=None):
    mask = np.ones(len(a), bool) if mask is None else mask
    a, b = a[mask], b[mask]; r = np.corrcoef(a, b)[0, 1]; rs = pd.Series(a).corr(pd.Series(b), method="spearman")
    z = np.arctanh(r); se = 1 / np.sqrt(len(a) - 3)
    return r, rs, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def resid_quad(y, x):
    return y - np.polyval(np.polyfit(x, y, 2), x)


def fit_s(Xin, spec, k):
    """fit any spec; D uses the rank-1 benchmark-demeaned fit as its base"""
    return fit(Xin, "D", k, base_ov=fit(Xin, "A", 1)["pred"]) if spec == "D" else fit(Xin, spec, k)


# ---------- 1. cross-validated predictive power ----------
KS = list(range(0, 7))
cvres = {s: cv(Xv, lambda Xt, k, s=s: fit_s(Xt, s, k)["pred"], KS) for s in SPECS}
cv_eci = {"ECI linear": cv(Xv, lambda Xt, k: eci_pred(Xt, k), [1])[1], "ECI quadratic": cv(Xv, lambda Xt, k: eci_pred(Xt, k, True), [1])[1]}
tot = float(np.nanvar(Xv))
cvt = pd.DataFrame(cvres).round(3); cvt.index.name = "k"
print("\nHeld-out MSE by rank k (lower is better); total variance of X = %.3f" % tot); print(cvt.to_string())
print("ECI-as-summary CV (LEAKY: ECI was fit on the held-out cells, do not compare):", {k: round(v, 3) for k, v in cv_eci.items()})
cvt.to_csv(f"{out}/cv_mse.csv"); pd.Series(cv_eci).to_csv(f"{out}/cv_mse_eci.csv")
kcv = {s: min(cvres[s], key=cvres[s].get) for s in SPECS}; print("CV-best k:", kcv)

# ---------- 2. Q1: how is ECI not the first PC? ----------
f1 = fit(Xv, "A", 1); pc1 = f1["scores"][:, 0]; pc1 = pc1 if np.corrcoef(pc1, eci)[0, 1] > 0 else -pc1
a_fe, b_fe = fe(np.where(obs, Xv, 0), obs)
r1 = corr(pc1, eci); ra = corr(a_fe, eci)
print(f"\nQ1  ECI vs PC1 (rank-1 benchmark-demeaned): r={r1[0]:.3f} [{r1[2]:.2f},{r1[3]:.2f}] spearman={r1[1]:.3f}  R2={r1[0]**2:.3f}")
print(f"    ECI vs two-way-FE model effect: r={ra[0]:.3f} spearman={ra[1]:.3f}")
q = np.polyfit(pc1, eci, 2); print(f"    R2 of ECI on PC1 + PC1^2: {1 - resid_quad(eci, pc1).var() / eci.var():.3f}")
terc = pd.qcut(nobs, 3, duplicates="drop", labels=False)
for t in np.unique(terc):
    mk = terc == t; print(f"    coverage tercile {t} (obs/model {nobs[mk].min()}-{nobs[mk].max()}, n={mk.sum()}): r(ECI,PC1)={corr(pc1, eci, mk)[0]:.3f}")
rs = resid_quad(eci, pc1)
pd.Series(pc1, index=M.index, name="pc1_rank1").to_csv(f"{out}/scores_A_rank1.csv")
easy = np.array([b_fe[obs[i]].mean() for i in range(n)])    # mean easiness of benchmarks the model took
print("    ECI residual (after PC1 quadratic) correlates with:")
q1rows = []
for nm, v in [("# benchmarks evaluated", nobs.astype(float)), ("mean easiness of benchmarks taken", easy), ("release date (yrs)", date), ("closed weights", closed)]:
    r = corr(v, rs); print(f"      {nm:36s} r={r[0]:+.2f} [{r[2]:+.2f},{r[3]:+.2f}]"); q1rows.append((nm, r[0], r[2], r[3]))
pd.DataFrame(q1rows, columns=["variable", "r", "lo", "hi"]).to_csv(f"{out}/q1_eci_residual_correlates.csv", index=False)
pr = []
for j, b in enumerate(M.columns):
    o = obs[:, j]
    if o.sum() < 15: continue
    sres = Xv[o, j] - np.polyval(np.polyfit(pc1[o], Xv[o, j], 1), pc1[o])
    r = np.corrcoef(sres, rs[o])[0, 1]; pr.append((b, r, int(o.sum())))
pr = pd.DataFrame(pr, columns=["benchmark", "partial_r_with_ECI_resid", "n"]).sort_values("partial_r_with_ECI_resid")
pr.round(3).to_csv(f"{out}/q1_benchmark_partial_corr.csv", index=False)
print("    benchmarks where models doing better than PC1 predicts also have HIGHER ECI than PC1 predicts (top):\n",
      pr.tail(6).round(2).to_string(index=False), "\n    LOWEST:\n", pr.head(6).round(2).to_string(index=False))

# ---------- 3. components under each specification ----------
KSHOW = 3
fits = {s: fit_s(Xv, s, KSHOW) for s in SPECS}
ordr = np.argsort(b_fe)                                  # hardest -> easiest benchmarks
L = {}; S = {}
for s in SPECS:
    Ls, Ss = fits[s]["load"].copy(), fits[s]["scores"].copy()
    for j in range(KSHOW):
        if (s, j) == ("A", 0) or (s == "B" and j == 0):
            sg = 1 if Ls[:, j].sum() > 0 else -1
        else:
            sg = 1 if Ls[np.abs(Ls[:, j]).argmax(), j] > 0 else -1
        Ls[:, j] *= sg; Ss[:, j] *= sg
    L[s], S[s] = Ls, Ss
    pd.DataFrame(Ls, index=M.columns, columns=[f"PC{j+1}" for j in range(KSHOW)]).assign(n_models=obs.sum(0)).round(3).to_csv(f"{out}/loadings_{s}.csv")
    pd.DataFrame(Ss, index=M.index, columns=[f"PC{j+1}" for j in range(KSHOW)]).to_csv(f"{out}/scores_{s}.csv")
    tv = (fits[s]["sv"] ** 2) / (fits[s]["sv"] ** 2).sum()
    print(f"\n{NAMES[s]}  (CV-best k={kcv[s]})")
    for j in range(KSHOW):
        o = np.argsort(Ls[:, j]); print(f"  PC{j+1}  +: " + ", ".join(f"{M.columns[i]} {Ls[i,j]:.2f}" for i in o[::-1][:5]) + "\n        -: " + ", ".join(f"{M.columns[i]} {Ls[i,j]:.2f}" for i in o[:5]))

# robustness: cosine between EM loadings of C and CJ
print("\ncosine C vs CJ loadings:", [round(abs(float(L["C"][:, j] @ L["CJ"][:, j])), 2) for j in range(KSHOW)])

# ---------- 4. Is there real structure in the FE residuals? (parallel analysis with FE-aware null) ----------
def res_eig(Xr):
    C = pd.DataFrame(Xr).corr(min_periods=12).fillna(0).to_numpy(copy=True); np.fill_diagonal(C, 1)
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-6, None); C = (V * w) @ V.T; s_ = np.sqrt(np.diag(C)); C /= np.outer(s_, s_)
    return np.sort(np.linalg.eigvalsh(C))[::-1]


Xo = np.where(obs, Xv, 0); a0, b0 = fe(Xo, obs); B0 = a0[:, None] + b0[None, :]
R0 = np.where(obs, Xv - B0, np.nan); w_real = res_eig(R0)
rng = np.random.default_rng(1); nulls = []
for _ in range(100):
    Rp = R0.copy()
    for j in range(m):
        o = obs[:, j]; Rp[o, j] = rng.permutation(R0[o, j])
    Xs = np.where(obs, B0 + np.nan_to_num(Rp), 0); a1, b1 = fe(Xs, obs); nulls.append(res_eig(np.where(obs, Xs - (a1[:, None] + b1[None, :]), np.nan)))
n95 = np.percentile(nulls, 95, axis=0)
print("\nFE-residual correlation matrix: eigenvalues vs null (null re-fits FE, so it includes FE-induced bias)")
for i in range(6): print(f"  eig{i+1}: {w_real[i]:.2f}   null95 {n95[i]:.2f}   {'ABOVE' if w_real[i] > n95[i] else ''}")

# ---------- 5. Q2: ECI vs each component ----------
ra_res = resid_quad(eci, a_fe)
rows = []
print("\nQ2  correlation of ECI with model scores on each component (n=%d). 'ECI|FE' = ECI after removing its quadratic relation to the model FE" % n)
for s in SPECS:
    for j in range(KSHOW):
        r = corr(S[s][:, j], eci); r2 = corr(S[s][:, j], ra_res); rd = corr(S[s][:, j], date)
        dres = lambda v: v - np.polyval(np.polyfit(date, v, 1), date)
        r3 = corr(S[s][:, j], rs); r4 = corr(dres(S[s][:, j]), dres(rs))
        rows.append((s, f"PC{j+1}", r[0], r[2], r[3], r2[0], r2[2], r2[3], rd[0], r3[0], r3[2], r3[3], r4[0], r4[2], r4[3]))
q2 = pd.DataFrame(rows, columns=["spec", "pc", "r_ECI", "lo", "hi", "r_ECIgivenFE", "lo2", "hi2", "r_date", "r_ECIgivenPC1", "lo3", "hi3", "r_ECIgivenPC1_date", "lo4", "hi4"]); q2.round(3).to_csv(f"{out}/q2_eci_vs_components.csv", index=False)
print(q2.round(2).to_string(index=False))

# ---------- 6. in-sample information: ECI as a 1-D summary vs data-driven rank-k ----------
cm = np.where(obs, Xv, 0).sum(0) / obs.sum(0); ss0 = float(((Xv - cm) ** 2)[obs].sum())
def r2_of(P): return 1 - float(((Xv - P) ** 2)[obs].sum()) / ss0
z_ = (eci - eci.mean()) / eci.std()
ins = {"benchmark means only": r2_of(np.tile(cm, (n, 1))), "FE additive (k=0 C)": r2_of(fit(Xv, "C", 0)["pred"]),
       "ECI, per-benchmark linear": r2_of(eci_pred(Xv, 1)), "ECI, per-benchmark quadratic": r2_of(eci_pred(Xv, 1, True))}
for k in (1, 2, 3, 4): ins[f"PCA rank-{k} (A)"] = r2_of(fit(Xv, "A", k)["pred"])
print("\nIn-sample share of within-benchmark variance explained (observed cells):")
for k_, v_ in ins.items(): print(f"  {k_:32s} {v_:.3f}")
pd.Series(ins).round(4).to_csv(f"{out}/insample_r2.csv")

# ---------- 7. diagnostics: are C's components just benchmark-slope heterogeneity? ----------
gam = np.array([np.polyfit(a_fe[obs[:, j]], Xv[obs[:, j], j], 1)[0] for j in range(m)])   # slope of each benchmark on model effect
print("\nbenchmark slopes on model FE: min %.2f median %.2f max %.2f" % (gam.min(), np.median(gam), gam.max()))
for s_ in ("C", "CJ", "D"):
    print(f"  {s_}: corr(loading_j, slope_b) =", [round(float(np.corrcoef(L[s_][:, j], gam)[0, 1]), 2) for j in range(KSHOW)],
          "| corr(score_j, model FE) =", [round(float(np.corrcoef(S[s_][:, j], a_fe)[0, 1]), 2) for j in range(KSHOW)])
pd.DataFrame({"benchmark": M.columns, "slope_on_FE": gam, "easiness": b_fe}).round(3).to_csv(f"{out}/benchmark_slopes.csv", index=False)
LA1 = fit(Xv, "A", 1)["load"][:, 0]; LA1 = LA1 if LA1.sum() > 0 else -LA1
pd.Series(LA1, index=M.columns).round(3).sort_values().to_csv(f"{out}/loadings_A_rank1.csv"); print("rank-1 PC1 loading range: %.2f to %.2f" % (LA1.min(), LA1.max()))

# ---------- 8. bootstrap stability of loadings (resample models) ----------
rngb = np.random.default_rng(2); stab = {s_: [] for s_ in ("A", "C", "D")}
for _ in range(60):
    ib = rngb.choice(n, n); Xb = Xv[ib]
    if (np.isnan(Xb).all(0)).any(): continue
    for s_ in stab:
        Lb = fit_s(Xb, s_, KSHOW)["load"]; Cm = np.abs(L[s_].T @ Lb); stab[s_].append(Cm.max(1))
print("\nBootstrap stability: mean best-match |cosine| of each full-data component with bootstrap components")
for s_ in stab: print(f"  {s_}: ", np.round(np.mean(stab[s_], 0), 2), " 10th pct:", np.round(np.percentile(stab[s_], 10, 0), 2))

# ---------- figures ----------
# F1: CV
fig, ax = plt.subplots(figsize=(8, 5))
for s in SPECS: ax.plot(KS, [cvres[s][k] for k in KS], "o-", color=COL[s], lw=2, ms=6, label=NAMES[s])
ax.set(xlabel="Number of components k (k=0: base model only; D k=0 is rank-1)", ylabel="Held-out MSE (logit scale)", title="How much does each extra component predict held-out scores?")
ax.legend(frameon=False, fontsize=9); plt.tight_layout(); plt.savefig(f"{out}/fig_cv.png", dpi=150); plt.close()
# F2: ECI vs PC1
fig, ax = plt.subplots(figsize=(7, 5.5)); sc = ax.scatter(pc1, eci, c=nobs, cmap="Blues", s=28, edgecolor="#555", linewidth=.3, vmin=0)
xs = np.linspace(pc1.min(), pc1.max(), 100); ax.plot(xs, np.polyval(q, xs), color=INK, lw=1.2)
ax.set(xlabel="PC1 score (rank-1, benchmark-demeaned)", ylabel="Official ECI", title=f"ECI vs first PC of benchmarks  (r = {r1[0]:.2f})"); plt.colorbar(sc, label="# benchmarks evaluated")
plt.tight_layout(); plt.savefig(f"{out}/fig_eci_vs_pc1.png", dpi=150); plt.close()
# F3: loadings heatmaps
fig, axs = plt.subplots(1, 5, figsize=(17, 9), sharey=True)
for ax, s in zip(axs, SPECS):
    ax.grid(False); im = ax.imshow(L[s][ordr], cmap=DIV, vmin=-.6, vmax=.6, aspect="auto")
    ax.set_xticks(range(KSHOW)); ax.set_xticklabels([f"PC{j+1}" for j in range(KSHOW)]); ax.set_title(textwrap.fill(NAMES[s], 20), fontsize=9)
axs[0].set_yticks(range(m)); axs[0].set_yticklabels(M.columns[ordr], fontsize=8)
fig.suptitle("Loadings; benchmarks ordered hardest (top) to easiest (bottom)", fontsize=11); fig.colorbar(im, ax=axs, shrink=.4, label="loading")
plt.savefig(f"{out}/fig_loadings.png", dpi=150, bbox_inches="tight"); plt.close()
# F4: eigenvalues vs null
fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(range(1, 11), w_real[:10], "o-", color=COL["C"], lw=2, label="FE residuals"); ax.plot(range(1, 11), n95[:10], "s--", color=MUTED, label="null 95th pct")
ax.set(xlabel="Component", ylabel="Eigenvalue", title="Spec C: structure beyond model + benchmark effects?"); ax.legend(frameon=False); plt.tight_layout(); plt.savefig(f"{out}/fig_fe_eigs.png", dpi=150); plt.close()
# F5: ECI vs components (C and A)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, key, ttl in [(axs[0], "r_ECI", "ECI"), (axs[1], "r_ECIgivenPC1", "ECI after removing its relation to PC1"), (axs[2], "r_ECIgivenPC1_date", "... and with release date partialled out of both")]:
    for si, s in enumerate(SPECS):
        sub = q2[q2.spec == s]; x = np.arange(KSHOW) + (si - 2) * .15
        lo, hi = {"r_ECI": ("lo", "hi"), "r_ECIgivenPC1": ("lo3", "hi3")}.get(key, ("lo4", "hi4"))
        ax.errorbar(x, sub[key], yerr=[sub[key] - sub[lo], sub[hi] - sub[key]], fmt="o", color=COL[s], ms=6, lw=1.2, capsize=2, label=NAMES[s])
    ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks(range(KSHOW)); ax.set_xticklabels([f"PC{j+1}" for j in range(KSHOW)]); ax.set_title(ttl, fontsize=10)
axs[0].set_ylabel("Correlation (95% CI)"); axs[0].legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig(f"{out}/fig_eci_vs_components.png", dpi=150); plt.close()
print("\nfigures written to", out)
