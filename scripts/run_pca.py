"""PCA on ECI benchmarks. Usage: python -I scripts/run_pca.py <extracted_dir> <out_dir>"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ex, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
MIN_MODELS, MIN_OVERLAP = 25, 12
rng = np.random.default_rng(0)

d = pd.read_csv(f"{ex}/epoch_capabilities_index/processed_data_for_eci.csv")
meta = pd.read_csv(f"{ex}/benchmark_metadata.csv")
eci = pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv")

# drop benchmarks superseded by another benchmark present in the data
present = set(d.benchmark)
sup = meta[meta.superseded_by.isin(present) & meta.benchmark.isin(present)].benchmark.tolist()
d = d[~d.benchmark.isin(sup)]
print("dropped superseded:", sup)

M = d.pivot_table(index="model", columns="benchmark", values="performance", aggfunc="mean")
M = M.loc[:, M.notna().sum() >= MIN_MODELS]
# connectivity filter: old-era benchmarks share ~no models with the rest, so their
# pairwise correlations are not estimable. Analyse them separately.
O = M.notna().astype(int).T @ M.notna().astype(int)
med_ov = pd.Series({b: O.loc[b].drop(b).median() for b in O.index})
print("median overlap per benchmark (lowest 14):\n", med_ov.sort_values().head(14).to_string())
legacy = med_ov[med_ov < 7].index.tolist()
print("legacy (poorly connected) benchmarks, analysed separately:", legacy)
M_leg = M[legacy]
M = M.drop(columns=legacy)
M = M.loc[M.notna().sum(axis=1) >= 2]
nbad = int((M.notna().astype(int).T @ M.notna().astype(int) < MIN_OVERLAP).to_numpy().sum())
print("benchmark pairs with overlap <", MIN_OVERLAP, ":", nbad)
print("analysis matrix", M.shape, "density", round(M.notna().mean().mean(), 3))


def zs(X):
    return (X - X.mean()) / X.std()


def pairwise_pca(X):
    C = X.corr(min_periods=MIN_OVERLAP)
    C = C.fillna(0).to_numpy(copy=True)
    np.fill_diagonal(C, 1)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)  # nearest-PSD by eigenvalue clipping
    C = (V * w) @ V.T
    s = np.sqrt(np.diag(C)); C = C / np.outer(s, s)
    w, V = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    return w[o], V[:, o]


def em_pca(Z, k, iters=500, mask=None):
    """Iterative SVD on z-scored matrix (NaN = missing)."""
    obs = ~np.isnan(Z) if mask is None else mask
    X = np.where(obs, Z, 0.0)
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        R = (U[:, :k] * s[:k]) @ Vt[:k]
        Xn = np.where(obs, Z, R)
        if np.abs(Xn - X).max() < 1e-6:
            X = Xn; break
        X = Xn
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] * s[:k], Vt[:k].T, s ** 2 / (len(X) - 1)


def cv_k(Z, kmax=6, folds=5):
    idx = np.argwhere(~np.isnan(Z))
    rng.shuffle(idx)
    parts = np.array_split(idx, folds)
    res = {}
    for k in range(0, kmax + 1):
        err = []
        for p in parts:
            Zt = Z.copy(); Zt[p[:, 0], p[:, 1]] = np.nan
            # skip folds that blank a whole row/col
            if (np.isnan(Zt).all(0)).any() or (np.isnan(Zt).all(1)).any():
                continue
            if k == 0:
                R = np.zeros_like(Z)
            else:
                sc, ld, _ = em_pca(Zt, k, iters=200)
                R = sc @ ld.T
            err.append(np.mean((R[p[:, 0], p[:, 1]] - Z[p[:, 0], p[:, 1]]) ** 2))
        res[k] = np.mean(err)
    return res


def run(M, tag):
    Z = zs(M).values
    w, V = pairwise_pca(zs(M))
    cvr = cv_k(Z)
    kbest = min(cvr, key=cvr.get)
    print(f"[{tag}] CV MSE by k:", {k: round(v, 3) for k, v in cvr.items()}, "best k", kbest)
    k = max(kbest, 3)
    sc, ld, ev = em_pca(Z, k)
    # sign convention: PC1 loadings mostly positive
    for j in range(k):
        if ld[:, j].sum() < 0:
            ld[:, j] *= -1; sc[:, j] *= -1
    for j in range(V.shape[1]):
        if V[:, j].sum() < 0:
            V[:, j] *= -1
    # EM loadings are right singular vectors; scale to correlation-like loadings
    load_em = pd.DataFrame(ld, index=M.columns, columns=[f"PC{i+1}" for i in range(k)])
    load_pw = pd.DataFrame(V[:, :k], index=M.columns, columns=[f"PC{i+1}" for i in range(k)])
    scores = pd.DataFrame(sc, index=M.index, columns=load_em.columns)
    return w, load_pw, load_em, scores, cvr, kbest


w, L_pw, L_em, S, cvr, kbest = run(M, "raw")
n = M.notna().sum()
L_pw.assign(n_models=n).round(3).to_csv(f"{out}/loadings_pairwise.csv")
L_em.assign(n_models=n).round(3).to_csv(f"{out}/loadings_em.csv")
S.to_csv(f"{out}/model_scores_em.csv")

print("\nPairwise-corr eigenvalue share:", (w / w.sum())[:6].round(3))
print("Loading agreement pairwise vs EM (|corr|):",
      {c: round(abs(np.corrcoef(L_pw[c], L_em[c])[0, 1]), 3) for c in L_pw.columns})

# validate PC1 against official ECI
e = eci.set_index("Model")
ecol = ["eci"]
print("ECI columns:", eci.columns.tolist()[:8])
try:
    j = S.join(e[ecol[0]], how="inner")
    print("PC1 vs ECI corr:", round(j["PC1"].corr(j[ecol[0]]), 3), "n =", len(j))
except Exception as ex_:
    print("ECI join failed", ex_)

# sensitivity: logit transform
Ml = np.log(M.clip(0.01, 0.99) / (1 - M.clip(0.01, 0.99)))
_, Lp_l, Le_l, _, _, kb_l = run(Ml, "logit")
print("logit PC1/PC2 agreement with raw (pairwise):",
      {c: round(abs(np.corrcoef(L_pw[c], Lp_l[c])[0, 1]), 3) for c in ["PC1", "PC2", "PC3"]})

# --- plots ---
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(range(1, 11), (w / w.sum())[:10], color="#3b6fb6")
ax[0].set(xlabel="Component", ylabel="Variance share", title="Scree (pairwise-corr PCA)")
ks = list(cvr); ax[1].plot(ks, [cvr[k] for k in ks], "o-", color="#3b6fb6")
ax[1].set(xlabel="k components", ylabel="Held-out MSE", title="CV choice of k")
plt.tight_layout(); plt.savefig(f"{out}/scree_cv.png", dpi=150); plt.close()

order = L_pw["PC2"].sort_values().index
fig, ax = plt.subplots(figsize=(6, 13))
im = ax.imshow(L_pw.loc[order, ["PC1", "PC2", "PC3"]].values, cmap="RdBu_r", vmin=-.5, vmax=.5, aspect="auto")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(["PC1", "PC2", "PC3"])
plt.colorbar(im, shrink=.5); ax.set_title("Benchmark loadings (sorted by PC2)")
plt.tight_layout(); plt.savefig(f"{out}/loadings_heatmap.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(L_pw.PC1, L_pw.PC2, s=n.values * 1.5, alpha=.6, color="#3b6fb6")
for b in L_pw.index:
    ax.annotate(b, (L_pw.PC1[b], L_pw.PC2[b]), fontsize=7)
ax.axhline(0, c="gray", lw=.5); ax.set(xlabel="PC1 loading", ylabel="PC2 loading",
                                      title="Benchmarks in PC1/PC2 space (size = #models)")
plt.tight_layout(); plt.savefig(f"{out}/benchmark_map.png", dpi=150); plt.close()

print("\nTop/bottom loadings per component (pairwise):")
for c in ["PC1", "PC2", "PC3"]:
    s_ = L_pw[c].sort_values()
    print(c, "NEG:", list(s_.index[:5]), "\n    POS:", list(s_.index[-5:]))
print("\nPC1 loadings sorted:\n", L_pw.PC1.sort_values().round(2).to_string())

# legacy-era PCA (separate)
Zl = zs(M_leg.loc[M_leg.notna().sum(axis=1) >= 3])
wl, Vl = pairwise_pca(Zl)
print("\nLEGACY benchmarks", Zl.shape, "eigen share:", (wl / wl.sum())[:3].round(3))
if Vl[:, 0].sum() < 0: Vl[:, 0] *= -1
print(pd.Series(Vl[:, 0], index=Zl.columns).round(2).sort_values().to_string())
pd.Series(Vl[:, 0], index=Zl.columns).to_csv(f"{out}/legacy_pc1_loadings.csv")
