"""PCA of benchmark residuals after removing general capability (official ECI) via per-benchmark logistic fit.
Usage: python -I scripts/residual_pca.py <extracted_dir> <out_dir>"""
import sys, numpy as np, pandas as pd
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ex, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(1)
d = pd.read_csv(f"{ex}/epoch_capabilities_index/processed_data_for_eci.csv")
meta = pd.read_csv(f"{ex}/benchmark_metadata.csv").set_index("benchmark")
eci = pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv").set_index("Model")["eci"]
sup = meta[meta.superseded_by.isin(set(d.benchmark))].index
d = d[~d.benchmark.isin(sup)]
M = d.pivot_table(index="model", columns="benchmark", values="performance")
M = M.loc[:, M.notna().sum() >= 25]
O = M.notna().astype(int).T @ M.notna().astype(int)
med = pd.Series({b: O.loc[b].drop(b).median() for b in O.index})
M = M.drop(columns=med[med < 7].index)           # same legacy exclusion as main run
e = eci.reindex(M.index)
R = pd.DataFrame(index=M.index, columns=M.columns, dtype=float)
fit_r2 = {}
for b in M:
    y = M[b].dropna(); x = e[y.index]
    lo = float(meta.loc[b, "random_baseline"]) if pd.notna(meta.loc[b, "random_baseline"]) else 0.0
    f = lambda x, a, c: lo + (1 - lo) * (1 / (1 + np.exp(-a * (x - c))))
    try:
        p, _ = curve_fit(f, x, y, p0=[0.05, x.median()], bounds=([1e-4, x.min() - 80], [2, x.max() + 80]), maxfev=20000)
    except Exception as ex_:
        print("fit failed", b, ex_); continue
    res = y - f(x, *p); R.loc[y.index, b] = res
    fit_r2[b] = 1 - res.var() / y.var()
print("logistic-on-ECI R^2 per benchmark:\n", pd.Series(fit_r2).round(2).sort_values().to_string())
R = R.dropna(axis=1, how="all")
Z = (R - R.mean()) / R.std()

def eig(Z):
    C = Z.corr(min_periods=12).fillna(0).to_numpy(copy=True); np.fill_diagonal(C, 1)
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-6, None); C = (V * w) @ V.T
    s = np.sqrt(np.diag(C)); C /= np.outer(s, s); w, V = np.linalg.eigh(C); o = np.argsort(w)[::-1]
    return w[o], V[:, o]
w, V = eig(Z)
# parallel analysis: shuffle each column's observed values among its observed rows (preserves missingness)
null = []
for _ in range(200):
    Zs = Z.copy()
    for c in Zs:
        idx = Zs[c].dropna().index; Zs.loc[idx, c] = rng.permutation(Zs.loc[idx, c].values)
    null.append(eig(Zs)[0])
null = np.array(null); n95 = np.percentile(null, 95, axis=0)
print("\nResidual PCA eigenvalues (real vs null 95th pct):")
for i in range(6): print(f"  PC{i+1}: {w[i]:.2f}  null95 {n95[i]:.2f}  {'<-- above null' if w[i] > n95[i] else ''}")
L = pd.DataFrame(V[:, :4], index=Z.columns, columns=[f"R{i+1}" for i in range(4)])
for c in L:
    if abs(L[c].min()) > L[c].max(): L[c] *= -1   # sign for readability only
L.assign(n=M.notna().sum()).round(2).to_csv(f"{out}/residual_loadings.csv")
for c in ["R1", "R2", "R3"]:
    s = L[c].sort_values(); print(f"\n{c}  (-):", ", ".join(f"{k} {v:.2f}" for k, v in s.head(6).items()), "\n     (+):", ", ".join(f"{k} {v:.2f}" for k, v in s.tail(6).items()))
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].plot(range(1, 11), w[:10], "o-", label="residual data"); ax[0].plot(range(1, 11), n95[:10], "s--", c="gray", label="null 95th pct")
ax[0].set(xlabel="Component", ylabel="Eigenvalue", title="Residual PCA vs shuffled null"); ax[0].legend()
ax[1].scatter(L.R1, L.R2, color="#3b6fb6")
for b in L.index: ax[1].annotate(b, (L.R1[b], L.R2[b]), fontsize=7)
ax[1].axhline(0, c="gray", lw=.5); ax[1].axvline(0, c="gray", lw=.5)
ax[1].set(xlabel="R1 loading", ylabel="R2 loading", title="Benchmarks after removing general capability")
plt.tight_layout(); plt.savefig(f"{out}/residual_pca.png", dpi=150)
