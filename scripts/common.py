"""Shared code: data loading, low-rank PCA with missing data, cross-validation, one-dimensional scores under different losses.

Scripts import this with  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  (needed under python -I).
See METHODOLOGY.md for the equations.
"""
from types import SimpleNamespace
import numpy as np, pandas as pd
from scipy.optimize import minimize


# ------------------------------------------------------------------ data
def load(ex, min_b=25, min_obs_model=4, legacy_overlap=7):
    """Benchmark x model table used everywhere.  ex = folder where benchmark_data.zip was extracted.
    Returns logit scores X (NaN = not evaluated), the official ECI, and the benchmark/model names."""
    d = pd.read_csv(f"{ex}/epoch_capabilities_index/processed_data_for_eci.csv")
    meta = pd.read_csv(f"{ex}/benchmark_metadata.csv").set_index("benchmark")
    E = pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv").set_index("Model")
    d = d[~d.benchmark.isin(meta[meta.superseded_by.isin(set(d.benchmark))].index)]          # drop superseded benchmarks
    M = d.pivot_table(index="model", columns="benchmark", values="performance")
    M = M.loc[:, M.notna().sum() >= min_b]                                                    # benchmarks with enough models
    O = M.notna().astype(int).T @ M.notna().astype(int)                                       # O[b, c] = models scored on both
    med = pd.Series({b: O.loc[b].drop(b).median() for b in O.index})
    M = M.drop(columns=med[med < legacy_overlap].index)                                       # older benchmarks that never overlap with the rest
    for _ in range(5):
        M = M.loc[M.notna().sum(axis=1) >= min_obs_model]
        M = M.loc[:, M.notna().sum() >= 15]
    Mc = M.clip(.01, .99)
    X = np.log(Mc / (1 - Mc))
    half = ((E["eci_ci_high"] - E["eci_ci_low"]) / 2).reindex(M.index)
    return SimpleNamespace(M=M, X=X, Xv=X.to_numpy(float), obs=~X.isna().to_numpy(), bench=list(M.columns),
                           eci=E["eci"].reindex(M.index).to_numpy(float), eci_halfwidth=float(half.mean()), meta=meta, raw=d)


def corr(a, b):
    """Pearson r with 95% interval (Fisher z), and Spearman"""
    r = np.corrcoef(a, b)[0, 1]; se = 1 / np.sqrt(len(a) - 3)
    return r, np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se), pd.Series(a).corr(pd.Series(b), method="spearman")


# ------------------------------------------------------------------ low-rank PCA with missing data
def fit_lowrank(Xin, k, base="joint", iters=400, tol=1e-6):
    """Rank-k PCA of Xin (NaN = missing) by iterative SVD on observed cells.
    base = "fixed" : subtract each benchmark's observed mean              (spec A, conventional PCA)
           "joint" : benchmark intercepts re-estimated with the factor    (spec AJ)
           ndarray : subtract this full matrix                            (spec E: the part ECI explains)
    Columns of the remainder are divided by their RMS.  Returns pred (= base + low rank), scores, loadings."""
    ob = ~np.isnan(Xin); Xo = np.where(ob, Xin, 0.0); R = np.zeros_like(Xo); cnt = ob.sum(0).clip(1)
    colmean = lambda Y: np.tile((np.where(ob, Y, 0.0)).sum(0) / cnt, (Xin.shape[0], 1))
    fixed = colmean(Xo) if isinstance(base, str) and base == "fixed" else (base if not isinstance(base, str) else None)

    def parts(R):
        B = colmean(Xo - R) if fixed is None else fixed
        D = np.where(ob, Xo - B, 0.0); s = np.sqrt((D ** 2).sum(0) / cnt); s[s < 1e-8] = 1
        return B, s, np.where(ob, D, R) / s

    for _ in range(iters if k > 0 else 0):
        B, s, Z = parts(R)
        U, sv, Vt = np.linalg.svd(Z, full_matrices=False)
        Rn = (U[:, :k] * sv[:k]) @ Vt[:k] * s
        done = np.abs(Rn - R).max() < tol; R = Rn
        if done: break
    B, s, Z = parts(R); res = dict(pred=B + R, B=B, R=R, s=s)
    if k > 0:
        U, sv, Vt = np.linalg.svd(Z, full_matrices=False); res.update(scores=U[:, :k] * sv[:k], load=Vt[:k].T)
    return res


def cv_mse(Xin, pred_fn, ks, folds=5, reps=2):
    """Held-out-cell mean squared error for each k.  A cell is testable if its model keeps >= 2 and its benchmark >= 5 training scores."""
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


def ols_on_eci(Xt, eci):
    """Per-benchmark least squares of the score on standardized ECI (free intercept and slope); full-matrix prediction.
    On the logit scale this is the same S-curve as ECI's  sigmoid(a (capability - difficulty))."""
    z = (eci - eci.mean()) / eci.std(); Dm = np.c_[np.ones(len(z)), z]; P = np.zeros_like(Xt)
    for j in range(Xt.shape[1]):
        o = ~np.isnan(Xt[:, j]); P[:, j] = Dm @ np.linalg.lstsq(Dm[o], Xt[o, j], rcond=None)[0]
    return P


def pairwise_eigs(Xr, min_periods=12):
    """Eigenvalues of the pairwise-complete correlation matrix (missing pairs set to 0, then made positive semi-definite)"""
    C = pd.DataFrame(Xr).corr(min_periods=min_periods).fillna(0).to_numpy(copy=True); np.fill_diagonal(C, 1)
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-6, None); C = (V * w) @ V.T; s_ = np.sqrt(np.diag(C)); C /= np.outer(s_, s_)
    return np.sort(np.linalg.eigvalsh(C))[::-1]


# ------------------------------------------------------------------ one score per model under different losses
def prep(X):
    idx = [np.where(~np.isnan(X[:, j]))[0] for j in range(X.shape[1])]
    yc = [X[i, j] - X[i, j].mean() for j, i in enumerate(idx)]
    return idx, yc, [np.linalg.norm(y) for y in yc]


def benchmark_r(theta, data):
    """per-benchmark correlation of the score with the benchmark over the models scored on it (+ pieces for the gradient)"""
    idx, yc, ny = data; r = np.zeros(len(idx)); parts = []
    for j, (i, y, l) in enumerate(zip(idx, yc, ny)):
        c = theta[i] - theta[i].mean(); nc = np.linalg.norm(c) + 1e-12; r[j] = c @ y / (nc * l); parts.append((c, nc))
    return r, parts


def r2_by_benchmark(theta, data):
    r, _ = benchmark_r(theta, data); return r * np.abs(r)


def pnorm_obj(theta, p, data):
    """p-norm over benchmarks of u_b = 1 - r_b|r_b| (= 1 - R2_b), and its gradient.  p = 1: average fit; p large: worst benchmark."""
    idx, yc, ny = data; r, parts = benchmark_r(theta, data)
    u = np.clip(1 - r * np.abs(r), 1e-9, None); um = u.max(); v = u / um; S = np.mean(v ** p); F = um * S ** (1 / p)
    w = S ** (1 / p - 1) * v ** (p - 1) / len(u); g = np.zeros_like(theta)
    for j, (i, y, l) in enumerate(zip(idx, yc, ny)):
        c, nc = parts[j]; g[i] += w[j] * (-2 * abs(r[j])) * (y / (nc * l) - r[j] * c / nc ** 2)
    return F, g


def worst_benchmark_path(X, ps, init):
    """warm-started solutions for p in ps (each starts from the previous); returns {p: standardized score vector}"""
    data = prep(X); th = (init - init.mean()) / init.std(); path = {}
    for p in ps:
        res = minimize(pnorm_obj, th, args=(p, data), jac=True, method="L-BFGS-B", options=dict(maxiter=3000, maxfun=6000))
        th = (res.x - res.x.mean()) / res.x.std(); path[p] = th.copy()
    return path


def cell_loss_solve(X, loss, init, deltas=(0.1, 0.03, 0.01)):
    """Score theta_m with a free intercept a_b and slope g_b per benchmark, jointly minimising a cell-level loss on
    normalised scores y_mb = (x_mb - mean_b)/sd_b, errors e_mb = y_mb - a_b - g_b theta_m:
      mse_cells : sum_b sd_b^2 sum_m e^2 / N      squared error in raw logit units (big / high-spread benchmarks count more)
      mse_bench : (1/B) sum_b (1/n_b) sum_m e^2   squared error, benchmarks weighted equally
      mae_bench : (1/B) sum_b (1/n_b) sum_m |e|   absolute-value loss, |e| smoothed as sqrt(e^2 + delta^2) - delta"""
    n, m = X.shape; I, J = np.where(~np.isnan(X)); Y0 = X[I, J]; nb = np.bincount(J, minlength=m)
    mu = np.bincount(J, Y0, minlength=m) / nb; sd = np.sqrt(np.bincount(J, (Y0 - mu[J]) ** 2, minlength=m) / nb); Y = (Y0 - mu[J]) / sd[J]
    W = sd[J] ** 2 / len(Y) if loss == "mse_cells" else 1 / (m * nb[J])
    th = (init - init.mean()) / init.std(); a = np.zeros(m); g = np.zeros(m)
    for j in range(m):
        o = J == j; g[j], a[j] = np.polyfit(th[I[o]], Y[o], 1)

    def f(par, delta):
        t, aa, gg = par[:n], par[n:n + m], par[n + m:]; e = Y - aa[J] - gg[J] * t[I]
        if loss == "mae_bench":
            s = np.sqrt(e * e + delta * delta); L = (W * (s - delta)).sum(); d = W * e / s
        else:
            L = (W * e * e).sum(); d = 2 * W * e
        return L, np.r_[-np.bincount(I, d * gg[J], minlength=n), -np.bincount(J, d, minlength=m), -np.bincount(J, d * t[I], minlength=m)]

    par = np.r_[th, a, g]
    for dl in (deltas if loss == "mae_bench" else (None,)):
        par = minimize(f, par, args=(dl,), jac=True, method="L-BFGS-B", options=dict(maxiter=5000, maxfun=10000)).x
    t = par[:n]; return (t - t.mean()) / t.std()
