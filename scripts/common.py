"""Shared code: data loading, PCA with missing data, the fit measure (logit R2), one-dimensional scores under a p-norm loss.

Scripts import this with  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  (needed under python -I).
See METHODOLOGY.md for the equations.

One weighting is used throughout: every observed score counts once, after scaling each benchmark to unit spread
(so benchmarks scored on more models count more).  This is what PCA with missing data does.
"""
from types import SimpleNamespace
import numpy as np, pandas as pd
from scipy.optimize import minimize


# ------------------------------------------------------------------ data
def load(ex, min_b=25, min_obs_model=4, legacy_overlap=7):
    """Benchmark x model table used everywhere.  ex = folder where benchmark_data.zip was extracted.
    Returns raw scores M, logit scores X (NaN = not evaluated), the official ECI, and the benchmark/model names."""
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
    half = ((E["eci_ci_high"] - E["eci_ci_low"]) / 2).reindex(M.index)                       # Epoch publishes 90% intervals
    return SimpleNamespace(M=M, Mv=M.to_numpy(float), X=X, Xv=X.to_numpy(float), obs=~X.isna().to_numpy(), bench=list(M.columns),
                           eci=E["eci"].reindex(M.index).to_numpy(float), eci_halfwidth=float(half.mean()), meta=meta, raw=d,
                           dates=pd.to_datetime(E["date"].reindex(M.index)))


BREAK_DATE = pd.Timestamp("2024-04-08")   # breakpoint Epoch estimated for the ECI frontier (epoch.ai/data-insights/ai-capabilities-progress-has-sped-up)


def frontier_idx(score, dates, keep=None):
    """indices of models that beat every earlier-released model (running maximum by release date); keep = optional boolean filter"""
    best, rec = -np.inf, []
    for i in np.argsort(np.asarray(dates), kind="stable"):
        if (keep is None or keep[i]) and score[i] > best: best = score[i]; rec.append(i)
    return np.array(rec, dtype=int)


def hinge_fit(t, y, t0):
    """continuous two-segment line y = a + b1 t + b2 max(t - t0, 0), break fixed at t0.
    Returns slope before, slope after, and the coefficients (a, b1, b2)."""
    A = np.c_[np.ones(len(t)), t, np.clip(t - t0, 0, None)]; c = np.linalg.lstsq(A, y, rcond=None)[0]
    return c[1], c[1] + c[2], c


def corr(a, b):
    """Pearson r with 95% interval (Fisher z), and Spearman"""
    r = np.corrcoef(a, b)[0, 1]; se = 1 / np.sqrt(len(a) - 3)
    return r, np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se), pd.Series(a).corr(pd.Series(b), method="spearman")


# ------------------------------------------------------------------ PCA with missing data
def fit_lowrank(Xin, k, base="joint", iters=400, tol=1e-6):
    """Rank-k PCA of Xin (NaN = missing) by iterative SVD on observed cells.
    base = "joint" : benchmark intercepts estimated together with the components; each benchmark divided by its observed SD
           "fixed" : subtract each benchmark's observed mean (conventional PCA; only used to show the selection bias)
           ndarray : subtract this full matrix (the part the ECI explains); each benchmark divided by the RMS of what is left
    Returns pred (= base + low rank), scores, loadings."""
    ob = ~np.isnan(Xin); Xo = np.where(ob, Xin, 0.0); R = np.zeros_like(Xo); cnt = ob.sum(0).clip(1)
    colmean = lambda Y: np.tile((np.where(ob, Y, 0.0)).sum(0) / cnt, (Xin.shape[0], 1))
    fixed = colmean(Xo) if isinstance(base, str) and base == "fixed" else (base if not isinstance(base, str) else None)
    sd_obs = np.sqrt((np.where(ob, Xo - colmean(Xo), 0.0) ** 2).sum(0) / cnt)

    def parts(R):
        B = colmean(Xo - R) if fixed is None else fixed
        D = np.where(ob, Xo - B, 0.0); s = sd_obs.copy() if fixed is None else np.sqrt((D ** 2).sum(0) / cnt); s[s < 1e-8] = 1
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


def ols_on_eci(Xt, eci):
    """Per-benchmark least squares of the logit score on standardized ECI (free intercept and slope); full-matrix prediction.
    On the logit scale this is the same S-curve as ECI's  sigmoid(a (capability - difficulty))."""
    z = (eci - eci.mean()) / eci.std(); Dm = np.c_[np.ones(len(z)), z]; P = np.zeros_like(Xt)
    for j in range(Xt.shape[1]):
        o = ~np.isnan(Xt[:, j]); P[:, j] = Dm @ np.linalg.lstsq(Dm[o], Xt[o, j], rcond=None)[0]
    return P


# ------------------------------------------------------------------ fit measure: logit-score R2
def logit_r2(S, X):
    """How much of each benchmark the score(s) explain, on logit scores.
    For each benchmark, least squares of the logit score on the score(s) with an intercept (a straight line on the logit scale
    is the ECI's S-curve).  S = one score per model (vector) or n x k matrix of scores.
    Returns per-benchmark R2_b, and the overall R2 = average of R2_b weighted by the number of models scored on each benchmark
    (= every score counts once, after scaling each benchmark to unit spread)."""
    S = np.atleast_2d(np.asarray(S, float).T).T; A = np.c_[np.ones(len(S)), S]; o = ~np.isnan(X); n = o.sum(0); r2 = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        oj = o[:, j]; x = X[oj, j]; e = x - A[oj] @ np.linalg.lstsq(A[oj], x, rcond=None)[0]; r2[j] = 1 - (e ** 2).sum() / ((x - x.mean()) ** 2).sum()
    return r2, (n * r2).sum() / n.sum()


# ------------------------------------------------------------------ one score per model under a p-norm loss
def prep(X):
    idx = [np.where(~np.isnan(X[:, j]))[0] for j in range(X.shape[1])]
    yc = [X[i, j] - X[i, j].mean() for j, i in enumerate(idx)]
    return idx, yc, [np.linalg.norm(y) for y in yc], np.array([len(i) for i in idx], float)


def benchmark_r(theta, data):
    """per-benchmark correlation of the score with the (logit) benchmark over the models scored on it (+ pieces for the gradient)"""
    idx, yc, ny, _ = data; r = np.zeros(len(idx)); parts = []
    for j, (i, y, l) in enumerate(zip(idx, yc, ny)):
        c = theta[i] - theta[i].mean(); nc = np.linalg.norm(c) + 1e-12; r[j] = c @ y / (nc * l); parts.append((c, nc))
    return r, parts


def pnorm_obj(theta, p, data):
    """Weighted power mean over benchmarks of u_b = 1 - r_b|r_b| (= 1 - R2_b on logit scores), and its gradient.
    Weights = number of models scored on each benchmark.
      p = 1   : arithmetic mean  (= the first principal component)
      p large : approaches the worst benchmark"""
    idx, yc, ny, nb = data; r, parts = benchmark_r(theta, data); wn = nb / nb.sum()
    u = np.clip(1 - r * np.abs(r), 1e-9, None)
    um = u.max(); v = u / um; S = (wn * v ** p).sum(); F = um * S ** (1 / p); dF = S ** (1 / p - 1) * v ** (p - 1) * wn
    g = np.zeros_like(theta)
    for j, (i, y, l) in enumerate(zip(idx, yc, ny)):
        c, nc = parts[j]; g[i] += dF[j] * (-2 * abs(r[j])) * (y / (nc * l) - r[j] * c / nc ** 2)
    return F, g


def pnorm_solve(X, p, init):
    """standardized score vector minimizing the p-norm loss, starting from init"""
    th = (init - init.mean()) / init.std()
    res = minimize(pnorm_obj, th, args=(p, prep(X)), jac=True, method="L-BFGS-B", options=dict(maxiter=3000, maxfun=6000))
    return (res.x - res.x.mean()) / res.x.std()


def pnorm_path(X, ps, init):
    """warm-started solutions for increasing p >= 1 (each starts from the previous); returns {p: standardized score vector}"""
    path = {}; th = init
    for p in ps: th = pnorm_solve(X, p, th); path[p] = th
    return path
