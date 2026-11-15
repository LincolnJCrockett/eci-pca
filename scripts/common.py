"""Shared data loading and low-rank estimators (iterative SVD on observed cells).

Import from a script with:  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  (needed under python -I)
"""
from types import SimpleNamespace
import numpy as np, pandas as pd
from scipy.optimize import curve_fit


def load(ex, tr="logit", min_b=25, min_obs_model=4, leg=7):
    d = pd.read_csv(f"{ex}/epoch_capabilities_index/processed_data_for_eci.csv")
    meta = pd.read_csv(f"{ex}/benchmark_metadata.csv").set_index("benchmark")
    E = pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv").set_index("Model")
    d = d[~d.benchmark.isin(meta[meta.superseded_by.isin(set(d.benchmark))].index)]
    M = d.pivot_table(index="model", columns="benchmark", values="performance")
    M = M.loc[:, M.notna().sum() >= min_b]
    O = M.notna().astype(int).T @ M.notna().astype(int)
    med = pd.Series({b: O.loc[b].drop(b).median() for b in O.index})
    M = M.drop(columns=med[med < leg].index)
    for _ in range(5):
        M = M.loc[M.notna().sum(axis=1) >= min_obs_model]
        M = M.loc[:, M.notna().sum() >= 15]
    Mc = M.clip(.01, .99)
    X = np.log(Mc / (1 - Mc)) if tr == "logit" else M
    lo = pd.to_numeric(meta["random_baseline"].reindex(M.columns), errors="coerce").fillna(0.0).to_numpy(float)
    return SimpleNamespace(
        M=M, X=X, Xv=X.to_numpy(float), Mv=M.to_numpy(float), obs=~np.isnan(X.to_numpy(float)),
        eci=E["eci"].reindex(M.index).to_numpy(float),
        date=pd.to_datetime(E["date"].reindex(M.index)).astype("int64").to_numpy() / 86400e9 / 365.25,
        floor=lo)


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
    """spec: A (benchmark means), B (model means), C (two-way FE), CJ / AJ (FE or benchmark intercepts re-estimated jointly
    with the low-rank part), or any label with base_ov = a full-matrix base to subtract (D, E)."""
    ob = ~np.isnan(Xin); Xo = np.where(ob, Xin, 0.0); R = np.zeros_like(Xo)
    joint = spec in ("CJ", "AJ"); jb = "C" if spec == "CJ" else "A"
    Bf = base_ov if base_ov is not None else (None if joint else base(Xo, ob, spec))
    iters = 150 if joint else iters

    def parts(R):
        B = base(np.where(ob, Xo - R, 0.0), ob, jb) if joint else Bf
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
    """held-out-cell MSE by k; identical folds for identical Xin"""
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


def cached(fn):
    store = {}
    def g(Xt, *a):
        key = hash(Xt.tobytes())
        if key not in store: store[key] = fn(Xt, *a)
        return store[key]
    return g


def ols_on_eci(Xt, eci, quad=False):
    """per-benchmark OLS of score on standardized ECI (free intercept and slope); returns full-matrix prediction"""
    n = len(eci); z = (eci - eci.mean()) / eci.std(); P = np.zeros_like(Xt)
    Dm = np.c_[np.ones(n), z, z ** 2] if quad else np.c_[np.ones(n), z]
    for j in range(Xt.shape[1]):
        o = ~np.isnan(Xt[:, j]); P[:, j] = Dm @ np.linalg.lstsq(Dm[o], Xt[o, j], rcond=None)[0]
    return P


def logistic_on_eci(Xt, eci, floor):
    """per-benchmark logistic of score on ECI: floor + (1-floor) * sigmoid(a (z - c)); returns full-matrix prediction (raw scale)"""
    n, m = Xt.shape; z = (eci - eci.mean()) / eci.std(); P = np.zeros((n, m))
    for j in range(m):
        o = ~np.isnan(Xt[:, j]); x, y, lo = z[o], Xt[o, j], floor[j]
        f = lambda x, a, c, lo=lo: lo + (1 - lo) / (1 + np.exp(-a * (x - c)))
        try:
            p, _ = curve_fit(f, x, y, p0=[1.0, np.median(x)], bounds=([1e-3, x.min() - 4], [50, x.max() + 4]), maxfev=20000)
            P[:, j] = f(z, *p)
        except Exception:
            P[:, j] = np.c_[np.ones(n), z] @ np.linalg.lstsq(np.c_[np.ones(o.sum()), x], y, rcond=None)[0]
    return P


def corr(a, b, mask=None):
    mask = np.ones(len(a), bool) if mask is None else mask
    a, b = a[mask], b[mask]; r = np.corrcoef(a, b)[0, 1]; rs = pd.Series(a).corr(pd.Series(b), method="spearman")
    z = np.arctanh(r); se = 1 / np.sqrt(len(a) - 3)
    return r, rs, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def pairwise_eigs(Xr, min_periods=12):
    """eigenvalues of the pairwise-complete correlation matrix, made PSD by eigenvalue clipping"""
    C = pd.DataFrame(Xr).corr(min_periods=min_periods).fillna(0).to_numpy(copy=True); np.fill_diagonal(C, 1)
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-6, None); C = (V * w) @ V.T; s_ = np.sqrt(np.diag(C)); C /= np.outer(s_, s_)
    return np.sort(np.linalg.eigvalsh(C))[::-1]
