"""Footnote to point 3: how does residual component E1 compare with Epoch's "Claudiness" (second principal component)?

Usage: python -I scripts/05_claudiness.py <extracted_dir> <results_dir>      -> <results_dir>/05_claudiness/
Reads residual_loadings.csv and residual_scores.csv written by 02_eci_residual.py.
Source for Claudiness: G. Burnham, "Benchmark Scores = General Capability + Claudiness", Epoch AI Gradient Updates, 20 Nov 2025,
https://epoch.ai/gradient-updates/benchmark-scores-general-capability-claudiness . The post publishes only its seven largest
PC2 weights (below); sign as published (agentic computer-use benchmarks positive).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import *

ex, res = sys.argv[1], sys.argv[2]; out = f"{res}/05_claudiness"; os.makedirs(out, exist_ok=True)
D = load(ex); Xv, bench = D.Xv, D.bench
L = pd.read_csv(f"{res}/02_eci_residual/residual_loadings.csv", index_col=0); S = pd.read_csv(f"{res}/02_eci_residual/residual_scores.csv", index_col=0)
org = pd.read_csv(f"{ex}/epoch_capabilities_index/eci_scores.csv").set_index("Model").Organization.reindex(S.index)
nb = pd.read_csv(f"{ex}/epoch_capabilities_index/processed_data_for_eci.csv").groupby("benchmark").model.nunique()

# ---- 1. loadings: Burnham's published PC2 weights vs E1, on the benchmarks both include ----
BURNHAM_PC2 = {"The Agent Company": .42, "OSUniverse": .37, "VideoMME": -.35, "OSWorld": .35, "Factorio Learning Environment": .33, "VPCT": -.28, "FrontierMath Tier 4": -.26}
HERE = {"VPCT": "VPCT", "FrontierMath Tier 4": "FrontierMath-Tier-4-v2-Private"}           # names in this sample
rows = []
for b, w in BURNHAM_PC2.items():
    mine = HERE.get(b); hit = nb.index.str.contains(b.split(" Learning")[0], case=False, regex=False)
    n = int(nb[mine]) if mine else (int(nb[hit].max()) if hit.any() else 0)                 # models scored on it in Epoch's data
    rows.append(dict(benchmark=b, burnham_PC2_weight=w, E1_loading=L.E1[mine] if mine else np.nan, models_in_Epoch_data=n, in_this_sample=mine is not None))
LB = pd.DataFrame(rows).set_index("benchmark"); print("Burnham's PC2 weights vs E1 loadings:"); print(LB.to_string()); LB.to_csv(f"{out}/loadings_vs_burnham.csv")

# the analogue of Burnham's analysis in this sample: the second component of the ordinary PCA (point 2)
V2 = pd.Series(fit_lowrank(Xv, 2, "joint")["load"][:, 1], index=bench); cos = abs(L.E1 @ V2) / np.linalg.norm(L.E1) / np.linalg.norm(V2)
print(f"\ncosine between E1 loadings and the loadings of PC2 of the ordinary PCA: {cos:.2f}")

# ---- 2. models Burnham names as most / least Claude-like: their E1 percentile ----
NAMED = {"most": ["Claude 3.5 Sonnet (October 2024)", "Claude 3.5 Sonnet", "Claude 3.7 Sonnet", "Claude Sonnet 4", "Claude Opus 4"],
         "least": ["GPT-5", "GPT-4o mini", "GPT-4o (Nov 2024)", "GPT-4o (Aug 2024)", "o4-mini"]}
pct = lambda m: 100 * (S.E1 < S.E1[m]).mean()
NM = pd.DataFrame([dict(burnham_group=g, model=m, E1=S.E1[m], E1_percentile=pct(m)) for g, ms in NAMED.items() for m in ms]).set_index("model")
opus = [m for m in S.index if m.startswith("Claude Opus")]
NM = pd.concat([NM, pd.DataFrame([dict(burnham_group="(other Claude Opus)", model=m, E1=S.E1[m], E1_percentile=pct(m)) for m in opus if m not in NM.index]).set_index("model")])
print("\nE1 percentile of the models Burnham names, and of the other Claude Opus models:"); print(NM.round(1).to_string()); NM.round(3).to_csv(f"{out}/named_models_E1.csv")

# ---- 3. how much of E1 is the developer? ----
def eta2(x, g):
    """share of the variance of x explained by group means"""
    mu = x.mean(); return sum(len(v) * (v.mean() - mu) ** 2 for _, v in x.groupby(g)) / ((x - mu) ** 2).sum()
big = org.value_counts()[lambda c: c >= 5].index; keep = org.isin(big); x, g = S.E1[keep], org[keep]
rng = np.random.default_rng(0); e_dev = eta2(x, g); p_dev = np.mean([eta2(x, pd.Series(rng.permutation(g.values), index=g.index)) >= e_dev for _ in range(2000)])
anth = (org == "Anthropic"); e_anth = np.corrcoef(anth.astype(float), S.E1)[0, 1] ** 2
a, b = S.E1[anth], S.E1[~anth]; auc = (a.values[:, None] > b.values[None, :]).mean()
print(f"\nshare of E1 explained by developer ({len(big)} developers with >= 5 models): {e_dev:.3f} (permutation p = {p_dev:.4f})")
print(f"share of E1 explained by Anthropic vs everyone else: {e_anth:.3f}; Anthropic mean {a.mean():+.2f} vs {b.mean():+.2f} "
      f"({(a.mean() - b.mean()) / S.E1.std():.2f} SD); P(random Claude > random other) = {auc:.2f}")
DV = S.E1[keep].groupby(g).agg(["mean", "count"]).rename(columns={"mean": "mean_E1", "count": "n_models"}).sort_values("mean_E1")
print("\nmean E1 by developer:"); print(DV.round(2).to_string()); DV.round(3).to_csv(f"{out}/E1_by_developer.csv")
pd.Series({"developers_with_5plus_models": len(big), "share_E1_explained_by_developer": e_dev, "permutation_p": p_dev, "share_E1_explained_by_anthropic_vs_rest": e_anth,
           "anthropic_minus_rest_in_SD": (a.mean() - b.mean()) / S.E1.std(), "prob_claude_above_other": auc, "cosine_E1_vs_ordinary_PC2": cos}).round(4).to_csv(f"{out}/summary.csv", header=["value"])
