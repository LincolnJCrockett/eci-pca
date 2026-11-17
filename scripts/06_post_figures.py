"""The three figures used in the blog post. Reads the CSVs written by scripts 01-03.
Usage: python -I scripts/06_post_figures.py <extracted_dir> <results_dir>      -> <results_dir>/figures/post/
"""
import sys, os, time, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import Patch
from common import *

ex, res = sys.argv[1], sys.argv[2]; fd = f"{res}/figures/post"; os.makedirs(fd, exist_ok=True)
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, VIOLET, LIGHT = "#2a78d6", "#eb6834", "#6250d6", "#c3c2b7"
plt.rcParams.update({"axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
W = lambda s, w=60: textwrap.fill(s, w)
def save(fig, name):
    fig.tight_layout()
    for attempt in range(6):
        try: fig.savefig(f"{fd}/{name}.png", dpi=150); break
        except OSError:
            if attempt == 5: raise
            time.sleep(1)
    plt.close(fig); print("wrote", name)
R = lambda d, f: pd.read_csv(f"{res}/{d}/{f}", index_col=0)
D = load(ex); eci = D.eci

# ---- Figure 1: share of each benchmark explained by ECI; the 10 benchmarks Epoch's model treats as noisiest highlighted ----
pb = R("01_pc1_vs_eci", "per_benchmark_explained.csv").sort_values("R2_ECI", ascending=False); noisy = pb.noise_rank_of_29 <= 10
fig, ax = plt.subplots(figsize=(8, 8.5)); y = np.arange(len(pb)); r = pb.R2_ECI.clip(0, 1)
ax.barh(y, r, color=np.where(noisy, ORANGE, BLUE), height=.6, edgecolor=SURF, linewidth=2)
for yi, v in zip(y, r): ax.text(v + .01, yi, f"{v:.0%}", va="center", fontsize=7, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels(pb.index, fontsize=8); ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
ax.legend(handles=[Patch(color=ORANGE, label="one of the 10 noisiest benchmarks (Epoch's own noise estimate)"), Patch(color=BLUE, label="other benchmarks")], frameon=False, fontsize=8, loc="lower right")
ax.set(xlabel="Share of each benchmark's variation across models explained by ECI",
       title=W("ECI explains most benchmarks well, but a few much less, and those are among the noisiest", 60)); save(fig, "fig1_eci_by_benchmark")

# ---- Figure 2: the main pattern in what ECI leaves out (first component of the ECI residuals, E1) ----
# Sign as in E1 / Burnham's "Claudiness": positive = agentic coding and proofs, negative = geolocation, chess, factual recall, visual physics.
e1 = R("02_eci_residual", "residual_loadings.csv").E1.sort_values()
fig, ax = plt.subplots(figsize=(8, 8.5)); y = np.arange(len(e1))
ax.barh(y, e1.values, color=np.where(e1.values < 0, ORANGE, BLUE), height=.6, edgecolor=SURF, linewidth=2)
ax.set_yticks(y); ax.set_yticklabels(e1.index, fontsize=8); ax.grid(axis="y", visible=False); ax.axvline(0, color=MUTED, lw=.8)
lim = np.abs(e1).max() * 1.15; ax.set_xlim(-lim, lim)
ax.text(-lim * .97, len(e1) - 1, "Models high on this pattern score\nhigher than their ECI predicts\non the blue benchmarks...", fontsize=8.5, color=BLUE, va="top")
ax.text(lim * .97, 0, "...and lower than their ECI\npredicts on the orange ones", fontsize=8.5, color=ORANGE, ha="right", va="bottom")
ax.set(xlabel="Weight in the main pattern of what ECI leaves out", ylim=(-1, len(e1)),
       title=W("What ECI leaves out is not random: its main pattern trades agentic coding and proofs against geolocation, chess, trivia and visual tasks", 60)); save(fig, "fig2_what_eci_leaves_out")

# ---- Figure 3: all models under the three scoring schemes, frontier and its trend from April 2024 ----
OT = R("03_loss_functions", "loss_over_time.csv"); SC = R("03_loss_functions", "loss_score_change_points.csv"); dates = D.dates.to_numpy()
L1, LI = "p=1 (average = first PC)", "p=inf (worst benchmark)"
schemes = {"Official ECI": ("Official ECI", BLUE), L1: ("Best single number (first principal component)", VIOLET), LI: ("Fit the worst-explained benchmark best", ORANGE)}
yrs = ((D.dates - D.dates.min()).dt.days / 365.25).to_numpy(); tb = (BREAK_DATE - D.dates.min()).days / 365.25
to_date = lambda t: D.dates.min() + pd.to_timedelta(t * 365.25, unit="D")
fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), sharex=True, sharey=True)
for a, (k, (title, col)) in zip(axs, schemes.items()):
    s = eci + SC[k].to_numpy(); rec = frontier_idx(s, dates)
    a.scatter(dates, s, s=12, color=col, alpha=.45, label="all models")
    a.step(dates[rec], s[rec], where="post", color=INK, lw=1.3); a.scatter(dates[rec], s[rec], s=20, color=INK, zorder=3, label="frontier (record at release)")
    _, post, c = hinge_fit(yrs[rec], s[rec], tb); tt = np.linspace(tb, yrs[rec].max(), 100)
    a.plot(to_date(tt), c[0] + c[1] * tt + c[2] * (tt - tb), color=INK, lw=1.6, ls="--", label=f"frontier trend since Apr 2024: {post:+.1f} points/yr")
    a.set_title(title, fontsize=10); a.legend(frameon=False, fontsize=8, loc="upper left")
    a.xaxis.set_major_locator(mdates.YearLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axs[0].set_ylabel("Score (ECI points)")
fig.suptitle("Whichever way models are scored, the frontier has risen about 16-17 points a year since April 2024", x=.01, ha="left", fontweight="bold", fontsize=12); save(fig, "fig3_progress_by_scheme")
