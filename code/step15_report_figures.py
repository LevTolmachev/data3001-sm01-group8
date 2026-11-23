"""
DATA3001 Group S01-G8 - Step 15: the figures in the final report
=================================================================
Run AFTER steps 2-14 (step 13 needs the wind files; without them Figure 7 is skipped). Reads their CSV / NPZ outputs and writes report-sized PNGs (300 dpi, every label,
legend and title drawn into the image) to <BASE>/models/report_figures.
Coastlines: Natural Earth (public domain), data/geo/south_america.geojson.
Run:  uv run python code/step15_report_figures.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
sys.path.insert(0, _here)
import step2_models_M0_M1_M2 as s2
from step2_models_M0_M1_M2 import NCELL, NX, NY, EXITS, CELL_LAT, CELL_LON, OUT, MIN_DRIFTERS, as_grid
import step4_drogue_season_gam_nn as s4

FIG = os.path.join(OUT, "report_figures")
os.makedirs(FIG, exist_ok=True)
S5, S12, S13, S14 = (os.path.join(OUT, d) for d in ("step5", "step12", "step13", "step14"))
INK, MUTED, GRIDC = "#1B2A3A", "#5B6B7A", "#E3E8EF"
BLUE, ORANGE, PURPLE, GREEN, GREY = "#1F6FB2", "#E07B1F", "#7A2E7E", "#1E9E6A", "#8C96A3"
LAND, NODATA = "#D3DAE3", "#F1F3F6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlecolor": INK, "axes.titlelocation": "left"})
EXT = [s2.LON_MIN, s2.LON_MAX, s2.LAT_MIN, s2.LAT_MAX]
XC = np.linspace(s2.LON_MIN + 0.5, s2.LON_MAX - 0.5, NX)
YC = np.linspace(s2.LAT_MAX - 0.5, s2.LAT_MIN + 0.5, NY)
GEO = json.load(open(os.path.join(os.path.dirname(_here), "data", "geo", "south_america.geojson")))
REL = LogNorm(1e-4, 1e-1)
W1, W2 = 6.9, 6.9            # full text width (inches)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print("saved", name)


def land(ax):
    for f in GEO["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polys:
            ax.add_patch(Polygon(np.array(p[0]), closed=True, facecolor=LAND, edgecolor="white", lw=0.5, zorder=3))


def map_ax(ax, title="", pad=0.0, ylab=True):
    ax.set_title(title, pad=4)
    ax.set_aspect(1 / np.cos(np.deg2rad(40)))
    ax.set_facecolor(NODATA)
    ax.set_xlim(s2.LON_MIN - pad, s2.LON_MAX + pad)
    ax.set_ylim(s2.LAT_MIN - pad, s2.LAT_MAX + pad)
    ax.set_xticks([-60, -50, -40], ["60°W", "50°W", "40°W"])
    ax.set_yticks([-50, -45, -40, -35, -30], ["50°S", "45°S", "40°S", "35°S", "30°S"] if ylab else [])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    land(ax)


def grid_img(ax, v, **kw):
    return ax.imshow(np.ma.masked_invalid(as_grid(v)), origin="upper", extent=EXT, interpolation="nearest", zorder=1, **kw)


def hpd(vec, q):
    v = vec[:NCELL] / max(vec[:NCELL].sum(), 1e-12)
    order = np.argsort(v)[::-1]
    keep = order[: np.searchsorted(np.cumsum(v[order]), q) + 1]
    m = np.zeros(NCELL); m[keep] = 1
    return as_grid(m)


def release_map(ax, snap, c0, masked, title, ylab=True):
    gr = snap[:NCELL].copy()
    gr[(gr < 1e-4) | masked] = np.nan
    im = grid_img(ax, gr, cmap="Blues", norm=REL)
    if snap[:NCELL].sum() > 0.01:
        ax.contour(XC, YC, hpd(snap, 0.5), levels=[0.5], colors=ORANGE, linewidths=1.6, zorder=4)
        ax.contour(XC, YC, hpd(snap, 0.9), levels=[0.5], colors=ORANGE, linewidths=1.0, linestyles="--", zorder=4)
    map_ax(ax, title, ylab=ylab)
    ax.plot(CELL_LON[c0], CELL_LAT[c0], marker="*", color=INK, ms=10, mec="white", mew=0.8, zorder=6)
    return im


def gain_plot(ax, rows, labels, colours=None, xlim=(-0.5, 0.2)):
    for i, (_, r) in enumerate(rows.iterrows()):
        col = GREEN if r["gain_p05"] > 0 else (ORANGE if r["gain_p95"] < 0 else GREY)
        v, lo, hi = r["gain"], r["gain_p05"], r["gain_p95"]
        if v < xlim[0]:
            ax.annotate("", xy=(xlim[0] + 0.01, i), xytext=(xlim[0] + 0.12, i), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
            ax.text(xlim[0] + 0.13, i, f"{v:.2f}", color=col, va="center", fontsize=7.5)
        else:
            ax.plot([max(lo, xlim[0]), hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
            ax.plot(v, i, "o", color=col, ms=5.5, mec="white", mew=0.8, zorder=3)
    ax.axvline(0, color=PURPLE, lw=1.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.grid(axis="x", color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- load
daily, tr, ids, SIG, ALP = s4.setup()
fit_ids = ids["train"] | ids["val"]
OP = {d: np.load(os.path.join(S5, f"operator_final_{d}.npz"), allow_pickle=True) for d in ("drogued", "undrogued")}
masked, ocean = OP["drogued"]["masked"], OP["drogued"]["ocean"]
cellfate = pd.read_csv(os.path.join(S5, "rq1_cell_fate.csv"))
summ = pd.read_csv(os.path.join(S5, "rq1_summary.csv"))
pos = pd.read_csv(os.path.join(S5, "final_test_position.csv"))
days = list(OP["drogued"]["show_days"])
c_conf = int(OP["drogued"]["release_cells"][0])

# ----------------------------------------------------------------------------- Figure 1: region and data
d = daily.sort_values(["id", "time"])
same = (d["id"].values[1:] == d["id"].values[:-1]) & (np.diff(d["time"].values) == np.timedelta64(1, "D"))
xy = np.column_stack([d["lon"].values, d["lat"].values])
segs = np.stack([xy[:-1], xy[1:]], axis=1)[same]
drg = d["drogued"].values[:-1][same].astype(bool)
m_all = s4.moves_for(tr[tr["id"].isin(fit_ids)], "all")
g = m_all.groupby("from_cell").agg(de=("de_km", "mean"), dn=("dn_km", "mean")).reindex(range(NCELL))
spd = np.hypot(g["de"], g["dn"]).values.astype(float); spd[masked] = np.nan
r = daily[daily["inR"] & daily["id"].isin(fit_ids)].copy()
r["cell"] = s2.cell_of(r["lat"].values, r["lon"].values)
cnt = r.groupby("cell")["id"].nunique().reindex(range(NCELL)).values.astype(float); cnt[~ocean] = np.nan
fig, ax = plt.subplots(1, 3, figsize=(W1, 3.15), layout="constrained")
a = ax[0]
for mk, col in ((drg, BLUE), (~drg, ORANGE)):
    a.add_collection(LineCollection(segs[mk], colors=col, linewidths=0.2, alpha=0.35, zorder=1))
map_ax(a, "(a) Daily drifter tracks", pad=2.0)
a.set_facecolor("white")
a.plot([-60, -40, -40, -60, -60], [-50, -50, -30, -30, -50], color=INK, lw=1.3, zorder=5)
a.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="drogued (15 m)"), Line2D([], [], color=ORANGE, lw=2, label="undrogued (surface)")],
         loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=1, frameon=False)
a = ax[1]
im = grid_img(a, spd, cmap="Blues", vmin=0, vmax=np.nanpercentile(spd, 99))
okq = ~masked
a.quiver(CELL_LON[okq], CELL_LAT[okq], g["de"].values[okq], g["dn"].values[okq], color=INK, scale=500, width=0.005,
         headwidth=3.5, pivot="mid", zorder=4, alpha=0.8)
map_ax(a, "(b) Mean daily drift", ylab=False)
cb = fig.colorbar(im, ax=a, orientation="horizontal", shrink=0.85, pad=0.02); cb.set_label("km per day"); cb.outline.set_visible(False)
a = ax[2]
im = grid_img(a, cnt, cmap="Blues", norm=LogNorm(3, np.nanmax(cnt)))
mk = masked & ocean
a.scatter(CELL_LON[mk], CELL_LAT[mk], marker="x", s=12, color=ORANGE, lw=1.0, zorder=5)
map_ax(a, "(c) Distinct drifters per cell", ylab=False)
cb = fig.colorbar(im, ax=a, orientation="horizontal", shrink=0.85, pad=0.02, ticks=[3, 10, 30, 100])
cb.ax.set_xticklabels(["3", "10", "30", "100"]); cb.ax.minorticks_off(); cb.outline.set_visible(False)
cb.set_label(f"× = fewer than {MIN_DRIFTERS} drifters (masked)")
save(fig, "fig01_region_data.png")

# ----------------------------------------------------------------------------- Figure 2: grid, time step, Markov (RQ3)
gp = pd.read_csv(os.path.join(S12, "grid_pooled.csv"))
mk_ = pd.read_csv(os.path.join(S12, "timestep_markov_pooled.csv"))
GC = {"square 0.5°": "#9ECAE1", "square 1°": BLUE, "square 2°": "#08306B", "hex ≈1°": ORANGE, "hex ≈2°": "#8C4A12"}
fig, ax = plt.subplots(1, 3, figsize=(W1, 2.6), layout="constrained")
for nm, gg in gp.groupby("grid", sort=False):
    x = gg["horizon_days"].values
    kw = dict(color=GC.get(nm, "k"), marker="o", ms=3.5, lw=1.4, label=nm)
    ax[0].plot(x, gg["gain_vs_square1"], **kw)
    if nm != "square 1°":
        ax[0].fill_between(x, gg["gain_p05"], gg["gain_p95"], color=kw["color"], alpha=0.15, lw=0)
    ax[1].plot(x, gg["in_90pct_area_%"], **kw)
ax[0].axhline(0, color=MUTED, lw=0.8)
ax[0].set_title("(a) Skill vs 1° squares"); ax[0].set_ylabel("Log score per km² − 1° squares")
ax[1].axhline(90, color=MUTED, ls=":", lw=1); ax[1].set_ylim(60, 100)
ax[1].set_title("(b) Calibration"); ax[1].set_ylabel("Drifters inside 90% area (%)")
ax[0].legend(frameon=False, fontsize=7, loc="lower right")
dd = mk_[mk_["step"] == "direct"]
ax[2].plot(dd["horizon_days"], dd["tv_split_half_reference"], color=GREY, marker="o", ms=3.5, lw=1.4, label="two random halves of the data")
ax[2].plot(dd["horizon_days"], dd["tv_iterated_vs_direct"], color=BLUE, marker="o", ms=3.5, lw=1.4, label="1-day matrix iterated vs direct fit")
ax[2].set_ylim(0, 1); ax[2].set_title("(c) Markov check"); ax[2].set_ylabel("Difference between forecasts (TV)")
ax[2].legend(frameon=False, fontsize=7, loc="upper left")
for a, xt in zip(ax, ([7, 30, 90], [7, 30, 90], [6, 30, 60, 90])):
    a.set_xticks(xt); a.set_xlabel("Horizon (days)"); a.grid(axis="y", color=GRIDC, lw=0.6); a.set_axisbelow(True)
save(fig, "fig02_grid_markov.png")

# ----------------------------------------------------------------------------- Figure 3: model ladder (RQ5)
allm = pd.read_csv(os.path.join(S12, "complexity_test_pooled.csv")).query("set == 'test / all'")
ROWS = [("M0b displacement", "M0b one pattern"), ("M1 empirical", "M1 counts"), ("M5 GAM", "M5 GAM"),
        ("M6 neural net", "M6 network"), ("M7 memory, counts", "M7 counts + memory"),
        ("M7 memory, GAM", "M7 GAM + memory"), ("M7 memory, network", "M7 network + memory")]
fig, ax = plt.subplots(1, 4, figsize=(W1, 2.6), layout="constrained", sharey=True)
for a, k in zip(ax, (1, 7, 30, 90)):
    a.axhspan(3.5, 6.5, color="#F1ECF3", zorder=0, lw=0)
    sub = pd.concat([allm[(allm["model"] == m) & (allm["horizon_days"] == k)] for m, _ in ROWS])
    sub = sub.rename(columns={"gain_vs_M2": "gain"})
    gain_plot(a, sub, [l for _, l in ROWS], xlim=(-0.5, 0.2))
    a.set_title({1: "1 day", 7: "1 week", 30: "1 month", 90: "3 months"}[k], loc="center")
    a.set_yticks(range(len(ROWS)), [l for _, l in ROWS])
    a.set_xticks([-0.4, -0.2, 0, 0.2]); a.set_xlabel("Log score − M2")
fig.legend(handles=[Line2D([], [], color=GREEN, marker="o", lw=2, label="clearly better than M2"),
                    Line2D([], [], color=GREY, marker="o", lw=2, label="no clear difference"),
                    Line2D([], [], color=ORANGE, marker="o", lw=2, label="clearly worse")],
           loc="outside lower center", ncol=3, frameon=False)
save(fig, "fig03_ladder.png")

# ----------------------------------------------------------------------------- Figure 4: depth (RQ4)
fig, ax = plt.subplots(1, 3, figsize=(W1, 2.75), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1, 1]})
a = ax[0]; dd_ = np.arange(366)
hd = summ[summ["quantity"] == "days until half has left"].set_index("operator")
for kind, col, nm in (("drogued", BLUE, "15 m"), ("undrogued", ORANGE, "surface")):
    z = OP[kind]
    a.fill_between(dd_, 100 * z["even_curve_p05"], 100 * z["even_curve_p95"], color=col, alpha=0.25, lw=0)
    a.plot(dd_, 100 * z["even_curve"], color=col, lw=1.8, label=f"{nm}: half gone by day {hd.loc[kind, 'value']:.0f}")
a.axhline(50, color=MUTED, ls="--", lw=0.8)
a.set_xlim(0, 365); a.set_ylim(0, 100); a.set_xticks([0, 90, 180, 270, 365])
a.set_title("(a) Even start: share still in R"); a.set_xlabel("Days after release"); a.set_ylabel("%")
a.legend(frameon=False, loc="upper right", fontsize=7.5); a.grid(axis="y", color=GRIDC, lw=0.6)
for a, kind, nm, yl in ((ax[1], "drogued", "(b) 15 m, day 90", True), (ax[2], "undrogued", "(c) Surface, day 90", False)):
    snap = OP[kind]["release_snaps"][0][days.index(90)]
    im = release_map(a, snap, c_conf, masked, f"{nm}\n{snap[:NCELL].sum():.0%} still in R", ylab=yl)
cb = fig.colorbar(im, ax=ax[1:], orientation="horizontal", shrink=0.7, pad=0.02, ticks=[1e-4, 1e-3, 1e-2, 1e-1])
cb.ax.set_xticklabels(["0.01%", "0.1%", "1%", "10%"]); cb.ax.minorticks_off(); cb.outline.set_visible(False)
cb.set_label("Probability per cell (orange: areas holding 50% solid, 90% dashed)")
save(fig, "fig04_depth.png")

# ----------------------------------------------------------------------------- Figure 5: accuracy on held-out drifters (RQ2)
M4 = "M4 + other drogue type"
hz, x = [1, 7, 30, 90], np.arange(4)
fig, ax = plt.subplots(1, 2, figsize=(W1, 2.7), layout="constrained")
a = ax[0]
for sname, col, off, lab in (("test / drogued", BLUE, -0.09, "15 m, test"), ("test / undrogued", ORANGE, -0.03, "surface, test"),
                             ("recent_test / drogued", BLUE, 0.03, "15 m, 2019–22"), ("recent_test / undrogued", ORANGE, 0.09, "surface, 2019–22")):
    q = pos[(pos["set"] == sname) & (pos["model"] == M4)].set_index("horizon_days").loc[hz]
    k = "in_90pct_area_%"
    rec = sname.startswith("recent")
    a.errorbar(x + off, q[k], yerr=[q[k] - q[k + "_p05"], q[k + "_p95"] - q[k]], color=col, marker="s" if rec else "o",
               mfc="white" if rec else col, ms=5, lw=0 if rec else 1.4, elinewidth=1, capsize=2, label=lab)
a.axhline(90, color=INK, ls="--", lw=1)
a.set_ylim(70, 100); a.set_xticks(x, ["1 day", "1 week", "1 month", "3 months"])
a.set_title("(a) Share of drifters inside the 90% area"); a.set_ylabel("%"); a.legend(frameon=False, ncol=2, fontsize=7, loc="lower left")
a.grid(axis="y", color=GRIDC, lw=0.6)
a = ax[1]
q = pos[(pos["set"] == "test / drogued") & (pos["model"] == M4)].set_index("horizon_days").loc[hz]
qu = pos[(pos["set"] == "test / undrogued") & (pos["model"] == M4)].set_index("horizon_days").loc[hz]
a.plot(x, q["median_persistence_error_km"], marker="o", ms=4, lw=1.6, color=GREY, label="persistence (no movement), 15 m")
a.plot(x, q["median_centre_error_km"], marker="o", ms=4, lw=1.6, color=BLUE, label="M4, 15 m")
a.plot(x, qu["median_persistence_error_km"], marker="o", ms=4, lw=1.2, ls=":", color=GREY, label="persistence, surface")
a.plot(x, qu["median_centre_error_km"], marker="o", ms=4, lw=1.6, color=ORANGE, label="M4, surface")
a.set_xticks(x, ["1 day", "1 week", "1 month", "3 months"]); a.set_ylabel("km")
a.set_title("(b) Median distance error of the forecast centre"); a.legend(frameon=False, fontsize=7, loc="upper left")
a.grid(axis="y", color=GRIDC, lw=0.6)
save(fig, "fig05_accuracy.png")

# ----------------------------------------------------------------------------- extra: release forecast at four times (RQ1), not in the report
fig, ax = plt.subplots(1, 4, figsize=(W1, 2.45), layout="constrained")
for j, (t, lab) in enumerate(((7, "1 week"), (30, "1 month"), (90, "3 months"), (365, "1 year"))):
    snap = OP["drogued"]["release_snaps"][0][days.index(t)]
    im = release_map(ax[j], snap, c_conf, masked, f"{lab}\n{snap[:NCELL].sum():.0%} still in R", ylab=(j == 0))
cb = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.8, pad=0.01, ticks=[1e-4, 1e-3, 1e-2, 1e-1])
cb.ax.set_yticklabels(["0.01%", "0.1%", "1%", "10%"]); cb.ax.minorticks_off(); cb.outline.set_visible(False)
save(fig, "extra_release_15m.png")

# ----------------------------------------------------------------------------- Figure 6: residence and relative density (RQ1)
cf = cellfate[cellfate["operator"] == "drogued"].reset_index(drop=True)
fig, ax = plt.subplots(1, 3, figsize=(W1, 2.75), layout="constrained")
md = cf["median_days_in_R"].values.copy(); md[masked] = np.nan
im = grid_img(ax[0], md, cmap="viridis", vmin=0, vmax=np.nanmax(md))
map_ax(ax[0], "(a) Days until half has left")
cb = fig.colorbar(im, ax=ax[0], orientation="horizontal", shrink=0.85, pad=0.02); cb.outline.set_visible(False); cb.set_label("days, by release cell")
ex = cf[[f"p_left_{e}_within_year" for e in EXITS]].values
cols = np.array([matplotlib.colors.to_rgba(c) for c in (BLUE, ORANGE, PURPLE, GREEN)])
rgb = np.zeros((NCELL, 4)); rgb[:, :3] = cols[ex.argmax(axis=1), :3]
rgb[:, 3] = np.clip(ex.max(axis=1) / np.maximum(ex.sum(axis=1), 1e-9), 0.3, 1); rgb[masked] = 0
ax[1].imshow(rgb.reshape(NY, NX, 4), origin="upper", extent=EXT, interpolation="nearest", zorder=1)
map_ax(ax[1], "(b) Most likely exit edge", ylab=False)
ax[1].legend(handles=[Patch(color=cols[j], label=f"{e}") for j, e in enumerate(EXITS)], loc="upper center",
             bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False, handlelength=1, columnspacing=0.8)
rho = cf["rho_day90"].values.copy(); rho[masked] = np.nan
im = grid_img(ax[2], rho, cmap="RdBu_r", norm=LogNorm(0.2, 5))
sure = (~masked) & ((cf["rho_day90_p05"] > 1) | (cf["rho_day90_p95"] < 1)).values
ax[2].scatter(CELL_LON[sure], CELL_LAT[sure], s=3, color=INK, lw=0, zorder=4)
map_ax(ax[2], "(c) Gathers or loses, 3 months", ylab=False)
cb = fig.colorbar(im, ax=ax[2], orientation="horizontal", shrink=0.85, pad=0.02, ticks=[0.2, 0.5, 1, 2, 5])
cb.ax.set_xticklabels(["0.2", "0.5", "1", "2", "5"]); cb.ax.minorticks_off(); cb.outline.set_visible(False)
cb.set_label("relative density (dot: 90% interval excludes 1)")
save(fig, "fig06_fate_density.png")

# ----------------------------------------------------------------------------- Figure 7: wind and oil (RQ6), if step 13 has run
if os.path.exists(os.path.join(S13, "validation_undrogued.csv")):
    val = pd.read_csv(os.path.join(S13, "validation_undrogued.csv"))
    sc = pd.read_csv(os.path.join(S13, "oil_scenario.csv"))
    zz = np.load(os.path.join(S13, "oil_scenario_snaps.npz"), allow_pickle=True)
    snaps = dict(zip(zz["keys"], zz["snaps"]))
    c0 = int(zz["release_cell"])
    fig = plt.figure(figsize=(W1, 4.4), layout="constrained")
    gsp = fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
    a = fig.add_subplot(gsp[0, :2])
    MODS = [("15 m matrix + wind difference", BLUE, "15 m matrix + wind"), ("surface matrix (fitted on surface drifters)", ORANGE, "surface matrix (own data)"),
            ("15 m matrix + actual wind that day", GREEN, "15 m + actual wind (1 day)")]
    v = val[val["set"] == "test / undrogued"]
    for j, (mname, col, lab) in enumerate(MODS):
        q = v[v["model"] == mname].set_index("horizon_days")
        if q.empty:
            continue
        xs = np.array([hz.index(h) for h in q.index]) + (j - 1) * 0.17
        a.errorbar(xs, q["gain_vs_15m"], yerr=[q["gain_vs_15m"] - q["gain_p05"], q["gain_p95"] - q["gain_vs_15m"]],
                   fmt="o", color=col, ms=4.5, elinewidth=1.2, capsize=2, label=lab)
    a.axhline(0, color=PURPLE, lw=1.2)
    a.set_xticks(range(4), ["1 day", "1 week", "1 month", "3 months"])
    a.set_ylabel("Log score − 15 m matrix")
    a.set_title("(a) Predicting surface test drifters")
    a.legend(frameon=False, fontsize=7, loc="upper left"); a.grid(axis="y", color=GRIDC, lw=0.6)
    a = fig.add_subplot(gsp[0, 2:])
    for mat, col, lab in (("drifter at 15 m", BLUE, "drifter at 15 m"), ("drifter at the surface", ORANGE, "drifter at the surface")):
        q = sc[(sc["release_month"] == "January") & (sc["material"] == mat)]
        a.plot(q["day"], 100 * q["still_in_R"], color=col, marker="o", ms=3, lw=1.5, label=lab)
    for mth, col in (("January", INK), ("July", PURPLE)):
        lo = sc[(sc["release_month"] == mth) & (sc["material"] == "oil, 4% of wind")]
        hi = sc[(sc["release_month"] == mth) & (sc["material"] == "oil, 2% of wind")]
        mid = sc[(sc["release_month"] == mth) & (sc["material"] == "oil, 3% of wind")]
        a.fill_between(mid["day"], 100 * lo["still_in_R"].values, 100 * hi["still_in_R"].values, color=col, alpha=0.15, lw=0)
        a.plot(mid["day"], 100 * mid["still_in_R"], color=col, marker="o", ms=3, lw=1.5, ls="--" if mth == "July" else "-",
               label=f"oil, 3% of wind, {mth} (band 2–4%)")
    a.set_xticks([7, 30, 90]); a.set_xlabel("Days after release"); a.set_ylabel("Still in R (%)")
    a.set_ylim(25, 102)
    a.set_title("(b) Confluence release: still in R"); a.legend(frameon=False, fontsize=6.8, loc="lower left")
    a.grid(axis="y", color=GRIDC, lw=0.6)
    for j, (key, lab) in enumerate((("January|drifter at 15 m|90", "(c) Drifter at 15 m"),
                                     ("January|drifter at the surface|90", "(d) Drifter at surface"),
                                     ("January|oil, 3% of wind|90", "(e) Oil 3%, January"),
                                     ("July|oil, 3% of wind|90", "(f) Oil 3%, July"))):
        a = fig.add_subplot(gsp[1, j])
        snap = snaps[key]
        im = release_map(a, snap, c0, masked, f"{lab}\nday 90: {snap[:NCELL].sum():.0%} in R", ylab=(j == 0))
    save(fig, "fig07_wind_oil.png")
# Figure 9 is drawn by step 14; copy it here so every report figure sits in one folder.
# Figure 8 is a screenshot of viewer/bmc_drift_viewer.html (release at 38.5°S 53.5°W, 15 m, 30 days), taken by hand.
import shutil
# The method equation (Section 4.1), drawn with matplotlib's mathtext
plt.rcParams.update({"mathtext.fontset": "cm"})
fig = plt.figure(figsize=(6.9, 0.62))
fig.text(0.5, 0.5, r"$\hat{P}_{ij}=\dfrac{n_{ij}}{n_i},\qquad p_k=e_i\,P^{\,k},\qquad \rho_k(c)=\dfrac{(u P^{\,k})_c\;/\;\sum_{c'\in R}(u P^{\,k})_{c'}}{u_c}$",
         ha="center", va="center", fontsize=13)
save(fig, "equation.png")
if os.path.exists(os.path.join(S14, "fig_drifter_age.png")):
    shutil.copy(os.path.join(S14, "fig_drifter_age.png"), os.path.join(FIG, "fig09_drifter_age.png"))
print("Report figures in", FIG)
