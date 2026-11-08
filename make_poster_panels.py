"""
DATA3001 Group S01-G8 - poster panels as separate high-resolution images
=========================================================================
Run in Colab AFTER make_initial_analysis.py has run in the same session.
(If the runtime was reset: Cell 1 !pip install -q "zarr<3", Cell 2 make_initial_analysis.py,
 then this cell.)

Saves four PNGs at 450 dpi to MyDrive/DATA3001/initial_analysis/poster_panels:
  panel_tracks.png             -> box 3, left map
  panel_drifters_per_cell.png  -> box 3, the two drifters-per-cell maps
  panel_release.png            -> box 6 (a), and the map inside the viewer mock-up
  panel_relative_density.png   -> box 6 (b)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = os.path.join(BASE, "poster_panels")
os.makedirs(OUT, exist_ok=True)
DPI = 450


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("saved", name)


df = prepare(data)
daily, tr = daily_transitions(df)

# ---- 1. drifter tracks with region R
fig, ax = plt.subplots(figsize=(3.6, 4.0), layout="constrained")
track_map(ax, df)
ax.set_title("")
save(fig, "panel_tracks.png")

# ---- 2. distinct drifters per cell, drogued and undrogued, shared colour scale
r = df[df["inR"]].copy()
r["cell"] = cell_of(r["lat"].values, r["lon"].values)
grids = {}
for flag in (True, False):
    cnt = r[r["drogued"] == flag].groupby("cell")["id"].nunique()
    g = np.full(NCELL, np.nan)
    g[cnt.index.values] = cnt.values
    grids[flag] = as_grid(g)
vmax = np.nanmax([np.nanmax(grids[True]), np.nanmax(grids[False])])
fig, axs = plt.subplots(1, 2, figsize=(6.2, 3.9), layout="constrained")
for ax, flag, name in zip(axs, (True, False), ("Drogued", "Undrogued")):
    im = show_grid(ax, grids[flag], cmap="Blues", vmin=0, vmax=vmax)
    map_axes(ax, f"{name} drifters per cell")
fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.6, label="Distinct drifters")
save(fig, "panel_drifters_per_cell.png")

# ---- operator from drogued drifters (same as Figure 2)
P, counts_out = estimate_P(tr[tr["drogued"]])
supported = counts_out > 0
xc = np.linspace(LON_MIN + CELL_DEG / 2, LON_MAX - CELL_DEG / 2, NX)
yc = np.linspace(LAT_MAX - CELL_DEG / 2, LAT_MIN + CELL_DEG / 2, NY)

# ---- 3. release forecast at day 30, with 90% area at day 7
p = np.zeros(NCELL + 4)
p[cell_of(np.array([RELEASE[0]]), np.array([RELEASE[1]]))[0]] = 1.0
snaps = {}
for k in range(1, H_LONG + 1):
    p = p @ P
    if k in (H_SHORT, H_LONG):
        snaps[k] = p.copy()
v = snaps[H_SHORT][:NCELL] / snaps[H_SHORT][:NCELL].sum()
order = np.argsort(v)[::-1]
keep = order[: np.searchsorted(np.cumsum(v[order]), 0.9) + 1]
mask = np.zeros(NCELL); mask[keep] = 1
pl = as_grid(snaps[H_LONG][:NCELL]); pl[pl <= 0] = np.nan
fig, ax = plt.subplots(figsize=(3.4, 4.3), layout="constrained")
im = show_grid(ax, pl, cmap="Blues", norm=LogNorm(1e-4, max(1e-3, np.nanmax(pl))))
ax.contour(xc, yc, as_grid(mask), levels=[0.5], colors="#d9822b", linewidths=1.4)
ax.plot(RELEASE[1], RELEASE[0], marker="*", color="black", markersize=9)
map_axes(ax, "")
ax.set_xticks([-60, -55, -50, -45, -40])
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.85, pad=0.02, label=f"Probability (orange: 90% area, day {H_SHORT})")
save(fig, "panel_release.png")

# ---- 4. relative density at day 30 from an even, area-weighted start
u = np.zeros(NCELL + 4)
u[:NCELL] = area_weights() * supported
u /= u.sum()
vv = u @ np.linalg.matrix_power(P, H_LONG)
inside = vv[:NCELL] / vv[:NCELL].sum()
rho = np.where(supported, inside / np.where(u[:NCELL] > 0, u[:NCELL], np.nan), np.nan)
fig, ax = plt.subplots(figsize=(3.4, 4.3), layout="constrained")
im = show_grid(ax, as_grid(rho), cmap="RdBu_r", norm=LogNorm(0.1, 10))
map_axes(ax, "")
ax.set_xticks([-60, -55, -50, -45, -40])
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.85, pad=0.02, label="Relative density (>1 gathers)")
save(fig, "panel_relative_density.png")

print("\nDone. Download the four PNGs from", OUT)
