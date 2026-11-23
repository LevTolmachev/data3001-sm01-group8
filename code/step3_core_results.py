"""
DATA3001 Group S01-G8 - Step 3: core results (RQ1, RQ2, RQ3) and the final operator
=====================================================================================

Run AFTER step2_models_M0_M1_M2.py (it re-uses step 2's functions, its frozen splits.csv
and its chosen M2 settings in m2_tuning.csv).

What it does
  A. RQ3  Which cell size and shape does the data support?  (validation drifters, M1)
          square 0.5 deg, 1 deg, 2 deg and hexagons with the same area as 1 deg and 2 deg cells,
          plus M2 at 1 deg as the reference. Scores are made comparable across grids by
          scoring density per km2 instead of probability per cell.
  B. RQ3  Which time step, and does the Markov assumption hold?  (validation drifters)
          operators with a 1, 2, 3 and 6 day step, iterated to 6-90 days, against an operator
          fitted DIRECTLY on 6-90 day moves. If iterating a 1-day operator is as good as fitting
          the long move directly, the Markov (memoryless) assumption is good enough.
  C. RQ1  Where material goes, how long it stays, where it leaves, where R gathers / loses it.
          The FINAL operator (M2, 1 deg, refitted on training + validation drifters) with
          bootstrap uncertainty (whole drifters resampled). Exports the operator and 30
          bootstrap operators for the viewer (D2 / D4).
  D. RQ2  The final held-out test, run ONCE (RUN_FINAL_TEST = True) on the test drifters and
          on the most recent years (recent_test). Every number gets a 90% interval from
          resampling whole test drifters, and the paired difference between models tells us
          whether the extra complexity is worth it.

How to run in Google Colab
  Cell 1:  !pip install -q "zarr<3"
  Cell 2:  paste step2_models_M0_M1_M2.py and run it (once is enough: it saves splits and tuning)
  Cell 3:  paste THIS file and run it.
  Locally:  put both .py files in one folder and run  python step3_core_results.py
            (set DATA3001_BASE to the folder holding gdp_bmc_extract.parquet).
  Outputs:  MyDrive/DATA3001/initial_analysis/models/step3  (CSVs, PNGs, operator files).
  Takes roughly 5-15 minutes with the real extract.

Leave RUN_FINAL_TEST = False until sections A-C are agreed by the group. Then set it to True,
run once, and report those numbers whatever they are.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="divide by zero")

# ----------------------------------------------------------------------------- step 2 functions
if "m2_smoothed" not in globals():           # not pasted after step 2 in the same Colab session
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _here = os.getcwd()
    sys.path.insert(0, _here)
    try:
        from step2_models_M0_M1_M2 import *          # noqa: F401,F403  (functions + CONFIG)
    except ImportError as e:
        raise RuntimeError("Run step2_models_M0_M1_M2.py first (paste it in the cell above), "
                           "or keep it in the same folder as this file.") from e

# ----------------------------------------------------------------------------- CONFIG
RUN_FINAL_TEST = False               # True ONLY once, at the very end (section D)
S3_SQUARE_DEG = [0.5, 1.0, 2.0]      # square cell sizes (degrees)
S3_HEX_EQUIV_DEG = [1.0, 2.0]        # hexagons with the same area as these square cells (at 40S)
S3_GRID_HORIZONS = [7, 30, 90]       # horizons for the cell-size comparison (days)
S3_DT = [1, 2, 3, 6]                 # time steps (days)
S3_DT_HORIZONS = [6, 12, 30, 60, 90] # divisible by every time step above
S3_TEST_HORIZONS = [1, 7, 30, 90]    # horizons for the final test
S3_N_BOOT_SCORE = 500                # bootstrap resamples of drifters for score intervals
S3_N_BOOT_OPERATOR = 100             # bootstrap refits of the final operator
S3_N_SAVE_OPERATORS = 30             # bootstrap operators saved for the viewer
S3_RELEASES = {                      # example release points (lat, lon); checked against the mask
    "Confluence 38S 54W": (-38.0, -54.0),
    "Brazil Current 33S 50W": (-33.0, -50.0),
    "Malvinas Current 44S 58W": (-44.0, -58.0),
}
S3_SHOW_DAYS = [7, 30, 365]          # a week, a month, a year
EARTH_KM = 6371.0

OUT3 = os.path.join(OUT, "step3")
os.makedirs(OUT3, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
GRID_COL = {"square 0.5°": "#9ecae1", "square 1°": "#1f6fb2", "square 2°": "#08306b",
            "hex ≈1°": "#d9822b", "hex ≈2°": "#8c4a12", "M2 square 1° (reference)": "#2e9e6a"}
DT_COL = {1: "#1f6fb2", 2: "#6baed6", 3: "#8a5cc2", 6: "#d9822b", "direct": "#000000"}
MODEL_COL = {"M0a persistence": "#8c8c8c", "M0b displacement": "#5b6b7a",
             "M1 empirical": "#1f6fb2", "M2 smoothed": "#d9822b"}


def save3(fig, name):
    fig.savefig(os.path.join(OUT3, name), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("  saved", name)


# ----------------------------------------------------------------------------- grids
def _cell_area_samples(step=0.02):
    """Centres and exact areas (km2) of a fine lat/lon lattice covering R (for hex areas)."""
    lat = np.arange(LAT_MIN + step / 2, LAT_MAX, step)
    lon = np.arange(LON_MIN + step / 2, LON_MAX, step)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    a = EARTH_KM ** 2 * np.deg2rad(step) * (np.sin(np.deg2rad(LA + step / 2)) - np.sin(np.deg2rad(LA - step / 2)))
    return LA.ravel(), LO.ravel(), a.ravel()


class SquareGrid:
    """Square lat/lon cells, indexed row-major from the north-west corner (same as step 2)."""
    kind = "square"

    def __init__(self, deg):
        self.deg = deg
        self.name = f"square {deg:g}°"
        self.nx = int(round((LON_MAX - LON_MIN) / deg))
        self.ny = int(round((LAT_MAX - LAT_MIN) / deg))
        self.ncell = self.nx * self.ny
        iy, ix = np.divmod(np.arange(self.ncell), self.nx)
        self.clat = LAT_MAX - (iy + 0.5) * deg
        self.clon = LON_MIN + (ix + 0.5) * deg
        top, bot = np.deg2rad(self.clat + deg / 2), np.deg2rad(self.clat - deg / 2)
        self.area = EARTH_KM ** 2 * np.deg2rad(deg) * (np.sin(top) - np.sin(bot))

    def cell_of(self, lat, lon):
        ix = np.clip(np.floor((np.asarray(lon) - LON_MIN) / self.deg).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor((LAT_MAX - np.asarray(lat)) / self.deg).astype(int), 0, self.ny - 1)
        return iy * self.nx + ix


class HexGrid:
    """Pointy-top hexagons of equal area on a local equal-distance projection centred on R.
    Cells are clipped to R; their areas and centres are the clipped ones."""
    kind = "hex"

    def __init__(self, equiv_deg):
        self.name = f"hex ≈{equiv_deg:g}°"
        self.lat0 = 0.5 * (LAT_MIN + LAT_MAX)
        self.lon0 = 0.5 * (LON_MIN + LON_MAX)
        self.coslat0 = np.cos(np.deg2rad(self.lat0))
        target = EARTH_KM ** 2 * np.deg2rad(equiv_deg) ** 2 * self.coslat0   # area of the square cell at 40S
        self.s = np.sqrt(2 * target / (3 * np.sqrt(3)))                      # hexagon side (km)
        la, lo, a = _cell_area_samples()
        keys = self._key(la, lo)
        self.keys, inv = np.unique(keys, return_inverse=True)
        self.ncell = len(self.keys)
        self.area = np.bincount(inv, weights=a, minlength=self.ncell)
        self.clat = np.bincount(inv, weights=a * la, minlength=self.ncell) / self.area
        self.clon = np.bincount(inv, weights=a * lo, minlength=self.ncell) / self.area

    def _xy(self, lat, lon):
        x = EARTH_KM * np.deg2rad(np.asarray(lon, float) - self.lon0) * self.coslat0
        y = EARTH_KM * np.deg2rad(np.asarray(lat, float) - self.lat0)
        return x, y

    def _key(self, lat, lon):
        x, y = self._xy(lat, lon)
        q = (np.sqrt(3) / 3 * x - y / 3) / self.s
        r = (2 / 3 * y) / self.s
        cx, cz = q, r
        cy = -cx - cz
        rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
        dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
        fx = (dx > dy) & (dx > dz)
        fz = ~fx & (dz >= dy)
        rx = np.where(fx, -ry - rz, rx)
        rz = np.where(fz, -rx - ry, rz)
        return (rx.astype(np.int64) + 10_000) * 100_000 + (rz.astype(np.int64) + 10_000)

    def cell_of(self, lat, lon):
        k = self._key(lat, lon)
        i = np.clip(np.searchsorted(self.keys, k), 0, self.ncell - 1)
        miss = self.keys[i] != k                       # a sliver of hexagon too thin to be sampled
        if miss.any():
            x, y = self._xy(np.asarray(lat)[miss], np.asarray(lon)[miss])
            cx, cy = self._xy(self.clat, self.clon)
            i[miss] = np.argmin((x[:, None] - cx) ** 2 + (y[:, None] - cy) ** 2, axis=1)
        return i


# ----------------------------------------------------------------------------- cases and operators
def starts_for(daily, ids):
    """Every drifter-day inside R (stride 1) with how its visit ends (from step 2)."""
    st = visit_starts(daily, ids, stride=1)
    snap = daily[daily["id"].isin(ids)][["id", "time", "lat", "lon"]].rename(
        columns={"time": "t_end", "lat": "lat_end", "lon": "lon_end"})
    return st, snap


def cases_at(starts, k):
    """What happened k days after each start: the position if it stayed in R throughout,
    or the edge it first left through. Starts whose record stops inside R before day k are dropped."""
    st0, snap = starts
    gone = st0["exited"] & (st0["days"] <= k)
    stay = (st0["exited"] & (st0["days"] > k)) | (~st0["exited"] & (st0["days"] >= k))
    a = st0[stay].assign(t_end=lambda x: x["time"] + pd.Timedelta(days=k)).merge(snap, on=["id", "t_end"])
    b = st0[gone].assign(lat_end=np.nan, lon_end=np.nan)
    c = pd.concat([a.drop(columns="t_end"), b], ignore_index=True)
    c["inn"] = np.r_[np.ones(len(a), bool), np.zeros(len(b), bool)]
    return c


def encode(c, grid):
    """Start cell and end state (cell, or ncell + exit edge) on a given grid."""
    frm = grid.cell_of(c["lat"].values, c["lon"].values)
    state = np.empty(len(c), dtype=int)
    inn = c["inn"].values
    state[inn] = grid.cell_of(c.loc[inn, "lat_end"].values, c.loc[inn, "lon_end"].values)
    state[~inn] = [grid.ncell + EXITS.index(e) for e in c.loc[~inn, "edge"]]
    return frm, state


def fit_operator(frm, to, n_cell, w=None):
    """Empirical operator (M1) on any grid: rows with no moves hold their mass, exits absorb."""
    C = np.zeros((n_cell + 4, n_cell + 4))
    np.add.at(C, (frm, to), np.ones(len(frm)) if w is None else w)
    tot = C[:n_cell].sum(axis=1)
    has = tot > 0
    C[:n_cell][has] /= tot[has, None]
    empty = np.flatnonzero(~has)
    C[empty, empty] = 1.0
    C[n_cell:] = 0
    C[n_cell:, n_cell:] = np.eye(4)
    return C, tot


def mpow(P, k):
    return np.linalg.matrix_power(P, int(k))


# ----------------------------------------------------------------------------- scoring
D_FLOOR = LOG_FLOOR / (EARTH_KM ** 2 * np.deg2rad(1.0) ** 2 * np.cos(np.deg2rad(40)))   # per km2


def evaluate(Pk, grid, c):
    """Per-case scores of a k-day forecast Pk for cases c. Work is done once per start cell."""
    frm, state = encode(c, grid)
    n = grid.ncell
    u, inv = np.unique(frm, return_inverse=True)
    R = Pk[u]
    order = np.argsort(-R, axis=1, kind="stable")
    csum = np.cumsum(np.take_along_axis(R, order, axis=1), axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(R.shape[1]), R.shape), axis=1)
    area_full = np.r_[grid.area, np.zeros(4)]
    out = {}
    for q in (0.5, 0.9):
        nkeep = (csum < q - 1e-12).sum(axis=1) + 1
        out[f"hit{int(q * 100)}"] = (rank[inv, state] < nkeep[inv]).astype(float)
        if q == 0.9:
            inside_set = rank < nkeep[:, None]
            out["area90_km2"] = (inside_set * area_full[None, :]).sum(axis=1)[inv]
    p = R[inv, state]
    in_state = state < n
    out["logp"] = np.log(np.maximum(p, LOG_FLOOR))
    out["logd"] = np.where(in_state, np.log(np.maximum(p / area_full[np.minimum(state, n - 1)], D_FLOOR)),
                           np.log(np.maximum(p, LOG_FLOOR)))
    pin = R[:, :n]
    mass = pin.sum(axis=1)
    cl = (pin @ grid.clat) / np.where(mass > 0, mass, 1)
    co = (pin @ grid.clon) / np.where(mass > 0, mass, 1)
    ok = c["inn"].values & (mass[inv] > 0)
    err = np.full(len(c), np.nan)
    pers = np.full(len(c), np.nan)
    err[ok] = haversine_km(cl[inv][ok], co[inv][ok], c["lat_end"].values[ok], c["lon_end"].values[ok])
    inn = c["inn"].values
    pers[inn] = haversine_km(c["lat"].values[inn], c["lon"].values[inn], c["lat_end"].values[inn], c["lon_end"].values[inn])
    out["err_km"], out["pers_km"] = err, pers
    out["id"] = c["id"].values
    return out


STAT = {"log_score": ("logp", np.nanmean), "log_score_per_km2": ("logd", np.nanmean),
        "in_50pct_area_%": ("hit50", lambda x: 100 * np.nanmean(x)),
        "in_90pct_area_%": ("hit90", lambda x: 100 * np.nanmean(x)),
        "area_90pct_1000km2": ("area90_km2", lambda x: np.nanmedian(x) / 1e3),
        "median_centre_error_km": ("err_km", np.nanmedian),
        "median_persistence_error_km": ("pers_km", np.nanmedian)}


def drifter_index(ids, seed=SEED, n_boot=S3_N_BOOT_SCORE):
    """Bootstrap resamples of whole drifters, as lists of case indices (shared across models,
    so differences between models are paired)."""
    codes, uniq = pd.factorize(pd.Series(ids))
    groups = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
            for _ in range(n_boot)]


def summarise(res, boot=None, stats=STAT):
    row = {"cases": len(res["id"]), "drifters": len(np.unique(res["id"]))}
    for name, (key, f) in stats.items():
        row[name] = f(res[key])
        if boot is not None:
            vals = np.array([f(res[key][ix]) for ix in boot])
            row[name + "_p05"], row[name + "_p95"] = np.nanpercentile(vals, [5, 95])
    return row


def paired_diff(res_a, res_b, boot, key="logp"):
    """Mean score of A minus B, with a 90% interval from resampled drifters."""
    d = res_a[key] - res_b[key]
    vals = np.array([np.nanmean(d[ix]) for ix in boot])
    return np.nanmean(d), *np.nanpercentile(vals, [5, 95])


# ----------------------------------------------------------------------------- setup
t_start = time.time()
if "data" not in globals():
    print(f"Reading cached extract {INPUT_FILE}")
    data = pd.read_parquet(INPUT_FILE)
df3 = prepare(data)
daily3, tr3 = daily_transitions(df3)
if not os.path.exists(SPLIT_FILE) or not os.path.exists(os.path.join(OUT, "m2_tuning.csv")):
    raise RuntimeError("splits.csv / m2_tuning.csv not found: run step 2 once first.")
split3 = make_splits(daily3)
daily3["split"] = daily3["id"].map(split3)
tr3["split"] = tr3["id"].map(split3)
ids3 = {s: set(daily3.loc[daily3["split"] == s, "id"]) for s in ("train", "val", "test", "recent_test")}
tune3 = pd.read_csv(os.path.join(OUT, "m2_tuning.csv"))
best3 = tune3.loc[tune3["val_log_score"].idxmax()]
SIG3, ALP3 = float(best3["sigma"]), float(best3["alpha"])
print(f"Drogue set: {DROGUE_SET}. M2 settings from step 2: sigma = {SIG3}, alpha = {ALP3}")
print("Drifters per split: " + ", ".join(f"{s} {len(v)}" for s, v in ids3.items()))

print("\nCollecting starts (every drifter-day inside R) ...")
starts_train = starts_for(daily3, ids3["train"])
starts_val = starts_for(daily3, ids3["val"])
print(f"  training starts {len(starts_train[0]):,}, validation starts {len(starts_val[0]):,}")

# sanity check: the 1-day M1 here must equal step 2's M1
_g1 = SquareGrid(CELL_DEG)
_c1 = cases_at(starts_train, 1)
_P1_here, _ = fit_operator(*encode(_c1, _g1), _g1.ncell)
_P1_step2 = m1_empirical(move_table(tr3[tr3["split"] == "train"]))
print(f"  check: max |M1 here - M1 step 2| = {np.abs(_P1_here - _P1_step2).max():.2e} (should be ~0)")


# ============================================================================= A. RQ3 grids
print("\n" + "=" * 78)
print("A. RQ3  CELL SIZE AND SHAPE (M1 on training drifters, scored on validation drifters)")
print("=" * 78)
grids = [SquareGrid(d) for d in S3_SQUARE_DEG] + [HexGrid(d) for d in S3_HEX_EQUIV_DEG]
val_cases = {k: cases_at(starts_val, k) for k in sorted(set(S3_GRID_HORIZONS + S3_DT_HORIZONS + S3_TEST_HORIZONS))}
train_cases_1 = _c1
boot_val = {k: drifter_index(val_cases[k]["id"].values) for k in S3_GRID_HORIZONS}

grid_rows, grid_res = [], {}
for g in grids:
    frm, to = encode(train_cases_1, g)
    P, moves = fit_operator(frm, to, g.ncell)
    drifters_per_cell = (pd.DataFrame({"cell": frm, "id": train_cases_1["id"].values})
                         .groupby("cell")["id"].nunique().reindex(range(g.ncell)).fillna(0).values)
    visited = moves > 0
    support = {"cells_visited": int(visited.sum()),
               f"cells_with_{MIN_DRIFTERS}+_drifters_%": 100 * (drifters_per_cell[visited] >= MIN_DRIFTERS).mean(),
               "median_moves_per_cell": float(np.median(moves[visited])),
               "median_cell_area_1000km2": float(np.median(g.area)) / 1e3}
    Pk, last = np.eye(g.ncell + 4), 0
    for k in S3_GRID_HORIZONS:
        Pk = Pk @ mpow(P, k - last)
        last = k
        r = evaluate(Pk, g, val_cases[k])
        grid_res[(g.name, k)] = r
        grid_rows.append({"grid": g.name, "cells": g.ncell, "horizon_days": k, **support,
                          **summarise(r, boot_val[k])})
    print(f"  {g.name:11s} done: {g.ncell} cells, {support['cells_visited']} visited, "
          f"{support[f'cells_with_{MIN_DRIFTERS}+_drifters_%']:.0f}% of them with >= {MIN_DRIFTERS} drifters")

# M2 at 1 deg (step 2 model, training only) as the reference
ocean_tr = ocean_mask(daily3[daily3["split"] == "train"])
m_tr = move_table(tr3[tr3["split"] == "train"])
D_tr = Displacements(m_tr, ocean_tr)
P_m2_tr = m2_smoothed(D_tr, SIG3, ALP3)
Pk, last = np.eye(NSTATE), 0
for k in S3_GRID_HORIZONS:
    Pk = Pk @ mpow(P_m2_tr, k - last)
    last = k
    r = evaluate(Pk, _g1, val_cases[k])
    grid_res[("M2 square 1° (reference)", k)] = r
    grid_rows.append({"grid": "M2 square 1° (reference)", "cells": NCELL, "horizon_days": k,
                      **summarise(r, boot_val[k])})
grid_tab = pd.DataFrame(grid_rows)
grid_tab.to_csv(os.path.join(OUT3, "rq3_cell_size_shape.csv"), index=False)
pd.set_option("display.width", 220)
show = ["grid", "horizon_days", "log_score_per_km2", "log_score_per_km2_p05", "log_score_per_km2_p95",
        "in_90pct_area_%", "area_90pct_1000km2", "median_centre_error_km"]
print(grid_tab[show].round(2).to_string(index=False))

print("\nPaired difference in log score per km2 against square 1° M1 (positive = better, 90% interval):")
gain = []
for k in S3_GRID_HORIZONS:
    for name in grid_tab["grid"].unique():
        d, lo, hi = paired_diff(grid_res[(name, k)], grid_res[("square 1°", k)], boot_val[k], key="logd")
        gain.append((name, k, d, lo, hi))
        if name == "square 1°":
            continue
        verdict = "better" if lo > 0 else ("worse" if hi < 0 else "no clear difference")
        print(f"  day {k:3d}  {name:26s} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")
gain = pd.DataFrame(gain, columns=["grid", "horizon_days", "gain_vs_square1", "gain_p05", "gain_p95"])
grid_tab = grid_tab.merge(gain, on=["grid", "horizon_days"])
grid_tab.to_csv(os.path.join(OUT3, "rq3_cell_size_shape.csv"), index=False)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
for name, g in grid_tab.groupby("grid", sort=False):
    kw = dict(color=GRID_COL.get(name, "k"), marker="o", lw=1.8, ms=4, label=name,
              ls="--" if name.startswith("M2") else "-")
    x = g["horizon_days"].values
    ax[0].plot(x, g["gain_vs_square1"], **kw)
    if name != "square 1°":
        ax[0].fill_between(x, g["gain_p05"], g["gain_p95"], color=kw["color"], alpha=0.15, lw=0)
    ax[1].plot(x, g["in_90pct_area_%"], **kw)
    ax[2].plot(x, g["area_90pct_1000km2"], **kw)
ax[1].axhline(90, color="#5b6b7a", ls=":", lw=1)
ax[1].set_ylim(0, 100)
ax[0].axhline(0, color="#5b6b7a", lw=0.8)
for a, t, yl in zip(ax, ["(a) Skill vs square 1° (above 0 = better)", "(b) Calibration (target 90%)",
                         "(c) Size of the 90% area (smaller = sharper)"],
                    ["Gain in log score per km² (90% interval)", "Validation drifters in 90% area (%)",
                     "Median area in R (1000 km²; 0 = mostly left R)"]):
    a.set_title(t, loc="left"); a.set_xlabel("Horizon (days)"); a.set_ylabel(yl)
    a.set_xticks(S3_GRID_HORIZONS); a.grid(axis="y", color="#e6e6e6", lw=0.6); a.set_axisbelow(True)
ax[0].legend(frameon=False, fontsize=8)
save3(fig, "fig_rq3_cell_size_shape.png")


# ============================================================================= B. RQ3 time step / Markov
print("\n" + "=" * 78)
print("B. RQ3  TIME STEP AND MARKOV CHECK (square 1°, M1)")
print("=" * 78)
g = _g1
boot_dt = {H: drifter_index(val_cases[H]["id"].values) for H in S3_DT_HORIZONS}
P_dt = {}
for dt in S3_DT:
    c = cases_at(starts_train, dt)
    P_dt[dt], _ = fit_operator(*encode(c, g), g.ncell)
dt_rows, dt_res = [], {}
for H in S3_DT_HORIZONS:
    cH = cases_at(starts_train, H)
    P_direct, cnt = fit_operator(*encode(cH, g), g.ncell)
    variants = {dt: mpow(P_dt[dt], H // dt) for dt in S3_DT} | {"direct": P_direct}
    for name, Pk in variants.items():
        r = evaluate(Pk, g, val_cases[H])
        dt_res[(name, H)] = r
        dt_rows.append({"step": name, "horizon_days": H, **summarise(r, boot_dt[H])})
    # distance between "iterate 1 day H times" and "fit H days directly", and a split-half reference
    rows = cnt >= 20
    tv = 0.5 * np.abs(variants[1][:g.ncell] - P_direct[:g.ncell]).sum(axis=1)
    ids_tr = np.array(sorted(ids3["train"]))
    half = set(np.random.default_rng(SEED).permutation(ids_tr)[: len(ids_tr) // 2])
    inA = cH["id"].isin(half).values
    PA, cA = fit_operator(*encode(cH[inA], g), g.ncell)
    PB, cB = fit_operator(*encode(cH[~inA], g), g.ncell)
    both = (cA >= 10) & (cB >= 10)
    tv_half = 0.5 * np.abs(PA[:g.ncell] - PB[:g.ncell]).sum(axis=1)
    dt_rows[-1].update({"tv_iterated_vs_direct": np.average(tv[rows], weights=cnt[rows]) if rows.any() else np.nan,
                        "tv_split_half_reference": np.average(tv_half[both], weights=(cA + cB)[both]) if both.any() else np.nan})
dt_tab = pd.DataFrame(dt_rows)
dt_tab.to_csv(os.path.join(OUT3, "rq3_time_step_markov.csv"), index=False)
print(dt_tab[["step", "horizon_days", "log_score", "log_score_p05", "log_score_p95", "in_90pct_area_%",
              "median_centre_error_km"]].round(2).to_string(index=False))
mk = dt_tab[dt_tab["step"] == "direct"][["horizon_days", "tv_iterated_vs_direct", "tv_split_half_reference"]]
print("\nMarkov check: how different is 'iterate the 1-day operator' from 'fit the H-day move directly'?")
print("(total-variation distance per start cell, 0 = identical, 1 = no overlap; the split-half value is")
print(" the difference between two halves of the training drifters, i.e. what sampling noise alone gives)")
print(mk.round(3).to_string(index=False))
print("\nPaired difference in log score, iterated 1-day operator minus direct fit (positive = iterating is better):")
for H in S3_DT_HORIZONS:
    d, lo, hi = paired_diff(dt_res[(1, H)], dt_res[("direct", H)], boot_dt[H])
    print(f"  day {H:3d}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
for name, gg in dt_tab.groupby("step", sort=False):
    lab = "fit directly (no Markov)" if name == "direct" else f"{name}-day step"
    kw = dict(color=DT_COL[name], marker="o", lw=1.8, ms=4, label=lab, ls="--" if name == "direct" else "-")
    ax[0].plot(gg["horizon_days"], gg["log_score"], **kw)
    ax[1].plot(gg["horizon_days"], gg["in_90pct_area_%"], **kw)
ax[2].plot(mk["horizon_days"], mk["tv_iterated_vs_direct"], marker="o", color="#1f6fb2", lw=1.8, label="1-day operator iterated vs direct")
ax[2].plot(mk["horizon_days"], mk["tv_split_half_reference"], marker="o", color="#8c8c8c", lw=1.8, ls="--", label="sampling noise (split-half)")
ax[1].axhline(90, color="#5b6b7a", ls=":", lw=1); ax[1].set_ylim(0, 100)
for a, t, yl in zip(ax, ["(a) Log score by time step", "(b) Calibration (target 90%)", "(c) Markov check"],
                    ["Mean log score", "Validation drifters in 90% area (%)", "Mean distance between forecasts"]):
    a.set_title(t, loc="left"); a.set_xlabel("Horizon (days)"); a.set_ylabel(yl)
    a.set_xticks(S3_DT_HORIZONS); a.grid(axis="y", color="#e6e6e6", lw=0.6); a.set_axisbelow(True)
ax[0].legend(frameon=False, fontsize=8); ax[2].legend(frameon=False, fontsize=8)
save3(fig, "fig_rq3_time_step_markov.png")


# ============================================================================= C. RQ1 final operator
print("\n" + "=" * 78)
print("C. RQ1  FINAL OPERATOR (M2, 1°, training + validation drifters) WITH BOOTSTRAP")
print("=" * 78)
fit_ids = ids3["train"] | ids3["val"]
ocean_f = ocean_mask(daily3[daily3["id"].isin(fit_ids)])
m_f = move_table(tr3[tr3["id"].isin(fit_ids)])
D_f = Displacements(m_f, ocean_f)
P_final = m2_smoothed(D_f, SIG3, ALP3)
per_cell_f = m_f.groupby("from_cell").agg(moves=("id", "size"), drifters=("id", "nunique"))
n_drift_f = per_cell_f["drifters"].reindex(range(NCELL)).fillna(0).values
n_moves_f = per_cell_f["moves"].reindex(range(NCELL)).fillna(0).values
masked_f = (n_drift_f < MIN_DRIFTERS) | ~ocean_f
print(f"Fitted on {m_f['id'].nunique()} drifters, {len(m_f):,} daily moves; masked cells: {int(masked_f.sum())} of {NCELL}")

# release points: move to the nearest supported cell if needed
rel_cells = {}
for name, (la, lo) in S3_RELEASES.items():
    c = int(cell_of(np.array([la]), np.array([lo]))[0])
    if masked_f[c]:
        ok = np.flatnonzero(~masked_f)
        c2 = ok[np.argmin(haversine_km(la, lo, CELL_LAT[ok], CELL_LON[ok]))]
        print(f"  note: {name} is in a masked cell; using the nearest supported cell "
              f"({CELL_LAT[c2]:.1f}, {CELL_LON[c2]:.1f})")
        c = int(c2)
    rel_cells[name] = c

YEAR = max(S3_SHOW_DAYS)
w_even = area_weights() * ocean_f
u_even = np.r_[w_even / w_even.sum(), np.zeros(4)]


def rq1_outputs(P):
    """Everything RQ1 reports, from one operator."""
    o = {}
    for name, c in rel_cells.items():
        p = np.zeros(NSTATE); p[c] = 1.0
        curve, snaps = [1.0], {}
        for t in range(1, YEAR + 1):
            p = p @ P
            curve.append(p[:NCELL].sum())
            if t in S3_SHOW_DAYS:
                snaps[t] = p.copy()
        o[name] = {"curve": np.array(curve), "snaps": snaps}
    v, rho = u_even.copy(), {}
    for t in range(1, 91):
        v = v @ P
        if t in (30, 90):
            inside = v[:NCELL] / max(v[:NCELL].sum(), 1e-12)
            rho[t] = np.where(ocean_f, inside / np.where(u_even[:NCELL] > 0, u_even[:NCELL], np.nan), np.nan)
    o["rho"] = rho
    return o


base = rq1_outputs(P_final)
still_c, M_year = propagate_cells(P_final, YEAR)           # every start cell, up to a year
expected_days = still_c[:, 1:].sum(axis=1)
below = still_c <= 0.5
median_days = np.where(below.any(axis=1), below.argmax(axis=1), np.nan)   # NaN = more than a year
exit_prob = M_year[:, NCELL:]                               # left via N/S/E/W by day 365
left_year = exit_prob.sum(axis=1)

print(f"Bootstrap: refitting the final operator on {S3_N_BOOT_OPERATOR} resamples of whole drifters ...")
rng_b = np.random.default_rng(SEED)
ids_f = np.array(sorted(fit_ids))
row_id = pd.Series(np.arange(len(ids_f)), index=ids_f).reindex(m_f["id"].values).values
boot_out, saved_P, t0b = [], [], time.time()
for b in range(S3_N_BOOT_OPERATOR):
    mult = np.bincount(rng_b.integers(0, len(ids_f), len(ids_f)), minlength=len(ids_f))
    Pb = m2_smoothed(D_f, SIG3, ALP3, w=mult[row_id].astype(float))
    boot_out.append(rq1_outputs(Pb))
    if b < S3_N_SAVE_OPERATORS:
        saved_P.append(Pb.astype(np.float32))
    if (b + 1) % 25 == 0:
        print(f"  {b + 1}/{S3_N_BOOT_OPERATOR} ({time.time() - t0b:.0f} s)")

# release summary with 90% bands
rel_rows = []
for name in rel_cells:
    for t in S3_SHOW_DAYS:
        vals = np.array([[bo[name]["snaps"][t][:NCELL].sum()] + list(bo[name]["snaps"][t][NCELL:]) for bo in boot_out])
        est = [base[name]["snaps"][t][:NCELL].sum()] + list(base[name]["snaps"][t][NCELL:])
        lo, hi = np.percentile(vals, [5, 95], axis=0)
        row = {"release": name, "lat": CELL_LAT[rel_cells[name]], "lon": CELL_LON[rel_cells[name]], "day": t}
        for j, lab in enumerate(["still_in_R"] + [f"left_{e}" for e in EXITS]):
            row[lab], row[lab + "_p05"], row[lab + "_p95"] = est[j], lo[j], hi[j]
        rel_rows.append(row)
rel_tab = pd.DataFrame(rel_rows)
rel_tab.to_csv(os.path.join(OUT3, "rq1_release_summary.csv"), index=False)
print("\nRelease forecasts (share of material; 90% bootstrap interval in brackets):")
for _, r in rel_tab.iterrows():
    print(f"  {r['release']:26s} day {int(r['day']):3d}: still in R {r['still_in_R']:.0%} "
          f"[{r['still_in_R_p05']:.0%}-{r['still_in_R_p95']:.0%}]; left "
          + ", ".join(f"{e} {r['left_' + e]:.0%}" for e in EXITS))

rho_b = {t: np.array([bo["rho"][t] for bo in boot_out]) for t in (30, 90)}
cell_fate = pd.DataFrame({"cell": np.arange(NCELL), "lat_centre": CELL_LAT, "lon_centre": CELL_LON,
                          "ocean": ocean_f, "masked": masked_f, "moves": n_moves_f.astype(int),
                          "distinct_drifters": n_drift_f.astype(int),
                          "median_days_in_R": median_days, "expected_days_in_R_first_year": expected_days,
                          "p_left_within_year": left_year,
                          **{f"p_left_{e}_within_year": exit_prob[:, j] for j, e in enumerate(EXITS)}})
for t in (30, 90):
    cell_fate[f"rho_day{t}"] = base["rho"][t]
    cell_fate[f"rho_day{t}_p05"] = np.nanpercentile(rho_b[t], 5, axis=0)
    cell_fate[f"rho_day{t}_p95"] = np.nanpercentile(rho_b[t], 95, axis=0)
cell_fate.to_csv(os.path.join(OUT3, "rq1_cell_fate.csv"), index=False)
ok = ~masked_f
for t in (30, 90):
    g_ = (cell_fate[f"rho_day{t}_p05"] > 1) & ok
    l_ = (cell_fate[f"rho_day{t}_p95"] < 1) & ok
    print(f"Relative density, day {t}: {int(g_.sum())} supported cells confidently GATHER material "
          f"(whole 90% interval above 1), {int(l_.sum())} confidently LOSE it, "
          f"{int(ok.sum() - g_.sum() - l_.sum())} not distinguishable from 1")
print(f"Median time in R from a supported cell: {np.nanmedian(median_days[ok]):.0f} days "
      f"(cells where half the material is still in R after a year: {int(np.isnan(median_days[ok]).sum())})")
dom = np.array(EXITS)[exit_prob.argmax(axis=1)]
print("Most likely exit edge (supported cells): " +
      ", ".join(f"{e} {int((dom[ok] == e).sum())}" for e in EXITS))


def hpd_mask(vec, q):
    v = vec[:NCELL] / max(vec[:NCELL].sum(), 1e-12)
    order = np.argsort(v)[::-1]
    keep = order[: np.searchsorted(np.cumsum(v[order]), q) + 1]
    m = np.zeros(NCELL); m[keep] = 1
    return as_grid(m)


xc = np.linspace(LON_MIN + CELL_DEG / 2, LON_MAX - CELL_DEG / 2, NX)
yc = np.linspace(LAT_MAX - CELL_DEG / 2, LAT_MIN + CELL_DEG / 2, NY)

REL_NORM = LogNorm(1e-4, 1e-1)   # one colour scale for every release map (darkest = 10% or more per cell)
# Figure C1: release forecasts (rows = release points, columns = a week, a month, a year, curve)
nr = len(rel_cells)
fig, ax = plt.subplots(nr, len(S3_SHOW_DAYS) + 1, figsize=(4.0 * (len(S3_SHOW_DAYS) + 1), 3.9 * nr),
                       layout="constrained", squeeze=False)
for i, (name, c) in enumerate(rel_cells.items()):
    for j, t in enumerate(S3_SHOW_DAYS):
        a = ax[i, j]
        snap = base[name]["snaps"][t]
        gr = as_grid(snap[:NCELL])
        gr[(gr < 1e-4) | masked_f.reshape(NY, NX)] = np.nan          # below 1 in 10,000: not shown
        im_rel = show_grid(a, gr, cmap="Blues", norm=REL_NORM)
        if snap[:NCELL].sum() < 0.01:
            a.text(0.5, 0.5, "Almost all material\nhas left R", transform=a.transAxes,
                   ha="center", va="center", fontsize=10, color="#5b6b7a")
        else:
            a.contour(xc, yc, hpd_mask(snap, 0.5), levels=[0.5], colors="#d9822b", linewidths=1.6)
            a.contour(xc, yc, hpd_mask(snap, 0.9), levels=[0.5], colors="#d9822b", linewidths=1.0, linestyles="--")
        a.plot(CELL_LON[c], CELL_LAT[c], marker="*", color="black", ms=11)
        r = rel_tab[(rel_tab["release"] == name) & (rel_tab["day"] == t)].iloc[0]
        map_axes(a, f"Day {t}: {r['still_in_R']:.0%} still in R")
        if i == 0 and j == 0:
            a.legend(handles=[plt.Line2D([], [], color="#d9822b", lw=1.6, label="50% area"),
                              plt.Line2D([], [], color="#d9822b", lw=1.0, ls="--", label="90% area")],
                     frameon=True, loc="lower right", fontsize=8)
    a = ax[i, -1]
    days = np.arange(YEAR + 1)
    bc = np.array([bo[name]["curve"] for bo in boot_out])
    a.fill_between(days, 100 * np.percentile(bc, 5, axis=0), 100 * np.percentile(bc, 95, axis=0), color="#1f6fb2", alpha=0.2, lw=0)
    a.plot(days, 100 * base[name]["curve"], color="#1f6fb2", lw=2)
    rY = rel_tab[(rel_tab["release"] == name) & (rel_tab["day"] == YEAR)].iloc[0]
    a.text(0.97, 0.95, "Left by day %d:\n" % YEAR + "\n".join(f"{e} {rY['left_' + e]:.0%}" for e in EXITS),
           transform=a.transAxes, ha="right", va="top", fontsize=9)
    a.set_ylim(0, 100); a.set_xlim(0, YEAR)
    a.set_xlabel("Days after release"); a.set_ylabel("Still in R (%)")
    a.set_title(name, loc="left", fontweight="bold")
    a.grid(axis="y", color="#e6e6e6", lw=0.6)
fig.colorbar(im_rel, ax=ax[:, :len(S3_SHOW_DAYS)], orientation="horizontal", shrink=0.5, pad=0.01,
             label="Probability of being in each 1° cell (orange: smallest areas holding 50% / 90% of what is still in R)")
fig.suptitle("Release forecasts from the final operator (drogued, M2, 1°). Shaded band: 90% bootstrap interval. "
             f"Grey: no data or fewer than {MIN_DRIFTERS} drifters.", x=0.01, ha="left", fontsize=10)
save3(fig, "fig_rq1_release_forecasts.png")

# Figure C2: per-cell fate and where R gathers / loses material
fig, ax = plt.subplots(1, 4, figsize=(17, 4.6), layout="constrained")
gmd = as_grid(np.where(masked_f, np.nan, np.where(np.isnan(median_days), YEAR, median_days)))
im = show_grid(ax[0], gmd, cmap="viridis", vmin=0, vmax=min(YEAR, np.nanmax(gmd)))
map_axes(ax[0], "(a) Days until half has left R")
fig.colorbar(im, ax=ax[0], orientation="horizontal", shrink=0.85, pad=0.02, label="Median days in R")
rgb = np.ones((NCELL, 4)); rgb[:, 3] = 0
cols = np.array([matplotlib.colors.to_rgba(c) for c in ["#1f6fb2", "#d9822b", "#8a5cc2", "#2e9e6a"]])  # N, S, E, W
cond = exit_prob / np.maximum(left_year[:, None], 1e-12)
rgb[:, :3] = cols[exit_prob.argmax(axis=1), :3]
rgb[:, 3] = np.clip(cond.max(axis=1), 0.25, 1.0)
rgb[masked_f] = [0, 0, 0, 0]
ax[1].imshow(rgb.reshape(NY, NX, 4), origin="upper", extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], interpolation="nearest")
map_axes(ax[1], "(b) Most likely exit edge (within a year)")
ax[1].legend(handles=[Patch(color=cols[j], label=f"leaves via {e}") for j, e in enumerate(EXITS)],
             loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8)
for k_, t in enumerate((30, 90)):
    a = ax[2 + k_]
    gr = cell_fate[f"rho_day{t}"].values.copy(); gr[masked_f] = np.nan
    im = show_grid(a, as_grid(gr), cmap="RdBu_r", norm=LogNorm(0.1, 10))
    sure = ok & ((cell_fate[f"rho_day{t}_p05"] > 1) | (cell_fate[f"rho_day{t}_p95"] < 1)).values
    a.scatter(CELL_LON[sure], CELL_LAT[sure], s=3, color="black")
    map_axes(a, f"({'cd'[k_]}) Gathers (red) or loses (blue), day {t}")
    fig.colorbar(im, ax=a, orientation="horizontal", shrink=0.85, pad=0.02,
                 label="Relative density (dot: 90% interval excludes 1)")
save3(fig, "fig_rq1_fate_maps.png")

# D2 export: final operator + bootstrap operators for the viewer
np.savez_compressed(os.path.join(OUT3, f"operator_M2_final_{DROGUE_SET}.npz"),
                    P=P_final, P_boot=np.array(saved_P), cell_lat=CELL_LAT, cell_lon=CELL_LON,
                    ocean=ocean_f, masked=masked_f,
                    states=np.array([f"cell_{i}" for i in range(NCELL)] + [f"exit_{e}" for e in EXITS]),
                    config=json.dumps({"region": [LAT_MIN, LAT_MAX, LON_MIN, LON_MAX], "cell_deg": CELL_DEG,
                                       "dt_days": DT_DAYS, "drogue_set": DROGUE_SET, "sigma": SIG3,
                                       "alpha": ALP3, "fitted_on": "train+val", "n_boot_saved": len(saved_P)}))
print(f"Saved operator_M2_final_{DROGUE_SET}.npz (P plus {len(saved_P)} bootstrap operators for the viewer)")


# ============================================================================= D. RQ2 final test
print("\n" + "=" * 78)
print("D. RQ2  FINAL HELD-OUT TEST")
print("=" * 78)
if not RUN_FINAL_TEST:
    print("Skipped (RUN_FINAL_TEST = False). Set it to True once sections A-C are agreed, and run once.")
else:
    P0b_f = m0_displacement(D_f)
    final_models = {"M0a persistence": m0_persistence(), "M0b displacement": P0b_f,
                    "M1 empirical": m1_empirical(m_f), "M2 smoothed": P_final}
    test_rows, diff_rows = [], []
    for s in ("test", "recent_test"):
        st_s = starts_for(daily3, ids3[s])
        if len(st_s[0]) == 0:
            print(f"  {s}: no drogued starts, skipped")
            continue
        for H in S3_TEST_HORIZONS:
            cH = cases_at(st_s, H)
            if cH.empty:
                continue
            boot = drifter_index(cH["id"].values)
            res = {n: evaluate(mpow(P, H), _g1, cH) for n, P in final_models.items()}
            for n, r in res.items():
                test_rows.append({"split": s, "model": n, "horizon_days": H, **summarise(r, boot)})
            for a_, b_ in [("M1 empirical", "M0b displacement"), ("M2 smoothed", "M1 empirical"),
                           ("M2 smoothed", "M0b displacement")]:
                d, lo, hi = paired_diff(res[a_], res[b_], boot)
                diff_rows.append({"split": s, "horizon_days": H, "comparison": f"{a_} minus {b_}",
                                  "log_score_gain": d, "p05": lo, "p95": hi,
                                  "verdict": "better" if lo > 0 else ("worse" if hi < 0 else "no clear difference")})
        fc = fate_cases(daily3, ids3[s])
        Sk = kaplan_meier(fc, FATE_DAYS)
        fr = pd.DataFrame([score_fate(P, fc, n, Sk)[0] for n, P in final_models.items() if n != "M0a persistence"])
        fr.to_csv(os.path.join(OUT3, f"final_{s}_fate.csv"), index=False)
        print(f"\n{s}: fate (exit edge and time in R)")
        print(fr[FATE_SHOW].round(3).to_string(index=False))
    test_tab = pd.DataFrame(test_rows)
    diff_tab = pd.DataFrame(diff_rows)
    test_tab.to_csv(os.path.join(OUT3, "final_test_position.csv"), index=False)
    diff_tab.to_csv(os.path.join(OUT3, "final_test_model_comparison.csv"), index=False)
    print("\nPosition scores on held-out drifters (90% intervals from resampling drifters):")
    print(test_tab[["split", "model", "horizon_days", "cases", "drifters", "log_score", "log_score_p05", "log_score_p95",
                    "in_90pct_area_%", "in_90pct_area_%_p05", "in_90pct_area_%_p95",
                    "median_centre_error_km", "median_persistence_error_km"]].round(2).to_string(index=False))
    print("\nIs the extra complexity worth it? (paired log-score gain, 90% interval)")
    print(diff_tab.round(3).to_string(index=False))

    for s, gs in test_tab.groupby("split", sort=False):
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
        for n, gg in gs.groupby("model", sort=False):
            x = gg["horizon_days"].values
            kw = dict(color=MODEL_COL[n], marker="o", lw=1.8, ms=4, label=n)
            if n != "M0a persistence":
                ax[0].plot(x, gg["log_score"], **kw)
                ax[0].fill_between(x, gg["log_score_p05"], gg["log_score_p95"], color=kw["color"], alpha=0.15, lw=0)
                ax[1].plot(x, gg["in_90pct_area_%"], **kw)
                ax[1].fill_between(x, gg["in_90pct_area_%_p05"], gg["in_90pct_area_%_p95"], color=kw["color"], alpha=0.15, lw=0)
                ax[2].plot(x, gg["median_centre_error_km"], **kw)
        gg = gs[gs["model"] == "M1 empirical"]
        ax[2].plot(gg["horizon_days"], gg["median_persistence_error_km"], color=MODEL_COL["M0a persistence"],
                   marker="o", lw=1.8, ms=4, ls="--", label="M0a persistence")
        ax[1].axhline(90, color="#5b6b7a", ls=":", lw=1); ax[1].set_ylim(0, 100)
        for a, t, yl in zip(ax, ["(a) Log score", "(b) Calibration (target 90%)", "(c) Distance error"],
                            ["Mean log score (higher is better)", "Test drifters in 90% area (%)", "Median error (km)"]):
            a.set_title(t, loc="left"); a.set_xlabel("Horizon (days)"); a.set_ylabel(yl)
            a.set_xscale("log"); a.set_xticks(S3_TEST_HORIZONS); a.set_xticklabels(S3_TEST_HORIZONS)
            a.grid(axis="y", color="#e6e6e6", lw=0.6); a.set_axisbelow(True)
        ax[2].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Final test on {'held-out drifters' if s == 'test' else 'the most recent years'} "
                     "(models refitted on training + validation; bands: 90% interval over drifters)",
                     x=0.01, ha="left", fontsize=10)
        save3(fig, f"fig_final_{s}.png")

print(f"\nStep 3 done in {(time.time() - t_start) / 60:.1f} min. Outputs in {OUT3}")
print("Please send back the printed output and the fig_*.png files from that folder.")
