"""
DATA3001 Group S01-G8 - Step 2: frozen splits, M0 baselines, M1 and M2 operators,
validation, bootstrap uncertainty and the D2 operator files
===================================================================================

What this script does (in order)
  1. Loads the cached extract (gdp_bmc_extract.parquet in Drive) and builds daily moves.
  2. FREEZES the drifter-level splits once and saves them to models/splits.csv:
        recent_test : drifters that first enter R in the most recent years (about 10%)
        train / val / test : the rest, 60 / 20 / 20, split by whole drifter
     If splits.csv already exists it is re-used, so every group member uses the same split.
  3. Fits the model ladder on TRAINING drifters only:
        M0a persistence      : "it does not move" (identity matrix)
        M0b displacement     : one displacement distribution used everywhere
        M1  empirical        : P_ij = n_ij / n_i
        M2  smoothed         : each cell borrows the moves of nearby cells (Gaussian, width sigma)
                               and is shrunk towards M0b (strength alpha). sigma and alpha are
                               chosen on the VALIDATION drifters.
  4. Validates every model on the validation drifters:
        POSITION at 7, 14, 30, 60 and 90 days (headline: 1 week and 1 month)
          log score, share of drifters inside the 50% and 90% areas, size of the 90% area,
          median error of the cloud's centre, and persistence error for comparison
        FATE at 1 year (few drifters stay in R for a year, so we test what happened to them)
          share still in R over 0-365 days (observed Kaplan-Meier curve vs each model),
          outcome at day 365 (still in R / left N, S, E, W): Brier score and shares
  5. Bootstrap (whole training drifters resampled with replacement) for M2:
        uncertainty bands for the release forecast, exits and relative density,
        and the cell mask (cells seen by fewer than MIN_DRIFTERS distinct drifters).
  6. Saves D2: the M1 and M2 operators as .npz + a cell table (.csv) + a README.

The TEST drifters are not touched unless RUN_TEST = True. Set it only once, at the very
end, after every modelling choice is fixed (as promised in the proposal).

How to run in Google Colab
  Cell 1:  !pip install -q "zarr<3"
  Cell 2:  paste this file and run it (needs gdp_bmc_extract.parquet in
           MyDrive/DATA3001/initial_analysis, created by the initial-analysis script).
  Takes a few minutes. Everything is written to MyDrive/DATA3001/initial_analysis/models.
  Please send back: the printed output, and the four figures fig_*.png.
"""

import os
import json
import time
import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

try:                                   # save everything in Google Drive when in Colab
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/DATA3001/initial_analysis"
except ImportError:                    # not in Colab
    BASE = os.environ.get("DATA3001_BASE", ".")
except Exception as e:
    print("WARNING: Google Drive did not mount (", e, "). Using temporary storage.")
    BASE = "/content/DATA3001/initial_analysis"

# ----------------------------------------------------------------------------- CONFIG
LAT_MIN, LAT_MAX = -50.0, -30.0
LON_MIN, LON_MAX = -60.0, -40.0
BUFFER_DEG = 2.0
CELL_DEG = 1.0
DT_DAYS = 1                  # keep at 1: horizons below are counted in daily steps
DROGUE_SET = "drogued"       # "drogued", "undrogued" or "all" (M4 will compare these)
SEED = 42
SPLIT_FRAC = (0.6, 0.2, 0.2)  # train / val / test, by whole drifter
RECENT_SHARE = 0.10          # about this share of drifters (the most recent) form recent_test
POS_HORIZONS = [7, 14, 30, 60, 90]   # position checks (days); headline 7 and 30
FATE_DAYS = 365              # fate check (days)
FATE_STRIDE = 5              # use every 5th day of a visit as a start (fewer, less correlated)
MIN_DRIFTERS = 10            # cells seen by fewer distinct drifters are masked
MIN_AT_RISK = 30             # observed survival curve shown only while this many starts are tracked
M2_SIGMAS = [0.0, 0.5, 1.0, 1.5]     # pooling width (cells); 0 = no pooling
M2_ALPHAS = [0, 2, 10, 50]           # pull towards M0b (pseudo-moves); 0 = none
N_BOOT = 200
RELEASE = (-38.0, -54.0)     # example release point (same as the poster)
RUN_TEST = False             # True ONLY once, at the end
LOG_FLOOR = 1e-6             # probabilities below this count as 1e-6 in the log score

INPUT_FILE = os.path.join(BASE, "gdp_bmc_extract.parquet")
OUT = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)
SPLIT_FILE = os.path.join(OUT, "splits.csv")

NX = int(round((LON_MAX - LON_MIN) / CELL_DEG))
NY = int(round((LAT_MAX - LAT_MIN) / CELL_DEG))
NCELL = NX * NY
NSTATE = NCELL + 4
EXITS = ["N", "S", "E", "W"]
IY, IX = np.divmod(np.arange(NCELL), NX)
CELL_LAT = LAT_MAX - (IY + 0.5) * CELL_DEG
CELL_LON = LON_MIN + (IX + 0.5) * CELL_DEG

COL = {"M0a persistence": "#8c8c8c", "M0b displacement": "#5b6b7a",
       "M1 empirical": "#1f6fb2", "M2 smoothed": "#d9822b", "Observed": "#000000"}
FATE_SHOW = ["model", "observed_up_to_day", "still_in_R_abs_error", "pred_still_in_R_day_30",
             "obs_still_in_R_day_30", "pred_still_in_R_day_%d" % FATE_DAYS, "obs_still_in_R_day_%d" % FATE_DAYS,
             "exit_edge_brier", "exit_edge_accuracy"] + [f"pred_share_left_{e}" for e in EXITS]
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------------------- helpers
def in_R(lat, lon):
    return (lat >= LAT_MIN) & (lat < LAT_MAX) & (lon >= LON_MIN) & (lon < LON_MAX)


def cell_of(lat, lon):
    """Cell index, row-major from the NORTH-WEST corner (rows run north -> south)."""
    ix = np.clip(np.floor((lon - LON_MIN) / CELL_DEG).astype(int), 0, NX - 1)
    iy = np.clip(np.floor((LAT_MAX - lat) / CELL_DEG).astype(int), 0, NY - 1)
    return iy * NX + ix


def raw_ij(lat, lon):
    """Row and column of a position WITHOUT clipping (outside R gives indices off the grid)."""
    return (np.floor((LAT_MAX - lat) / CELL_DEG).astype(int),
            np.floor((lon - LON_MIN) / CELL_DEG).astype(int))


def exit_edge(lat, lon):
    v = np.stack([lat - LAT_MAX, LAT_MIN - lat, lon - LON_MAX, LON_MIN - lon], axis=1)
    return np.array(EXITS)[np.argmax(v, axis=1)]


def as_grid(values):
    return np.array(values, dtype=float).reshape(NY, NX)


def haversine_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def area_weights():
    return np.cos(np.deg2rad(CELL_LAT))


def drogue_ok(flags):
    flags = np.asarray(flags).astype(bool)
    if DROGUE_SET == "drogued":
        return flags
    if DROGUE_SET == "undrogued":
        return ~flags
    return np.ones_like(flags, dtype=bool)


def map_axes(ax, title):
    ax.set_title(title, loc="left")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal")
    ax.set_facecolor("#e6e6e6")
    ax.set_xticks([-60, -55, -50, -45, -40])


def show_grid(ax, grid, **kw):
    return ax.imshow(np.ma.masked_invalid(grid), origin="upper",
                     extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], interpolation="nearest", **kw)


# ----------------------------------------------------------------------------- data
def prepare(df):
    df = df.dropna(subset=["lat", "lon", "time"]).copy()
    df["id"] = df["id"].astype(str)
    df["time"] = pd.to_datetime(df["time"]).dt.round("h")
    df = df.drop_duplicates(["id", "time"]).sort_values(["id", "time"]).reset_index(drop=True)
    df["inR"] = in_R(df["lat"].values, df["lon"].values)
    return df


def daily_transitions(df):
    """One snapshot per drifter per day (00 UTC) and the move to the next day."""
    d = df[(df["time"].dt.hour == 0) & (df["time"].dt.minute == 0)].copy()
    d = d.sort_values(["id", "time"]).reset_index(drop=True)
    g = d.groupby("id")
    for col in ("time", "lat", "lon", "inR"):
        d[col + "_next"] = g[col].shift(-DT_DAYS)
    tr = d[d["inR"]].copy()
    complete = (tr["time_next"] - tr["time"]) == pd.Timedelta(days=DT_DAYS)
    tr["outcome"] = np.where(complete, "move", "censored")
    tr["from_cell"] = cell_of(tr["lat"].values, tr["lon"].values)
    mv = (tr["outcome"] == "move").values
    nxt_in = tr["inR_next"].fillna(False).astype(bool).values
    tr["to"] = -1
    inside, leave = mv & nxt_in, mv & ~nxt_in
    tr.loc[inside, "to"] = cell_of(tr.loc[inside, "lat_next"].values, tr.loc[inside, "lon_next"].values)
    edges = exit_edge(tr.loc[leave, "lat_next"].values, tr.loc[leave, "lon_next"].values)
    tr.loc[leave, "to"] = [NCELL + EXITS.index(e) for e in edges]
    return d, tr


# ----------------------------------------------------------------------------- 2. splits
def make_splits(daily):
    if os.path.exists(SPLIT_FILE):
        s = pd.read_csv(SPLIT_FILE, dtype={"id": str})
        print(f"Re-using frozen splits from {SPLIT_FILE}")
        return dict(zip(s["id"], s["split"]))
    first = daily[daily["inR"]].groupby("id")["time"].min().dt.year
    cut = max(y for y in sorted(first.unique()) if (first >= y).mean() >= RECENT_SHARE)
    recent = first.index[first >= cut].values
    rest = np.random.default_rng(SEED).permutation(first.index[first < cut].values)
    a = int(round(SPLIT_FRAC[0] * len(rest)))
    b = a + int(round(SPLIT_FRAC[1] * len(rest)))
    split = {i: "train" for i in rest[:a]}
    split.update({i: "val" for i in rest[a:b]})
    split.update({i: "test" for i in rest[b:]})
    split.update({i: "recent_test" for i in recent})
    pd.DataFrame({"id": list(split), "split": list(split.values()),
                  "first_year_in_R": first.reindex(list(split)).values}).to_csv(SPLIT_FILE, index=False)
    print(f"Created and froze splits in {SPLIT_FILE} (recent_test = drifters first in R from {cut})")
    return split


# ----------------------------------------------------------------------------- 3. models
def ocean_mask(daily_train):
    """Cells that training drifters actually visited (everything else is land or never sampled)."""
    m = np.zeros(NCELL, dtype=bool)
    r = daily_train[daily_train["inR"]]
    m[np.unique(cell_of(r["lat"].values, r["lon"].values))] = True
    return m


def move_table(tr):
    """Training moves with their displacement (rows, columns) from the start cell."""
    m = tr[(tr["outcome"] == "move") & drogue_ok(tr["drogued"])].copy()
    fy, fx = np.divmod(m["from_cell"].values, NX)
    ny, nx_ = raw_ij(m["lat_next"].values.astype(float), m["lon_next"].values.astype(float))
    m["dy"], m["dx"] = ny - fy, nx_ - fx
    return m


def finish(P_counts, extra=None):
    """Turn a count matrix into probabilities; empty cells hold their mass; exits absorb."""
    P = P_counts.copy()
    tot = P[:NCELL].sum(axis=1)
    has = tot > 0
    P[:NCELL][has] /= tot[has, None]
    empty = np.flatnonzero(~has)
    P[empty] = 0
    P[empty, empty] = 1.0
    P[NCELL:] = 0
    P[NCELL:, NCELL:] = np.eye(4)
    return P


def m0_persistence():
    return np.eye(NSTATE)


def m1_empirical(m, w=None):
    C = np.zeros((NSTATE, NSTATE))
    np.add.at(C, (m["from_cell"].values, m["to"].values.astype(int)),
              np.ones(len(m)) if w is None else w)
    return finish(C)


class Displacements:
    """Look-up of where each displacement lands from each cell (cell, exit, or invalid)."""
    def __init__(self, m, ocean):
        self.R = int(min(10, max(1, np.abs(m["dy"]).max(), np.abs(m["dx"]).max())))
        off = np.arange(-self.R, self.R + 1)
        DY, DX = np.meshgrid(off, off, indexing="ij")
        self.DY, self.DX = DY.ravel(), DX.ravel()
        self.K = len(self.DY)
        ty, tx = IY[:, None] + self.DY[None, :], IX[:, None] + self.DX[None, :]
        inside = (ty >= 0) & (ty < NY) & (tx >= 0) & (tx < NX)
        c = np.where(inside, ty * NX + tx, 0)
        tgt = np.full(ty.shape, -1)
        okc = inside & ocean[c]
        tgt[okc] = c[okc]
        edge = np.argmax(np.stack([-ty, ty - (NY - 1), tx - (NX - 1), -tx], axis=-1), axis=-1)
        tgt[~inside] = NCELL + edge[~inside]
        tgt[~ocean] = -1                       # never move FROM land / unsampled cells
        self.tgt, self.ocean = tgt, ocean
        keep = (np.abs(m["dy"].values) <= self.R) & (np.abs(m["dx"].values) <= self.R)
        if (~keep).any():
            print(f"  note: {int((~keep).sum())} moves longer than {self.R} cells ignored")
        self.k = np.where(keep, (m["dy"].values + self.R) * (2 * self.R + 1) + (m["dx"].values + self.R), -1)
        self.cell = m["from_cell"].values

    def histogram(self, w=None):
        H = np.zeros((NCELL, self.K))
        ok = self.k >= 0
        ww = np.ones(len(self.k)) if w is None else w
        np.add.at(H, (self.cell[ok], self.k[ok]), ww[ok])
        return H

    def to_counts(self, H):
        C = np.zeros((NSTATE, NSTATE))
        rows = np.repeat(np.arange(NCELL), self.K)
        cols = self.tgt.ravel()
        v = H.ravel()
        ok = (cols >= 0) & (v > 0)
        np.add.at(C, (rows[ok], cols[ok]), v[ok])
        return C


def m0_displacement(D, w=None):
    """One displacement distribution (pooled over all training moves) applied from every cell."""
    k = D.histogram(w).sum(axis=0)
    H = np.tile(k / k.sum(), (NCELL, 1))
    C = D.to_counts(H)                          # invalid (land) targets dropped, rows renormalised
    return finish(C)


def m2_smoothed(D, sigma, alpha, w=None, P0b=None):
    """Borrow moves from nearby cells (as displacements), then shrink towards M0b."""
    H = D.histogram(w)
    if sigma > 0:
        H = gaussian_filter(H.reshape(NY, NX, D.K), sigma=(sigma, sigma, 0),
                            mode="constant", cval=0.0, truncate=2.0).reshape(NCELL, D.K)
    C = D.to_counts(H)
    if alpha > 0:
        P0b = m0_displacement(D, w) if P0b is None else P0b
        C[:NCELL] += alpha * P0b[:NCELL] * D.ocean[:, None]
    return finish(C)


# ----------------------------------------------------------------------------- 4a. position check
def position_cases(daily, ids):
    """Start = a drifter-day inside R. Truth k days later = its cell if it stayed in R the whole
    time, or the edge it FIRST left through (exits are final, as in the operator). Starts whose
    track stops inside R before day k are left out, because we do not know what happened."""
    st0 = visit_starts(daily, ids, stride=1)
    snap = daily[daily["id"].isin(ids)][["id", "time", "lat", "lon"]].rename(
        columns={"time": "t_end", "lat": "lat_end", "lon": "lon_end"})
    cases = {}
    for k in POS_HORIZONS:
        gone = st0["exited"] & (st0["days"] <= k)              # left R by day k
        stay = (st0["exited"] & (st0["days"] > k)) | (~st0["exited"] & (st0["days"] >= k))
        a = st0[stay].assign(t_end=lambda x: x["time"] + pd.Timedelta(days=k)).merge(snap, on=["id", "t_end"])
        b = st0[gone].assign(lat_end=np.nan, lon_end=np.nan)
        st = pd.concat([a, b], ignore_index=True)
        if st.empty:
            continue
        inn = np.r_[np.ones(len(a), bool), np.zeros(len(b), bool)]
        state = np.empty(len(st), dtype=int)
        state[inn] = cell_of(st.loc[inn, "lat_end"].values, st.loc[inn, "lon_end"].values)
        state[~inn] = [NCELL + EXITS.index(e) for e in st.loc[~inn, "edge"]]
        cases[k] = dict(cell=st["cell"].values, state=state, inn=inn, id=st["id"].values,
                        lat=st["lat"].values, lon=st["lon"].values,
                        lat_end=st["lat_end"].values, lon_end=st["lon_end"].values)
    return cases


def coverage(probs, state, q):
    order = np.argsort(-probs, axis=1)
    csum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    n_keep = (csum < q - 1e-12).sum(axis=1) + 1
    rank = np.argmax(order == state[:, None], axis=1)
    return rank < n_keep, n_keep


def score_position(P, cases, name):
    rows = []
    for k, c in cases.items():
        probs = np.linalg.matrix_power(P, k)[c["cell"]]
        hit50, _ = coverage(probs, c["state"], 0.5)
        hit90, n90 = coverage(probs, c["state"], 0.9)
        pin = probs[:, :NCELL]
        mass = pin.sum(axis=1)
        ok = c["inn"] & (mass > 0)
        cl = (pin * CELL_LAT).sum(axis=1) / np.where(mass > 0, mass, 1)
        co = (pin * CELL_LON).sum(axis=1) / np.where(mass > 0, mass, 1)
        err = haversine_km(cl[ok], co[ok], c["lat_end"][ok], c["lon_end"][ok])
        pers = haversine_km(c["lat"][ok], c["lon"][ok], c["lat_end"][ok], c["lon_end"][ok])
        logp = np.log(np.maximum(probs[np.arange(len(c["state"])), c["state"]], LOG_FLOOR))
        rows.append({"model": name, "horizon_days": k, "cases": len(c["state"]),
                     "drifters": len(np.unique(c["id"])),
                     "log_score": logp.mean(), "in_50pct_area_%": 100 * hit50.mean(),
                     "in_90pct_area_%": 100 * hit90.mean(), "cells_in_90pct_area": float(np.median(n90)),
                     "median_centre_error_km": float(np.median(err)) if ok.any() else np.nan,
                     "median_persistence_error_km": float(np.median(pers)) if ok.any() else np.nan})
    return rows


# ----------------------------------------------------------------------------- 4b. fate check
def visit_starts(daily, ids, stride):
    """Every `stride`-th drifter-day inside R, with what happened next:
         exited = True  -> it first left R after `days` days, through `edge`
         exited = False -> it was still in R `days` days later, then the record stops (censored)"""
    out = []
    d = daily[daily["id"].isin(ids)].sort_values(["id", "time"])
    one = np.timedelta64(1, "D")
    for did, g in d.groupby("id", sort=False):
        t = g["time"].values
        inr, lat, lon, drg = g["inR"].values, g["lat"].values, g["lon"].values, g["drogued"].values
        n, i = len(g), 0
        while i < n:
            if not inr[i]:
                i += 1
                continue
            s = i
            while i + 1 < n and inr[i + 1] and (t[i + 1] - t[i]) == one:
                i += 1
            e = i                                     # last consecutive day inside R
            exited = bool(i + 1 < n and (t[i + 1] - t[i]) == one and not inr[i + 1])
            edge = exit_edge(np.array([lat[i + 1]]), np.array([lon[i + 1]]))[0] if exited else ""
            for j in range(s, e + 1, stride):
                if drogue_ok([drg[j]])[0]:
                    out.append((did, t[j], lat[j], lon[j], exited, (e - j + 1) if exited else (e - j), edge))
            i += 1
    out = pd.DataFrame(out, columns=["id", "time", "lat", "lon", "exited", "days", "edge"])
    out["time"] = pd.to_datetime(out["time"])
    out["cell"] = cell_of(out["lat"].values, out["lon"].values)
    return out


def fate_cases(daily, ids):
    return visit_starts(daily, ids, FATE_STRIDE)


def propagate_cells(P, days):
    """still[c, t] = chance that material starting in cell c is still in R after t days,
    and the full state distribution after `days` days, for every start cell."""
    M = np.eye(NSTATE)[:NCELL]
    still = np.ones((NCELL, days + 1))
    for t in range(1, days + 1):
        M = M @ P
        still[:, t] = M[:, :NCELL].sum(axis=1)
    return still, M


def kaplan_meier(fc, days):
    """Observed share still in R after u days, allowing for records that stop (censoring).
    Days where fewer than MIN_AT_RISK starts are still being tracked are set to NaN."""
    T, ev = fc["days"].values, fc["exited"].values.astype(bool)
    S, s = np.full(days + 1, np.nan), 1.0
    S[0] = 1.0
    for u in range(1, days + 1):
        at_risk = (T >= u).sum()
        if at_risk < MIN_AT_RISK:
            break
        s *= 1 - ((T == u) & ev).sum() / at_risk
        S[u] = s
    return S


def score_fate(P, fc, name, S_obs):
    """Two fate checks that respect censoring (few drifters are tracked for a whole year):
       (1) how long material stays: predicted share still in R vs the observed Kaplan-Meier curve
       (2) where it leaves: for drifters SEEN to exit, the predicted exit edge (given that it
           leaves within a year) vs the edge it actually used: Brier score and accuracy."""
    still, M = propagate_cells(P, FATE_DAYS)
    pred_curve = still[fc["cell"].values].mean(axis=0)
    ex = fc[fc["exited"]]
    pe = M[ex["cell"].values, NCELL:]
    tot = pe.sum(axis=1)
    okx = tot > 1e-9                                   # model lets it leave at all
    pe = pe[okx] / tot[okx, None]
    edges = ex["edge"].to_numpy(dtype=str)[okx]
    obs = (edges[:, None] == np.array(EXITS)[None, :]).astype(float)
    row = {"model": name, "fate_starts": len(fc), "seen_to_exit": len(ex),
           "observed_up_to_day": int(np.flatnonzero(np.isfinite(S_obs)).max()),
           "still_in_R_abs_error": float(np.nanmean(np.abs(pred_curve - S_obs))),
           "pred_still_in_R_day_30": pred_curve[30], "pred_still_in_R_day_%d" % FATE_DAYS: pred_curve[-1],
           "obs_still_in_R_day_30": S_obs[30], "obs_still_in_R_day_%d" % FATE_DAYS: S_obs[-1],
           "exit_edge_brier": float(((pe - obs) ** 2).sum(axis=1).mean()) if okx.any() else np.nan,
           "exit_edge_accuracy": float((pe.argmax(axis=1) == obs.argmax(axis=1)).mean()) if okx.any() else np.nan}
    for j, e in enumerate(EXITS):
        row[f"pred_share_left_{e}"] = pe[:, j].mean() if okx.any() else np.nan
        row[f"obs_share_left_{e}"] = obs[:, j].mean() if okx.any() else np.nan
    return row, pred_curve


# ----------------------------------------------------------------------------- 5. bootstrap
def summaries(P, ocean):
    """Key outputs of one operator: release forecast, even-start fate, relative density."""
    out = {}
    p = np.zeros(NSTATE)
    p[cell_of(np.array([RELEASE[0]]), np.array([RELEASE[1]]))[0]] = 1.0
    u = np.zeros(NSTATE)
    u[:NCELL] = area_weights() * ocean
    u /= u.sum()
    v = u.copy()
    for t in range(1, FATE_DAYS + 1):
        p, v = p @ P, v @ P
        if t in (7, 30, FATE_DAYS):
            out[f"release_still_in_R_day_{t}"] = p[:NCELL].sum()
            out[f"even_still_in_R_day_{t}"] = v[:NCELL].sum()
        if t == 30:
            inside = v[:NCELL] / max(v[:NCELL].sum(), 1e-12)
            rho = np.where(ocean, inside / np.where(u[:NCELL] > 0, u[:NCELL], np.nan), np.nan)
            out["release_day_30"] = p[:NCELL].copy()
    for j, e in enumerate(EXITS):
        out[f"release_left_{e}_day_{FATE_DAYS}"] = p[NCELL + j]
        out[f"even_left_{e}_day_{FATE_DAYS}"] = v[NCELL + j]
    out["rho_day_30"] = rho
    return out


def bootstrap(train_ids, m, D, ocean, sigma, alpha):
    rng = np.random.default_rng(SEED)
    ids = np.array(sorted(train_ids))
    pos = pd.Series(np.arange(len(ids)), index=ids)
    row_id = pos.reindex(m["id"].values).values
    scal, rhos = [], []
    t0 = time.time()
    for b in range(N_BOOT):
        mult = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        w = mult[row_id].astype(float)
        P = m2_smoothed(D, sigma, alpha, w=w)
        s = summaries(P, ocean)
        rhos.append(s.pop("rho_day_30"))
        s.pop("release_day_30")
        scal.append(s)
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{N_BOOT} ({time.time() - t0:.0f} s)")
    scal = pd.DataFrame(scal)
    band = scal.quantile([0.05, 0.5, 0.95]).T
    band.columns = ["p05", "median", "p95"]
    return band, np.array(rhos)


# ----------------------------------------------------------------------------- figures
HL = {"M0b displacement": "o", "M1 empirical": "s", "M2 smoothed": "D"}


def fig_position(pos):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1), layout="constrained")
    for name, g in pos.groupby("model", sort=False):
        if name == "M0a persistence":
            continue
        kw = dict(color=COL[name], marker=HL.get(name, "o"), ms=4, lw=1.8, label=name)
        ax[0].plot(g["horizon_days"], g["log_score"], **kw)
        ax[1].plot(g["horizon_days"], g["in_90pct_area_%"], **kw)
        ax[2].plot(g["horizon_days"], g["median_centre_error_km"], **kw)
    g = pos[pos["model"] == "M1 empirical"]
    ax[2].plot(g["horizon_days"], g["median_persistence_error_km"], color=COL["M0a persistence"],
               marker="o", ms=4, lw=1.8, ls="--", label="M0a persistence")
    ax[1].axhline(90, color="#5b6b7a", ls=":", lw=1)
    ax[0].set_ylabel("Mean log score (higher is better)")
    ax[1].set_ylabel("Validation drifters in 90% area (%)")
    ax[1].set_ylim(0, 100)
    ax[2].set_ylabel("Median error (km)")
    for a, t in zip(ax, ["(a) Log score", "(b) Calibration (target 90%)", "(c) Distance error"]):
        a.set_title(t, loc="left")
        a.set_xlabel("Horizon (days)")
        a.set_xticks(POS_HORIZONS)
        a.grid(axis="y", color="#e6e6e6", lw=0.6)
        a.set_axisbelow(True)
    ax[2].legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(OUT, "fig_validation_position.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def fig_fate(curves, fate):
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2), layout="constrained",
                           gridspec_kw={"width_ratios": [1.2, 1]})
    days = np.arange(FATE_DAYS + 1)
    last = int(np.flatnonzero(np.isfinite(curves["Observed"])).max())
    if last < FATE_DAYS:
        ax[0].axvspan(last, FATE_DAYS, color="#f2f2f2", zorder=0)
        ax[0].text(min(last + 5, FATE_DAYS - 5), 95, f"too few drifters\nstill tracked\nafter day {last}",
                   fontsize=7, color="#5b6b7a", va="top")
    for name, c in curves.items():
        ax[0].plot(days, 100 * c, color=COL[name], lw=2.2 if name == "Observed" else 1.6,
                   ls="-" if name != "M0b displacement" else "--", label=name)
    ax[0].set_title("(a) How long material stays in R (validation drifters)", loc="left")
    ax[0].set_xlabel("Days after start")
    ax[0].set_ylabel("Still in R (%)")
    ax[0].set_xlim(0, FATE_DAYS)
    ax[0].set_ylim(0, 100)
    ax[0].legend(frameon=False)
    names = ["Observed"] + [n for n in fate["model"] if n != "M0a persistence"]
    width = 0.8 / len(names)
    x = np.arange(len(EXITS))
    obs_row = fate.dropna(subset=["obs_share_left_N"]).iloc[0]
    for i, n in enumerate(names):
        r = obs_row if n == "Observed" else fate[fate["model"] == n].iloc[0]
        key = "obs_share_left_" if n == "Observed" else "pred_share_left_"
        vals = [r[key + e] for e in EXITS]
        ax[1].bar(x + (i - (len(names) - 1) / 2) * width, 100 * np.array(vals), width,
                  color=COL[n], label=n)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"Left {e}" for e in EXITS])
    ax[1].set_ylabel("Share of exits (%)")
    ax[1].set_title("(b) Exit edge, for drifters seen to leave R", loc="left")
    ax[1].legend(frameon=False, fontsize=7)
    for a in ax:
        a.grid(axis="y", color="#e6e6e6", lw=0.6)
        a.set_axisbelow(True)
    fig.savefig(os.path.join(OUT, "fig_validation_fate.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def fig_maps(P, ocean, masked, rhos):
    p = np.zeros(NSTATE)
    p[cell_of(np.array([RELEASE[0]]), np.array([RELEASE[1]]))[0]] = 1.0
    snaps = {}
    for t in range(1, FATE_DAYS + 1):
        p = p @ P
        if t in (7, 30, FATE_DAYS):
            snaps[t] = p.copy()
    rho_med = np.nanmedian(rhos, axis=0)
    spread = np.nanpercentile(rhos, 95, axis=0) / np.nanpercentile(rhos, 5, axis=0)
    fig, ax = plt.subplots(1, 5, figsize=(15, 3.7), layout="constrained")
    for a, t in zip(ax[:3], (7, 30, FATE_DAYS)):
        g = as_grid(snaps[t][:NCELL])
        g[(g <= 0) | masked.reshape(NY, NX)] = np.nan
        im = show_grid(a, g, cmap="Blues", norm=LogNorm(1e-4, max(1e-3, np.nanmax(g) if np.isfinite(g).any() else 1)))
        a.plot(RELEASE[1], RELEASE[0], marker="*", color="black", ms=8)
        left = 100 * snaps[t][NCELL:].sum()
        map_axes(a, f"Release, day {t} ({left:.0f}% has left R)")
        fig.colorbar(im, ax=a, orientation="horizontal", shrink=0.85, pad=0.02, label="Probability")
    g = rho_med.copy()
    g[masked] = np.nan
    im = show_grid(ax[3], as_grid(g), cmap="RdBu_r", norm=LogNorm(0.1, 10))
    map_axes(ax[3], "Relative density, day 30")
    fig.colorbar(im, ax=ax[3], orientation="horizontal", shrink=0.85, pad=0.02, label=">1 gathers, <1 loses")
    g = spread.copy()
    g[masked] = np.nan
    im = show_grid(ax[4], as_grid(g), cmap="viridis", norm=LogNorm(1, max(2, np.nanpercentile(g, 98))))
    map_axes(ax[4], "Bootstrap spread of relative density")
    fig.colorbar(im, ax=ax[4], orientation="horizontal", shrink=0.85, pad=0.02,
                 label="95th / 5th percentile (1 = certain)")
    fig.suptitle(f"M2 smoothed operator ({DROGUE_SET} drifters, training set). Grey = no data or "
                 f"fewer than {MIN_DRIFTERS} distinct drifters (masked).", fontsize=9, x=0.01, ha="left")
    fig.savefig(os.path.join(OUT, "fig_M2_maps.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def fig_tuning(tune):
    piv = tune.pivot(index="sigma", columns="alpha", values="val_log_score")
    fig, ax = plt.subplots(figsize=(4.2, 3.0), layout="constrained")
    im = ax.imshow(piv.values, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if piv.values[i, j] < np.nanmedian(piv.values) else "black")
    ax.set_xlabel("alpha (pull towards M0b)")
    ax.set_ylabel("sigma (pooling width, cells)")
    ax.set_title("M2 tuning: validation log score\n(mean of 7 and 30 days, higher is better)", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(OUT, "fig_M2_tuning.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------- D2 export
def export_operator(P, name, ocean, cell_table, params):
    np.savez_compressed(os.path.join(OUT, f"operator_{name}_{DROGUE_SET}.npz"), P=P,
                        cell_lat=CELL_LAT, cell_lon=CELL_LON, ocean=ocean,
                        states=np.array([f"cell_{i}" for i in range(NCELL)] + [f"exit_{e}" for e in EXITS]),
                        config=json.dumps({"region": [LAT_MIN, LAT_MAX, LON_MIN, LON_MAX],
                                           "cell_deg": CELL_DEG, "dt_days": DT_DAYS,
                                           "drogue_set": DROGUE_SET, **params}))


README = """D2 transport operators for region R (30-50 S, 60-40 W), DATA3001 Group S01-G8
Files
  operator_<model>_<drogue>.npz   P (404 x 404): row i = where material in state i is one day later.
                                  States 0-399 = 1-degree cells, row-major from the north-west corner
                                  (cell = row*20 + col, rows run north to south); 400-403 = left R via
                                  N, S, E, W (absorbing). Also cell_lat, cell_lon (centres), ocean mask, config.
  cell_table_<drogue>.csv         per cell: centre, moves and distinct drifters used, mask flag,
                                  relative density at day 30 with its bootstrap 5-95% band.
Use (Python)
  import numpy as np
  z = np.load("operator_M2_smoothed_drogued.npz"); P = z["P"]
  p = np.zeros(P.shape[0]); p[cell_index] = 1          # release in one cell
  for day in range(30): p = p @ P                      # p is now the day-30 forecast
  p[:400].sum() = still in R, p[400:] = left via N, S, E, W
Cells flagged masked=True had fewer than {m} distinct drifters and should be shown as unsupported.
"""


# ----------------------------------------------------------------------------- main
def main(raw):
    t0 = time.time()
    df = prepare(raw)
    daily, tr = daily_transitions(df)
    split = make_splits(daily)
    daily["split"] = daily["id"].map(split)
    tr["split"] = tr["id"].map(split)
    ids = {s: set(daily.loc[daily["split"] == s, "id"]) for s in ("train", "val", "test", "recent_test")}
    print("Drifters per split: " + ", ".join(f"{s} {len(v)}" for s, v in ids.items()))

    ocean = ocean_mask(daily[daily["split"] == "train"])
    m = move_table(tr[tr["split"] == "train"])
    D = Displacements(m, ocean)
    print(f"Training moves ({DROGUE_SET}): {len(m):,} from {m['id'].nunique()} drifters; "
          f"ocean cells: {int(ocean.sum())}; displacement window ±{D.R} cells")

    per_cell = m.groupby("from_cell").agg(moves=("id", "size"), drifters=("id", "nunique"))
    n_moves = per_cell["moves"].reindex(range(NCELL)).fillna(0).values
    n_drift = per_cell["drifters"].reindex(range(NCELL)).fillna(0).values
    masked = (n_drift < MIN_DRIFTERS) | ~ocean
    print(f"Masked cells (fewer than {MIN_DRIFTERS} distinct drifters, or never visited): "
          f"{int(masked.sum())} of {NCELL}")

    print("\nBuilding position and fate checks for the validation drifters...")
    vcases = position_cases(daily, ids["val"])
    vfate = fate_cases(daily, ids["val"])
    print(f"  position cases at {POS_HORIZONS} days: "
          + ", ".join(str(len(c['state'])) for c in vcases.values())
          + f"; fate starts: {len(vfate)} ({int(vfate['exited'].sum())} seen to exit)")

    # ---- M2 tuning on validation (log score, mean of 7 and 30 days)
    print("\nTuning M2 on validation drifters...")
    P0b = m0_displacement(D)
    tune_cases = {k: vcases[k] for k in (7, 30) if k in vcases}
    tune = []
    for s in M2_SIGMAS:
        for a in M2_ALPHAS:
            P = m2_smoothed(D, s, a, P0b=P0b)
            sc = pd.DataFrame(score_position(P, tune_cases, "M2"))
            tune.append({"sigma": s, "alpha": a, "val_log_score": sc["log_score"].mean(),
                         "val_in_90pct_%": sc["in_90pct_area_%"].mean()})
    tune = pd.DataFrame(tune)
    tune.to_csv(os.path.join(OUT, "m2_tuning.csv"), index=False)
    best = tune.loc[tune["val_log_score"].idxmax()]
    SIG, ALP = float(best["sigma"]), float(best["alpha"])
    print(tune.round(3).to_string(index=False))
    print(f"Chosen M2: sigma = {SIG}, alpha = {ALP}")
    fig_tuning(tune)

    models = {"M0a persistence": m0_persistence(), "M0b displacement": P0b,
              "M1 empirical": m1_empirical(m), "M2 smoothed": m2_smoothed(D, SIG, ALP, P0b=P0b)}

    # ---- validation: position
    pos = pd.DataFrame([r for n, P in models.items() for r in score_position(P, vcases, n)])
    pos.to_csv(os.path.join(OUT, "validation_position.csv"), index=False)
    print("\n=== Validation: position (headline 7 and 30 days) ===")
    pd.set_option("display.width", 200)
    print(pos.round(2).to_string(index=False))
    fig_position(pos)

    # ---- validation: fate at one year
    print(f"\n=== Validation: fate up to day {FATE_DAYS} ===")
    S = kaplan_meier(vfate, FATE_DAYS)
    fate_rows, curves = [], {"Observed": S}
    for n, P in models.items():
        row, curve = score_fate(P, vfate, n, S)
        fate_rows.append(row)
        if n != "M0a persistence":
            curves[n] = curve
    fate = pd.DataFrame(fate_rows)
    fate.to_csv(os.path.join(OUT, "validation_fate.csv"), index=False)
    last = int(np.flatnonzero(np.isfinite(S)).max())
    fmt = lambda d: f"{S[d]:.0%}" if np.isfinite(S[d]) else "not enough data"
    print(f"Observed (Kaplan-Meier, allows for records that stop): still in R after 7 d {fmt(7)}, "
          f"30 d {fmt(30)}, {FATE_DAYS} d {fmt(FATE_DAYS)}")
    print(f"  the observed curve is reliable up to day {last} (at least {MIN_AT_RISK} starts still tracked); "
          f"beyond that the 1-year check rests on exit edges only")
    print(fate[FATE_SHOW].round(3).to_string(index=False))
    print("Observed exit edges (drifters seen to leave): "
          + ", ".join(f"{e} {fate.dropna(subset=['obs_share_left_N']).iloc[0]['obs_share_left_' + e]:.0%}" for e in EXITS))
    fig_fate(curves, fate)

    # ---- bootstrap for M2
    print(f"\nBootstrap of M2 ({N_BOOT} resamples of whole training drifters)...")
    band, rhos = bootstrap(ids["train"], m, D, ocean, SIG, ALP)
    band.to_csv(os.path.join(OUT, "bootstrap_M2_summary.csv"))
    print(band.round(3).to_string())
    fig_maps(models["M2 smoothed"], ocean, masked, rhos)

    # ---- D2 export (operators fitted on the training drifters; refit on train+val at the end)
    cell_table = pd.DataFrame({"cell": np.arange(NCELL), "row": IY, "col": IX,
                               "lat_centre": CELL_LAT, "lon_centre": CELL_LON, "ocean": ocean,
                               "moves": n_moves.astype(int), "distinct_drifters": n_drift.astype(int),
                               "masked": masked,
                               "rho_day30_median": np.nanmedian(rhos, axis=0),
                               "rho_day30_p05": np.nanpercentile(rhos, 5, axis=0),
                               "rho_day30_p95": np.nanpercentile(rhos, 95, axis=0)})
    cell_table.to_csv(os.path.join(OUT, f"cell_table_{DROGUE_SET}.csv"), index=False)
    export_operator(models["M1 empirical"], "M1_empirical", ocean, cell_table, {})
    export_operator(models["M2 smoothed"], "M2_smoothed", ocean, cell_table, {"sigma": SIG, "alpha": ALP})
    with open(os.path.join(OUT, "README.txt"), "w") as f:
        f.write(README.replace("{m}", str(MIN_DRIFTERS)))

    # ---- test set: ONLY when every choice is final
    if RUN_TEST:
        print("\n=== TEST SET (final, run once) ===")
        tv = tr[tr["split"].isin(["train", "val"])]
        m_tv = move_table(tv)
        D_tv = Displacements(m_tv, ocean)
        final = {"M0b displacement": m0_displacement(D_tv), "M1 empirical": m1_empirical(m_tv),
                 "M2 smoothed": m2_smoothed(D_tv, SIG, ALP)}
        for s in ("test", "recent_test"):
            c = position_cases(daily, ids[s])
            fc = fate_cases(daily, ids[s])
            res = pd.DataFrame([r for n, P in final.items() for r in score_position(P, c, n)])
            Sk = kaplan_meier(fc, FATE_DAYS)
            fr = pd.DataFrame([score_fate(P, fc, n, Sk)[0] for n, P in final.items()])
            res.to_csv(os.path.join(OUT, f"{s}_position.csv"), index=False)
            fr.to_csv(os.path.join(OUT, f"{s}_fate.csv"), index=False)
            print(f"\n{s}:")
            print(res.round(2).to_string(index=False))
            print(fr[FATE_SHOW].round(3).to_string(index=False))

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs in {OUT}")
    return models, pos, fate


if __name__ == "__main__":
    if "data" not in globals():
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(f"{INPUT_FILE} not found. Run the initial-analysis script once "
                                    "first: it creates this cached extract.")
        print(f"Reading cached extract {INPUT_FILE}")
        data = pd.read_parquet(INPUT_FILE)
    models, pos, fate = main(data)
