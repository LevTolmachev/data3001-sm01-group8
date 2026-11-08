"""
DATA3001 Group 8 - Initial analysis for the Project Proposal (Section 6)
========================================================================

Produces (saved in Google Drive under MyDrive/DATA3001/initial_analysis)
  fig1_data_support.png          -> Figure 1 in the proposal
  fig2_preliminary_operator.png  -> Figure 2 in the proposal
  table3_data_adequacy.csv       -> numbers for Table 3 (also printed)
  + a printed list of the other numbers to paste into Section 6

How to run in Google Colab (same set-up as our region_check notebook)
  Start from a fresh runtime (Runtime -> Disconnect and delete runtime).
  Cell 1:   !pip install -q "zarr<3"
            (do NOT install clouddrift: it upgrades pandas/requests and breaks Colab)
  Cell 2:   paste this whole file and run it. Approve the Google Drive prompt.
  - The first run streams the GDP hourly product once and keeps only R plus a buffer
    (roughly 20-60 min, keep the tab open). The extract is cached in your Drive as
    gdp_bmc_extract.parquet, so later runs take seconds, even in a new session.
  - Figures and the CSV are written to the same Drive folder.

Using an extract you already have
  Rename its columns to  id, time, lat, lon, drogued  (drogued = True/False per row) and
  call main(your_df). It must cover R PLUS A BUFFER (BUFFER_DEG), because the first
  position after a drifter leaves R is how exits are detected.

Everything depends only on CONFIG, so the same code works for any region R.
"""

import os
try:                                   # save everything in Google Drive when in Colab
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/DATA3001/initial_analysis"
except ImportError:                    # not in Colab
    BASE = "."
except Exception as e:                 # Drive did not mount: files vanish when the session ends
    print("WARNING: Google Drive did not mount (", e, "). Using temporary storage.")
    BASE = "/content/DATA3001/initial_analysis"
os.makedirs(BASE, exist_ok=True)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ----------------------------------------------------------------------------- CONFIG
LAT_MIN, LAT_MAX = -50.0, -30.0
LON_MIN, LON_MAX = -60.0, -40.0
BUFFER_DEG = 2.0            # extra margin extracted around R so exits can be seen
CELL_DEG = 1.0              # grid resolution for the preliminary operator
DT_DAYS = 1                 # operator time step (days)
RELEASE = (-38.0, -54.0)    # (lat, lon) of the example release point in Fig. 2b
H_SHORT, H_LONG = 7, 30     # horizons (days) shown in Fig. 2b/2c
CURVE_DAYS = 180            # length of the retention curve in Fig. 2d
MIN_DRIFTERS = 10           # "well supported" cell threshold for Table 3
INPUT_FILE = os.path.join(BASE, "gdp_bmc_extract.parquet")   # created on the first run
GDP_URL = "https://noaa-oar-hourly-gdp-pds.s3.amazonaws.com/latest/gdp-v2.01.1.zarr"
BLOCK = 5_000_000           # rows streamed per block (lower this if Colab runs out of memory)

# Colours (checked for colour-blind separation)
C_DROGUED, C_UNDROGUED = "#1f6fb2", "#d9822b"
EXIT_COLOURS = {"N": "#1f6fb2", "S": "#d9822b", "E": "#8a5cc2", "W": "#2e9e6a"}
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})

NX = int(round((LON_MAX - LON_MIN) / CELL_DEG))
NY = int(round((LAT_MAX - LAT_MIN) / CELL_DEG))
NCELL = NX * NY
EXITS = ["N", "S", "E", "W"]


# ----------------------------------------------------------------------------- loader
def load_from_gdp():
    """Stream the GDP hourly product once and keep only rows inside R plus the buffer.

    Opens NOAA's public Zarr store directly with xarray (what clouddrift's gdp1h() does),
    so clouddrift does not need to be installed."""
    import time as _time
    import xarray as xr
    ds = xr.open_dataset(GDP_URL, engine="zarr")
    n_obs = ds.sizes["obs"]
    traj_end = np.cumsum(ds["rowsize"].values.astype(np.int64))   # row where each drifter ends
    drifter_id = ds["id"].values
    use_status = "drogue_status" in ds
    lost_dates = None if use_status else ds["drogue_lost_date"].values
    lat_lo, lat_hi = LAT_MIN - BUFFER_DEG, LAT_MAX + BUFFER_DEG
    lon_lo, lon_hi = LON_MIN - BUFFER_DEG, LON_MAX + BUFFER_DEG
    parts, start = [], _time.time()
    for a in range(0, n_obs, BLOCK):
        b = min(a + BLOCK, n_obs)
        sl = slice(a, b)
        lon = ds["lon"].isel(obs=sl).values
        lat = ds["lat"].isel(obs=sl).values
        keep = np.flatnonzero((lat >= lat_lo) & (lat <= lat_hi) & (lon >= lon_lo) & (lon <= lon_hi))
        if keep.size:
            t = ds["time"].isel(obs=sl).values[keep]
            traj = np.searchsorted(traj_end, a + keep, side="right")
            if use_status:
                drogued = ds["drogue_status"].isel(obs=sl).values[keep].astype(bool)
            else:
                lost = lost_dates[traj]
                drogued = np.isnat(lost) | (t < lost)
            parts.append(pd.DataFrame({"id": drifter_id[traj], "time": t, "lat": lat[keep],
                                       "lon": lon[keep], "drogued": drogued}))
        print(f"  streamed {b:,}/{n_obs:,} rows ({100 * b / n_obs:5.1f}%), kept {sum(len(p) for p in parts):,}, "
              f"{(_time.time() - start) / 60:5.1f} min", flush=True)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- helpers
def in_R(lat, lon):
    return (lat >= LAT_MIN) & (lat < LAT_MAX) & (lon >= LON_MIN) & (lon < LON_MAX)


def cell_of(lat, lon):
    """Cell index, row-major from the NORTH-WEST corner (rows run north -> south)."""
    ix = np.floor((lon - LON_MIN) / CELL_DEG).astype(int)
    iy = np.floor((LAT_MAX - lat) / CELL_DEG).astype(int)
    ix = np.clip(ix, 0, NX - 1)
    iy = np.clip(iy, 0, NY - 1)
    return iy * NX + ix


def exit_edge(lat, lon):
    """Which boundary of R a point outside R lies beyond (largest violation wins)."""
    v = np.stack([lat - LAT_MAX,            # N
                  LAT_MIN - lat,            # S
                  lon - LON_MAX,            # E
                  LON_MIN - lon], axis=1)   # W
    return np.array(EXITS)[np.argmax(v, axis=1)]


def as_grid(values):
    return np.array(values, dtype=float).reshape(NY, NX)   # copy, never a view


def map_axes(ax, title):
    ax.set_title(title, loc="left")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal")
    ax.set_facecolor("#e6e6e6")          # grey = no data (mostly land / inner shelf)


def show_grid(ax, grid, **kw):
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    return ax.imshow(np.ma.masked_invalid(grid), origin="upper", extent=extent,
                     interpolation="nearest", **kw)


# ----------------------------------------------------------------------------- preparation
def prepare(df):
    df = df.dropna(subset=["lat", "lon", "time"]).copy()
    df["time"] = pd.to_datetime(df["time"]).dt.round("h")   # snap to whole hours
    df = df.drop_duplicates(["id", "time"]).sort_values(["id", "time"]).reset_index(drop=True)
    df["inR"] = in_R(df["lat"].values, df["lon"].values)
    return df


def daily_transitions(df):
    """One snapshot per drifter per day (00 UTC) and the move to the next day."""
    d = df[(df["time"].dt.hour == 0) & (df["time"].dt.minute == 0)].copy()
    d = d.sort_values(["id", "time"])
    g = d.groupby("id")
    d["t_next"] = g["time"].shift(-DT_DAYS)
    d["lat_next"] = g["lat"].shift(-DT_DAYS)
    d["lon_next"] = g["lon"].shift(-DT_DAYS)
    d["inR_next"] = g["inR"].shift(-DT_DAYS)
    tr = d[d["inR"]].copy()
    complete = (tr["t_next"] - tr["time"]) == pd.Timedelta(days=DT_DAYS)
    tr["outcome"] = np.where(complete, "move", "censored")       # gap or track end
    tr["from_cell"] = cell_of(tr["lat"].values, tr["lon"].values)
    mv = tr["outcome"] == "move"
    inside = mv & tr["inR_next"].astype(bool)
    tr["to"] = -1
    tr.loc[inside, "to"] = cell_of(tr.loc[inside, "lat_next"].values, tr.loc[inside, "lon_next"].values)
    leave = mv & ~tr["inR_next"].astype(bool)
    edges = exit_edge(tr.loc[leave, "lat_next"].values, tr.loc[leave, "lon_next"].values)
    tr.loc[leave, "to"] = [NCELL + EXITS.index(e) for e in edges]
    tr["edge"] = ""
    tr.loc[leave, "edge"] = edges
    return d, tr


def estimate_P(tr):
    """Empirical transition matrix with 4 absorbing exit states (NCELL+4 square)."""
    m = tr[tr["outcome"] == "move"]
    n = NCELL + 4
    C = np.zeros((n, n))
    np.add.at(C, (m["from_cell"].values, m["to"].values.astype(int)), 1)
    counts_out = C[:NCELL].sum(axis=1)
    P = np.zeros_like(C)
    has = counts_out > 0
    P[:NCELL][has] = C[:NCELL][has] / counts_out[has, None]
    P[np.flatnonzero(~has), np.flatnonzero(~has)] = 1.0   # unobserved cells hold their mass
    P[NCELL:, NCELL:] = np.eye(4)                          # exits are absorbing
    return P, counts_out


# ----------------------------------------------------------------------------- Table 3
def visit_stats(daily):
    """Split each drifter's daily snapshots into visits to R and record how each visit ends."""
    rows = []
    for did, g in daily.groupby("id", sort=False):
        t = g["time"].values
        inr = g["inR"].values
        lat, lon, drg = g["lat"].values, g["lon"].values, g["drogued"].values
        i = 0
        while i < len(g):
            if not inr[i]:
                i += 1
                continue
            start = i
            while (i + 1 < len(g) and inr[i + 1]
                   and (t[i + 1] - t[i]) == np.timedelta64(DT_DAYS, "D")):
                i += 1
            nxt = i + 1
            if nxt < len(g) and (t[nxt] - t[i]) == np.timedelta64(DT_DAYS, "D") and not inr[nxt]:
                end = exit_edge(np.array([lat[nxt]]), np.array([lon[nxt]]))[0]
            else:
                end = "censored"
            rows.append({"id": did, "days": (i - start + 1) * DT_DAYS, "end": end,
                         "drogued": bool(drg[i])})
            i += 1
    return pd.DataFrame(rows)


def table3(df, tr, visits):
    r = df[df["inR"]].copy()
    r["cell"] = cell_of(r["lat"].values, r["lon"].values)
    ocean_cells = r["cell"].nunique()
    out = {}
    for label, flag in [("Drogued", True), ("Undrogued", False)]:
        rr = r[r["drogued"] == flag]
        t = tr[(tr["drogued"] == flag) & (tr["outcome"] == "move")]
        v = visits[visits["drogued"] == flag]
        ex = v[v["end"] != "censored"]
        per_cell = rr.groupby("cell")["id"].nunique()
        shares = (ex["end"].value_counts(normalize=True) * 100).reindex(EXITS).fillna(0)
        out[label] = {
            "Hourly observations": f"{len(rr):,}",
            "Distinct drifters": f"{rr['id'].nunique():,}",
            f"Daily ({DT_DAYS}-day) transitions": f"{len(t):,}",
            f"Ocean cells with >= {MIN_DRIFTERS} distinct drifters":
                f"{int((per_cell >= MIN_DRIFTERS).sum())} of {ocean_cells}",
            "Median days in R before first exit": f"{ex['days'].median():.0f}" if len(ex) else "n/a",
            "First exit N / S / E / W (%)": " / ".join(f"{s:.0f}" for s in shares.values),
            "Visits ending inside R (censored, %)":
                f"{(v['end'] == 'censored').mean() * 100:.0f}" if len(v) else "n/a",
        }
    tab = pd.DataFrame(out)
    tab.to_csv(os.path.join(BASE, "table3_data_adequacy.csv"))
    print("\n=== Table 3: data adequacy ===")
    print(tab.to_string())
    # Support for the seasonal split (RQ4)
    season = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
              6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    m = tr[tr["outcome"] == "move"].copy()
    m["season"] = m["time"].dt.month.map(season)
    print("\nDaily transitions by season and drogue status (support for RQ4):")
    print(pd.crosstab(m["season"], m["drogued"].map({True: "drogued", False: "undrogued"}),
                      rownames=["season"], colnames=["drogue status"]))
    return tab


def track_map(ax, df):
    """Daily drifter positions joined into tracks, coloured by drogue status, with R outlined."""
    from matplotlib.collections import LineCollection
    d = df[(df["time"].dt.hour == 0) & (df["time"].dt.minute == 0)].sort_values(["id", "time"])
    same = (d["id"].values[1:] == d["id"].values[:-1]) & \
           (np.diff(d["time"].values) == np.timedelta64(1, "D"))
    xy = np.column_stack([d["lon"].values, d["lat"].values])
    segs = np.stack([xy[:-1], xy[1:]], axis=1)[same]
    drg = d["drogued"].values[:-1][same].astype(bool)
    for mask, colour, label in [(drg, C_DROGUED, "Drogued"), (~drg, C_UNDROGUED, "Undrogued")]:
        ax.add_collection(LineCollection(segs[mask], colors=colour, linewidths=0.25, alpha=0.35))
        ax.plot([], [], color=colour, label=label)
    try:                                   # optional coastline if cartopy is installed
        import cartopy.feature as cfeature
        for geom in cfeature.LAND.with_scale("50m").geometries():
            for poly in getattr(geom, "geoms", [geom]):
                x, y = poly.exterior.xy
                ax.fill(x, y, color="#d9d9d9", zorder=3, linewidth=0)
    except Exception:
        pass
    ax.plot([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
            [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN], color="black", linewidth=1.2, zorder=4)
    ax.set_xlim(LON_MIN - BUFFER_DEG, LON_MAX + BUFFER_DEG)
    ax.set_ylim(LAT_MIN - BUFFER_DEG, LAT_MAX + BUFFER_DEG)
    ax.set_aspect("equal")
    ax.set_title("(a) Drifter tracks and region R", loc="left")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              handlelength=1.2)


# ----------------------------------------------------------------------------- Figure 1
def figure1(df):
    r = df[df["inR"]].copy()
    r["cell"] = cell_of(r["lat"].values, r["lon"].values)
    grids = {}
    for flag in (True, False):
        counts = r[r["drogued"] == flag].groupby("cell")["id"].nunique()
        g = np.full(NCELL, np.nan)
        g[counts.index.values] = counts.values
        grids[flag] = as_grid(g)
    vmax = np.nanmax([np.nanmax(grids[True]), np.nanmax(grids[False])])

    fig = plt.figure(figsize=(12.5, 3.7), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1, 1, 1.45])

    # (a) drifter tracks (daily positions) with the study box, as in a "data" map
    ax = fig.add_subplot(gs[0, 0])
    track_map(ax, df)

    for k, (flag, name) in enumerate([(True, "drogued"), (False, "undrogued")]):
        ax = fig.add_subplot(gs[0, k + 1])
        im = show_grid(ax, grids[flag], cmap="Blues", vmin=0, vmax=vmax)
        map_axes(ax, f"({'bc'[k]}) {name.capitalize()} drifters per cell")
        fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.85, pad=0.02,
                     label="Distinct drifters")

    ax = fig.add_subplot(gs[0, 3])
    yr = r.groupby([r["time"].dt.year, "drogued"]).size().unstack(fill_value=0)
    years = yr.index.values
    d = yr.get(True, pd.Series(0, index=yr.index)).values / 1e3
    u = yr.get(False, pd.Series(0, index=yr.index)).values / 1e3
    ax.bar(years, d, width=0.8, color=C_DROGUED, label="Drogued")
    ax.bar(years, u, width=0.8, bottom=d, color=C_UNDROGUED, label="Undrogued",
           edgecolor="white", linewidth=0.6)
    ax.set_title("(d) Hourly observations per year in R", loc="left")
    ax.set_ylabel("Observations (thousands)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(BASE, "fig1_data_support.png"), dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------- Figure 2
def area_weights():
    lat_c = LAT_MAX - (np.arange(NY) + 0.5) * CELL_DEG
    return np.repeat(np.cos(np.deg2rad(lat_c)), NX)


def figure2(P, counts_out):
    supported = counts_out > 0
    # (a) transition matrix between observed ocean cells
    idx = np.flatnonzero(supported)
    Psub = P[np.ix_(idx, idx)]

    # (b) release forecast
    p = np.zeros(NCELL + 4)
    p[cell_of(np.array([RELEASE[0]]), np.array([RELEASE[1]]))[0]] = 1.0
    snaps = {}
    for k in range(1, H_LONG + 1):
        p = p @ P
        if k in (H_SHORT, H_LONG):
            snaps[k] = p.copy()

    def top_mass_mask(vec, q=0.9):
        v = vec[:NCELL] / vec[:NCELL].sum()
        order = np.argsort(v)[::-1]
        keep = order[: np.searchsorted(np.cumsum(v[order]), q) + 1]
        m = np.zeros(NCELL)
        m[keep] = 1
        return as_grid(m)

    # (c) relative density from an even, area-weighted start
    u = np.zeros(NCELL + 4)
    u[:NCELL] = area_weights() * supported
    u /= u.sum()
    v = u.copy()
    still, exits = [1.0], {e: [0.0] for e in EXITS}
    for k in range(1, CURVE_DAYS + 1):
        v = v @ P
        still.append(v[:NCELL].sum())
        for j, e in enumerate(EXITS):
            exits[e].append(v[NCELL + j])
        if k == H_LONG:
            v_long = v.copy()
    inside = v_long[:NCELL] / v_long[:NCELL].sum()
    rho = np.where(supported, inside / np.where(u[:NCELL] > 0, u[:NCELL], np.nan), np.nan)

    fig = plt.figure(figsize=(10.5, 3.6), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15])

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(np.ma.masked_less_equal(Psub, 0), norm=LogNorm(1e-3, 1), cmap="Blues",
                   interpolation="nearest")
    ax.set_title("(a) Transition matrix P", loc="left")
    ax.set_xlabel("To cell (north → south)")
    ax.set_ylabel("From cell (north → south)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.85, pad=0.02,
                 label="Transition probability")

    ax = fig.add_subplot(gs[0, 1])
    pl = as_grid(snaps[H_LONG][:NCELL])
    pl[pl <= 0] = np.nan
    im = show_grid(ax, pl, cmap="Blues", norm=LogNorm(1e-4, max(1e-3, np.nanmax(pl))))
    ax.contour(np.linspace(LON_MIN + CELL_DEG / 2, LON_MAX - CELL_DEG / 2, NX),
               np.linspace(LAT_MAX - CELL_DEG / 2, LAT_MIN + CELL_DEG / 2, NY),
               top_mass_mask(snaps[H_SHORT]), levels=[0.5], colors="#d9822b", linewidths=1.2)
    ax.plot(RELEASE[1], RELEASE[0], marker="*", color="black", markersize=8)
    map_axes(ax, f"(b) Release, day {H_LONG}")
    fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.85, pad=0.02,
                 label=f"Probability (orange: 90% area, day {H_SHORT})")

    ax = fig.add_subplot(gs[0, 2])
    im = show_grid(ax, as_grid(rho), cmap="RdBu_r", norm=LogNorm(0.1, 10))
    map_axes(ax, f"(c) Relative density, day {H_LONG}")
    fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.85, pad=0.02,
                 label="Relative density (>1 gathers)")

    ax = fig.add_subplot(gs[0, 3])
    days = np.arange(CURVE_DAYS + 1)
    layers = [np.array(still) * 100] + [np.array(exits[e]) * 100 for e in EXITS]
    labels = ["Still in R"] + [f"Left via {e}" for e in EXITS]
    ax.stackplot(days, *layers, colors=["#c9c9c9"] + [EXIT_COLOURS[e] for e in EXITS],
                 labels=labels, edgecolor="white", linewidth=0.5)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              handlelength=1.2, columnspacing=0.8)
    ax.set_xlim(0, CURVE_DAYS)
    ax.set_ylim(0, 100)
    ax.set_title("(d) Even start: still in R or exited", loc="left")
    ax.set_xlabel("Days after release")
    ax.set_ylabel("Share of material (%)")
    fig.savefig(os.path.join(BASE, "fig2_preliminary_operator.png"), dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # numbers for the Section 6 text
    rel = snaps[H_LONG]
    print("\n=== Numbers for Section 6 text ===")
    print(f"Release at {RELEASE}: still in R after {H_SHORT} d = {snaps[H_SHORT][:NCELL].sum():.0%}, "
          f"after {H_LONG} d = {rel[:NCELL].sum():.0%}; exits by day {H_LONG}: "
          + ", ".join(f"{e} {rel[NCELL + j]:.0%}" for j, e in enumerate(EXITS)))
    for k in (H_LONG, 90, CURVE_DAYS):
        print(f"Even start: still in R after {k} d = {still[k]:.0%}; left via "
              + ", ".join(f"{e} {exits[e][k]:.0%}" for e in EXITS))
    print(f"Cells with relative density > 2 at day {H_LONG}: {int(np.nansum(rho > 2))}, "
          f"< 0.5: {int(np.nansum(rho < 0.5))}")


# ----------------------------------------------------------------------------- main
def main(df):
    df = prepare(df)
    daily, tr = daily_transitions(df)
    visits = visit_stats(daily)
    table3(df, tr, visits)
    figure1(df)
    P, counts_out = estimate_P(tr[tr["drogued"]])            # drogued operator for Fig. 2
    figure2(P, counts_out)
    print(f"\nSaved fig1_data_support.png, fig2_preliminary_operator.png and table3_data_adequacy.csv to {BASE}")


if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        print(f"Reading cached extract {INPUT_FILE}")
        data = pd.read_parquet(INPUT_FILE)
    else:
        print("No cached extract yet: streaming R plus buffer from the GDP hourly product...")
        data = load_from_gdp()
        data.to_parquet(INPUT_FILE)
        print(f"Saved {len(data):,} rows to {INPUT_FILE}")
    main(data)
