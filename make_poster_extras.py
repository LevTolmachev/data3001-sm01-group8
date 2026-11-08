"""
DATA3001 Group S01-G8 - extra checks and high-resolution figures for the poster
================================================================================

Run this in Google Colab AFTER make_initial_analysis.py:

  Cell 1:  !pip install -q "zarr<3"
  Cell 2:  paste make_initial_analysis.py and run it (reads the cached extract from Drive,
           takes about a minute)
  Cell 3:  paste THIS file and run it

It does three things:
  A. Verifies the claims in box 4 "Building the Data Product" (duplicates, missing values,
     timestamps, drogue labels, gaps, exits) and prints a short report.
  B. A first held-out check (preliminary evidence for RQ2): fit P on 80% of drogued drifters,
     test on the other 20%, at horizons of 1, 3, 7, 14 and 30 days.
  C. Saves every poster figure again at 450 dpi so they print sharply at A1.

Everything is written to MyDrive/DATA3001/initial_analysis. Please send back:
  - the printed report (copy the cell output),
  - fig1_data_support.png, fig2_preliminary_operator.png and fig_heldout_check.png.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- A. checks
print("=" * 70)
print("A. DATA PRODUCT CHECKS")
print("=" * 70)
raw = data.copy()                                   # `data` was created by make_initial_analysis.py
raw["time"] = pd.to_datetime(raw["time"])
n_raw = len(raw)
n_dup = int(raw.duplicated(["id", "time"]).sum())
n_missing = int(raw[["lat", "lon", "time"]].isna().any(axis=1).sum())
off_hour = int(((raw["time"].dt.minute != 0) | (raw["time"].dt.second != 0)).sum())
print(f"Rows in extract (R + {BUFFER_DEG:.0f} deg buffer): {n_raw:,}")
print(f"Duplicate drifter-time rows:            {n_dup:,}")
print(f"Rows with missing lat/lon/time:         {n_missing:,}")
print(f"Timestamps not exactly on the hour:     {off_hour:,}")

df = prepare(raw)
inR = df[df["inR"]]
print(f"Rows after cleaning:                    {len(df):,}")
print(f"Rows inside R:                          {len(inR):,}  "
      f"(drogued {int(inR['drogued'].sum()):,}, undrogued {int((~inR['drogued']).sum()):,})")
print(f"Distinct drifters in R:                 {inR['id'].nunique():,}  "
      f"(drogued {inR.loc[inR['drogued'], 'id'].nunique():,}, undrogued {inR.loc[~inR['drogued'], 'id'].nunique():,})")
both = inR.groupby("id")["drogued"].nunique()
print(f"Drifters that lose their drogue in R:   {int((both == 2).sum()):,}")
print(f"Years covered:                          {inR['time'].dt.year.min()}-{inR['time'].dt.year.max()}")

# gaps inside each track (hourly product should be gap-free while a drifter reports)
dt_h = df.sort_values(["id", "time"]).groupby("id")["time"].diff().dt.total_seconds() / 3600
print(f"Gaps > 1 h within tracks:               {int((dt_h > 1.01).sum()):,}  (> 24 h: {int((dt_h > 24).sum()):,})")

daily, tr = daily_transitions(df)
for flag, name in [(True, "drogued"), (False, "undrogued")]:
    t_ = tr[tr["drogued"] == flag]
    mv = t_[t_["outcome"] == "move"]
    ex = mv[mv["edge"] != ""]["edge"].value_counts().reindex(EXITS).fillna(0).astype(int)
    print(f"Daily transitions, {name:9s}: moves {len(mv):,}, censored {int((t_['outcome'] == 'censored').sum()):,}, "
          f"exits N/S/E/W {'/'.join(str(v) for v in ex.values)}")
print("\nIf 'Duplicate drifter-time rows' is 0, the poster should say 'check for duplicates (none found)'.")

# ----------------------------------------------------------------------------- B. held-out check
print("\n" + "=" * 70)
print("B. HELD-OUT CHECK (drogued drifters, 80/20 split by drifter)")
print("=" * 70)
HORIZONS = [1, 3, 7, 14, 30]
rng = np.random.default_rng(42)
trd = tr[tr["drogued"]]
ids = trd["id"].unique()
test_ids = set(rng.choice(ids, size=max(1, int(0.2 * len(ids))), replace=False))
P_train, counts_train = estimate_P(trd[~trd["id"].isin(test_ids)])
print(f"Training drifters: {len(ids) - len(test_ids)}, test drifters: {len(test_ids)}")

lat_c = LAT_MAX - (np.arange(NY) + 0.5) * CELL_DEG
lon_c = LON_MIN + (np.arange(NX) + 0.5) * CELL_DEG
cell_lat = np.repeat(lat_c, NX)
cell_lon = np.tile(lon_c, NY)


def haversine_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


snap = daily[daily["id"].isin(test_ids)][["id", "time", "lat", "lon", "inR", "drogued"]]
start = snap[snap["inR"] & snap["drogued"]].copy()
start["cell"] = cell_of(start["lat"].values, start["lon"].values)
rows = []
for k in HORIZONS:
    Pk = np.linalg.matrix_power(P_train, k)
    later = snap.rename(columns={"time": "t_end", "lat": "lat_end", "lon": "lon_end", "inR": "inR_end"})[["id", "t_end", "lat_end", "lon_end", "inR_end"]]
    st = start.assign(t_end=start["time"] + pd.Timedelta(days=k)).merge(later, on=["id", "t_end"], how="inner")
    if st.empty:
        continue
    # true state: cell if inside R, otherwise the exit edge (buffer position)
    in_end = st["inR_end"].astype(bool).values
    state = np.empty(len(st), dtype=int)
    state[in_end] = cell_of(st.loc[in_end, "lat_end"].values, st.loc[in_end, "lon_end"].values)
    edges = exit_edge(st.loc[~in_end, "lat_end"].values, st.loc[~in_end, "lon_end"].values)
    state[~in_end] = [NCELL + EXITS.index(e) for e in edges]
    probs = Pk[st["cell"].values]                                   # one predicted row per case
    # 90% predicted area: smallest set of states holding 90% of the probability
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    n_keep = (np.cumsum(sorted_p, axis=1) < 0.9).sum(axis=1) + 1
    rank_of_truth = np.argmax(order == state[:, None], axis=1)
    hit90 = rank_of_truth < n_keep
    # centre of the predicted cloud (cells only) vs persistence, for cases still inside R
    pin = probs[:, :NCELL]
    mass = pin.sum(axis=1)
    ok = in_end & (mass > 0)
    cen_lat = (pin * cell_lat).sum(axis=1) / np.where(mass > 0, mass, 1)
    cen_lon = (pin * cell_lon).sum(axis=1) / np.where(mass > 0, mass, 1)
    err_op = haversine_km(cen_lat[ok], cen_lon[ok], st.loc[ok, "lat_end"].values, st.loc[ok, "lon_end"].values)
    err_ps = haversine_km(st.loc[ok, "lat"].values, st.loc[ok, "lon"].values, st.loc[ok, "lat_end"].values, st.loc[ok, "lon_end"].values)
    logp = np.log(np.maximum(probs[np.arange(len(st)), state], 1e-6))
    rows.append(dict(horizon_days=k, cases=len(st), drifters=st["id"].nunique(),
                     coverage90_pct=100 * hit90.mean(), median_cells_in_90pct_area=float(np.median(n_keep)),
                     operator_centre_err_km=float(np.median(err_op)) if ok.any() else np.nan,
                     persistence_err_km=float(np.median(err_ps)) if ok.any() else np.nan,
                     mean_log_score=float(logp.mean())))
res = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(res.round(2).to_string(index=False))
res.to_csv(f"{BASE}/heldout_check.csv", index=False)
print("Coverage near 90% means the stated 90% areas are honest (calibrated).")
print("Operator error below persistence error means the operator adds skill over 'it stays where it started'.")

# figure for the poster: two small panels, no dual axis
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
ax[0].plot(res["horizon_days"], res["coverage90_pct"], marker="o", color=C_DROGUED, lw=2)
ax[0].axhline(90, color="#5b6b7a", ls="--", lw=1)
ax[0].text(res["horizon_days"].max(), 87, "target 90%", ha="right", va="top", fontsize=8, color="#5b6b7a")
ax[0].set_xscale("log"); ax[0].set_xticks(HORIZONS); ax[0].set_xticklabels(HORIZONS)
ax[0].set_ylim(0, 100); ax[0].set_xlabel("Horizon (days)"); ax[0].set_ylabel("Test drifters inside 90% area (%)")
ax[0].set_title("Calibration on held-out drifters", loc="left")
ax[1].plot(res["horizon_days"], res["persistence_err_km"], marker="o", color="#5b6b7a", lw=2, label="Persistence")
ax[1].plot(res["horizon_days"], res["operator_centre_err_km"], marker="o", color=C_UNDROGUED, lw=2, label="Operator")
ax[1].set_xscale("log"); ax[1].set_xticks(HORIZONS); ax[1].set_xticklabels(HORIZONS)
ax[1].set_xlabel("Horizon (days)"); ax[1].set_ylabel("Median error (km)")
ax[1].set_title("Error vs simple baseline", loc="left"); ax[1].legend(frameon=False)
for a in ax:
    a.grid(axis="y", color="#e6e6e6", lw=0.6); a.set_axisbelow(True)
fig.savefig(f"{BASE}/fig_heldout_check.png", dpi=450, bbox_inches="tight")
plt.show(); plt.close(fig)

# ----------------------------------------------------------------------------- C. high-res figures
print("\n" + "=" * 70)
print("C. RE-SAVING POSTER FIGURES AT 450 DPI")
print("=" * 70)
_orig_savefig = matplotlib.figure.Figure.savefig
def _hi_res(self, *a, **k):
    k["dpi"] = 450
    return _orig_savefig(self, *a, **k)
matplotlib.figure.Figure.savefig = _hi_res
try:
    figure1(df)
    P_all, counts_all = estimate_P(tr[tr["drogued"]])
    figure2(P_all, counts_all)
finally:
    matplotlib.figure.Figure.savefig = _orig_savefig
print(f"Saved fig1_data_support.png, fig2_preliminary_operator.png and fig_heldout_check.png (450 dpi) to {BASE}")
