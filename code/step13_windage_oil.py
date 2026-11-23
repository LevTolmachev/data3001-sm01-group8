"""
DATA3001 Group S01-G8 - Step 13: wind, the difference between drogued and undrogued drifters, and an
oil scenario (RQ6)
=====================================================================================================
Run AFTER steps 2, 4 and 5, and after code/get_winds.py has produced winds_at_drifters.parquet and
ncep_winds_region.nc in the data folder (NCEP/NCAR Reanalysis 1 daily-mean 10 m winds, Kalnay et al. 1996).

There are no observed oil tracks to test an oil model against, so the wind part is validated on something
we can test: undrogued (surface) drifters. The logic is
  A. Estimate how each drifter type responds to the wind. For every daily move we take the anomaly of the
     displacement from its cell mean and the anomaly of the wind from its cell mean (so the mean current
     cancels), and regress one on the other with a complex coefficient c (km/day per m/s): |c| is the
     share of the wind speed, arg(c) the deflection (positive = to the left, as expected south of the
     equator). The extra response of undrogued drifters is  delta = c_undrogued - c_drogued.
  B. Test it: forecast UNDROGUED test drifters with the DROGUED matrix plus a wind shift of delta x wind,
     and compare with (i) the drogued matrix alone and (ii) a matrix fitted on undrogued drifters
     themselves. If the wind-shifted drogued matrix closes most of the gap, the "current + wind"
     decomposition works on data the model never saw. Wind enters as the distribution of daily winds in
     each cell and calendar month (1994-2022), so no wind forecast is needed. At 1 day we also test the
     actual wind on the start day.
  C. Oil scenario: the 15 m current (drogued matrix) plus 3% of the wind speed, the usual oil drift factor
     (Samuels et al. 1982; ASCE 1996), with 2% and 4% as a sensitivity range, and a simple evaporation
     curve (Fingas 2004). This part cannot be validated against oil and is reported as a scenario.

All settings are fixed in advance (no tuning). Fitting uses training + validation drifters; scoring uses
the test and recent-test drifters (a further, disclosed use of the test set).
Outputs: <BASE>/models/step13.      Run:  uv run python code/step13_windage_oil.py
"""
import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
sys.path.insert(0, _here)
import step2_models_M0_M1_M2 as s2
from step2_models_M0_M1_M2 import (NCELL, NSTATE, NX, NY, EXITS, CELL_LAT, CELL_LON, SEED, OUT, BASE,
                                   ocean_mask, finish, Displacements, m2_smoothed, m0_displacement, haversine_km)
import step4_drogue_season_gam_nn as s4
from step4_drogue_season_gam_nn import (moves_for, starts_for, cases_at, score, drifter_boot, summarise, paired)
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
OUT13 = os.path.join(OUT, "step13")
os.makedirs(OUT13, exist_ok=True)
KM = 111.19                      # km per degree of latitude
KMDAY = 86.4                     # km per day for 1 m/s
HORIZONS = [1, 7, 30, 90]
N_BOOT_COEF = 200
OIL_FACTORS = [0.02, 0.03, 0.04]  # share of wind speed, no deflection (sensitivity range)
RK = 3                            # largest wind shift kept, in cells per day
EVAP_D, EVAP_T = 20.0, 15.0       # % distilled at 180 C (a medium crude) and water temperature (C)
RELEASE = (-38.0, -54.0)          # same release as the presentation (confluence)


# ----------------------------------------------------------------------------- A. wind response
def attach_wind(m, w):
    """Daily-mean wind on the day each move starts (00 UTC position)."""
    w = w[["id", "time", "u10", "v10"]].copy()
    w["id"] = w["id"].astype(str)
    w["time"] = pd.to_datetime(w["time"])
    out = m.merge(w, on=["id", "time"], how="left")
    return out[np.isfinite(out["u10"]) & np.isfinite(out["v10"])].reset_index(drop=True)


def response(m):
    """Complex wind-response coefficient from within-cell anomalies (km/day per m/s)."""
    z = m["de_km"].values + 1j * m["dn_km"].values
    W = m["u10"].values + 1j * m["v10"].values
    g = pd.Series(m["from_cell"].values)
    za = z - pd.Series(z).groupby(g.values).transform("mean").values
    Wa = W - pd.Series(W).groupby(g.values).transform("mean").values
    return np.sum(np.conj(Wa) * za) / np.sum(np.abs(Wa) ** 2)


def describe(c):
    """Share of wind speed (%) and deflection (degrees, positive = to the left of the wind)."""
    return 100 * np.abs(c) / KMDAY, np.degrees(np.angle(c))


def fit_responses(m_all, n_boot=N_BOOT_COEF):
    drg = m_all["drogued"].values.astype(bool)
    cd, cu = response(m_all[drg]), response(m_all[~drg])
    ids = m_all["id"].values
    uid, inv = np.unique(ids, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uid))]
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(n_boot):
        ix = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        mb = m_all.iloc[ix]
        db = mb["drogued"].values.astype(bool)
        boot.append((response(mb[db]), response(mb[~db])))
    boot = np.array(boot)
    rows = []
    for name, est, b in (("drogued (15 m)", cd, boot[:, 0]), ("undrogued (surface)", cu, boot[:, 1]),
                         ("difference (undrogued - drogued)", cu - cd, boot[:, 1] - boot[:, 0])):
        s, a = describe(est)
        sb, ab = describe(b)
        rows.append({"quantity": name, "share_of_wind_%": s, "p05": np.percentile(sb, 5), "p95": np.percentile(sb, 95),
                     "deflection_deg_left": a, "defl_p05": np.percentile(ab, 5), "defl_p95": np.percentile(ab, 95),
                     "moves": int((drg if "drogued (15" in name else ~drg).sum()) if "difference" not in name else len(m_all)})
    return cd, cu, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- wind at cell centres
def cell_winds(grid_file):
    import xarray as xr
    g = xr.open_dataset(grid_file).load()
    lon360 = CELL_LON % 360
    u = g["u10"].interp(lat=xr.DataArray(CELL_LAT, dims="c"), lon=xr.DataArray(lon360, dims="c")).values
    v = g["v10"].interp(lat=xr.DataArray(CELL_LAT, dims="c"), lon=xr.DataArray(lon360, dims="c")).values
    days = pd.to_datetime(g["time"].values)
    return days, u + 1j * v                                     # (days, NCELL) complex, m/s


# ----------------------------------------------------------------------------- wind-shifted matrices
def shift_cells(z_km):
    """Displacement (complex km, east + i north) to (east cells, south cells) at every cell centre."""
    return z_km.real / (KM * np.cos(np.deg2rad(CELL_LAT))), -z_km.imag / KM


def kernels(shifts_e, shifts_s):
    """Bilinear weights of many (east, south) shifts per cell on a (2RK+1)^2 grid of whole-cell offsets.
    shifts_* have shape (samples, NCELL); returns (NCELL, 2RK+1, 2RK+1) averaged over samples."""
    n = 2 * RK + 1
    K = np.zeros((NCELL, n, n))
    e, s = np.clip(shifts_e, -RK, RK - 1e-9), np.clip(shifts_s, -RK, RK - 1e-9)
    e0, s0 = np.floor(e).astype(int), np.floor(s).astype(int)
    fe, fs = e - e0, s - s0
    cells = np.broadcast_to(np.arange(NCELL), e.shape)
    for dy, dx, wgt in ((0, 0, (1 - fs) * (1 - fe)), (0, 1, (1 - fs) * fe), (1, 0, fs * (1 - fe)), (1, 1, fs * fe)):
        np.add.at(K, (cells.ravel(), (s0 + dy + RK).ravel(), (e0 + dx + RK).ravel()), wgt.ravel())
    return K / np.maximum(K.sum(axis=(1, 2), keepdims=True), 1e-12)


def convolve(H, D, K):
    """Shift every cell's displacement histogram by its wind kernel: H (NCELL, D.K) -> same shape."""
    n, R = 2 * D.R + 1, D.R
    Hg = H.reshape(NCELL, n, n)
    out = np.zeros((NCELL, n + 2 * RK, n + 2 * RK))
    for a in range(2 * RK + 1):
        for b in range(2 * RK + 1):
            w = K[:, a, b]
            if not w.any():
                continue
            out[:, a:a + n, b:b + n] += w[:, None, None] * Hg
    return out[:, RK:RK + n, RK:RK + n].reshape(NCELL, n * n)


def smoothed_hist(D, w, sigma):
    H = D.histogram(w)
    if sigma > 0:
        H = gaussian_filter(H.reshape(NY, NX, D.K), sigma=(sigma, sigma, 0), mode="constant", cval=0.0,
                            truncate=2.0).reshape(NCELL, D.K)
    return H


def shifted_operator(D, Hs, K, alpha, P0b):
    C = D.to_counts(convolve(Hs, D, K))
    if alpha > 0:
        C[:NCELL] += alpha * P0b[:NCELL] * D.ocean[:, None]
    return finish(C)


def monthly_operators(D, Hs, alpha, P0b, days, cw, coef):
    """Twelve matrices: the drogued displacements shifted by coef x the daily winds of each calendar month."""
    ops = []
    month = days.month.values
    for mth in range(1, 13):
        z = coef * cw[month == mth]                                  # coef: km/day per m/s -> km/day
        se, ss = shift_cells(z)
        ops.append(shifted_operator(D, Hs, kernels(se, ss), alpha, P0b))
    return ops


def forecast_monthly(ops, c, k):
    X = np.zeros((len(c), NSTATE))
    X[np.arange(len(c)), c["cell"].values] = 1.0
    t0 = c["time"].values.astype("datetime64[D]")
    for t in range(k):
        mth = (t0 + np.timedelta64(t, "D")).astype("datetime64[M]").astype(int) % 12
        for m_ in range(12):
            r = mth == m_
            if r.any():
                X[r] = X[r] @ ops[m_]
    return X


def one_day_actual(D, Hs, alpha, P0b, c, coef, u10, v10):
    """1-day forecast rows using the actual daily-mean wind at each start (cases x states)."""
    z = coef * (u10 + 1j * v10)
    cells = c["cell"].values
    e = np.clip(z.real / (KM * np.cos(np.deg2rad(CELL_LAT[cells]))), -RK, RK - 1e-9)
    s = np.clip(-z.imag / KM, -RK, RK - 1e-9)
    n = 2 * D.R + 1
    rows = np.zeros((len(c), NSTATE))
    for i, ci in enumerate(cells):
        hg = Hs[ci].reshape(n, n)
        out = np.zeros((n + 2 * RK, n + 2 * RK))
        e0, s0 = int(np.floor(e[i])), int(np.floor(s[i]))
        fe, fs = e[i] - e0, s[i] - s0
        for dy, dx, wgt in ((0, 0, (1 - fs) * (1 - fe)), (0, 1, (1 - fs) * fe), (1, 0, fs * (1 - fe)), (1, 1, fs * fe)):
            a, b = s0 + dy + RK, e0 + dx + RK
            out[a:a + n, b:b + n] += wgt * hg
        h = out[RK:RK + n, RK:RK + n].ravel()
        tgt = D.tgt[ci]
        ok = (tgt >= 0) & (h > 0)
        np.add.at(rows[i], tgt[ok], h[ok])
        if alpha > 0:
            rows[i] += alpha * P0b[ci]
        rows[i] /= rows[i].sum()
    return rows


# ----------------------------------------------------------------------------- C. evaporation
def evaporated_share(days):
    """Fingas (2004) logarithmic equation for crude oil: % evaporated = (0.165 %D + 0.045 (T - 15)) ln(t, minutes)."""
    t = np.maximum(np.asarray(days, float) * 1440.0, 1.0)
    return np.clip((0.165 * EVAP_D + 0.045 * (EVAP_T - 15)) * np.log(t), 0, 100) / 100


# ============================================================================= main
def main():
    t0 = time.time()
    wind_file = os.path.join(BASE, "winds_at_drifters.parquet")
    grid_file = os.path.join(BASE, "ncep_winds_region.nc")
    for f in (wind_file, grid_file):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found: run code/get_winds.py first (Google Colab or locally).")
    daily, tr, ids, SIG, ALP = s4.setup()
    fit_ids = ids["train"] | ids["val"]
    ocean = ocean_mask(daily[daily["id"].isin(fit_ids)])
    m_all = moves_for(tr[tr["id"].isin(fit_ids)], "all")
    D = Displacements(m_all, ocean)
    drg = m_all["drogued"].values.astype(bool)
    w = pd.read_parquet(wind_file)
    pd.set_option("display.width", 220)

    # ---- A. wind response of each drifter type (training + validation drifters)
    mw = attach_wind(m_all, w)
    print(f"Moves with wind: {len(mw):,} of {len(m_all):,}")
    cd, cu, coef_tab = fit_responses(mw)
    coef_tab.to_csv(os.path.join(OUT13, "wind_response.csv"), index=False)
    print("\nA. Response to the daily wind (share of wind speed, deflection to the left; 90% interval over drifters)")
    print(coef_tab.round(2).to_string(index=False))
    delta = cu - cd

    # ---- matrices
    Hd = smoothed_hist(D, drg.astype(float), SIG)
    P0b_d = m0_displacement(D, drg.astype(float))
    days, cw = cell_winds(grid_file)
    U0 = m2_smoothed(D, SIG, ALP, w=(~drg).astype(float))                 # surface, fitted on surface drifters
    D0 = m2_smoothed(D, SIG, ALP, w=drg.astype(float))                    # 15 m, no wind
    DW = monthly_operators(D, Hd, ALP, P0b_d, days, cw, delta)            # 15 m + delta x wind
    print(f"\nMatrices built ({(time.time() - t0) / 60:.1f} min)")

    # ---- B. validation on undrogued test drifters
    print("\nB. Forecasting UNDROGUED test drifters (log score gain over the drogued matrix without wind)")
    rows, gains = [], []
    for split in ("test", "recent_test"):
        st = starts_for(daily, ids[split], "undrogued")
        for k in HORIZONS:
            c = cases_at(st, k)
            boot = drifter_boot(c["id"].values)
            res = {"15 m matrix (no wind)": score(s4.forecast(D0, c, k), c),
                   "15 m matrix + wind difference": score(forecast_monthly(DW, c, k), c),
                   "surface matrix (fitted on surface drifters)": score(s4.forecast(U0, c, k), c)}
            if k == 1:
                key = pd.MultiIndex.from_arrays([c["id"].astype(str).values, pd.to_datetime(c["time"]).values])
                ww = w.assign(id=w["id"].astype(str), time=pd.to_datetime(w["time"])).set_index(["id", "time"])
                uu = ww["u10"].reindex(key).values
                vv = ww["v10"].reindex(key).values
                okw = np.isfinite(uu) & np.isfinite(vv)
                uu, vv = np.where(okw, uu, 0.0), np.where(okw, vv, 0.0)          # missing wind: no shift
                res["15 m matrix + actual wind that day"] = score(one_day_actual(D, Hd, ALP, P0b_d, c, delta, uu, vv), c)
            for n_, r in res.items():
                g = paired(r, res["15 m matrix (no wind)"], boot)
                rows.append({"set": f"{split} / undrogued", "model": n_, "horizon_days": k, **summarise(r, boot),
                             "gain_vs_15m": g[0], "gain_p05": g[1], "gain_p95": g[2],
                             "verdict": "reference" if n_.startswith("15 m matrix (no") else g[3]})
            gap = res["surface matrix (fitted on surface drifters)"]["logp"] - res["15 m matrix (no wind)"]["logp"]
            got = res["15 m matrix + wind difference"]["logp"] - res["15 m matrix (no wind)"]["logp"]
            share = [np.nanmean(got[ix]) / np.nanmean(gap[ix]) for ix in boot if np.nanmean(gap[ix]) > 0]
            gains.append({"set": split, "horizon_days": k, "gap_closed_%": 100 * np.nanmean(got) / np.nanmean(gap),
                          "p05": 100 * np.percentile(share, 5), "p95": 100 * np.percentile(share, 95)})
    val = pd.DataFrame(rows)
    val.to_csv(os.path.join(OUT13, "validation_undrogued.csv"), index=False)
    gains = pd.DataFrame(gains)
    gains.to_csv(os.path.join(OUT13, "gap_closed.csv"), index=False)
    print(val[["set", "model", "horizon_days", "cases", "drifters", "log_score", "in_90pct_area_%",
               "median_centre_error_km", "gain_vs_15m", "gain_p05", "gain_p95", "verdict"]].round(3).to_string(index=False))
    print("\nShare of the gap between the 15 m and the surface matrix closed by adding the wind difference:")
    print(gains.round(1).to_string(index=False))

    # ---- C. oil scenario from the confluence release (January vs July winds)
    print("\nC. Oil scenario (current at 15 m + share of the wind), release at", RELEASE)
    c0 = int(s2.cell_of(np.array([RELEASE[0]]), np.array([RELEASE[1]]))[0])
    snaps, scen = {}, []
    variants = {"drifter at 15 m": None, "drifter at the surface": "U0"}
    variants.update({f"oil, {int(100 * f)}% of wind": f for f in OIL_FACTORS})
    for mth, lab in ((1, "January"), (7, "July")):
        start = pd.Timestamp(f"2001-{mth:02d}-15")
        for vname, f in variants.items():
            if f is None:
                ops = [D0] * 12
            elif f == "U0":
                ops = [U0] * 12
            else:
                ops = monthly_operators(D, Hd, ALP, P0b_d, days, cw, f * KMDAY + 0j)
            p = np.zeros(NSTATE); p[c0] = 1.0
            for t in range(1, 91):
                p = p @ ops[(start + pd.Timedelta(days=t - 1)).month - 1]
                if t in (7, 30, 90):
                    snaps[(lab, vname, t)] = p.copy()
                    scen.append({"release_month": lab, "material": vname, "day": t, "still_in_R": p[:NCELL].sum(),
                                 **{f"left_{e}": p[NCELL + j] for j, e in enumerate(EXITS)},
                                 "mean_lat_in_R": float(p[:NCELL] @ CELL_LAT / max(p[:NCELL].sum(), 1e-12)),
                                 "mean_lon_in_R": float(p[:NCELL] @ CELL_LON / max(p[:NCELL].sum(), 1e-12)),
                                 "oil_floating_share": (1 - evaporated_share(t)) * p[:NCELL].sum() if isinstance(f, float) else np.nan})
    scen = pd.DataFrame(scen)
    scen.to_csv(os.path.join(OUT13, "oil_scenario.csv"), index=False)
    print(scen.round(3).to_string(index=False))
    np.savez_compressed(os.path.join(OUT13, "oil_scenario_snaps.npz"),
                        keys=np.array(["|".join(map(str, k)) for k in snaps]), snaps=np.array(list(snaps.values())),
                        release_cell=c0)
    ev = pd.DataFrame({"day": [1, 7, 30, 90], "evaporated_%": 100 * evaporated_share([1, 7, 30, 90])})
    ev.to_csv(os.path.join(OUT13, "evaporation.csv"), index=False)
    with open(os.path.join(OUT13, "settings.json"), "w") as f_:
        json.dump({"oil_factors": OIL_FACTORS, "evap_percent_distilled_180C": EVAP_D, "water_temp_C": EVAP_T,
                   "release": RELEASE, "delta_complex_km_per_day_per_ms": [delta.real, delta.imag],
                   "sigma": SIG, "alpha": ALP}, f_)
    print(f"\nStep 13 done in {(time.time() - t0) / 60:.1f} min. Outputs in {OUT13}")


if __name__ == "__main__":
    main()
