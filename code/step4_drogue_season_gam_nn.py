"""
DATA3001 Group S01-G8 - Step 4: RQ4 (drogue status, season) and RQ5 (GAM, neural network)
==========================================================================================

Run AFTER step2_models_M0_M1_M2.py (it re-uses step 2's functions, the frozen splits.csv and the
M2 settings in m2_tuning.csv). Step 3 is not needed for this file.

Every model here ends up as the same object as M1 and M2: a 404 x 404 one-day transition matrix on
the 1 degree grid. So every model is scored in exactly the same way, on the same validation
drifters, and "is the extra complexity worth it?" is answered by the paired difference in log
score against M2 with a 90% interval from resampling whole validation drifters.

  RQ4a  Drogue status.  M2 fitted on drogued moves, on undrogued moves and on all moves,
        each scored on drogued starts and on undrogued starts.
  RQ4b  Season.  One M2 per season (DJF, MAM, JJA, SON), with each season pulled towards the
        all-year operator by lam pseudo-moves (lam = 0 is a fully separate operator per season).
        Forecasts step through the seasons day by day.
  RQ5a  GAM.  Daily displacement (km east, km north) ~ Gaussian whose mean and covariance are
        smooth tensor-product spline functions of longitude and latitude (a location-scale GAM,
        penalised). The Gaussian is integrated over the destination cells to give a matrix.
  RQ5b  Neural network.  A small network maps a position to the probability of each daily
        cell displacement (softmax). Averaged over start points in a cell this gives a matrix.
        "NN+" also gets day of year and drogue status as inputs and is trained on all moves.

Nothing here touches the test drifters. Outputs go to <BASE>/models/step4.

Run:  python step4_drogue_season_gam_nn.py     (set DATA3001_BASE to the data folder)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
sys.path.insert(0, _here)
import step2_models_M0_M1_M2 as s2
from step2_models_M0_M1_M2 import (NCELL, NSTATE, NX, NY, EXITS, CELL_LAT, CELL_LON, LAT_MAX, LON_MIN,
                                   CELL_DEG, SEED, LOG_FLOOR, MIN_DRIFTERS, OUT, INPUT_FILE, SPLIT_FILE,
                                   prepare, daily_transitions, make_splits, ocean_mask, finish,
                                   Displacements, m0_displacement, m1_empirical, m2_smoothed,
                                   haversine_km, propagate_cells)
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- CONFIG
HORIZONS = [1, 7, 30, 90]            # days
N_BOOT_SCORE = 500                   # resamples of whole drifters for score intervals
SEASON_LAMS = [0, 20, 100, 500]      # pull of each season towards the all-year operator (pseudo-moves)
GAM_KNOTS = [4, 5, 8, 12]            # spline knots per axis
GAM_RIDGE = [1.0, 10.0, 100.0, 1000.0]   # smoothing penalty
DROGUE_W = [0.25, 0.5]               # weight given to the other drogue type's moves (0 = separate, 1 = pooled)
NN_HIDDEN = [(32, 32), (64, 64)]
NN_L2 = [1e-4, 1e-2]
NN_SEEDS = 3                         # networks averaged per setting
SUB = 4                              # sub-points per cell side when turning a model into a matrix
KM_PER_DEG = 111.19
# index by month-1: Jan, Feb -> 0 (DJF); Mar-May -> 1 (MAM); Jun-Aug -> 2 (JJA); Sep-Nov -> 3 (SON); Dec -> 0
SEASON_OF_MONTH = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0])
SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON_MID_DOY = {0: [15, 46, 349], 1: [74, 105, 135], 2: [166, 196, 227], 3: [258, 288, 319]}

OUT4 = os.path.join(OUT, "step4")
os.makedirs(OUT4, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------------------- data helpers
def moves_for(tr, drogue_set):
    """Daily moves (with displacement in cells and in km) for one drogue set."""
    old = s2.DROGUE_SET
    s2.DROGUE_SET = drogue_set
    try:
        m = s2.move_table(tr)
    finally:
        s2.DROGUE_SET = old
    lat, lon = m["lat"].values, m["lon"].values
    m["dn_km"] = (m["lat_next"].values.astype(float) - lat) * KM_PER_DEG
    m["de_km"] = (m["lon_next"].values.astype(float) - lon) * KM_PER_DEG * np.cos(np.deg2rad(lat))
    m["season"] = SEASON_OF_MONTH[m["time"].dt.month.values - 1]
    m["doy"] = m["time"].dt.dayofyear.values
    return m


def starts_for(daily, ids, drogue_set):
    """Every drifter-day inside R for one drogue set, with how its visit ends (step 2 logic)."""
    old = s2.DROGUE_SET
    s2.DROGUE_SET = drogue_set
    try:
        st = s2.visit_starts(daily, ids, stride=1)
    finally:
        s2.DROGUE_SET = old
    snap = daily[daily["id"].isin(ids)][["id", "time", "lat", "lon"]].rename(
        columns={"time": "t_end", "lat": "lat_end", "lon": "lon_end"})
    return st, snap


def cases_at(starts, k):
    """Truth k days after each start: the cell if it stayed in R throughout, or the edge it first
    left through. Starts whose record stops inside R before day k are dropped (unknown outcome)."""
    st0, snap = starts
    gone = st0["exited"] & (st0["days"] <= k)
    stay = (st0["exited"] & (st0["days"] > k)) | (~st0["exited"] & (st0["days"] >= k))
    a = st0[stay].assign(t_end=lambda x: x["time"] + pd.Timedelta(days=k)).merge(snap, on=["id", "t_end"])
    b = st0[gone].assign(lat_end=np.nan, lon_end=np.nan)
    c = pd.concat([a.drop(columns="t_end"), b], ignore_index=True)
    c["inn"] = np.r_[np.ones(len(a), bool), np.zeros(len(b), bool)]
    state = np.empty(len(c), dtype=int)
    inn = c["inn"].values
    state[inn] = s2.cell_of(c.loc[inn, "lat_end"].values, c.loc[inn, "lon_end"].values)
    state[~inn] = [NCELL + EXITS.index(e) for e in c.loc[~inn, "edge"]]
    c["state"] = state
    return c


# ----------------------------------------------------------------------------- forecasts and scores
def forecast(model, c, k):
    """Forecast rows (cases x states) k days ahead. `model` is one matrix, or a list of four
    seasonal matrices (DJF, MAM, JJA, SON) that are stepped through day by day."""
    if isinstance(model, np.ndarray):
        return np.linalg.matrix_power(model, k)[c["cell"].values]
    X = np.zeros((len(c), NSTATE))
    X[np.arange(len(c)), c["cell"].values] = 1.0
    t0 = c["time"].values.astype("datetime64[D]")
    for t in range(k):
        month = (t0 + np.timedelta64(t, "D")).astype("datetime64[M]").astype(int) % 12
        sea = SEASON_OF_MONTH[month]
        for s in range(4):
            r = sea == s
            if r.any():
                X[r] = X[r] @ model[s]
    return X


def score(probs, c):
    """Per-case scores: log score, inside the 50% / 90% area, distance error of the cloud centre."""
    state = c["state"].values
    n = len(c)
    order = np.argsort(-probs, axis=1, kind="stable")
    csum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
    rank = np.argmax(order == state[:, None], axis=1)
    out = {"id": c["id"].values}
    for q in (0.5, 0.9):
        out[f"hit{int(q * 100)}"] = (rank < (csum < q - 1e-12).sum(axis=1) + 1).astype(float)
    out["logp"] = np.log(np.maximum(probs[np.arange(n), state], LOG_FLOOR))
    pin = probs[:, :NCELL]
    mass = pin.sum(axis=1)
    cl = (pin @ CELL_LAT) / np.where(mass > 0, mass, 1)
    co = (pin @ CELL_LON) / np.where(mass > 0, mass, 1)
    ok = c["inn"].values & (mass > 0)
    err = np.full(n, np.nan)
    err[ok] = haversine_km(cl[ok], co[ok], c["lat_end"].values[ok], c["lon_end"].values[ok])
    out["err_km"] = err
    return out


def drifter_boot(ids, n_boot=N_BOOT_SCORE, seed=SEED):
    codes, uniq = pd.factorize(pd.Series(ids))
    groups = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
            for _ in range(n_boot)]


def summarise(r, boot):
    row = {"cases": len(r["id"]), "drifters": len(np.unique(r["id"]))}
    for name, key, f in [("log_score", "logp", np.nanmean),
                         ("in_50pct_area_%", "hit50", lambda x: 100 * np.nanmean(x)),
                         ("in_90pct_area_%", "hit90", lambda x: 100 * np.nanmean(x)),
                         ("median_centre_error_km", "err_km", np.nanmedian)]:
        row[name] = f(r[key])
        v = np.array([f(r[key][ix]) for ix in boot])
        row[name + "_p05"], row[name + "_p95"] = np.nanpercentile(v, [5, 95])
    return row


def paired(ra, rb, boot):
    d = ra["logp"] - rb["logp"]
    v = np.array([np.nanmean(d[ix]) for ix in boot])
    lo, hi = np.nanpercentile(v, [5, 95])
    return float(np.nanmean(d)), float(lo), float(hi), ("better" if lo > 0 else "worse" if hi < 0 else "no clear difference")


def evaluate_models(models, cases, reference, label=""):
    """Score every model on the same cases; paired gain against `reference`."""
    rows, res = [], {}
    for k, c in cases.items():
        boot = drifter_boot(c["id"].values)
        for name, mdl in models.items():
            res[(name, k)] = score(forecast(mdl, c, k), c)
        for name in models:
            g = paired(res[(name, k)], res[(reference, k)], boot)
            rows.append({"set": label, "model": name, "horizon_days": k, **summarise(res[(name, k)], boot),
                         "gain_vs_" + reference.split()[0]: g[0], "gain_p05": g[1], "gain_p95": g[2],
                         "verdict": "reference" if name == reference else g[3]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- RQ4a partial pooling
def drogue_weighted(D_all, m_all, target, w_other, sigma, alpha, w=None):
    """M2 for one drogue type that also uses the other type's moves, down-weighted by w_other."""
    own = m_all["drogued"].values.astype(bool) == (target == "drogued")
    ww = np.where(own, 1.0, w_other) * (1.0 if w is None else w)
    return m2_smoothed(D_all, sigma, alpha, w=ww)


# ----------------------------------------------------------------------------- RQ4b seasonal operators
def seasonal_operators(D, m, sigma, alpha, lam, P_all, P0b):
    """One smoothed operator per season. Each is pulled towards the all-year operator P_all by
    lam pseudo-moves per cell (lam = 0: separate operators; large lam: the all-year operator)."""
    ops = []
    for s in range(4):
        H = D.histogram((m["season"].values == s).astype(float))
        if sigma > 0:
            H = gaussian_filter(H.reshape(NY, NX, D.K), sigma=(sigma, sigma, 0),
                                mode="constant", cval=0.0, truncate=2.0).reshape(NCELL, D.K)
        C = D.to_counts(H)
        C[:NCELL] += (alpha * P0b[:NCELL] + lam * P_all[:NCELL]) * D.ocean[:, None]
        ops.append(finish(C))
    return ops


# ----------------------------------------------------------------------------- matrix from a point model
def sub_points():
    """SUB x SUB start points inside every cell: arrays (NCELL, SUB*SUB) of lat and lon."""
    off = (np.arange(SUB) + 0.5) / SUB - 0.5
    oy, ox = np.meshgrid(off, off, indexing="ij")
    lat = CELL_LAT[:, None] - oy.ravel()[None, :] * CELL_DEG
    lon = CELL_LON[:, None] + ox.ravel()[None, :] * CELL_DEG
    return lat, lon


# ----------------------------------------------------------------------------- RQ5a GAM
class GaussianGAM:
    """Location-scale GAM: displacement (east, north) in km ~ N(mu(lon, lat), Sigma(lon, lat)),
    with mu and Sigma penalised tensor-product cubic spline surfaces."""

    def __init__(self, n_knots, ridge):
        from sklearn.preprocessing import SplineTransformer
        self.n_knots, self.ridge = n_knots, ridge
        kn = lambda lo, hi: np.linspace(lo, hi, n_knots)[:, None]
        self.sx = SplineTransformer(degree=3, knots=kn(s2.LON_MIN, s2.LON_MAX), extrapolation="linear")
        self.sy = SplineTransformer(degree=3, knots=kn(s2.LAT_MIN, s2.LAT_MAX), extrapolation="linear")
        self.sx.fit(np.zeros((2, 1))); self.sy.fit(np.zeros((2, 1)))

    def basis(self, lat, lon):
        bx = self.sx.transform(np.asarray(lon, float).reshape(-1, 1))
        by = self.sy.transform(np.asarray(lat, float).reshape(-1, 1))
        return (bx[:, :, None] * by[:, None, :]).reshape(len(bx), -1)

    def _solve(self, B, Y, w, pen):
        A = (B * w[:, None]).T @ B + pen * np.eye(B.shape[1])
        return np.linalg.solve(A, (B * w[:, None]).T @ Y)

    def fit(self, m, w=None):
        w = np.ones(len(m)) if w is None else np.asarray(w, float)
        B = self.basis(m["lat"].values, m["lon"].values)
        Y = m[["de_km", "dn_km"]].values
        self.beta = self._solve(B, Y, w, self.ridge)
        r = Y - B @ self.beta
        S = np.column_stack([r[:, 0] ** 2, r[:, 1] ** 2, r[:, 0] * r[:, 1]])
        self.gamma = self._solve(B, S, w, 10 * self.ridge)        # the spread is smoothed more
        self.floor = 0.05 * np.average(S[:, :2], axis=0, weights=w)
        return self

    def predict(self, lat, lon):
        B = self.basis(lat, lon)
        mu = B @ self.beta
        S = B @ self.gamma
        vE, vN = np.maximum(S[:, 0], self.floor[0]), np.maximum(S[:, 1], self.floor[1])
        rho = np.clip(S[:, 2] / np.sqrt(vE * vN), -0.95, 0.95)
        return mu, vE, vN, rho

    def histogram(self, D):
        """Probability of each cell displacement from each cell (NCELL x D.K)."""
        lat, lon = sub_points()
        P = lat.shape[1]
        mu, vE, vN, rho = self.predict(lat.ravel(), lon.ravel())
        off = (np.arange(SUB) + 0.5) / SUB - 0.5
        H = np.zeros((NCELL, D.K))
        la0, lo0 = lat.ravel(), lon.ravel()
        cl, cc = np.repeat(CELL_LAT, P), np.repeat(CELL_LON, P)
        det = vE * vN * (1 - rho ** 2)
        for k in range(D.K):
            dens = np.zeros(len(la0))
            for oy in off:
                for ox in off:
                    tlat = cl - (D.DY[k] + oy) * CELL_DEG
                    tlon = cc + (D.DX[k] + ox) * CELL_DEG
                    zn = (tlat - la0) * KM_PER_DEG - mu[:, 1]
                    ze = (tlon - lo0) * KM_PER_DEG * np.cos(np.deg2rad(la0)) - mu[:, 0]
                    q = (ze ** 2 * vN - 2 * rho * np.sqrt(vE * vN) * ze * zn + zn ** 2 * vE) / det
                    dens += np.exp(-0.5 * q) / np.sqrt(det) * np.cos(np.deg2rad(tlat))
            H[:, k] = dens.reshape(NCELL, P).mean(axis=1)
        return H / np.maximum(H.sum(axis=1, keepdims=True), 1e-300)

    def operator(self, D):
        return finish(D.to_counts(self.histogram(D)))


# ----------------------------------------------------------------------------- RQ5b neural network
def nn_features(lat, lon, doy=None, drogued=None):
    X = [(np.asarray(lon, float) + 50) / 10, (np.asarray(lat, float) + 40) / 10]
    if doy is not None:
        a = 2 * np.pi * np.asarray(doy, float) / 365.25
        X += [np.sin(a), np.cos(a), np.asarray(drogued, float)]
    return np.column_stack(X)


class DisplacementNet:
    """Small network: position (and optionally day of year, drogue status) -> probability of each
    daily cell displacement. An average of NN_SEEDS networks with different random starts."""

    def __init__(self, hidden, l2, full=False):
        self.hidden, self.l2, self.full = hidden, l2, full

    def fit(self, m, D, w=None):
        from sklearn.neural_network import MLPClassifier
        ok = D.k >= 0
        rep = np.ones(ok.sum(), int) if w is None else np.asarray(w)[ok].astype(int)   # bootstrap weights
        X = nn_features(m["lat"].values, m["lon"].values,
                        m["doy"].values if self.full else None,
                        m["drogued"].values if self.full else None)[ok]
        X, y = np.repeat(X, rep, axis=0), np.repeat(D.k[ok], rep)
        self.K = D.K
        self.nets = []
        for s in range(NN_SEEDS):
            net = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.l2, early_stopping=True,
                                validation_fraction=0.15, n_iter_no_change=8, max_iter=150,
                                learning_rate_init=3e-3, batch_size=256, random_state=SEED + s)
            self.nets.append(net.fit(X, y))
        return self

    def proba(self, X):
        out = np.zeros((len(X), self.K))
        for net in self.nets:
            out[:, net.classes_] += net.predict_proba(X)
        return out / len(self.nets)

    def histogram(self, doys=None, drogued=None):
        lat, lon = sub_points()
        P = lat.shape[1]
        if not self.full:
            return self.proba(nn_features(lat.ravel(), lon.ravel())).reshape(NCELL, P, self.K).mean(axis=1)
        H = np.zeros((NCELL, self.K))
        for d in doys:
            X = nn_features(lat.ravel(), lon.ravel(), np.full(lat.size, d), np.full(lat.size, drogued))
            H += self.proba(X).reshape(NCELL, P, self.K).mean(axis=1)
        return H / len(doys)

    def operator(self, D, **kw):
        return finish(D.to_counts(self.histogram(**kw)))

    def seasonal(self, D, drogued):
        return [self.operator(D, doys=SEASON_MID_DOY[s], drogued=drogued) for s in range(4)]


# ----------------------------------------------------------------------------- set-up shared by step 5
def setup():
    data = pd.read_parquet(INPUT_FILE)
    df = prepare(data)
    daily, tr = daily_transitions(df)
    if not os.path.exists(SPLIT_FILE):
        raise RuntimeError("splits.csv not found: run step 2 once first.")
    split = make_splits(daily)
    daily["split"] = daily["id"].map(split)
    tr["split"] = tr["id"].map(split)
    ids = {s: set(daily.loc[daily["split"] == s, "id"]) for s in ("train", "val", "test", "recent_test")}
    tune = pd.read_csv(os.path.join(OUT, "m2_tuning.csv"))
    best = tune.loc[tune["val_log_score"].idxmax()]
    return daily, tr, ids, float(best["sigma"]), float(best["alpha"])


def gain_lines(tab, col):
    for _, r in tab[tab["verdict"] != "reference"].iterrows():
        print(f"  {r['set']:22s} day {int(r['horizon_days']):3d}  {r['model']:28s} {r[col]:+.3f} "
              f"[{r['gain_p05']:+.3f}, {r['gain_p95']:+.3f}]  {r['verdict']}")


# ============================================================================= main
def main():
    t_start = time.time()
    daily, tr, ids, SIG, ALP = setup()
    print(f"M2 settings from step 2: sigma = {SIG}, alpha = {ALP}")
    trn = tr[tr["split"] == "train"]
    ocean = ocean_mask(daily[daily["split"] == "train"])
    pd.set_option("display.width", 220)
    show = ["set", "model", "horizon_days", "cases", "drifters", "log_score", "in_90pct_area_%", "median_centre_error_km"]

    # ------------------------------------------------------------------ RQ4a drogue status
    print("\n" + "=" * 78 + "\nRQ4a  DROGUE STATUS\n" + "=" * 78)
    mset = {d: moves_for(trn, d) for d in ("drogued", "undrogued", "all")}
    Dset = {d: Displacements(mset[d], ocean) for d in mset}
    ops = {f"M2 fitted on {d}": m2_smoothed(Dset[d], SIG, ALP) for d in mset}
    for d in mset:
        print(f"  {d:9s}: {len(mset[d]):,} training moves from {mset[d]['id'].nunique()} drifters; "
              f"median daily displacement {np.median(np.hypot(mset[d]['de_km'], mset[d]['dn_km'])):.1f} km")
    drogue_tab = []
    vstarts = {d: starts_for(daily, ids["val"], d) for d in ("drogued", "undrogued")}
    for d in ("drogued", "undrogued"):
        cases = {k: cases_at(vstarts[d], k) for k in HORIZONS}
        these = dict(ops)
        for wo in DROGUE_W:                      # partial pooling: the other type's moves count for less
            these[f"M4 own type + {wo:g} x other"] = drogue_weighted(Dset["all"], mset["all"], d, wo, SIG, ALP)
        t = evaluate_models(these, cases, "M2 fitted on all", label=f"{d} starts")
        drogue_tab.append(t)
    drogue_tab = pd.concat(drogue_tab, ignore_index=True)
    drogue_tab.to_csv(os.path.join(OUT4, "rq4_drogue.csv"), index=False)
    print(drogue_tab[show].round(2).to_string(index=False))
    print("\nGain in log score over the pooled operator (positive = better, 90% interval):")
    gain_lines(drogue_tab, "gain_vs_M2")

    # physical difference between the two operators
    phys = []
    for d in ("drogued", "undrogued"):
        P = ops[f"M2 fitted on {d}"]
        still, M = propagate_cells(P, 365)
        w = s2.area_weights() * ocean
        w = w / w.sum()
        curve = w @ still
        phys.append({"operator": d, "even_start_still_in_R_day30": curve[30], "day90": curve[90], "day365": curve[365],
                     "days_until_half_left": int(np.argmax(curve <= 0.5)),
                     **{f"left_{e}_by_day365": float(w @ M[:, NCELL + j]) for j, e in enumerate(EXITS)}})
    phys = pd.DataFrame(phys)
    phys.to_csv(os.path.join(OUT4, "rq4_drogue_physical.csv"), index=False)
    print("\nWhat the two operators say (even start over R):")
    print(phys.round(3).to_string(index=False))

    # ------------------------------------------------------------------ RQ4b season
    print("\n" + "=" * 78 + "\nRQ4b  SEASON (drogued)\n" + "=" * 78)
    m, D = mset["drogued"], Dset["drogued"]
    P0b = m0_displacement(D)
    P_m2 = ops["M2 fitted on drogued"]
    print("  training moves per season: " + ", ".join(f"{SEASONS[s]} {(m['season'] == s).sum():,}" for s in range(4)))
    vcases = {k: cases_at(vstarts["drogued"], k) for k in HORIZONS}
    smodels = {"M2 all-year": P_m2}
    for lam in SEASON_LAMS:
        smodels[f"M4 seasonal, pull {lam}"] = seasonal_operators(D, m, SIG, ALP, lam, P_m2, P0b)
    season_tab = evaluate_models(smodels, vcases, "M2 all-year", label="drogued starts")
    season_tab.to_csv(os.path.join(OUT4, "rq4_season.csv"), index=False)
    print(season_tab[show].round(2).to_string(index=False))
    print("\nGain in log score over the all-year operator:")
    gain_lines(season_tab, "gain_vs_M2")

    # ------------------------------------------------------------------ RQ5a GAM
    print("\n" + "=" * 78 + "\nRQ5a  GAM (drogued)\n" + "=" * 78)
    tune_k = [7, 30]
    gtune = []
    for nk in GAM_KNOTS:
        for rg in GAM_RIDGE:
            P = GaussianGAM(nk, rg).fit(m).operator(D)
            ls = np.mean([score(forecast(P, vcases[k], k), vcases[k])["logp"].mean() for k in tune_k])
            gtune.append({"n_knots": nk, "ridge": rg, "val_log_score": ls})
    gtune = pd.DataFrame(gtune)
    gtune.to_csv(os.path.join(OUT4, "gam_tuning.csv"), index=False)
    print(gtune.round(3).to_string(index=False))
    gb = gtune.loc[gtune["val_log_score"].idxmax()]
    gam = GaussianGAM(int(gb["n_knots"]), float(gb["ridge"])).fit(m)
    P_gam = gam.operator(D)
    print(f"Chosen GAM: {int(gb['n_knots'])} knots per axis, ridge {gb['ridge']}")

    # ------------------------------------------------------------------ RQ5b neural network
    print("\n" + "=" * 78 + "\nRQ5b  NEURAL NETWORK\n" + "=" * 78)
    ntune, nets = [], {}
    for h in NN_HIDDEN:
        for l2 in NN_L2:
            net = DisplacementNet(h, l2).fit(m, D)
            P = net.operator(D)
            ls = np.mean([score(forecast(P, vcases[k], k), vcases[k])["logp"].mean() for k in tune_k])
            ntune.append({"hidden": str(h), "l2": l2, "val_log_score": ls})
            nets[(str(h), l2)] = P
    ntune = pd.DataFrame(ntune)
    ntune.to_csv(os.path.join(OUT4, "nn_tuning.csv"), index=False)
    print(ntune.round(3).to_string(index=False))
    nb = ntune.loc[ntune["val_log_score"].idxmax()]
    P_nn = nets[(nb["hidden"], nb["l2"])]
    print(f"Chosen network: hidden {nb['hidden']}, L2 {nb['l2']}")
    # NN+ : position + day of year + drogue status, trained on all moves
    m_all, D_all = mset["all"], Dset["all"]
    net_full = DisplacementNet(eval(nb["hidden"]), float(nb["l2"]), full=True).fit(m_all, D_all)
    P_nn_full = net_full.seasonal(D_all, drogued=1.0)

    ladder = {"M0b displacement": P0b, "M1 empirical": m1_empirical(m), "M2 smoothed": P_m2,
              "M5 GAM": P_gam, "M6 neural net": P_nn, "M6+ neural net (season, drogue)": P_nn_full}
    lad_tab = evaluate_models(ladder, vcases, "M2 smoothed", label="drogued starts")
    lad_tab.to_csv(os.path.join(OUT4, "rq5_ladder_validation.csv"), index=False)
    print("\n=== Whole ladder on the validation drifters ===")
    print(lad_tab[show].round(2).to_string(index=False))
    print("\nGain in log score over M2 (positive = better, 90% interval):")
    gain_lines(lad_tab, "gain_vs_M2")

    pd.DataFrame([{"gam_n_knots": int(gb["n_knots"]), "gam_ridge": float(gb["ridge"]),
                   "nn_hidden": nb["hidden"], "nn_l2": float(nb["l2"])}]).to_csv(
        os.path.join(OUT4, "chosen_settings.csv"), index=False)
    print(f"\nStep 4 done in {(time.time() - t_start) / 60:.1f} min. Outputs in {OUT4}")


if __name__ == "__main__":
    main()
