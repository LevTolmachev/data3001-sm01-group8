"""
DATA3001 Group S01-G8 - Step 10: models with memory (a third attempt at RQ5)
=============================================================================
Run AFTER steps 2, 4, 5 and 8.

Every model so far is first-order: tomorrow depends only on today's cell. A drifter also has
momentum: the direction it moved yesterday says something about today. Position alone cannot
carry that, which caps what a GAM or a network can learn from position. Here the state is
(cell, direction of the previous day's move), so the operator stays a (larger) transition matrix
and is scored exactly as before.

  direction class   8 compass sectors of the daily displacement, "slow" (< SLOW_KM), or "unknown"
  M7  memory, counts    P(today's direction | cell, yesterday's direction) and
                        P(cell move | cell, today's direction), both smoothed over neighbouring cells
                        and shrunk towards the memoryless M2
  M7g memory, GAM       multinomial GAM: spline of position + previous direction + their interaction
  M7n memory, network   network on Fourier features of position + previous direction

All settings are chosen on the VALIDATION drifters. Section C scores the chosen models on the test
drifters: this is a further, disclosed use of the test set (the earlier models are unchanged).
A forecast with memory needs the previous day's position, so it applies when a recent track is known.

Outputs: <BASE>/models/step10.      Run:  python step10_memory_models.py
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
sys.path.insert(0, _here)
import step2_models_M0_M1_M2 as s2
from step2_models_M0_M1_M2 import NCELL, NX, NY, SEED, OUT, ocean_mask, Displacements, m2_smoothed
import step4_drogue_season_gam_nn as s4
from step4_drogue_season_gam_nn import (HORIZONS, moves_for, starts_for, cases_at, forecast, score, drifter_boot,
                                        summarise, paired, sub_points, KM_PER_DEG)
from step8_stronger_gam_nn import fourier

warnings.filterwarnings("ignore")
OUT10 = os.path.join(OUT, "step10")
os.makedirs(OUT10, exist_ok=True)

SLOW_KM = 5.0
NA, UNK = 10, 9                      # 8 sectors + slow + unknown
NS = NCELL * NA + 4
BETAS = [10, 40, 160]                # pseudo-moves pulling P(direction | cell, previous direction) to the memoryless one
SIGMAS = [0.5, 1.0]                  # spatial smoothing of the memory counts (cells)
GAMMA = 5.0                          # pseudo-moves pulling P(cell move | cell, direction) to M2's
GAM_C = [0.3, 3.0]
NN_SET = [((64, 64), 1e-3), ((128, 128), 1e-3)]
TUNE_K = [7, 30]


def sector(de, dn):
    """Direction class of a daily displacement (km east, km north)."""
    de, dn = np.asarray(de, float), np.asarray(dn, float)
    a = (np.floor(((np.arctan2(dn, de) + np.pi / 8) % (2 * np.pi)) / (np.pi / 4)).astype(int)) % 8
    return np.where(np.hypot(de, dn) < SLOW_KM, 8, a)


def arrival_class(daily):
    """For every drifter-day: the direction class of the move that ARRIVED there (unknown if no previous day)."""
    d = daily.sort_values(["id", "time"])
    g = d.groupby("id")
    ok = (d["time"] - g["time"].shift(1)) == pd.Timedelta(days=1)
    lat0, lon0 = g["lat"].shift(1), g["lon"].shift(1)
    de = (d["lon"] - lon0) * KM_PER_DEG * np.cos(np.deg2rad(lat0))
    dn = (d["lat"] - lat0) * KM_PER_DEG
    a = np.where(ok, sector(de.fillna(0), dn.fillna(0)), UNK)
    return pd.Series(a, index=pd.MultiIndex.from_arrays([d["id"].values, d["time"].values]))


def smooth(T, sig):
    """Smooth a (NCELL, ...) count table over neighbouring cells."""
    if sig <= 0:
        return T
    shp = T.shape
    G = T.reshape((NY, NX) + shp[1:])
    return gaussian_filter(G, sigma=(sig, sig) + (0,) * (len(shp) - 1), mode="constant", cval=0.0, truncate=2.0).reshape(shp)


class Memory:
    """Shared pieces: tables from the training moves and the conversion of q[c, a, a', k] to a sparse matrix."""

    def __init__(self, m, D, sig2, alp2):
        self.m, self.D = m, D
        self.ok = D.k >= 0
        self.a = m["a_prev"].values
        self.a2 = sector(m["de_km"].values, m["dn_km"].values)
        H = smooth(D.histogram(), sig2)                       # M2's displacement distribution per cell
        glob = H.sum(axis=0) / H.sum()
        self.p2 = (H + alp2 * glob) / (H.sum(axis=1, keepdims=True) + alp2)
        B = np.zeros((NCELL, 9, D.K))
        np.add.at(B, (D.cell[self.ok], self.a2[self.ok], D.k[self.ok]), 1.0)
        self.B = B
        A = np.zeros((NCELL, NA, 9))
        np.add.at(A, (D.cell, self.a, self.a2), 1.0)
        self.A = A

    def qB(self, sig):
        B = smooth(self.B, sig)
        return (B + GAMMA * self.p2[:, None, :]) / (B.sum(axis=2, keepdims=True) + GAMMA)

    def pA0(self, sig):
        A0 = smooth(self.A.sum(axis=1), sig) + 1e-3
        return A0 / A0.sum(axis=1, keepdims=True)

    def counts_model(self, sig, beta):
        pA0 = self.pA0(sig)
        A = smooth(self.A[:, :9, :], sig)
        qA = np.empty((NCELL, NA, 9))
        qA[:, :9, :] = (A + beta * pA0[:, None, :]) / (A.sum(axis=2, keepdims=True) + beta)
        qA[:, UNK, :] = pA0
        return qA[:, :, :, None] * self.qB(sig)[:, None, :, :]            # (cell, a, a', k)

    def from_joint(self, pj, sig):
        """q from a model's joint probabilities pj[c, a(0..8), a', k]; unknown previous direction uses the counts."""
        q = np.empty((NCELL, NA, 9, self.D.K))
        q[:, :9] = pj
        q[:, UNK] = self.pA0(sig)[:, :, None] * self.qB(sig)
        return q

    def operator(self, q):
        D = self.D
        c, a, a2, k = np.meshgrid(np.arange(NCELL), np.arange(NA), np.arange(9), np.arange(D.K), indexing="ij")
        tgt = D.tgt[c, k]
        col = np.where(tgt >= NCELL, NCELL * NA + (tgt - NCELL), tgt * NA + a2)
        keep = (tgt >= 0) & (q > 0)
        P = sp.coo_matrix((q[keep], ((c * NA + a)[keep], col[keep])), shape=(NS, NS)).tocsr()
        tot = np.asarray(P.sum(axis=1)).ravel()
        scale = np.where(tot > 0, 1.0 / np.maximum(tot, 1e-300), 0.0)
        P = sp.diags(scale) @ P
        stay = np.flatnonzero(tot == 0)                         # land / unsampled cells and the four exits
        return (P + sp.coo_matrix((np.ones(len(stay)), (stay, stay)), shape=(NS, NS))).tocsr()

    # ---- the two learned versions
    def design(self, lat, lon, a, kind, basis=None):
        onehot = np.eye(9)[a]
        if kind == "nn":
            return np.column_stack([fourier(lat, lon, 3), onehot])
        Bf, Bc = basis
        f, c = Bf(lat, lon), Bc(lat, lon)
        return np.column_stack([f, onehot, (onehot[:, :, None] * c[:, None, :]).reshape(len(f), -1)])

    def fit_joint(self, kind, **kw):
        m, D = self.m, self.D
        rows = self.ok & (self.a < 9)
        y = self.a2[rows] * D.K + D.k[rows]
        basis = None
        if kind == "gam":
            from step8_stronger_gam_nn import MultinomialGAM
            basis = (MultinomialGAM(10, 1).basis, MultinomialGAM(5, 1).basis)
        X = self.design(m["lat"].values[rows], m["lon"].values[rows], self.a[rows], kind, basis)
        if kind == "gam":
            from sklearn.linear_model import LogisticRegression
            models = [LogisticRegression(C=kw["C"], max_iter=200, tol=1e-3).fit(X, y)]
        else:
            from sklearn.neural_network import MLPClassifier
            models = [MLPClassifier(hidden_layer_sizes=kw["hidden"], alpha=kw["l2"], early_stopping=True, validation_fraction=0.15,
                                    n_iter_no_change=10, max_iter=200, learning_rate_init=2e-3, batch_size=256,
                                    random_state=SEED + s).fit(X, y) for s in range(kw.get("seeds", 3))]
        lat, lon = sub_points()
        P_ = lat.shape[1]
        pj = np.zeros((NCELL, 9, 9 * D.K))
        for a in range(9):
            Xp = self.design(lat.ravel(), lon.ravel(), np.full(lat.size, a), kind, basis)
            for mdl in models:
                pr = np.zeros((len(Xp), 9 * D.K))
                pr[:, mdl.classes_] = mdl.predict_proba(Xp)
                pj[:, a, :] += pr.reshape(NCELL, P_, -1).mean(axis=1) / len(models)
        return pj.reshape(NCELL, 9, 9, D.K)


def forecast_memory(P, c, horizons):
    """Cell-level forecast rows (cases x 404) at each horizon for cases with start state (cell, a_start)."""
    s0 = c["cell"].values * NA + c["a_start"].values
    u, inv = np.unique(s0, return_inverse=True)
    X = np.zeros((len(u), NS), dtype=np.float32)
    X[np.arange(len(u)), u] = 1.0
    PT = P.T.tocsr().astype(np.float32)
    out, t = {}, 0
    for k in sorted(horizons):
        for _ in range(k - t):
            X = (PT @ X.T).T
        t = k
        cells = X[:, :NCELL * NA].reshape(len(u), NCELL, NA).sum(axis=2)
        out[k] = np.hstack([cells, X[:, NCELL * NA:]]).astype(float)[inv]
    return out


def add_start_class(c, arr):
    idx = pd.MultiIndex.from_arrays([c["id"].values, c["time"].values])
    c = c.copy()
    c["a_start"] = arr.reindex(idx).fillna(UNK).astype(int).values
    return c


def prep_moves(tr_sub, arr):
    m = moves_for(tr_sub, "drogued")
    m["a_prev"] = arr.reindex(pd.MultiIndex.from_arrays([m["id"].values, m["time"].values])).fillna(UNK).astype(int).values
    return m


def build_chosen(mem, ch):
    out = {"M7 memory, counts": mem.operator(mem.counts_model(ch["sig"], ch["beta"]))}
    out["M7g memory, GAM"] = mem.operator(mem.from_joint(mem.fit_joint("gam", C=ch["gam_C"]), ch["sig"]))
    out["M7n memory, network"] = mem.operator(mem.from_joint(mem.fit_joint("nn", hidden=eval(ch["nn_hidden"]), l2=ch["nn_l2"], seeds=5), ch["sig"]))
    return out


def compare(models, P_m2, cases_by_k, label):
    """Score memory models and M2 on the same cases; paired gain over M2."""
    rows = []
    for k, c in cases_by_k.items():
        boot = drifter_boot(c["id"].values)
        ref = score(forecast(P_m2, c, k), c)
        rows.append({"set": label, "model": "M2 smoothed", "horizon_days": k, **summarise(ref, boot), "gain_vs_M2": 0.0,
                     "gain_p05": 0.0, "gain_p95": 0.0, "verdict": "reference", "known_previous_day_%": 100 * (c["a_start"] != UNK).mean()})
        for n, P in models.items():
            r = score(forecast_memory(P, c, [k])[k], c)
            g = paired(r, ref, boot)
            rows.append({"set": label, "model": n, "horizon_days": k, **summarise(r, boot), "gain_vs_M2": g[0], "gain_p05": g[1],
                         "gain_p95": g[2], "verdict": g[3], "known_previous_day_%": 100 * (c["a_start"] != UNK).mean()})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    daily, tr, ids, SIG, ALP = s4.setup()
    arr = arrival_class(daily)
    ocean = ocean_mask(daily[daily["split"] == "train"])
    m = prep_moves(tr[tr["split"] == "train"], arr)
    D = Displacements(m, ocean)
    mem = Memory(m, D, SIG, ALP)
    P_m2 = m2_smoothed(D, SIG, ALP)
    vst = starts_for(daily, ids["val"], "drogued")
    vcases = {k: add_start_class(cases_at(vst, k), arr) for k in HORIZONS}
    print(f"Training moves {len(m):,}; previous day known for {100 * (m['a_prev'] != UNK).mean():.0f}%. "
          f"Chance today's direction equals yesterday's: {100 * (mem.a2[mem.a < 9] == mem.a[mem.a < 9]).mean():.0f}% (9 classes)")
    vs = lambda P: float(np.mean([score(forecast_memory(P, vcases[k], [k])[k], vcases[k])["logp"].mean() for k in TUNE_K]))
    ref = float(np.mean([score(forecast(P_m2, vcases[k], k), vcases[k])["logp"].mean() for k in TUNE_K]))
    print(f"Validation reference, M2: {ref:.3f} (mean log score at 7 and 30 days)")

    rows, best = [], (-9, None)
    for sig in SIGMAS:
        for beta in BETAS:
            v = vs(mem.operator(mem.counts_model(sig, beta)))
            rows.append({"model": "counts", "sigma": sig, "beta": beta, "val_log_score": v})
            if v > best[0]:
                best = (v, (sig, beta))
    sig, beta = best[1]
    bg = (-9, None)
    for C in GAM_C:
        v = vs(mem.operator(mem.from_joint(mem.fit_joint("gam", C=C), sig)))
        rows.append({"model": "GAM", "C": C, "val_log_score": v})
        if v > bg[0]:
            bg = (v, C)
    bn = (-9, None)
    for h, l2 in NN_SET:
        v = vs(mem.operator(mem.from_joint(mem.fit_joint("nn", hidden=h, l2=l2, seeds=3), sig)))
        rows.append({"model": "network", "hidden": str(h), "l2": l2, "val_log_score": v})
        if v > bn[0]:
            bn = (v, (h, l2))
    tune = pd.DataFrame(rows)
    tune.to_csv(os.path.join(OUT10, "tuning.csv"), index=False)
    pd.set_option("display.width", 200)
    print(tune.round(3).to_string(index=False))
    ch = {"sig": sig, "beta": beta, "gam_C": bg[1], "nn_hidden": str(bn[1][0]), "nn_l2": bn[1][1]}
    pd.DataFrame([ch]).to_csv(os.path.join(OUT10, "chosen_settings.csv"), index=False)
    print("Chosen:", ch)

    vt = compare(build_chosen(mem, ch), P_m2, vcases, "validation / drogued")
    vt.to_csv(os.path.join(OUT10, "memory_validation.csv"), index=False)
    show = lambda t: [print(f"  {r['set']:22s} day {int(r['horizon_days']):3d}  {r['model']:22s} {r['gain_vs_M2']:+.3f} [{r['gain_p05']:+.3f}, {r['gain_p95']:+.3f}]  "
                            f"{r['verdict']:20s} 90% area holds {r['in_90pct_area_%']:.1f}%  centre error {r['median_centre_error_km']:.0f} km")
                      for _, r in t.iterrows()]
    print("\nB. Validation: log score minus M2 (90% interval)")
    show(vt)

    print("\nC. TEST DRIFTERS (a further, disclosed use of the test set)")
    fit_ids = ids["train"] | ids["val"]
    ocean_f = ocean_mask(daily[daily["id"].isin(fit_ids)])
    mf = prep_moves(tr[tr["id"].isin(fit_ids)], arr)
    Df = Displacements(mf, ocean_f)
    memf = Memory(mf, Df, SIG, ALP)
    final = build_chosen(memf, ch)
    P_m2f = m2_smoothed(Df, SIG, ALP)
    out = []
    for split in ("test", "recent_test"):
        st = starts_for(daily, ids[split], "drogued")
        out.append(compare(final, P_m2f, {k: add_start_class(cases_at(st, k), arr) for k in HORIZONS}, f"{split} / drogued"))
    tt = pd.concat(out, ignore_index=True)
    tt.to_csv(os.path.join(OUT10, "memory_test.csv"), index=False)
    show(tt)
    print(f"\nStep 10 done in {(time.time() - t0) / 60:.1f} min. Outputs in {OUT10}")


if __name__ == "__main__":
    main()
