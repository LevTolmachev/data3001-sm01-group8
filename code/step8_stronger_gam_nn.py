"""
DATA3001 Group S01-G8 - Step 8: a second, stronger attempt at the GAM and the neural network (RQ5),
and season for undrogued drifters (RQ4)
==========================================================================================
Run AFTER steps 2, 4 and 5.

Step 4's GAM assumed a Gaussian daily displacement and its network was small with little tuning.
This file gives both a fairer chance, and adds two hybrids:

  M5b  Multinomial GAM   displacement class ~ multinomial, linear predictor = penalised tensor-product
                         cubic spline of (longitude, latitude). No Gaussian assumption.
  M6b  Wider network     Fourier features of position (lets a network fit sharp fronts), larger and
                         deeper layers, a wider search, an average of 5 networks.
  M2+GAM, M2+NN          The smoothed matrix M2, but shrunk towards the GAM / network instead of
                         towards the single displacement pattern M0b ("counts where there are data,
                         the model where there are not").
  Season for undrogued   The seasonal operator of step 4, fitted and scored on undrogued drifters,
                         because wind acts on surface material and wind is seasonal.

Everything is CHOSEN on the validation drifters only. Section C then scores the chosen models on the
test drifters. NOTE FOR THE REPORT: this is the second time the test drifters are used. The models
of step 5 were not changed; the models here were added afterwards, selected on validation alone.

Outputs: <BASE>/models/step8.      Run:  python step8_stronger_gam_nn.py
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
sys.path.insert(0, _here)
import step2_models_M0_M1_M2 as s2
from step2_models_M0_M1_M2 import (NCELL, NX, NY, SEED, OUT, ocean_mask, finish, Displacements,
                                   m0_displacement, m1_empirical, m2_smoothed)
import step4_drogue_season_gam_nn as s4
from step4_drogue_season_gam_nn import (HORIZONS, moves_for, starts_for, cases_at, forecast, score,
                                        evaluate_models, seasonal_operators, sub_points, gain_lines,
                                        GaussianGAM, DisplacementNet, nn_features)
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
OUT8 = os.path.join(OUT, "step8")
os.makedirs(OUT8, exist_ok=True)

MGAM_KNOTS = [6, 10, 14]
MGAM_C = [0.03, 0.3, 3.0]             # inverse penalty (smaller = smoother)
NN2_HIDDEN = [(64, 64), (128, 128), (64, 64, 64)]
NN2_L2 = [1e-4, 1e-3, 1e-2]
NN2_FREQS = [0, 3]                    # number of Fourier frequencies (0 = raw position only)
NN2_SEEDS = 5
HYB_ALPHA = [2, 10, 50, 200]          # pseudo-moves given to the model prior
HYB_SIGMA = [0.0, 0.5]
SEASON_LAMS = [0, 20, 100, 500]
TUNE_K = [7, 30]


class MultinomialGAM:
    """Multinomial GAM: class = daily cell displacement; smooth = tensor-product spline of position."""

    def __init__(self, n_knots, C):
        from sklearn.preprocessing import SplineTransformer
        self.C = C
        kn = lambda lo, hi: np.linspace(lo, hi, n_knots)[:, None]
        self.sx = SplineTransformer(degree=3, knots=kn(s2.LON_MIN, s2.LON_MAX), extrapolation="constant").fit(np.zeros((2, 1)))
        self.sy = SplineTransformer(degree=3, knots=kn(s2.LAT_MIN, s2.LAT_MAX), extrapolation="constant").fit(np.zeros((2, 1)))

    def basis(self, lat, lon):
        bx = self.sx.transform(np.asarray(lon, float).reshape(-1, 1))
        by = self.sy.transform(np.asarray(lat, float).reshape(-1, 1))
        return (bx[:, :, None] * by[:, None, :]).reshape(len(bx), -1)

    def fit(self, m, D):
        from sklearn.linear_model import LogisticRegression
        ok = D.k >= 0
        self.K = D.K
        self.clf = LogisticRegression(C=self.C, max_iter=300, tol=1e-4).fit(
            self.basis(m["lat"].values[ok], m["lon"].values[ok]), D.k[ok])
        return self

    def operator(self, D):
        lat, lon = sub_points()
        pr = np.zeros((lat.size, self.K))
        pr[:, self.clf.classes_] = self.clf.predict_proba(self.basis(lat.ravel(), lon.ravel()))
        return finish(D.to_counts(pr.reshape(NCELL, lat.shape[1], self.K).mean(axis=1)))


def fourier(lat, lon, nf):
    X = nn_features(lat, lon)
    if nf == 0:
        return X
    out = [X]
    for f in range(1, nf + 1):
        for fn in (np.sin, np.cos):
            out.append(fn(np.pi * f * X))
    return np.column_stack(out)


class FourierNet:
    def __init__(self, hidden, l2, nf, seeds=NN2_SEEDS):
        self.hidden, self.l2, self.nf, self.seeds = hidden, l2, nf, seeds

    def fit(self, m, D):
        from sklearn.neural_network import MLPClassifier
        ok = D.k >= 0
        X, y = fourier(m["lat"].values[ok], m["lon"].values[ok], self.nf), D.k[ok]
        self.K = D.K
        self.nets = [MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.l2, early_stopping=True,
                                   validation_fraction=0.15, n_iter_no_change=10, max_iter=200,
                                   learning_rate_init=2e-3, batch_size=256, random_state=SEED + s).fit(X, y)
                     for s in range(self.seeds)]
        return self

    def operator(self, D):
        lat, lon = sub_points()
        X = fourier(lat.ravel(), lon.ravel(), self.nf)
        pr = np.zeros((len(X), self.K))
        for net in self.nets:
            pr[:, net.classes_] += net.predict_proba(X) / len(self.nets)
        return finish(D.to_counts(pr.reshape(NCELL, lat.shape[1], self.K).mean(axis=1)))


def hybrid(D, sigma, alpha, P_prior):
    """M2, but shrunk towards a model's matrix instead of towards M0b."""
    H = D.histogram()
    if sigma > 0:
        H = gaussian_filter(H.reshape(NY, NX, D.K), sigma=(sigma, sigma, 0), mode="constant", cval=0.0,
                            truncate=2.0).reshape(NCELL, D.K)
    C = D.to_counts(H)
    C[:NCELL] += alpha * P_prior[:NCELL] * D.ocean[:, None]
    return finish(C)


def val_score(P, vcases):
    return float(np.mean([score(forecast(P, vcases[k], k), vcases[k])["logp"].mean() for k in TUNE_K]))


def fit_all(m, D, ch):
    """Fit the chosen step-8 models on a set of moves (used for validation and for the test refit)."""
    gam = MultinomialGAM(int(ch["mgam_knots"]), float(ch["mgam_C"])).fit(m, D).operator(D)
    net = FourierNet(eval(ch["nn_hidden"]), float(ch["nn_l2"]), int(ch["nn_freqs"])).fit(m, D).operator(D)
    return {"M5b multinomial GAM": gam, "M6b wider network": net,
            "M2 + GAM prior": hybrid(D, float(ch["hg_sigma"]), float(ch["hg_alpha"]), gam),
            "M2 + network prior": hybrid(D, float(ch["hn_sigma"]), float(ch["hn_alpha"]), net)}


def main():
    t0 = time.time()
    daily, tr, ids, SIG, ALP = s4.setup()
    old = pd.read_csv(os.path.join(OUT, "step4", "chosen_settings.csv")).iloc[0]
    trn = tr[tr["split"] == "train"]
    ocean = ocean_mask(daily[daily["split"] == "train"])
    m, mu = moves_for(trn, "drogued"), moves_for(trn, "undrogued")
    D, Du = Displacements(m, ocean), Displacements(mu, ocean)
    vst = {d: starts_for(daily, ids["val"], d) for d in ("drogued", "undrogued")}
    vcases = {k: cases_at(vst["drogued"], k) for k in HORIZONS}
    P_m2 = m2_smoothed(D, SIG, ALP)
    pd.set_option("display.width", 220)
    print(f"Reference on validation: M2 {val_score(P_m2, vcases):.3f} (mean log score at 7 and 30 days)")

    # ------------------------------------------------------------------ A. tuning on validation
    print("\nA1. Multinomial GAM")
    rows, best = [], (-9, None, None)
    for nk in MGAM_KNOTS:
        for C in MGAM_C:
            P = MultinomialGAM(nk, C).fit(m, D).operator(D)
            v = val_score(P, vcases)
            rows.append({"n_knots": nk, "C": C, "val_log_score": v})
            if v > best[0]:
                best = (v, (nk, C), P)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    pd.DataFrame(rows).to_csv(os.path.join(OUT8, "mgam_tuning.csv"), index=False)
    (g_nk, g_C), P_gam = best[1], best[2]

    print("\nA2. Wider network (3 networks per setting while searching, 5 for the chosen one)")
    rows, best = [], (-9, None)
    for nf in NN2_FREQS:
        for h in NN2_HIDDEN:
            for l2 in NN2_L2:
                v = val_score(FourierNet(h, l2, nf, seeds=3).fit(m, D).operator(D), vcases)
                rows.append({"fourier_freqs": nf, "hidden": str(h), "l2": l2, "val_log_score": v})
                if v > best[0]:
                    best = (v, (nf, h, l2))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    pd.DataFrame(rows).to_csv(os.path.join(OUT8, "nn_tuning.csv"), index=False)
    n_nf, n_h, n_l2 = best[1]
    P_net = FourierNet(n_h, n_l2, n_nf).fit(m, D).operator(D)

    print("\nA3. Hybrids: M2 shrunk towards the GAM / the network")
    rows, hb = [], {}
    for name, Pp in (("GAM", P_gam), ("network", P_net)):
        best = (-9, None)
        for sg in HYB_SIGMA:
            for al in HYB_ALPHA:
                v = val_score(hybrid(D, sg, al, Pp), vcases)
                rows.append({"prior": name, "sigma": sg, "alpha": al, "val_log_score": v})
                if v > best[0]:
                    best = (v, (sg, al))
        hb[name] = best[1]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    pd.DataFrame(rows).to_csv(os.path.join(OUT8, "hybrid_tuning.csv"), index=False)
    ch = {"mgam_knots": g_nk, "mgam_C": g_C, "nn_hidden": str(n_h), "nn_l2": n_l2, "nn_freqs": n_nf,
          "hg_sigma": hb["GAM"][0], "hg_alpha": hb["GAM"][1], "hn_sigma": hb["network"][0], "hn_alpha": hb["network"][1]}
    pd.DataFrame([ch]).to_csv(os.path.join(OUT8, "chosen_settings.csv"), index=False)
    print("Chosen:", ch)

    # ------------------------------------------------------------------ B. validation comparison
    P_gauss = GaussianGAM(int(old["gam_n_knots"]), float(old["gam_ridge"])).fit(m).operator(D)
    P_nn1 = DisplacementNet(eval(old["nn_hidden"]), float(old["nn_l2"])).fit(m, D).operator(D)
    models = {"M1 empirical": m1_empirical(m), "M2 smoothed": P_m2, "M5 Gaussian GAM": P_gauss, "M6 small network": P_nn1,
              **fit_all(m, D, ch)}
    vt = evaluate_models(models, vcases, "M2 smoothed", label="validation / drogued")
    vt.to_csv(os.path.join(OUT8, "rq5_validation.csv"), index=False)
    print("\nB. Validation: gain in log score over M2 (positive = better, 90% interval)")
    gain_lines(vt, "gain_vs_M2")

    print("\nB2. Season for UNDROGUED drifters (validation)")
    P0b_u = m0_displacement(Du)
    P_m2_u = m2_smoothed(Du, SIG, ALP)
    ucases = {k: cases_at(vst["undrogued"], k) for k in HORIZONS}
    sm = {"M2 all-year": P_m2_u}
    for lam in SEASON_LAMS:
        sm[f"M4 seasonal, pull {lam}"] = seasonal_operators(Du, mu, SIG, ALP, lam, P_m2_u, P0b_u)
    st = evaluate_models(sm, ucases, "M2 all-year", label="validation / undrogued")
    st.to_csv(os.path.join(OUT8, "rq4_season_undrogued.csv"), index=False)
    gain_lines(st, "gain_vs_M2")

    # ------------------------------------------------------------------ C. test (second look, disclosed)
    print("\n" + "=" * 78 + "\nC. TEST DRIFTERS, step-8 models only (second use of the test set: disclose)\n" + "=" * 78)
    fit_ids = ids["train"] | ids["val"]
    trf = tr[tr["id"].isin(fit_ids)]
    ocean_f = ocean_mask(daily[daily["id"].isin(fit_ids)])
    mf = moves_for(trf, "drogued")
    Df = Displacements(mf, ocean_f)
    tm = {"M2 smoothed": m2_smoothed(Df, SIG, ALP), **fit_all(mf, Df, ch)}
    out = []
    for split in ("test", "recent_test"):
        stt = starts_for(daily, ids[split], "drogued")
        cs = {k: cases_at(stt, k) for k in HORIZONS}
        out.append(evaluate_models(tm, cs, "M2 smoothed", label=f"{split} / drogued"))
    tt = pd.concat(out, ignore_index=True)
    tt.to_csv(os.path.join(OUT8, "rq5_test.csv"), index=False)
    gain_lines(tt, "gain_vs_M2")
    print(f"\nStep 8 done in {(time.time() - t0) / 60:.1f} min. Outputs in {OUT8}")


if __name__ == "__main__":
    main()
