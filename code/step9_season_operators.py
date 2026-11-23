"""
DATA3001 Group S01-G8 - Step 9: seasonal versions of the final operators (for the viewer)
=========================================================================================
Run AFTER steps 2, 4, 5 and 8. Step 8 showed on VALIDATION drifters that a seasonal operator, pulled
towards the all-year one, helps for undrogued (surface) drifters and makes no clear difference for
drogued ones. This file
  A. builds the seasonal version of each final M4 operator (training + validation drifters),
  B. scores seasonal against all-year on the test drifters (a further, disclosed use of the test set;
     the setting SEASON_PULL was fixed on validation in step 8),
  C. saves the seasonal operators with bootstrap copies for the viewer, and how the answer changes
     with the month of release.
Outputs: <BASE>/models/step9.      Run:  python step9_season_operators.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
sys.path.insert(0, _here)
import step2_models_M0_M1_M2 as s2
from step2_models_M0_M1_M2 import NCELL, NSTATE, NX, NY, EXITS, SEED, OUT, ocean_mask, finish, Displacements, m0_displacement
import step4_drogue_season_gam_nn as s4
from step4_drogue_season_gam_nn import HORIZONS, moves_for, starts_for, cases_at, evaluate_models, drogue_weighted, gain_lines, SEASON_OF_MONTH
from scipy.ndimage import gaussian_filter

W_OTHER = 0.25          # as in step 5
SEASON_PULL = 100       # pseudo-moves towards the all-year operator (validation, step 8)
N_BOOT = 10             # bootstrap copies saved for the viewer
OUT9 = os.path.join(OUT, "step9")
os.makedirs(OUT9, exist_ok=True)


def seasonal_m4(D, m, target, sig, alp, w=None):
    """All-year M4 operator for one drogue type and its four seasonal versions."""
    own = m["drogued"].values.astype(bool) == (target == "drogued")
    ww = np.where(own, 1.0, W_OTHER) * (1.0 if w is None else w)
    P_all = drogue_weighted(D, m, target, W_OTHER, sig, alp, w=w)
    P0b = m0_displacement(D, ww)
    ops = []
    for s in range(4):
        H = D.histogram(ww * (m["season"].values == s))
        if sig > 0:
            H = gaussian_filter(H.reshape(NY, NX, D.K), sigma=(sig, sig, 0), mode="constant", cval=0.0, truncate=2.0).reshape(NCELL, D.K)
        C = D.to_counts(H)
        C[:NCELL] += (alp * P0b[:NCELL] + SEASON_PULL * P_all[:NCELL]) * D.ocean[:, None]
        ops.append(finish(C))
    return P_all, ops


def even_curve(ops, month, ocean, days=365):
    """Share of an even start still in R, for a release on the 15th of `month`, stepping through seasons."""
    w = s2.area_weights() * ocean
    v = np.r_[w / w.sum(), np.zeros(4)]
    t0 = np.datetime64(f"2001-{month:02d}-15")
    out = [1.0]
    for t in range(days):
        mth = (t0 + np.timedelta64(t, "D")).astype("datetime64[M]").astype(int) % 12
        v = v @ ops[SEASON_OF_MONTH[mth]]
        out.append(v[:NCELL].sum())
    return np.array(out)


def main():
    t0 = time.time()
    daily, tr, ids, SIG, ALP = s4.setup()
    fit_ids = ids["train"] | ids["val"]
    ocean = ocean_mask(daily[daily["id"].isin(fit_ids)])
    m = moves_for(tr[tr["id"].isin(fit_ids)], "all")
    D = Displacements(m, ocean)
    fitted = {d: seasonal_m4(D, m, d, SIG, ALP) for d in ("drogued", "undrogued")}

    print("B. Test drifters: seasonal minus all-year M4 (log score, 90% interval)")
    out = []
    for split in ("test", "recent_test"):
        for d in ("drogued", "undrogued"):
            st = starts_for(daily, ids[split], d)
            cs = {k: cases_at(st, k) for k in HORIZONS}
            out.append(evaluate_models({"M4 all-year": fitted[d][0], "M4 seasonal": fitted[d][1]}, cs, "M4 all-year", label=f"{split} / {d}"))
    tt = pd.concat(out, ignore_index=True)
    tt.to_csv(os.path.join(OUT9, "season_test.csv"), index=False)
    gain_lines(tt, "gain_vs_M4")

    print("\nC. How the answer changes with the month of release (even start, days until half has left R)")
    ids_f = np.array(sorted(fit_ids))
    row_id = pd.Series(np.arange(len(ids_f)), index=ids_f).reindex(m["id"].values).values
    rows = []
    for d in ("drogued", "undrogued"):
        rng = np.random.default_rng(SEED)
        boots = []
        for b in range(N_BOOT):
            mult = np.bincount(rng.integers(0, len(ids_f), len(ids_f)), minlength=len(ids_f))
            boots.append(seasonal_m4(D, m, d, SIG, ALP, w=mult[row_id].astype(float))[1])
        for month, lab in ((1, "January"), (4, "April"), (7, "July"), (10, "October")):
            c = even_curve(fitted[d][1], month, ocean)
            hb = [int(np.argmax(even_curve(bo, month, ocean) <= 0.5)) for bo in boots]
            rows.append({"operator": d, "release_month": lab, "days_until_half_left": int(np.argmax(c <= 0.5)),
                         "boot_min": min(hb), "boot_max": max(hb), "still_in_R_day30": c[30], "still_in_R_day90": c[90]})
        np.savez_compressed(os.path.join(OUT9, f"operator_seasonal_{d}.npz"),
                            P_season=np.array(fitted[d][1]), P_season_boot=np.array(boots).astype(np.float32),
                            seasons=np.array(["DJF", "MAM", "JJA", "SON"]),
                            config=json.dumps({"model": "M4 seasonal", "target": d, "season_pull": SEASON_PULL, "other_type_weight": W_OTHER}))
    rows = pd.DataFrame(rows)
    rows.to_csv(os.path.join(OUT9, "season_release_month.csv"), index=False)
    print(rows.round(3).to_string(index=False))
    print(f"\nStep 9 done in {(time.time() - t0) / 60:.1f} min. Outputs in {OUT9}")


if __name__ == "__main__":
    main()
