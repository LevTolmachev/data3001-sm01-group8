"""
DATA3001 Group S01-G8 - Step 5: the final held-out test (run once) and the final operators
===========================================================================================

Run AFTER step 2 and step 4. Every modelling choice is fixed before this file runs:
  M2   sigma, alpha            from step 2 (m2_tuning.csv)
  M4   weight of the other drogue type's moves, seasonal pull    (constants below, chosen in step 4)
  M5   GAM knots and ridge,  M6 network size and L2              (step4/chosen_settings.csv)

  A. Refit every model on training + validation drifters.
  B. Score them ONCE on the test drifters and on the most recent years (recent_test):
     position at 1, 7, 30 and 90 days with 90% intervals from resampling whole drifters, the paired
     gain of every model over M2, and the fate check (time in R against Kaplan-Meier, exit edge).
  C. Build the final operators (M4, one for drogued and one for undrogued material) with bootstrap
     copies, and everything RQ1 reports from them. These files feed the viewer and the figures.

Outputs go to <BASE>/models/step5.   Run:  python step5_final_test_and_operator.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
sys.path.insert(0, _here)
import step2_models_M0_M1_M2 as s2
from step2_models_M0_M1_M2 import (NCELL, NSTATE, EXITS, CELL_LAT, CELL_LON, SEED, MIN_DRIFTERS, OUT,
                                   ocean_mask, Displacements, m0_displacement, m1_empirical, m2_smoothed,
                                   haversine_km, propagate_cells)
import step4_drogue_season_gam_nn as s4
from step4_drogue_season_gam_nn import (HORIZONS, moves_for, starts_for, cases_at, forecast, score,
                                        drifter_boot, summarise, paired, evaluate_models, drogue_weighted,
                                        seasonal_operators, GaussianGAM, DisplacementNet, gain_lines)

# ----------------------------------------------------------------------------- CONFIG (fixed in step 4)
W_OTHER = 0.25            # weight of the other drogue type's moves in M4
SEASON_PULL = 100         # pull of each season towards the all-year operator
N_BOOT_OPERATOR = 200     # bootstrap refits of the final operator
N_SAVE = 30               # bootstrap operators saved for the viewer
RELEASES = {"Confluence 38°S 54°W": (-38.0, -54.0),
            "Brazil Current 33°S 50°W": (-33.0, -50.0),
            "Malvinas Current 44°S 58°W": (-44.0, -58.0)}
SHOW_DAYS = [7, 30, 90, 365]
YEAR = 365

OUT5 = os.path.join(OUT, "step5")
os.makedirs(OUT5, exist_ok=True)
DONE_FLAG = os.path.join(OUT5, "FINAL_TEST_HAS_BEEN_RUN.txt")


def main():
    t_start = time.time()
    daily, tr, ids, SIG, ALP = s4.setup()
    ch = pd.read_csv(os.path.join(OUT, "step4", "chosen_settings.csv")).iloc[0]
    print(f"Fixed settings: M2 sigma {SIG}, alpha {ALP}; M4 other-type weight {W_OTHER}, seasonal pull {SEASON_PULL}; "
          f"GAM {int(ch['gam_n_knots'])} knots, ridge {ch['gam_ridge']}; network {ch['nn_hidden']}, L2 {ch['nn_l2']}")
    fit_ids = ids["train"] | ids["val"]
    trf = tr[tr["id"].isin(fit_ids)]
    ocean = ocean_mask(daily[daily["id"].isin(fit_ids)])
    mset = {d: moves_for(trf, d) for d in ("drogued", "undrogued", "all")}
    Dset = {d: Displacements(mset[d], ocean) for d in mset}
    pd.set_option("display.width", 240)

    # ------------------------------------------------------------------ A. refit
    print("\nA. Refitting every model on training + validation drifters ...")
    m, D = mset["drogued"], Dset["drogued"]
    P0b = m0_displacement(D)
    P_m2 = m2_smoothed(D, SIG, ALP)
    P_m4 = {d: drogue_weighted(Dset["all"], mset["all"], d, W_OTHER, SIG, ALP) for d in ("drogued", "undrogued")}
    models = {"M0b displacement": P0b,
              "M1 empirical": m1_empirical(m),
              "M2 smoothed": P_m2,
              "M4 + other drogue type": P_m4["drogued"],
              "M4 + season": seasonal_operators(D, m, SIG, ALP, SEASON_PULL, P_m2, P0b),
              "M5 GAM": GaussianGAM(int(ch["gam_n_knots"]), float(ch["gam_ridge"])).fit(m).operator(D),
              "M6 neural net": DisplacementNet(eval(ch["nn_hidden"]), float(ch["nn_l2"])).fit(m, D).operator(D)}
    models_u = {"M2 undrogued only": m2_smoothed(Dset["undrogued"], SIG, ALP),
                "M2 drogued only": P_m2,
                "M4 + other drogue type": P_m4["undrogued"]}
    print(f"  drogued moves {len(m):,} from {m['id'].nunique()} drifters; undrogued moves "
          f"{len(mset['undrogued']):,} from {mset['undrogued']['id'].nunique()} drifters")

    # ------------------------------------------------------------------ B. final test (once)
    print("\n" + "=" * 78 + "\nB. FINAL HELD-OUT TEST\n" + "=" * 78)
    if os.path.exists(DONE_FLAG):
        print(f"NOTE: {DONE_FLAG} exists, so the test set has been scored before. Re-running reproduces the\n"
              "same numbers (nothing is tuned here), but no modelling choice may be changed after this point.")
    show = ["set", "model", "horizon_days", "cases", "drifters", "log_score", "log_score_p05", "log_score_p95",
            "in_50pct_area_%", "in_90pct_area_%", "in_90pct_area_%_p05", "in_90pct_area_%_p95",
            "median_centre_error_km", "median_persistence_error_km"]
    pos, fate_rows, curves = [], [], {}
    for split in ("test", "recent_test"):
        for d, mods, ref in (("drogued", models, "M2 smoothed"), ("undrogued", models_u, "M2 undrogued only")):
            st = starts_for(daily, ids[split], d)
            if len(st[0]) == 0:
                print(f"  {split} / {d}: no starts, skipped")
                continue
            cases = {k: cases_at(st, k) for k in HORIZONS}
            t = evaluate_models(mods, cases, ref, label=f"{split} / {d}")
            pers = {}
            for k, c in cases.items():                      # persistence: it stays where it started
                inn = c["inn"].values
                pers[k] = float(np.median(haversine_km(c["lat"].values[inn], c["lon"].values[inn],
                                                       c["lat_end"].values[inn], c["lon_end"].values[inn])))
            t["median_persistence_error_km"] = t["horizon_days"].map(pers)
            pos.append(t)
            # fate: time in R and exit edge
            old = s2.DROGUE_SET
            s2.DROGUE_SET = d
            fc = s2.fate_cases(daily, ids[split])
            s2.DROGUE_SET = old
            S = s2.kaplan_meier(fc, YEAR)
            curves[(split, d, "Observed")] = S
            for name, P in mods.items():
                if isinstance(P, list):
                    continue
                row, cur = s2.score_fate(P, fc, name, S)
                fate_rows.append({"set": f"{split} / {d}", **row})
                curves[(split, d, name)] = cur
    pos = pd.concat(pos, ignore_index=True)
    fate = pd.DataFrame(fate_rows)
    pos.to_csv(os.path.join(OUT5, "final_test_position.csv"), index=False)
    fate.to_csv(os.path.join(OUT5, "final_test_fate.csv"), index=False)
    pd.DataFrame({f"{a}|{b}|{c}": v for (a, b, c), v in curves.items()}).to_csv(
        os.path.join(OUT5, "final_test_survival_curves.csv"), index_label="day")
    print(pos[show].round(2).to_string(index=False))
    print("\nIs the extra complexity worth it? Paired gain in log score over the reference (90% interval):")
    gain_lines(pos, [c for c in pos.columns if c.startswith("gain_vs_")][0])
    fshow = ["set", "model", "fate_starts", "seen_to_exit", "observed_up_to_day", "still_in_R_abs_error",
             "pred_still_in_R_day_30", "obs_still_in_R_day_30", "exit_edge_brier", "exit_edge_accuracy"] + \
            [f"pred_share_left_{e}" for e in EXITS] + [f"obs_share_left_{e}" for e in EXITS]
    print("\nFate (time in R, exit edge):")
    print(fate[fshow].round(3).to_string(index=False))
    with open(DONE_FLAG, "w") as f:
        f.write("The final test was scored on " + time.strftime("%Y-%m-%d %H:%M") + ". Do not tune anything after this.\n")

    # ------------------------------------------------------------------ C. final operators
    print("\n" + "=" * 78 + "\nC. FINAL OPERATORS (M4, training + validation drifters) WITH BOOTSTRAP\n" + "=" * 78)
    m_all, D_all = mset["all"], Dset["all"]
    per_cell = m_all.groupby("from_cell").agg(moves=("id", "size"), drifters=("id", "nunique"))
    n_drift = per_cell["drifters"].reindex(range(NCELL)).fillna(0).values
    n_moves = per_cell["moves"].reindex(range(NCELL)).fillna(0).values
    masked = (n_drift < MIN_DRIFTERS) | ~ocean
    print(f"Fitted on {m_all['id'].nunique()} drifters, {len(m_all):,} daily moves. "
          f"Supported cells: {int((~masked).sum())} of {NCELL} ({int(ocean.sum())} ever visited).")
    rel_cells = {}
    for name, (la, lo) in RELEASES.items():
        c = int(s2.cell_of(np.array([la]), np.array([lo]))[0])
        if masked[c]:
            okc = np.flatnonzero(~masked)
            c = int(okc[np.argmin(haversine_km(la, lo, CELL_LAT[okc], CELL_LON[okc]))])
            print(f"  note: {name} is in a masked cell; using the nearest supported cell")
        rel_cells[name] = c
    w_even = s2.area_weights() * ocean
    u_even = np.r_[w_even / w_even.sum(), np.zeros(4)]

    def outputs(P):
        o = {}
        still, M = propagate_cells(P, YEAR)
        o["still"] = still                                   # (cell, day)
        o["exit"] = M[:, NCELL:]                             # left via N,S,E,W by day 365
        below = still <= 0.5
        o["median_days"] = np.where(below.any(axis=1), below.argmax(axis=1), np.nan)
        o["even_curve"] = u_even[:NCELL] @ still
        v, rho, ev = u_even.copy(), {}, {}
        for t in range(1, YEAR + 1):
            v = v @ P
            if t in (30, 90):
                inside = v[:NCELL] / max(v[:NCELL].sum(), 1e-12)
                rho[t] = np.where(ocean, inside / np.where(u_even[:NCELL] > 0, u_even[:NCELL], np.nan), np.nan)
            if t in SHOW_DAYS:
                ev[t] = v[NCELL:].copy()
        o["rho"], o["even_exit"] = rho, ev
        snaps = {}
        for name, c in rel_cells.items():
            p = np.zeros(NSTATE); p[c] = 1.0
            for t in range(1, YEAR + 1):
                p = p @ P
                if t in SHOW_DAYS:
                    snaps[(name, t)] = p.copy()
        o["snaps"] = snaps
        return o

    ids_f = np.array(sorted(fit_ids))
    row_id = pd.Series(np.arange(len(ids_f)), index=ids_f).reindex(m_all["id"].values).values
    summary, cell_tabs = [], []
    for d in ("drogued", "undrogued"):
        P = P_m4[d]
        base = outputs(P)
        rng = np.random.default_rng(SEED)
        boots, saved = [], []
        for b in range(N_BOOT_OPERATOR):
            mult = np.bincount(rng.integers(0, len(ids_f), len(ids_f)), minlength=len(ids_f))
            Pb = drogue_weighted(D_all, m_all, d, W_OTHER, SIG, ALP, w=mult[row_id].astype(float))
            boots.append(outputs(Pb))
            if b < N_SAVE:
                saved.append(Pb.astype(np.float32))
        q = lambda f: np.nanpercentile(np.array([f(o) for o in boots]), [5, 95], axis=0)
        # summary numbers with 90% intervals
        for t in SHOW_DAYS:
            lo, hi = q(lambda o: o["even_curve"][t])
            summary.append({"operator": d, "start": "even over R", "day": t, "quantity": "still in R",
                            "value": base["even_curve"][t], "p05": lo, "p95": hi})
            for j, e in enumerate(EXITS):
                lo, hi = q(lambda o: o["even_exit"][t][j])
                summary.append({"operator": d, "start": "even over R", "day": t, "quantity": f"left via {e}",
                                "value": base["even_exit"][t][j], "p05": lo, "p95": hi})
            for name in rel_cells:
                lo, hi = q(lambda o: o["snaps"][(name, t)][:NCELL].sum())
                summary.append({"operator": d, "start": name, "day": t, "quantity": "still in R",
                                "value": base["snaps"][(name, t)][:NCELL].sum(), "p05": lo, "p95": hi})
                for j, e in enumerate(EXITS):
                    lo, hi = q(lambda o: o["snaps"][(name, t)][NCELL + j])
                    summary.append({"operator": d, "start": name, "day": t, "quantity": f"left via {e}",
                                    "value": base["snaps"][(name, t)][NCELL + j], "p05": lo, "p95": hi})
        half = lambda o: float(np.argmax(o["even_curve"] <= 0.5))
        lo, hi = q(half)
        summary.append({"operator": d, "start": "even over R", "day": np.nan, "quantity": "days until half has left",
                        "value": half(base), "p05": lo, "p95": hi})
        ct = pd.DataFrame({"operator": d, "cell": np.arange(NCELL), "lat_centre": CELL_LAT, "lon_centre": CELL_LON,
                           "ocean": ocean, "masked": masked, "moves": n_moves.astype(int),
                           "distinct_drifters": n_drift.astype(int), "median_days_in_R": base["median_days"],
                           "p_left_within_year": base["exit"].sum(axis=1),
                           **{f"p_left_{e}_within_year": base["exit"][:, j] for j, e in enumerate(EXITS)}})
        lo, hi = q(lambda o: o["median_days"])
        ct["median_days_p05"], ct["median_days_p95"] = lo, hi
        for t in (30, 90):
            lo, hi = q(lambda o: o["rho"][t])
            ct[f"rho_day{t}"], ct[f"rho_day{t}_p05"], ct[f"rho_day{t}_p95"] = base["rho"][t], lo, hi
        cell_tabs.append(ct)
        okc = ~masked
        for t in (30, 90):
            g = int(((ct[f"rho_day{t}_p05"] > 1) & okc).sum()); l = int(((ct[f"rho_day{t}_p95"] < 1) & okc).sum())
            print(f"  {d:9s} relative density day {t}: {g} supported cells confidently gather, {l} confidently lose, "
                  f"{int(okc.sum()) - g - l} not distinguishable from 1")
        dom = np.array(EXITS)[base["exit"].argmax(axis=1)]
        print(f"  {d:9s} most likely exit edge (supported cells): " + ", ".join(f"{e} {int((dom[okc] == e).sum())}" for e in EXITS))
        curves_b = np.array([o["even_curve"] for o in boots])
        rel_b = {name: np.array([[o["snaps"][(name, t)][:NCELL].sum() for t in SHOW_DAYS] for o in boots]) for name in rel_cells}
        np.savez_compressed(os.path.join(OUT5, f"operator_final_{d}.npz"),
                            P=P, P_boot=np.array(saved), cell_lat=CELL_LAT, cell_lon=CELL_LON, ocean=ocean, masked=masked,
                            distinct_drifters=n_drift, even_curve=base["even_curve"],
                            even_curve_p05=np.percentile(curves_b, 5, axis=0), even_curve_p95=np.percentile(curves_b, 95, axis=0),
                            release_names=np.array(list(rel_cells)), release_cells=np.array(list(rel_cells.values())),
                            release_snaps=np.array([[base["snaps"][(n, t)] for t in SHOW_DAYS] for n in rel_cells]),
                            show_days=np.array(SHOW_DAYS),
                            states=np.array([f"cell_{i}" for i in range(NCELL)] + [f"exit_{e}" for e in EXITS]),
                            config=json.dumps({"region": [s2.LAT_MIN, s2.LAT_MAX, s2.LON_MIN, s2.LON_MAX],
                                               "cell_deg": s2.CELL_DEG, "dt_days": 1, "model": "M4", "target": d,
                                               "other_type_weight": W_OTHER, "sigma": SIG, "alpha": ALP,
                                               "fitted_on": "train+val", "n_boot_saved": len(saved)}))
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(OUT5, "rq1_summary.csv"), index=False)
    pd.concat(cell_tabs, ignore_index=True).to_csv(os.path.join(OUT5, "rq1_cell_fate.csv"), index=False)
    print("\nRQ1 headline numbers (90% bootstrap interval):")
    s_ = summary[(summary["start"] == "even over R")]
    for _, r in s_.iterrows():
        if r["quantity"] == "days until half has left":
            print(f"  {r['operator']:9s} days until half of an even start has left R: {r['value']:.0f} [{r['p05']:.0f}-{r['p95']:.0f}]")
        elif r["quantity"] in ("still in R", "left via E"):
            print(f"  {r['operator']:9s} day {int(r['day']):3d} {r['quantity']:11s} {r['value']:.0%} [{r['p05']:.0%}-{r['p95']:.0%}]")
    print("\nRelease points, still in R (90% interval):")
    for _, r in summary[(summary["start"] != "even over R") & (summary["quantity"] == "still in R") & (summary["day"] >= 90)].iterrows():
        print(f"  {r['operator']:9s} {r['start']:28s} day {int(r['day']):3d}: {r['value']:.0%} [{r['p05']:.0%}-{r['p95']:.0%}]")
    print(f"\nStep 5 done in {(time.time() - t_start) / 60:.1f} min. Outputs in {OUT5}")


if __name__ == "__main__":
    main()
