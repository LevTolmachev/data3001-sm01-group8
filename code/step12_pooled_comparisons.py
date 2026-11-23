"""
DATA3001 Group S01-G8 - Step 12: grid and model-complexity comparisons on ALL drifters (pooled),
then the depth question (pooled vs split by drogue status)
==================================================================================================
Run AFTER steps 2, 4, 5, 8 and 10 (re-uses their functions and the frozen splits).

The research questions are answered in this order:
  1. Grid size      which cell size / shape / time step do the data support?   (all drifters, M1)
  2. Complexity     M0, M1, M2, GAM, neural network, memory models             (all drifters)
  3. Depth          does splitting the pooled M2 by drogue status help?        (M4 = the split model)
Steps 3, 8 and 10 did 1 and 2 on drogued drifters only. This file redoes them on all drifters, so drogue
status is not used until question 3, as in the research questions.

Every setting is chosen on the validation drifters (all of them), then scored on the test drifters and the
most recent years. NOTE FOR THE REPORT: the test drifters have been scored before (steps 5, 8, 9, 10); this is
a further, disclosed use. Outputs: <BASE>/models/step12.      Run:  python step12_pooled_comparisons.py
"""
import os, sys, ast, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
sys.path.insert(0, _here)
import step2_models_M0_M1_M2 as s2
from step2_models_M0_M1_M2 import (NCELL, OUT, SEED, MIN_DRIFTERS, M2_SIGMAS, M2_ALPHAS, ocean_mask, Displacements,
                                   m0_displacement, m1_empirical, m2_smoothed)
import step4_drogue_season_gam_nn as s4
from step4_drogue_season_gam_nn import (HORIZONS, moves_for, starts_for, cases_at, forecast, score, drifter_boot,
                                        summarise, evaluate_models, drogue_weighted, gain_lines)
from step8_stronger_gam_nn import MultinomialGAM, FourierNet, MGAM_KNOTS, MGAM_C
import step10_memory_models as s10

OUT12 = os.path.join(OUT, "step12")
os.makedirs(OUT12, exist_ok=True)
TUNE_K = [7, 30]
NN_SET = [(3, (64, 64), 1e-3), (3, (64, 64), 1e-2), (3, (128, 128), 1e-3), (3, (64, 64, 64), 1e-3), (0, (128, 128), 1e-3)]
W_OTHER = 0.25

# --------------------------------------------------------------------------- grid machinery from step 3
# (step 3 runs its analysis on import, so its class and function definitions are loaded on their own)
_src = open(os.path.join(_here, "step3_core_results.py")).read()
_keep = {"_cell_area_samples", "SquareGrid", "HexGrid", "encode", "fit_operator", "mpow", "evaluate", "EARTH_KM", "D_FLOOR"}
_tree = ast.parse(_src)
_nodes = [n for n in _tree.body if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _keep)
          or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in _keep for t in n.targets))]
G3 = {k: getattr(s2, k) for k in dir(s2) if not k.startswith("__")}
G3.update(np=np, pd=pd)
exec(compile(ast.Module(body=_nodes, type_ignores=[]), "step3_defs", "exec"), G3)
SquareGrid, HexGrid, encode, fit_operator, mpow, evaluate = (G3[k] for k in ("SquareGrid", "HexGrid", "encode", "fit_operator", "mpow", "evaluate"))


def gstat(r, boot):
    out = {}
    for name, key, f in [("log_score_per_km2", "logd", np.nanmean), ("in_90pct_area_%", "hit90", lambda x: 100 * np.nanmean(x))]:
        out[name] = f(r[key])
        v = np.array([f(r[key][ix]) for ix in boot])
        out[name + "_p05"], out[name + "_p95"] = np.nanpercentile(v, [5, 95])
    return out


def main():
    t0 = time.time()
    daily, tr, ids, SIG_D, ALP_D = s4.setup()          # SIG_D, ALP_D: settings of the final M4 (step 2)
    pd.set_option("display.width", 220)
    trn, fit_ids = tr[tr["split"] == "train"], ids["train"] | ids["val"]
    st_tr, st_val = starts_for(daily, ids["train"], "all"), starts_for(daily, ids["val"], "all")
    print(f"Training starts (all drifters) {len(st_tr[0]):,}; validation starts {len(st_val[0]):,}")

    # ======================================================================= 1. grid size and time step
    print("\n1. GRID SIZE (M1 on all training drifters, scored on all validation drifters)")
    c1 = cases_at(st_tr, 1)
    vc = {k: cases_at(st_val, k) for k in sorted(set([7, 30, 90, 6, 12, 60]))}
    rows, res = [], {}
    grids = [SquareGrid(0.5), SquareGrid(1.0), SquareGrid(2.0), HexGrid(1.0), HexGrid(2.0)]
    for g in grids:
        frm, to = encode(c1, g)
        P, moves = fit_operator(frm, to, g.ncell)
        nd = pd.DataFrame({"cell": frm, "id": c1["id"].values}).groupby("cell")["id"].nunique().reindex(range(g.ncell)).fillna(0).values
        visited = moves > 0
        usable = 100 * (nd[visited] >= MIN_DRIFTERS).mean()
        Pk, last = np.eye(g.ncell + 4), 0
        for k in (7, 30, 90):
            Pk = Pk @ mpow(P, k - last); last = k
            r = evaluate(Pk, g, vc[k]); res[(g.name, k)] = r
            rows.append({"grid": g.name, "cells": g.ncell, "usable_%": usable, "horizon_days": k, **gstat(r, drifter_boot(vc[k]["id"].values))})
    gt = pd.DataFrame(rows)
    gain = []
    for k in (7, 30, 90):
        boot = drifter_boot(vc[k]["id"].values)
        for name in gt["grid"].unique():
            dd = res[(name, k)]["logd"] - res[("square 1°", k)]["logd"]
            v = np.array([np.nanmean(dd[ix]) for ix in boot])
            gain.append({"grid": name, "horizon_days": k, "gain_vs_square1": np.nanmean(dd), "gain_p05": np.nanpercentile(v, 5), "gain_p95": np.nanpercentile(v, 95)})
    gt = gt.merge(pd.DataFrame(gain), on=["grid", "horizon_days"])
    gt.to_csv(os.path.join(OUT12, "grid_pooled.csv"), index=False)
    print(gt[["grid", "cells", "usable_%", "horizon_days", "in_90pct_area_%", "gain_vs_square1", "gain_p05", "gain_p95"]].round(2).to_string(index=False))

    print("\n1b. TIME STEP and MARKOV CHECK (1° squares)")
    g1 = SquareGrid(1.0)
    Pdt = {dt: fit_operator(*encode(cases_at(st_tr, dt), g1), g1.ncell)[0] for dt in (1, 2, 3, 6)}
    rows = []
    for H in (6, 12, 30, 60, 90):
        cH = cases_at(st_tr, H)
        Pdir, cnt = fit_operator(*encode(cH, g1), g1.ncell)
        boot = drifter_boot(vc[H]["id"].values)
        rr = {dt: evaluate(mpow(Pdt[dt], H // dt), g1, vc[H]) for dt in (1, 2, 3, 6)}
        rr["direct"] = evaluate(Pdir, g1, vc[H])
        for name, r in rr.items():
            d = r["logp"] - rr[1]["logp"]
            v = np.array([np.nanmean(d[ix]) for ix in boot])
            rows.append({"step": name, "horizon_days": H, "log_score": np.nanmean(r["logp"]), "gain_vs_1day": np.nanmean(d),
                         "gain_p05": np.nanpercentile(v, 5), "gain_p95": np.nanpercentile(v, 95)})
        rowsel = cnt >= 20
        tv = 0.5 * np.abs(mpow(Pdt[1], H)[:NCELL] - Pdir[:NCELL]).sum(axis=1)
        idt = np.array(sorted(ids["train"]))
        half = set(np.random.default_rng(SEED).permutation(idt)[: len(idt) // 2])
        inA = cH["id"].isin(half).values
        PA, cA = fit_operator(*encode(cH[inA], g1), g1.ncell)
        PB, cB = fit_operator(*encode(cH[~inA], g1), g1.ncell)
        both = (cA >= 10) & (cB >= 10)
        tvh = 0.5 * np.abs(PA[:NCELL] - PB[:NCELL]).sum(axis=1)
        rows[-1].update({"tv_iterated_vs_direct": np.average(tv[rowsel], weights=cnt[rowsel]),
                         "tv_split_half_reference": np.average(tvh[both], weights=(cA + cB)[both])})
    mt = pd.DataFrame(rows)
    mt.to_csv(os.path.join(OUT12, "timestep_markov_pooled.csv"), index=False)
    print(mt.round(3).to_string(index=False))

    # ======================================================================= 2. complexity, all drifters
    print("\n2. MODEL COMPLEXITY (all drifters)")
    arr = s10.arrival_class(daily)
    ocean = ocean_mask(daily[daily["split"] == "train"])

    def pooled_moves(sub):
        m = moves_for(sub, "all")
        m["a_prev"] = arr.reindex(pd.MultiIndex.from_arrays([m["id"].values, m["time"].values])).fillna(s10.UNK).astype(int).values
        return m

    m = pooled_moves(trn)
    D = Displacements(m, ocean)
    vcases = {k: s10.add_start_class(cases_at(st_val, k), arr) for k in HORIZONS}
    vs = lambda P: float(np.mean([score(forecast(P, vcases[k], k), vcases[k])["logp"].mean() for k in TUNE_K]))
    vsm = lambda P: float(np.mean([score(s10.forecast_memory(P, vcases[k], [k])[k], vcases[k])["logp"].mean() for k in TUNE_K]))
    tune = []
    best = (-9, None)
    for sg in M2_SIGMAS:
        for al in M2_ALPHAS:
            v = vs(m2_smoothed(D, sg, al)); tune.append({"model": "M2", "sigma": sg, "alpha": al, "val": v})
            if v > best[0]:
                best = (v, (sg, al))
    SIG, ALP = best[1]
    print(f"  M2 (pooled) chosen: sigma {SIG}, alpha {ALP}")
    best = (-9, None)
    for nk in MGAM_KNOTS:
        for C in MGAM_C:
            v = vs(MultinomialGAM(nk, C).fit(m, D).operator(D)); tune.append({"model": "GAM", "knots": nk, "C": C, "val": v})
            if v > best[0]:
                best = (v, (nk, C))
    GK, GC = best[1]
    print(f"  GAM chosen: {GK} knots, C {GC}")
    best = (-9, None)
    for nf, h, l2 in NN_SET:
        v = vs(FourierNet(h, l2, nf, seeds=3).fit(m, D).operator(D)); tune.append({"model": "network", "freqs": nf, "hidden": str(h), "l2": l2, "val": v})
        if v > best[0]:
            best = (v, (nf, h, l2))
    NF, NH, NL = best[1]
    print(f"  network chosen: freqs {NF}, hidden {NH}, L2 {NL}")
    mem = s10.Memory(m, D, SIG, ALP)
    best = (-9, None)
    for sg in s10.SIGMAS:
        for beta in s10.BETAS:
            v = vsm(mem.operator(mem.counts_model(sg, beta))); tune.append({"model": "memory counts", "sigma": sg, "beta": beta, "val": v})
            if v > best[0]:
                best = (v, (sg, beta))
    MS, MB = best[1]
    best = (-9, None)
    for C in s10.GAM_C:
        v = vsm(mem.operator(mem.from_joint(mem.fit_joint("gam", C=C), MS))); tune.append({"model": "memory GAM", "C": C, "val": v})
        if v > best[0]:
            best = (v, C)
    MGC = best[1]
    best = (-9, None)
    for h, l2 in s10.NN_SET:
        v = vsm(mem.operator(mem.from_joint(mem.fit_joint("nn", hidden=h, l2=l2, seeds=3), MS))); tune.append({"model": "memory network", "hidden": str(h), "l2": l2, "val": v})
        if v > best[0]:
            best = (v, (h, l2))
    MNH, MNL = best[1]
    tune = pd.DataFrame(tune)
    tune.to_csv(os.path.join(OUT12, "tuning_pooled.csv"), index=False)
    ch = dict(m2_sigma=SIG, m2_alpha=ALP, gam_knots=GK, gam_C=GC, nn_freqs=NF, nn_hidden=str(NH), nn_l2=NL,
              mem_sigma=MS, mem_beta=MB, mem_gam_C=MGC, mem_nn_hidden=str(MNH), mem_nn_l2=MNL)
    pd.DataFrame([ch]).to_csv(os.path.join(OUT12, "chosen_settings_pooled.csv"), index=False)
    print("  chosen:", ch)

    def fit_all(mm, DD):
        P2 = m2_smoothed(DD, SIG, ALP)
        mats = {"M0b displacement": m0_displacement(DD), "M1 empirical": m1_empirical(mm), "M2 smoothed": P2,
                "M5 GAM": MultinomialGAM(GK, GC).fit(mm, DD).operator(DD),
                "M6 neural net": FourierNet(NH, NL, NF).fit(mm, DD).operator(DD)}
        me = s10.Memory(mm, DD, SIG, ALP)
        mems = {"M7 memory, counts": me.operator(me.counts_model(MS, MB)),
                "M7 memory, GAM": me.operator(me.from_joint(me.fit_joint("gam", C=MGC), MS)),
                "M7 memory, network": me.operator(me.from_joint(me.fit_joint("nn", hidden=MNH, l2=MNL, seeds=5), MS))}
        return mats, mems, P2

    def score_set(mats, mems, P2, cases, label):
        a = evaluate_models(mats, cases, "M2 smoothed", label=label)
        b = s10.compare(mems, P2, cases, label)
        return pd.concat([a, b[b["model"] != "M2 smoothed"]], ignore_index=True)

    mats, mems, P2 = fit_all(m, D)
    val = score_set(mats, mems, P2, vcases, "validation / all")
    val.to_csv(os.path.join(OUT12, "complexity_validation_pooled.csv"), index=False)
    print("\n  validation, log score minus pooled M2:"); gain_lines(val, "gain_vs_M2")

    print("\n  TEST (refit on training + validation drifters)")
    ocean_f = ocean_mask(daily[daily["id"].isin(fit_ids)])
    mf = pooled_moves(tr[tr["id"].isin(fit_ids)])
    Df = Displacements(mf, ocean_f)
    mats, mems, P2f = fit_all(mf, Df)
    out = []
    for split in ("test", "recent_test"):
        cs = {k: s10.add_start_class(cases_at(starts_for(daily, ids[split], "all"), k), arr) for k in HORIZONS}
        out.append(score_set(mats, mems, P2f, cs, f"{split} / all"))
    tt = pd.concat(out, ignore_index=True)
    tt.to_csv(os.path.join(OUT12, "complexity_test_pooled.csv"), index=False)
    gain_lines(tt, "gain_vs_M2")

    # ======================================================================= 3. depth: pooled M2 vs split
    print("\n3. DEPTH: pooled M2 vs M2 split by drogue status (test drifters)")
    mset = {dd: moves_for(tr[tr["id"].isin(fit_ids)], dd) for dd in ("drogued", "undrogued")}
    Dset = {dd: Displacements(mset[dd], ocean_f) for dd in mset}
    out = []
    for split in ("test", "recent_test"):
        for dd in ("drogued", "undrogued"):
            models = {"M2 pooled": P2f, "M2 own depth only": m2_smoothed(Dset[dd], SIG, ALP),
                      "M2 other depth only": m2_smoothed(Dset["undrogued" if dd == "drogued" else "drogued"], SIG, ALP),
                      "M4 final (own depth + 0.25 x other)": drogue_weighted(Df, mf, dd, W_OTHER, SIG_D, ALP_D)}
            cs = {k: cases_at(starts_for(daily, ids[split], dd), k) for k in HORIZONS}
            out.append(evaluate_models(models, cs, "M2 pooled", label=f"{split} / {dd}"))
    dt = pd.concat(out, ignore_index=True)
    dt.to_csv(os.path.join(OUT12, "depth_test.csv"), index=False)
    gain_lines(dt, "gain_vs_M2")
    print(f"\nStep 12 done in {(time.time() - t0) / 60:.1f} min. Outputs in {OUT12}")


if __name__ == "__main__":
    main()
