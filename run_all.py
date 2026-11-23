"""
Run the whole S01-G8 pipeline in order, from the data files in ./data to every result, figure and the viewer.

    uv run python run_all.py              # steps 2-15 (about 30 minutes on a 2-core machine)
    uv run python run_all.py --from 5     # start at step 5 (earlier outputs must already exist)
    uv run python run_all.py --only 13    # one step

Steps 1 (download the GDP extract) and get_winds (download winds and metadata) need internet access to NOAA and
are skipped by default because their outputs are already in ./data. Add --download to run them first.
Each step's printed output is saved to logs/<step>.log, and run times to logs/times.txt.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    (2, "step2_models_M0_M1_M2"),
    (3, "step3_core_results"),
    (4, "step4_drogue_season_gam_nn"),
    (5, "step5_final_test_and_operator"),
    (8, "step8_stronger_gam_nn"),
    (9, "step9_season_operators"),
    (10, "step10_memory_models"),
    (11, "step11_model_pictures"),
    (12, "step12_pooled_comparisons"),
    (13, "step13_windage_oil"),
    (14, "step14_drifter_age"),
    (6, "step6_figures"),          # needs steps 3-5, 8, 10, 12
    (7, "step7_build_viewer"),     # needs steps 5 and 9
    (15, "step15_report_figures"), # needs steps 5, 12, 13, 14
]
DOWNLOADS = [(1, "step1_extract"), (0, "get_winds")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=None, help="first step number to run")
    ap.add_argument("--only", type=int, default=None, help="run only this step number")
    ap.add_argument("--download", action="store_true", help="also run step 1 and get_winds (needs NOAA access)")
    a = ap.parse_args()

    steps = (DOWNLOADS if a.download else []) + STEPS
    if a.only is not None:
        steps = [s for s in steps if s[0] == a.only]
    elif a.start is not None:
        order = [n for n, _ in steps]
        steps = steps[order.index(a.start):]

    env = dict(os.environ)
    env.setdefault("DATA3001_BASE", os.path.join(ROOT, "data"))
    env["MPLBACKEND"] = "Agg"
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    t_all = time.time()
    for num, name in steps:
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] step {num}: {name} ...", flush=True)
        with open(os.path.join(ROOT, "logs", f"{name}.log"), "w", encoding="utf-8") as log:
            rc = subprocess.call([sys.executable, os.path.join(ROOT, "code", f"{name}.py")],
                                 stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        secs = time.time() - t0
        with open(os.path.join(ROOT, "logs", "times.txt"), "a", encoding="utf-8") as f:
            f.write(f"{name} rc={rc} {secs:.0f}s\n")
        print(f"    {'ok' if rc == 0 else 'FAILED'} in {secs / 60:.1f} min", flush=True)
        if rc != 0:
            print(f"    see logs/{name}.log")
            sys.exit(rc)
    print(f"All done in {(time.time() - t_all) / 60:.1f} min. Results in data/models, viewer in viewer/.")


if __name__ == "__main__":
    main()
