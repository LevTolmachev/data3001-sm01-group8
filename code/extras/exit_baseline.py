"""
Exit-edge baseline for Section 6.4 of the report (DATA3001 S01-G8).

The final model's exit-edge Brier score (step 5) is compared with a forecast that gives every
start the same exit shares: the shares of exit edges among the fitting drifters (training +
validation) that were seen to leave R. The score uses the same fate cases and the same Brier
calculation as step 5 (s2.fate_cases and s2.score_fate).

Run after steps 2-5:  uv run python code/extras/exit_baseline.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import step2_models_M0_M1_M2 as s2
import step4_drogue_season_gam_nn as s4

EXITS = np.array(s2.EXITS)


def exit_edges(fc):
    return fc[fc["exited"]]["edge"].to_numpy(dtype=str)


def main():
    daily, tr, ids, _, _ = s4.setup()
    print("Brier score of a constant exit-share forecast (shares from training + validation drifters)")
    for d in ("drogued", "undrogued"):
        old = s2.DROGUE_SET
        s2.DROGUE_SET = d
        fit = exit_edges(s2.fate_cases(daily, ids["train"] | ids["val"]))
        shares = np.array([(fit == e).mean() for e in EXITS])
        for split in ("test", "recent_test"):
            e = exit_edges(s2.fate_cases(daily, ids[split]))
            obs = (e[:, None] == EXITS[None, :]).astype(float)
            brier = float(((shares[None, :] - obs) ** 2).sum(axis=1).mean())
            print(f"  {split:11s} / {d:9s}  Brier {brier:.3f}   shares "
                  + ", ".join(f"{x} {p:.1%}" for x, p in zip(EXITS, shares)))
        s2.DROGUE_SET = old


if __name__ == "__main__":
    main()
