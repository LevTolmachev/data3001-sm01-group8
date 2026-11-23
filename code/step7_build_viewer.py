"""
DATA3001 Group S01-G8 - Step 7: build the interactive viewer (D4)
==================================================================
Run AFTER steps 5 and 9. Packs the two final operators and their bootstrap copies into ONE self-contained
HTML file (no server, no install: double-click to open). Output: viewer/bmc_drift_viewer.html
Run:  python step7_build_viewer.py
"""
import os, sys, json, base64
import numpy as np
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
sys.path.insert(0, _here)
from step2_models_M0_M1_M2 import OUT, NCELL, NX, NY, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, MIN_DRIFTERS

N_BOOT_VIEWER = 20
S5 = os.path.join(OUT, "step5")
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def sparse(P):
    """Rows of P as (row pointer, columns, values), dropping entries below 1e-6."""
    P = np.where(P < 1e-6, 0, P)
    P = P / P.sum(axis=1, keepdims=True)
    rp, cols, vals = [0], [], []
    for i in range(P.shape[0]):
        j = np.flatnonzero(P[i])
        cols.append(j); vals.append(P[i, j]); rp.append(rp[-1] + len(j))
    return {"rp": b64(np.array(rp, "<u4")), "c": b64(np.concatenate(cols).astype("<u2")),
            "v": b64(np.concatenate(vals).astype("<f4"))}


data = {"nx": NX, "ny": NY, "region": [LAT_MIN, LAT_MAX, LON_MIN, LON_MAX], "min_drifters": MIN_DRIFTERS, "ops": {}}
for d in ("drogued", "undrogued"):
    z = np.load(os.path.join(S5, f"operator_final_{d}.npz"), allow_pickle=True)
    zs = np.load(os.path.join(OUT, "step9", f"operator_seasonal_{d}.npz"), allow_pickle=True)   # from step 9
    data["ops"][d] = {"P": sparse(z["P"].astype(float)),
                      "boot": [sparse(b.astype(float)) for b in z["P_boot"][:N_BOOT_VIEWER]],
                      "season": [sparse(x.astype(float)) for x in zs["P_season"]],
                      "sboot": [[sparse(x.astype(float)) for x in b] for b in zs["P_season_boot"]]}
    data["masked"] = z["masked"].astype(int).tolist()
    data["ocean"] = z["ocean"].astype(int).tolist()
    data["drifters"] = z["distinct_drifters"].astype(int).tolist()
    data["config"] = json.loads(str(z["config"]))
html = open(os.path.join(_here, "viewer_template.html"), encoding="utf-8").read()
out_dir = os.path.join(os.path.dirname(_here), "viewer")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "bmc_drift_viewer.html")
open(out, "w", encoding="utf-8").write(html.replace("/*__DATA__*/null", json.dumps(data)))
print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
