"""
DATA3001 Group S01-G8 - Step 11: what each model predicts for one cell (for the slide that explains the models)
==============================================================================================================
Run AFTER steps 2, 5, 8 and 10. Fits M2, the tuned GAM, the tuned network and the memory network on
training + validation drifters (drogued), then reads off, for the release cell at 38S 54W, the probability
of each of the 25 one-day moves (up to 2 cells in any direction). The memory model is shown for a drifter
whose previous day's move was to the east. Output: <BASE>/models/figures/model_pictures.json
Run:  python step11_model_pictures.py
"""
import os, sys, json
import numpy as np
import pandas as pd
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
sys.path.insert(0, _here)
from step2_models_M0_M1_M2 import NCELL, OUT, ocean_mask, Displacements, m2_smoothed
import step4_drogue_season_gam_nn as s4
from step4_drogue_season_gam_nn import moves_for, sub_points
from step8_stronger_gam_nn import MultinomialGAM, FourierNet
import step10_memory_models as s10

daily, tr, ids, SIG, ALP = s4.setup()
fit_ids = ids["train"] | ids["val"]
ocean = ocean_mask(daily[daily["id"].isin(fit_ids)])
arr = s10.arrival_class(daily)
m = s10.prep_moves(tr[tr["id"].isin(fit_ids)], arr)
D = Displacements(m, ocean)
c0 = int(np.load(os.path.join(OUT, "step5", "operator_final_drogued.npz"), allow_pickle=True)["release_cells"][0])
ch8 = pd.read_csv(os.path.join(OUT, "step8", "chosen_settings.csv")).iloc[0]
ch10 = pd.read_csv(os.path.join(OUT, "step10", "chosen_settings.csv")).iloc[0]
print("release cell", c0, "displacement classes", D.K, "window", D.R)


def window(hist_row):
    """(2R+1) x (2R+1) grid of move probabilities, rows north to south, columns west to east."""
    v = np.asarray(hist_row, float)
    v = v / v.sum()
    n = 2 * D.R + 1
    return v.reshape(n, n).round(4).tolist()


out = {"cell": c0, "R": int(D.R)}
# M2: smoothed histogram (same smoothing and shrinkage as the operator)
from scipy.ndimage import gaussian_filter
from step2_models_M0_M1_M2 import NX, NY
H = gaussian_filter(D.histogram().reshape(NY, NX, D.K), sigma=(SIG, SIG, 0), mode="constant", cval=0.0, truncate=2.0).reshape(NCELL, D.K)
glob = H.sum(axis=0) / H.sum()
out["M2"] = window((H[c0] + ALP * glob) / (H[c0].sum() + ALP))
# M5 GAM
gam = MultinomialGAM(int(ch8["mgam_knots"]), float(ch8["mgam_C"])).fit(m, D)
lat, lon = sub_points()
pr = np.zeros((lat.shape[1], D.K))
pr[:, gam.clf.classes_] = gam.clf.predict_proba(gam.basis(lat[c0], lon[c0]))
out["M5"] = window(pr.mean(axis=0))
# M6 network
net = FourierNet(eval(ch8["nn_hidden"]), float(ch8["nn_l2"]), int(ch8["nn_freqs"])).fit(m, D)
from step8_stronger_gam_nn import fourier
X = fourier(lat[c0], lon[c0], net.nf)
pr = np.zeros((len(X), D.K))
for nn in net.nets:
    pr[:, nn.classes_] += nn.predict_proba(X) / len(net.nets)
out["M6"] = window(pr.mean(axis=0))
# M7 memory network, previous move to the east (direction class 0)
mem = s10.Memory(m, D, SIG, ALP)
pj = mem.fit_joint("nn", hidden=eval(ch10["nn_hidden"]), l2=float(ch10["nn_l2"]), seeds=5)
out["M7_east"] = window(pj[c0, 0].sum(axis=0))
out["M7_west"] = window(pj[c0, 4].sum(axis=0))
for k in ("M2", "M5", "M6", "M7_east"):
    g = np.array(out[k])
    print(k, "stay %.2f" % g[D.R, D.R], "east 1 cell %.2f" % g[D.R, D.R + 1], "south 1 cell %.2f" % g[D.R + 1, D.R])
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
with open(os.path.join(OUT, "figures", "model_pictures.json"), "w") as f:
    json.dump(out, f)
print("saved model_pictures.json")
