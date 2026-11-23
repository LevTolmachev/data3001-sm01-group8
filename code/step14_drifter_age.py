"""
DATA3001 Group S01-G8 - Step 14: why drogued drifters dominate the south of R and undrogued ones the north
==========================================================================================================
Question raised at our presentation: is the north-south split of drogue status a sampling effect or an
environmental one? We join every daily position in R to the GDP per-drifter metadata (deployment date and
position, from code/get_winds.py: gdp_metadata.parquet) and ask how old drifters are, and where they were
deployed, in each latitude band.
Outputs: <BASE>/models/step14 (CSV tables and fig_drifter_age.png).
Run:  uv run python code/step14_drifter_age.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = os.getcwd()
sys.path.insert(0, _here)
from step2_models_M0_M1_M2 import BASE, OUT, INPUT_FILE, prepare, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

OUT14 = os.path.join(OUT, "step14")
os.makedirs(OUT14, exist_ok=True)
BANDS = [-50, -45, -40, -35, -30]
BAND_LAB = ["45–50°S", "40–45°S", "35–40°S", "30–35°S"]


def origin(r):
    if (-52 <= r.deploy_lat <= -28) and (-62 <= r.deploy_lon <= -38):
        return "inside R (+2°)"
    if r.deploy_lat < -50:
        return "south (Southern Ocean)"
    if r.deploy_lat > -28:
        return "north (tropical and subtropical Atlantic)"
    return "east (open South Atlantic)"


def main():
    meta_file = os.path.join(BASE, "gdp_metadata.parquet")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"{meta_file} not found: run code/get_winds.py first (it saves the metadata too).")
    df = prepare(pd.read_parquet(INPUT_FILE))
    d = df[(df["time"].dt.hour == 0) & df["inR"]].copy()
    m = pd.read_parquet(meta_file)
    m["id"] = m["id"].astype(str)
    d = d.merge(m[["id", "deploy_date", "deploy_lat", "deploy_lon"]], on="id", how="left")
    d["age_days"] = (d["time"] - pd.to_datetime(d["deploy_date"])).dt.days
    d["band"] = pd.cut(d["lat"], BANDS, labels=BAND_LAB)
    drogue_life = (pd.to_datetime(m["drogue_lost_date"]) - pd.to_datetime(m["deploy_date"])).dt.days
    drogue_life = drogue_life[drogue_life > 0]

    by_band = d.groupby("band", observed=True).agg(daily_positions=("id", "size"), drifters=("id", "nunique"),
                                                   undrogued_share=("drogued", lambda x: 1 - x.mean()),
                                                   median_age_days=("age_days", "median"))
    age_type = d.pivot_table(index="band", columns="drogued", values="age_days", aggfunc="median", observed=True)
    age_type.columns = ["median_age_undrogued", "median_age_drogued"]
    by_band = by_band.join(age_type)
    first = d.sort_values("time").groupby("id").first()
    first["deployed"] = first.apply(origin, axis=1)
    entry = pd.crosstab(first["deployed"], first["band"])
    status = pd.crosstab(first["deployed"], first["drogued"].map({True: "drogued", False: "undrogued"}))
    status["undrogued_share_on_entry"] = status["undrogued"] / status.sum(axis=1)
    status["median_age_on_entry_days"] = first.groupby("deployed")["age_days"].median()
    by_band.to_csv(os.path.join(OUT14, "by_latitude_band.csv"))
    entry.to_csv(os.path.join(OUT14, "entry_band_by_origin.csv"))
    status.to_csv(os.path.join(OUT14, "status_on_entry_by_origin.csv"))
    pd.set_option("display.width", 200)
    print("By latitude band (daily positions in R):\n", by_band.round(2).to_string())
    print("\nWhere drifters entering R were deployed, by the band they entered through:\n", entry.to_string())
    print("\nDrogue status and age on first entry, by origin:\n", status.round(2).to_string())
    print(f"\nDays from deployment to drogue loss (GDP metadata, drifters seen in R): median {drogue_life.median():.0f}, "
          f"quartiles {drogue_life.quantile(.25):.0f}-{drogue_life.quantile(.75):.0f}")

    # figure: (a) undrogued share and median age by band, (b) deployment positions coloured by status on entry
    BLUE, ORANGE, INK = "#1F6FB2", "#E07B1F", "#1B2A3A"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
    a = ax[0]
    y = np.arange(len(by_band))
    a.barh(y, 100 * by_band["undrogued_share"], color=ORANGE, height=0.6)
    for i, (s, ag) in enumerate(zip(by_band["undrogued_share"], by_band["median_age_days"])):
        a.text(100 * s + 1.5, i, f"{100 * s:.0f}%  (median age {ag:.0f} d)", va="center", fontsize=10, color=INK)
    a.set_yticks(y, by_band.index)
    a.set_xlim(0, 105)
    a.set_xlabel("Daily positions from undrogued drifters (%)")
    a.set_title("(a) Undrogued share and drifter age, by latitude band", loc="left", fontsize=11.5, fontweight="bold")
    a = ax[1]
    land_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "geo", "ne_110m_land.geojson")
    if os.path.exists(land_file):          # Natural Earth 1:110m land (public domain)
        import json
        from matplotlib.patches import Polygon
        for feat in json.load(open(land_file))["features"]:
            g = feat["geometry"]
            for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
                a.add_patch(Polygon(np.array(poly[0]), closed=True, facecolor="#D3DAE3", edgecolor="white", lw=0.4, zorder=0))
    for flag, col, lab in ((True, BLUE, "drogued on entry"), (False, ORANGE, "undrogued on entry")):
        f = first[first["drogued"] == flag]
        a.scatter(f["deploy_lon"], f["deploy_lat"], s=9, color=col, alpha=0.6, lw=0, label=lab)
    a.plot([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN], [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN], color=INK, lw=1.5)
    a.set_xlim(-80, 20); a.set_ylim(-70, 10); a.set_aspect(1.15)
    a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.0f}°{'W' if v < 0 else ('E' if v > 0 else '')}"))
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.0f}°{'S' if v < 0 else ('N' if v > 0 else '')}"))
    a.set_xlabel("Deployment longitude"); a.set_ylabel("Deployment latitude")
    a.set_title("(b) Where drifters that reached R were deployed (box = R)", loc="left", fontsize=11.5, fontweight="bold")
    a.legend(frameon=False, loc="lower right", markerscale=2)
    fig.savefig(os.path.join(OUT14, "fig_drifter_age.png"), dpi=300, bbox_inches="tight")
    print("Saved", os.path.join(OUT14, "fig_drifter_age.png"))


if __name__ == "__main__":
    main()
