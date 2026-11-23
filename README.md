# Predicting surface transport in the Brazil–Malvinas Confluence

UNSW DATA3001, Term 3 2026, group S01-G8: Ishaan Pala, Henrique Barroso, Lev Tolmachev, Haiming Zhang and Kai Nguyen. Client: Dr Shane Elipot. Lecturer: Dr Lachlan Astfalck.

If something floating enters the sea off southern Brazil, Uruguay or Argentina, where is it likely to be in a week, a month or three months, and which side of the region will it leave through? We answer that for the box 30–50°S, 60–40°W using 24 years of NOAA Global Drifter Program data. The core model is a daily transition matrix on 1° cells, which the report compares with GAMs, neural networks and models that remember the previous day's direction, all scored on drifters the models never saw.

Every table and figure in the report can be rebuilt from this repository with one command, and so can the interactive viewer. The input data are included, so you don't need internet access.

## Running it

You need [uv](https://docs.astral.sh/uv/), which installs the right Python version and the exact package versions for you.

```
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then:

```
git clone https://github.com/LevTolmachev/data3001-sm01-group8.git
cd data3001-sm01-group8
uv sync
uv run python run_all.py
```

The full run takes about 30 minutes on a 2-core laptop and needs roughly 4 GB of memory. Results go to `data/models/`, which isn't stored in the repository; the first run creates it. Each step's printed output is saved in `logs/`. To run a single step use `--only 13`, and to restart from a step use `--from 5` (the steps before it must have run at least once).

The viewer needs no setup: open `viewer/bmc_drift_viewer.html` in any browser. It works offline.

If you'd rather not install uv, create a virtual environment with `python -m venv .venv`, activate it, then run `pip install -r requirements.txt` and `python run_all.py`. You get the same package versions, because `requirements.txt` is exported from the lock file.

## Python environment

We ran everything on Python 3.12.3 under Linux. `pyproject.toml` lists the direct dependencies and allows Python 3.11 to 3.13; `uv.lock` pins all 48 packages to exact versions, and `.python-version` tells uv to use 3.12. The main versions are numpy 2.3.5, pandas 2.3.3, scipy 1.18.1, scikit-learn 1.7.2 and xarray 2026.9.0. zarr stays at 2.18.7, below version 3, because step 1 reads NOAA's data store through xarray's zarr 2 interface.

## Data

Everything in `data/` was downloaded once and committed, so the results don't depend on NOAA's servers being available.

- `gdp_bmc_extract.parquet` (47 MB): GDP hourly v2.01.1 positions inside the region plus a 2° buffer, with drifter id, time and drogue status. It holds 3,306,318 rows from 1,043 drifters, September 1998 to October 2022, and is made by `code/step1_extract.py`.
- `winds_at_drifters.parquet` and `ncep_winds_region.nc` (31 MB together): daily 10 m winds from NCEP/NCAR Reanalysis 1, at each drifter's position and on a grid over the region. Made by `code/get_winds.py`.
- `gdp_metadata.parquet`: where and when each drifter was deployed and when it lost its drogue, used in step 14.
- `geo/`: Natural Earth coastlines for the maps.

`uv run python run_all.py --download` rebuilds these files from NOAA before running the analysis, which adds roughly an hour. Set the environment variable `DATA3001_BASE` if you want the data kept in another folder.

## Steps

`run_all.py` runs the steps in the order below and stops at the first error. RQ numbers refer to Table 1 of the report.

| Step | Script | What it does | Time |
|---|---|---|---|
| 2 | `step2_models_M0_M1_M2` | Daily moves, frozen drifter splits, models M0–M2 | 0.5 min |
| 3 | `step3_core_results` | Grids, time steps and the Markov check on drogued drifters (RQ3) | 0.5 min |
| 4 | `step4_drogue_season_gam_nn` | Depth split (M4), season, first GAM and network (RQ4, RQ5) | 1 min |
| 5 | `step5_final_test_and_operator` | Test scores for the model ladder, final matrices with 200 bootstrap refits, transport results (RQ1, RQ2) | 6 min |
| 8 | `step8_stronger_gam_nn` | Second GAM and network (RQ5) | 3 min |
| 9 | `step9_season_operators` | Seasonal matrices (RQ4) | 0.5 min |
| 10 | `step10_memory_models` | Models with memory, M7 (RQ5) | 4 min |
| 11 | `step11_model_pictures` | Example forecasts used in the slides | 0.5 min |
| 12 | `step12_pooled_comparisons` | Grid and model comparisons on all drifters, as reported (RQ3–RQ5) | 10 min |
| 13 | `step13_windage_oil` | Wind response and the oil scenario (RQ6) | 0.5 min |
| 14 | `step14_drifter_age` | Why drogue status changes with latitude | 0.2 min |
| 6 | `step6_figures` | Poster and presentation figures | 0.5 min |
| 7 | `step7_build_viewer` | Builds the viewer | < 0.1 min |
| 15 | `step15_report_figures` | Report Figures 1–7 and 9 | 1 min |

Step 1 and `get_winds.py` only run with `--download`. The numbers aren't consecutive because steps were added as the project grew. `M3_M7_GUIDE.md` says which script and function holds each of models M3 to M7. `code/extras/exit_baseline.py` computes the constant exit-share baseline for the exit-edge Brier score in Section 6.4 (run it after step 5).

Settings were chosen on validation drifters. The test drifters were first scored in step 5 and scored again in steps 8, 10, 12 and 13; Section 4.3 of the report discusses this. Seeds are fixed, so a rerun gives the same numbers, although network results can differ in the last digit between operating systems.

## Using the matrices yourself

The final matrices are in `data/models/step5/`: `operator_final_drogued.npz` for water at 15 m and `operator_final_undrogued.npz` for the surface. Each file holds the 404 × 404 one-day matrix `P`, 30 bootstrap copies, the cell centres and a mask of cells with too few drifters. States 0–399 are the 1° cells, numbered row by row from the north-west corner; 400–403 mean the material has left through the north, south, east or west edge. A forecast is repeated multiplication:

```python
import numpy as np

z = np.load("data/models/step5/operator_final_drogued.npz", allow_pickle=True)
P = z["P"]
p = np.zeros(404)
p[8 * 20 + 6] = 1.0          # row 8 (38–39°S), column 6 (54–53°W)
for day in range(90):
    p = p @ P
print("still in R:", p[:400].sum(), "left via N, S, E, W:", p[400:])
```

## Credits

Drifter data: NOAA Global Drifter Program (Elipot et al., 2016, 2022). Winds: NCEP/NCAR Reanalysis 1 (Kalnay et al., 1996), from NOAA PSL. Coastlines: Natural Earth. We wrote the code with help from Claude, an AI assistant; Section 10 of the report describes how we used it and what we checked.
