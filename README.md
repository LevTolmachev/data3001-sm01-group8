# Predicting surface transport in the Brazil–Malvinas Confluence

DATA3001 capstone, Group S01-G8 (UNSW, Term 3 2026).

Where does floating material released off southern Brazil, Uruguay and Argentina go, how long does it stay in the region
R = 30–50°S, 60–40°W, and where does it leave? This repository builds daily transition matrices (Ulam's method,
1° cells, 404 states) from the NOAA Global Drifter Program hourly data, compares them with GAMs, neural networks and
models with memory on held-out drifters, extends them to wind-driven material such as oil, and packs the final model
into a single-file interactive viewer.

Everything in the final report is produced by the code here. Running `run_all.py` from a fresh clone regenerates
every table, figure and the viewer from the data files in `data/`.

## Quick start

1. **Install uv** (a fast Python installer that reads the pinned environment):
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. **Get the code and create the environment** (uv downloads Python 3.12 if you do not have it):
   ```
   git clone https://github.com/LevTolmachev/data3001-sm01-group8.git bmc-drift
   cd bmc-drift
   uv sync
   ```
3. **Run everything** (about 35 minutes on a 2-core machine; 4 GB of memory is enough):
   ```
   uv run python run_all.py
   ```
4. **Open the viewer**: double-click `viewer/bmc_drift_viewer.html` (any modern browser, works offline).

Without uv: `python -m venv .venv`, activate it, then `pip install -r requirements.txt` and `python run_all.py`
(`requirements.txt` is exported from `uv.lock`, so the versions are the same).

## The Python environment

| File | What it pins |
|---|---|
| `pyproject.toml` | Python ≥ 3.11, < 3.14, and the direct dependencies with version ranges |
| `uv.lock` | the exact version of every package, including dependencies of dependencies (48 packages) |
| `.python-version` | Python 3.12, which `uv sync` installs if needed |
| `requirements.txt` | the same pinned versions for plain pip |

The results in the report were produced with Python 3.12.3, numpy 2.3.5, pandas 2.3.3, scipy 1.18.1,
scikit-learn 1.7.2, xarray 2026.9.0 and zarr 2.18.7 on Linux. `zarr` is held below version 3 because step 1 opens the GDP store through
the zarr 2 interface of xarray.

## Data

All inputs are in `data/`, so no download is needed to reproduce the results.

| File | Size | What it is | Made by |
|---|---|---|---|
| `gdp_bmc_extract.parquet` | 47 MB | GDP hourly v2.01.1 rows inside R plus a 2° buffer: drifter id, time, lat, lon, drogue status (3,306,318 rows, 1,043 drifters, Sep 1998 – Oct 2022) | `code/step1_extract.py` |
| `winds_at_drifters.parquet` | 4 MB | daily-mean 10 m wind at every drifter's 00 UTC position (NCEP/NCAR Reanalysis 1) | `code/get_winds.py` |
| `ncep_winds_region.nc` | 27 MB | daily-mean 10 m wind grid over R plus a margin, 1994–2022 | `code/get_winds.py` |
| `gdp_metadata.parquet` | 0.1 MB | GDP per-drifter metadata: deployment date and place, drogue-loss date | `code/get_winds.py` |
| `geo/south_america.geojson`, `geo/ne_110m_land.geojson` | 0.4 MB | Natural Earth coastlines (public domain), for maps only | downloaded |

To rebuild the data from NOAA instead (needs internet access, about an hour in total):
`uv run python run_all.py --download`. Step 1 streams the public GDP Zarr store on AWS
(`https://noaa-oar-hourly-gdp-pds.s3.amazonaws.com/latest/gdp-v2.01.1.zarr`); `get_winds.py` downloads NCEP/NCAR daily winds from NOAA PSL.
Both also run unchanged in Google Colab, where they save to `MyDrive/DATA3001/initial_analysis`.

To keep the data somewhere else, set the environment variable `DATA3001_BASE` to that folder; every step reads
from and writes to it.

## What each step does

Run order and typical run times (2-core Linux machine). `run_all.py` runs them in this order and stops at the first
error; every step can also be run on its own with `uv run python code/<step>.py` once the steps it needs have run.

| Step | Time | Answers | What it does | Main outputs (in `data/models/`) |
|---|---|---|---|---|
| 1 `step1_extract` | 20–60 min | – | Download and cut the GDP data to R; proposal figures | `data/gdp_bmc_extract.parquet` |
| 2 `step2_models_M0_M1_M2` | 0.5 min | – | Daily moves, frozen drifter splits (60/20/20 + recent 2019–22), M0–M2, tuning of M2 | `splits.csv`, `operator_M2_*.npz`, `m2_tuning.csv` |
| 3 `step3_core_results` | 0.5 min | RQ3 | Grids (0.5°, 1°, 2°, hexagons), time steps, Markov check, drogued only | `step3/` |
| 4 `step4_drogue_season_gam_nn` | 1 min | RQ4, RQ5 | Depth split (M4), season, first GAM and network, chosen on validation | `step4/` |
| 5 `step5_final_test_and_operator` | 6 min | RQ1, RQ2 | Final test of the ladder; final operators with 200 bootstrap refits; release, residence, exits, density | `step5/operator_final_*.npz`, `step5/*.csv` |
| 8 `step8_stronger_gam_nn` | 3 min | RQ5 | Multinomial GAM and Fourier-feature network; seasonal tuning | `step8/` |
| 9 `step9_season_operators` | 0.5 min | RQ4 | Seasonal operators and their test | `step9/operator_seasonal_*.npz` |
| 10 `step10_memory_models` | 4 min | RQ5 | Models with memory (M7, 4,004 states) | `step10/` |
| 11 `step11_model_pictures` | 0.5 min | – | Example one-day forecasts of each model, for slides | `step11/` |
| 12 `step12_pooled_comparisons` | 10 min | RQ3, RQ4, RQ5 | Grid and complexity on all drifters; depth comparison | `step12/` |
| 13 `step13_windage_oil` | 0.5 min | RQ6 | Wind response, validation on undrogued drifters, oil scenario | `step13/` |
| 14 `step14_drifter_age` | 0.2 min | – | Why drogue status varies with latitude (presentation question) | `step14/` |
| 6 `step6_figures` | 0.5 min | – | Presentation and poster figures | `figures/` |
| 7 `step7_build_viewer` | < 0.1 min | – | Packs the operators into the single-file viewer | `viewer/bmc_drift_viewer.html` |
| 15 `step15_report_figures` | 1 min | – | Figures 1–7 and 9 and the method equation of the final report | `report_figures/` |

Settings were chosen on the validation drifters only. The test drifters were first scored in step 5; steps 8, 10, 12
and 13 score later additions on them again, which the report discloses. Random seeds are fixed, so a rerun gives the
same numbers; network training can differ in the last digit across operating systems.

Figure 8 of the report is a screenshot of the viewer (15 m, release at 38.5°S 53.5°W, 30 days).

## Repository layout

```
bmc-drift/
├── README.md              this file
├── run_all.py             runs every step in order
├── pyproject.toml         dependencies
├── uv.lock                exact versions
├── requirements.txt       the same versions for pip
├── code/
│   ├── step1_extract.py … step15_report_figures.py
│   ├── get_winds.py       wind and metadata download
│   ├── viewer_template.html
│   └── extras/            scripts used only for the poster
├── data/                  inputs (see Data); models/ is created by the steps
├── viewer/
│   └── bmc_drift_viewer.html
└── logs/                  printed output and run times of the last run
```

## Using the operators in other work

`data/models/step5/operator_final_drogued.npz` (15 m) and `operator_final_undrogued.npz` (surface) each hold the
404 × 404 one-day matrix `P` (rows: from, columns: to; states 0–399 are 1° cells numbered row by row from the
north-west corner, 400–403 are "left through N, S, E, W"), 30 bootstrap copies, the cell centres and the mask of
unsupported cells. A forecast is repeated multiplication:

```python
import numpy as np
z = np.load("data/models/step5/operator_final_drogued.npz", allow_pickle=True)
P = z["P"]
p = np.zeros(404); p[8 * 20 + 6] = 1.0        # row 8 (38–39°S), column 6 (54–53°W)
for day in range(90):
    p = p @ P
print("still in R:", p[:400].sum(), "left via N, S, E, W:", p[400:])
```

## Credits and licences

Data: NOAA Global Drifter Program hourly product v2.01.1 (Elipot et al., 2016, 2022); NCEP/NCAR Reanalysis 1
(Kalnay et al., 1996), provided by NOAA PSL; Natural Earth coastlines (public domain). Code written by the group with
help from an AI assistant (see Section 10 of the report).
