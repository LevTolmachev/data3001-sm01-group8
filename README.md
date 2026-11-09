# data3001-sm01-group8

UNSW DATA3001 Term 3 2026, Group S01-G8. Lecturer: Lachlan Astfalck. Client: Shane Elipot.

## Group members

- Ishaan Pala
- Henrique Barroso
- Lev Tolmachev
- Haiming Zhang
- Kai Nguyen

## Project

Something goes into the water in the Brazil-Malvinas Confluence (BMC): a container, a life raft or an oil slick.
Where has the surface flow taken it after a week, a month or a year? How long does it stay, and which edge does it leave through?

**Overarching question:** How well can historical drifter observations predict where floating material entering the BMC travels, how long it remains and where it leaves, and how much model complexity can the data justify?

**Region R:** 30-50°S, 60-40°W, split into 1° cells. We also extract a 2° buffer around R so we can see which edge a drifter leaves through.

**Data:** NOAA Global Drifter Program hourly product v2.01. Inside R this gives 2.39M hourly rows from 845 drifters (1.13M drogued, 1.25M undrogued).

## Method in brief

We build a transition matrix P (Ulam's method). Each row says where material in a cell is one day later, with four absorbing exit states (N, S, E, W).
Iterating P forward gives a release forecast (where material from one point ends up) and relative density (where R gathers or loses material).

Models are compared on held-out drifters, from simple to complex. A model is kept only if it clearly beats the simpler one.

| Model | Description | Status |
|---|---|---|
| M0a | Persistence (no movement) | Done |
| M0b | One displacement distribution used everywhere | Done |
| M1 | Empirical transition matrix | Done |
| M2 | Smoothed matrix, tuned on validation | Done |
| M3 | Other grids and time steps | Planned |
| M4 | Drogue status and season | Planned |
| M5, M6 | GAM and neural network | Optional |

Drifters are split by whole drifter into train, validation and test sets, plus a recent-years test set. The split is frozen in `models/splits.csv`, which everyone must use. The test sets are used once, at the end.

## Repository contents

| File | What it does |
|---|---|
| `make_initial_analysis.py` | Downloads the region from NOAA, cleans it, builds daily transitions, proposal figures |
| `make_poster_extras.py` | Data checks, first held-out check, high-resolution poster figures |
| `make_poster_panels.py` | Separate poster panels |
| `step2_models_M0_M1_M2.py` | Frozen splits, M0a to M2, validation, bootstrap uncertainty, operator export |
| `models/` | Outputs of the first real step 2 run (validation only, not final results) |

## How to run (Google Colab)

1. Start a fresh runtime. In the first cell run `!pip install -q "zarr<3"`. Do not install clouddrift, it breaks Colab.
2. Run `make_initial_analysis.py`. The first run streams the region from NOAA (20-60 min) and caches it in Drive as `MyDrive/DATA3001/initial_analysis/gdp_bmc_extract.parquet`. Later runs take seconds.
3. Copy `models/splits.csv` from this repo into `MyDrive/DATA3001/initial_analysis/models/`.
4. Run `step2_models_M0_M1_M2.py`. Outputs are written to the same `models/` folder in Drive.

To run outside Colab, set the environment variable `DATA3001_BASE` to the folder holding the parquet file.

## Current status (9 November)

- First real run of step 2 is complete (drogued drifters, validation set).
- M2 is best by a small margin. Its 90% areas hold 86% of validation drifters at 30 days (target 90%).
- Still to do: bootstrap test of the model differences, the Markov check, undrogued runs, M3 and M4, then the final test-set run.
