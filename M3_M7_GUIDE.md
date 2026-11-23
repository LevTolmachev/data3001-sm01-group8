# Where models M3-M7 live (DATA3001 S01-G8)

Every script imports shared helpers from step2_models_M0_M1_M2.py (data loading, the 1-degree grid,
the frozen drifter splits, M0-M2), so keep all files together in one code/ folder.

| Model | File | What to look for |
|---|---|---|
| M3 grids and time steps (0.5/1/2 degree squares, hexagons, 2-6 day steps, Markov check) | step3_core_results.py (drogued only), step12_pooled_comparisons.py (all drifters, used in the report) | SquareGrid, HexGrid, fit_operator, mpow |
| M4 depth split with borrowing, and season | step4_drogue_season_gam_nn.py (tuning), step5_final_test_and_operator.py (final operators), step9_season_operators.py (seasonal operators) | drogue_weighted, seasonal_operators |
| M5 GAM | step4 (first, Gaussian GAM), step8_stronger_gam_nn.py (final multinomial GAM) | GaussianGAM, MultinomialGAM |
| M6 neural network | step4 (first, DisplacementNet), step8 (final Fourier-feature network) | DisplacementNet, FourierNet, fourier |
| M7 memory (cell + yesterday's direction, 4,004 states) | step10_memory_models.py | sector, arrival_class, Memory, forecast_memory |
| Scoring used by all of them (log score, 90% areas, bootstrap over drifters) | step4_drogue_season_gam_nn.py | forecast, score, drifter_boot, paired |

All of these run as part of `uv run python run_all.py` (see the README).
