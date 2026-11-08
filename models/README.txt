D2 transport operators for region R (30-50 S, 60-40 W), DATA3001 Group S01-G8
Files
  operator_<model>_<drogue>.npz   P (404 x 404): row i = where material in state i is one day later.
                                  States 0-399 = 1-degree cells, row-major from the north-west corner
                                  (cell = row*20 + col, rows run north to south); 400-403 = left R via
                                  N, S, E, W (absorbing). Also cell_lat, cell_lon (centres), ocean mask, config.
  cell_table_<drogue>.csv         per cell: centre, moves and distinct drifters used, mask flag,
                                  relative density at day 30 with its bootstrap 5-95% band.
Use (Python)
  import numpy as np
  z = np.load("operator_M2_smoothed_drogued.npz"); P = z["P"]
  p = np.zeros(P.shape[0]); p[cell_index] = 1          # release in one cell
  for day in range(30): p = p @ P                      # p is now the day-30 forecast
  p[:400].sum() = still in R, p[400:] = left via N, S, E, W
Cells flagged masked=True had fewer than 10 distinct drifters and should be shown as unsupported.
