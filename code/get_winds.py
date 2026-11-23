"""
DATA3001 S01-G8 - download the extra data for RQ6 (winds) and the drogue question (drifter metadata).
Needs internet access to NOAA. Its outputs are already in ./data, so this only needs re-running to rebuild them.

What it makes (in the data folder, or MyDrive/DATA3001/initial_analysis when run in Google Colab):
  winds_at_drifters.parquet  daily-mean 10 m wind (u10 east, v10 north, m/s) at every drifter's
                             00 UTC position, for the day that starts at that position
  ncep_winds_region.nc       the daily-mean 10 m wind grid over the region plus a margin, 1994-2022 (about 27 MB)
  gdp_metadata.parquet       GDP per-drifter metadata: deployment date and position, drogue-loss date

Source: NCEP/NCAR Reanalysis 1 daily means (NOAA PSL, public, no account needed).
Cite: Kalnay et al. (1996), The NCEP/NCAR 40-Year Reanalysis Project, BAMS 77, 437-471.

Run locally:  uv run python code/get_winds.py      (downloads about 1.5 GB, roughly 15-30 minutes)
Run in Colab: !pip install -q "zarr<3" netCDF4, then paste this file into a cell. Downloads are cached, so a
re-run after a disconnect skips the years already fetched.
"""
import os
import time
import urllib.request
import numpy as np
import pandas as pd
import xarray as xr

try:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/DATA3001/initial_analysis"
    CACHE = "/content/ncep"
except ImportError:                                  # running locally
    BASE = os.environ.get("DATA3001_BASE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    CACHE = os.path.join(BASE, "ncep_cache")
os.makedirs(CACHE, exist_ok=True)

EXTRACT = os.path.join(BASE, "gdp_bmc_extract.parquet")
URLS = ["https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Dailies/surface_gauss/{v}.10m.gauss.{y}.nc",
        "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface_gauss/{v}.10m.gauss.{y}.nc"]
LAT_LO, LAT_HI = -58.0, -22.0          # region R (30-50 S, 60-40 W) plus an 8 degree margin
LON_LO, LON_HI = 292.0, 328.0          # in 0-360 longitude (68 W to 32 W)

# ---- daily drifter positions (00 UTC rows, as in every modelling step)
print("Reading", EXTRACT)
d = pd.read_parquet(EXTRACT, columns=["id", "time", "lat", "lon"])
d["time"] = pd.to_datetime(d["time"]).dt.round("h")
d = d[(d["time"].dt.hour == 0)].drop_duplicates(["id", "time"]).dropna()
d["day"] = d["time"].dt.floor("D")
print(f"{len(d):,} daily drifter positions, {d['day'].dt.year.min()}-{d['day'].dt.year.max()}")


def fetch(v, y):
    fn = os.path.join(CACHE, f"{v}.10m.gauss.{y}.nc")
    if os.path.exists(fn) and os.path.getsize(fn) > 1e6:
        return fn
    last = None
    for u in URLS:
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(u.format(v=v, y=y), fn + ".part")
                os.replace(fn + ".part", fn)
                return fn
            except Exception as e:                      # retry, then try the other address
                last = e
                time.sleep(3)
    raise RuntimeError(f"could not download {v} {y}: {last}")


parts, grids = [], []
t0 = time.time()
for y in sorted(d["day"].dt.year.unique()):
    sub = d[d["day"].dt.year == y].copy()
    fields = {}
    for v in ("uwnd", "vwnd"):
        ds = xr.open_dataset(fetch(v, y))
        da = ds[v].sortby("lat").sel(lat=slice(LAT_LO - 2, LAT_HI + 2), lon=slice(LON_LO - 2, LON_HI + 2)).load()
        da["time"] = pd.to_datetime(da["time"].values).floor("D")
        fields[v] = da
        ds.close()
    pts = dict(time=xr.DataArray(sub["day"].values, dims="p"),
               lat=xr.DataArray(sub["lat"].values, dims="p"),
               lon=xr.DataArray(sub["lon"].values % 360, dims="p"))
    sub["u10"] = fields["uwnd"].sel(time=pts["time"]).interp(lat=pts["lat"], lon=pts["lon"]).values
    sub["v10"] = fields["vwnd"].sel(time=pts["time"]).interp(lat=pts["lat"], lon=pts["lon"]).values
    parts.append(sub)
    grids.append(xr.Dataset({"u10": fields["uwnd"].sel(lat=slice(LAT_LO, LAT_HI), lon=slice(LON_LO, LON_HI)),
                             "v10": fields["vwnd"].sel(lat=slice(LAT_LO, LAT_HI), lon=slice(LON_LO, LON_HI))}))
    print(f"  {y}: {len(sub):,} positions, mean wind speed {np.nanmean(np.hypot(sub['u10'], sub['v10'])):.1f} m/s "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)

w = pd.concat(parts, ignore_index=True)[["id", "time", "lat", "lon", "u10", "v10"]]
w["id"] = w["id"].astype(str)
w.to_parquet(os.path.join(BASE, "winds_at_drifters.parquet"), index=False)
g = xr.concat(grids, dim="time").astype("float32")
g.attrs = {"source": "NCEP/NCAR Reanalysis 1, daily mean 10 m wind (Kalnay et al. 1996)", "units": "m/s"}
g.to_netcdf(os.path.join(BASE, "ncep_winds_region.nc"))
print(f"\nSaved winds_at_drifters.parquet ({len(w):,} rows, {int(w[['u10', 'v10']].isna().any(axis=1).sum())} missing) "
      f"and ncep_winds_region.nc ({g.sizes['time']} days, {g.sizes['lat']} x {g.sizes['lon']} grid) to {BASE}")
print("Mean wind (u10, v10) over all positions: %.2f, %.2f m/s" % (w["u10"].mean(), w["v10"].mean()))

# ---- per-drifter metadata (one row per drifter: where and when it was deployed, when it lost its drogue)
# Used to explain why drogued drifters dominate the south of R and undrogued ones the north (drifter age).
try:
    GDP_URL = "https://noaa-oar-hourly-gdp-pds.s3.amazonaws.com/latest/gdp-v2.01.1.zarr"
    meta = xr.open_dataset(GDP_URL, engine="zarr")
    keep = [v for v in ("id", "deploy_date", "deploy_lat", "deploy_lon", "start_date", "start_lat", "start_lon",
                        "end_date", "end_lat", "end_lon", "drogue_lost_date", "typedeath", "typebuoy", "rowsize")
            if v in meta.variables]
    traj_dim = meta["id"].dims[0]
    m = pd.DataFrame({v: meta[v].values for v in keep if meta[v].dims == (traj_dim,)})
    m["id"] = m["id"].astype(str)
    m = m[m["id"].isin(set(w["id"]))]
    for c in m.columns:
        if m[c].dtype == object and c != "id":
            m[c] = m[c].astype(str)
    m.to_parquet(os.path.join(BASE, "gdp_metadata.parquet"), index=False)
    print(f"Saved gdp_metadata.parquet: {len(m):,} drifters, columns {list(m.columns)}")
except Exception as e:
    print("Metadata step failed (the wind files above are still fine). Please send this message:", repr(e))
