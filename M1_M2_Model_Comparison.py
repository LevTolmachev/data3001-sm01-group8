"""
DATA3001 S01-G8: reproducible validation of M0/M1/M2 in the BMC.

Uses the group's existing NOAA GDP extract; never downloads or changes the source data.

Features:
* Fixed drifter-level train / validation / test / recent-test splits.
* First observed exit identified from hourly positions, even if a drifter re-enters.
* Tracks segmented at gaps and changes of drogue status.
* M0 persistence, M0 pooled-displacement, M1 empirical, M2 smoothed operators.
* M2 parameters selected on validation only; held-out tests require --final-test.
* All-state and within-region 50%/90% coverage, position error, log score, exit Brier.
* Model / grid support diagnostics, plots, reproducible NPZ outputs.

The data may not show a first exit when the observing record is censored or when
there is a gap in the extracted buffered domain; these cases are not scored.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


@dataclass(frozen=True)
class Config:
    lat_min: float = -50.0
    lat_max: float = -30.0
    lon_min: float = -60.0
    lon_max: float = -40.0
    cell_deg: float = 1.0
    seed: int = 42
    drifter_type: str = "drogued"
    horizons: tuple[int, ...] = (1, 3, 7, 14, 30)
    sigmas: tuple[float, ...] = (0.0, 0.75, 1.5)
    alphas: tuple[float, ...] = (0.0, 5.0, 20.0)
    minimum_unique_drifters: int = 10
    recent_test_share: float = 0.10
    release_lat: float = -38.0
    release_lon: float = -54.0
    log_floor: float = 1e-9

    @property
    def nx(self):
        return int(round((self.lon_max - self.lon_min) / self.cell_deg))

    @property
    def ny(self):
        return int(round((self.lat_max - self.lat_min) / self.cell_deg))

    @property
    def nc(self):
        return self.nx * self.ny

    @property
    def ns(self):
        return self.nc + 4

    @property
    def cell_centres(self):
        iy, ix = np.divmod(np.arange(self.nc), self.nx)
        lat = self.lat_max - (iy + 0.5) * self.cell_deg
        lon = self.lon_min + (ix + 0.5) * self.cell_deg
        return lat, lon


EXITS = ("N", "S", "E", "W")


def inside(lat, lon, c: Config):
    return ((np.asarray(lat) >= c.lat_min) & (np.asarray(lat) < c.lat_max)
            & (np.asarray(lon) >= c.lon_min) & (np.asarray(lon) < c.lon_max))


def cell_of(lat, lon, c: Config):
    x = np.floor((np.asarray(lon) - c.lon_min) / c.cell_deg).astype(int)
    y = np.floor((c.lat_max - np.asarray(lat)) / c.cell_deg).astype(int)
    return np.clip(y, 0, c.ny - 1) * c.nx + np.clip(x, 0, c.nx - 1)


def exit_edges(lat, lon, c: Config):
    v = np.stack([np.asarray(lat) - c.lat_max, c.lat_min - np.asarray(lat),
                  np.asarray(lon) - c.lon_max, c.lon_min - np.asarray(lon)], axis=-1)
    return np.argmax(v, axis=-1).astype(np.int16)


def haversine(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def find_input(path=None):
    candidates = []
    if path:
        candidates.append(Path(path).expanduser())
    env = os.getenv('BMC_INPUT')
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [
        Path('/content/drive/MyDrive/DATA3001/initial_analysis/gdp_bmc_extract.parquet'),
        Path('/content/gdp_bmc_extract.parquet'),
        Path('gdp_bmc_extract.parquet'),
        Path('4_Data/gdp_bmc_extract.parquet'),
        Path('DATA3001_S01-G8_files/4_Data/gdp_bmc_extract.parquet'),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        'GDP extract not found. Set --input PATH or BMC_INPUT. Expected '
        '/content/drive/MyDrive/DATA3001/initial_analysis/gdp_bmc_extract.parquet '
        'or a local 4_Data/gdp_bmc_extract.parquet. Do not run NOAA download again.'
    )


def load_prepare(path, c):
    print('Loading existing GDP extract:', path, flush=True)
    try:
        raw = pd.read_parquet(path, columns=['id', 'time', 'lat', 'lon', 'drogued'])
    except ImportError as e:
        raise RuntimeError('Parquet support missing. In Colab run: %pip install -q pyarrow') from e
    if raw.empty:
        raise ValueError('The input dataset is empty.')
    raw = raw.dropna(subset=['id', 'time', 'lat', 'lon', 'drogued']).copy()
    raw['id'] = raw['id'].astype(str)
    raw['time'] = pd.to_datetime(raw['time'], utc=True).dt.tz_localize(None).dt.round('h')
    raw['lat'] = pd.to_numeric(raw['lat'], errors='coerce')
    raw['lon'] = pd.to_numeric(raw['lon'], errors='coerce')
    raw = raw.dropna(subset=['time', 'lat', 'lon'])
    raw = raw[raw['lat'].between(-90, 90) & raw['lon'].between(-180, 180)].copy()
    # Do not silently coerce arbitrary strings ('False' -> True) to bool.
    if not pd.api.types.is_bool_dtype(raw['drogued']):
        allowed = {True, False, 0, 1, 'True', 'False', 'true', 'false', '0', '1'}
        if not raw['drogued'].isin(allowed).all():
            raise ValueError('Unsupported drogued labels; expected boolean flags.')
        raw['drogued'] = raw['drogued'].map({True: True, False: False, 0: False, 1: True,
                                            'True': True, 'False': False,
                                            'true': True, 'false': False, '0': False, '1': True})
    raw['drogued'] = raw['drogued'].astype(bool)
    raw = raw.drop_duplicates(['id','time'], keep='first').sort_values(['id','time']).reset_index(drop=True)
    raw['inR'] = inside(raw['lat'].to_numpy(), raw['lon'].to_numpy(), c)

    # A segment is a continuous hourly record with fixed drogue status.
    new_id = raw['id'].ne(raw['id'].shift())
    gap = raw['time'].diff().ne(pd.Timedelta(hours=1))
    change = raw['drogued'].ne(raw['drogued'].shift())
    raw['segment'] = (new_id | gap | change).cumsum().astype(np.int32)

    # Next out-of-R hourly row within the SAME continuous segment, if any.
    n = len(raw)
    idx = pd.Series(np.where(~raw['inR'].to_numpy(), np.arange(n), n), index=raw.index)
    next_idx = idx.iloc[::-1].groupby(raw['segment'].iloc[::-1], sort=False).cummin().iloc[::-1].to_numpy()
    valid = next_idx < n
    out_time = np.full(n, np.datetime64('NaT', 'ns'), dtype='datetime64[ns]')
    out_edge = np.full(n, -1, dtype=np.int16)
    out_lat = np.full(n, np.nan, dtype=float)
    out_lon = np.full(n, np.nan, dtype=float)
    time_values = raw['time'].to_numpy(dtype='datetime64[ns]')
    out_time[valid] = time_values[next_idx[valid]]
    if valid.any():
        positions = raw.iloc[next_idx[valid]]
        out_lat[valid] = positions['lat'].to_numpy()
        out_lon[valid] = positions['lon'].to_numpy()
        out_edge[valid] = exit_edges(out_lat[valid], out_lon[valid], c)
    raw['first_exit_lat'] = out_lat
    raw['first_exit_lon'] = out_lon
    raw['first_exit_time'] = out_time
    raw['first_exit_edge'] = out_edge
    print(f'Prepared {len(raw):,} hourly observations, {raw.id.nunique()} distinct drifters; '
          f'{int(raw.inR.sum()):,} observations in R; {raw.segment.nunique():,} continuous segments.',
          flush=True)
    return raw


def daily_snapshots(raw, c):
    # Timestamp rounding is already done, so midnight = 00:00 UTC exactly.
    d = raw.loc[raw['time'].dt.hour.eq(0),
                ['id', 'segment', 'time', 'lat', 'lon', 'drogued', 'inR',
                 'first_exit_time', 'first_exit_edge', 'first_exit_lat', 'first_exit_lon']].copy().reset_index(drop=True)
    d['cell'] = -1
    sel = d['inR'].to_numpy()
    d.loc[sel, 'cell'] = cell_of(d.loc[sel,'lat'].to_numpy(), d.loc[sel,'lon'].to_numpy(), c)
    if d.empty or not sel.any():
        raise ValueError('No daily (00 UTC) snapshots inside the region. Inspect timestamps.')
    return d


def make_splits(d, path, c):
    first = d.loc[d.inR].groupby('id')['time'].min().sort_values()
    if path.exists():
        saved = pd.read_csv(path, dtype={'id':str})
        mapping = dict(zip(saved['id'], saved['split']))
        if set(mapping) != set(first.index):
            raise RuntimeError('Frozen split IDs differ from dataset. Use a separate output folder '
                               'for a different extract; do not silently regenerate test IDs.')
        print('Reusing frozen drifter-level splits:', path, flush=True)
        return mapping
    ids = first.index.to_numpy()
    n_recent = min(len(ids)-3, max(1, int(round(len(ids) * c.recent_test_share)))) if len(ids) > 4 else 0
    recent_ids = set(ids[-n_recent:]) if n_recent else set()
    other = np.array([i for i in ids if i not in recent_ids], dtype=object)
    other = np.random.default_rng(c.seed).permutation(other)
    if len(other) < 3:
        raise ValueError('Need at least 3 drifters for separate train/validation/test.')
    ntr = max(1, int(round(0.6 * len(other))))
    nval = max(1, int(round(0.2 * len(other))))
    if ntr+nval >= len(other):
        ntr, nval = len(other)-2, 1
    mp = {i:'train' for i in other[:ntr]}
    mp.update({i:'val' for i in other[ntr:ntr+nval]})
    mp.update({i:'test' for i in other[ntr+nval:]})
    mp.update({i:'recent_test' for i in recent_ids})
    saved = pd.DataFrame({'id':list(mp), 'split':list(mp.values())})
    saved['first_entry_R'] = saved['id'].map(first)
    saved.to_csv(path,index=False)
    print('Created frozen split:', saved['split'].value_counts().to_dict(), flush=True)
    return mp


def candidate_starts(d, id_set, c):
    subset = d[d['id'].isin(id_set) & d['inR']]
    if c.drifter_type == 'drogued':
        subset = subset[subset['drogued']]
    elif c.drifter_type == 'undrogued':
        subset = subset[~subset['drogued']]
    return subset.copy()


def predict_cases(d, starts, k, c):
    """Observed first exit by k days, else exact same-segment location at k days.

    Every censored start is excluded; returns counts so reporting can disclose it.
    """
    if starts.empty:
        return pd.DataFrame(), 0
    s = starts[['id','segment','time','cell','lat','lon',
                'first_exit_time','first_exit_edge']].copy()
    s['target_time'] = s['time'] + pd.Timedelta(days=k)
    t = d[['segment','time','cell','lat','lon','inR']].rename(columns={
        'time':'target_time', 'cell':'target_cell', 'lat':'lat_end',
        'lon':'lon_end', 'inR':'inR_end'})
    s = s.merge(t, on=['segment','target_time'], how='left', validate='many_to_one')
    exited = s['first_exit_time'].notna() & (s['first_exit_time'] <= s['target_time'])
    remaining = (~exited) & s['target_cell'].notna() & s['inR_end'].eq(True)
    known = exited | remaining
    censored = int((~known).sum())
    s = s.loc[known].copy()
    is_exit = exited.loc[known].to_numpy()
    st = np.where(is_exit, c.nc + s['first_exit_edge'].to_numpy().astype(int),
                  s['target_cell'].fillna(0).to_numpy().astype(int))
    s['true_state'] = st
    s['true_exited'] = is_exit
    return s, censored


def observed_daily_moves(d, c):
    starts = d[d['inR']].copy()
    nxt = d.groupby('segment')[['time','cell','lat','lon','inR']].shift(-1)
    starts['next_time'] = nxt.loc[starts.index, 'time']
    starts['next_cell'] = nxt.loc[starts.index, 'cell']
    starts['lat_end'] = nxt.loc[starts.index, 'lat']
    starts['lon_end'] = nxt.loc[starts.index, 'lon']
    valid_end = starts['next_time'].eq(starts['time'] + pd.Timedelta(days=1))
    exits = starts['first_exit_time'].notna() & (
        starts['first_exit_time'] <= starts['time'] + pd.Timedelta(days=1))
    # A first observed exit can be earlier than next midnight; score it as exit.
    complete = exits | valid_end
    starts = starts.loc[complete].copy()
    exits = exits.loc[complete].to_numpy()
    starts.loc[exits,'lat_end'] = starts.loc[exits,'first_exit_lat']
    starts.loc[exits,'lon_end'] = starts.loc[exits,'first_exit_lon']
    starts['to'] = np.where(exits,
        c.nc + starts['first_exit_edge'].to_numpy().astype(int),
        starts['next_cell'].fillna(0).to_numpy().astype(int)).astype(int)
    if (starts['to'] < 0).any() or (starts['to'] >= c.ns).any():
        raise ValueError('Bad destination state in daily moves')
    return starts


def empirical(m, c, w=None):
    P = np.zeros((c.ns,c.ns),dtype=np.float64)
    weights = np.ones(len(m)) if w is None else np.asarray(w)
    np.add.at(P, (m['cell'].to_numpy(dtype=int), m['to'].to_numpy(dtype=int)), weights)
    return normalise(P, c)


def normalise(counts, c):
    P = np.array(counts, dtype=np.float64, copy=True)
    total = P[:c.nc].sum(axis=1)
    ok = total > 0
    P[:c.nc][ok] /= total[ok,None]
    empty = np.flatnonzero(~ok)
    P[empty,empty] = 1.0   # explicit unsupported-cell fallback, flagged separately
    P[c.nc:,:] = 0.0
    P[c.nc:,c.nc:] = np.eye(4)
    return P


class OffsetModel:
    """Displacement distribution projected from every training-supported start cell."""
    def __init__(self, m, supported, c):
        self.c, self.supported = c, supported
        if m.empty:
            raise ValueError('No training transitions for requested drogue type.')
        fy, fx = np.divmod(m['cell'].to_numpy(dtype=int), c.nx)
        ty = np.floor((c.lat_max - m['lat_end'].to_numpy()) / c.cell_deg).astype(int)
        tx = np.floor((m['lon_end'].to_numpy() - c.lon_min) / c.cell_deg).astype(int)
        dy, dx = ty - fy, tx - fx
        self.r = int(min(max(c.nx,c.ny), max(1,np.max(np.abs(dy)),np.max(np.abs(dx)))))
        if np.max(np.abs(dy)) > self.r or np.max(np.abs(dx)) > self.r:
            print('WARNING: some extreme daily displacements exceed the offset window.', flush=True)
        off = np.arange(-self.r,self.r+1)
        DY,DX = np.meshgrid(off,off,indexing='ij')
        self.dy,self.dx = DY.ravel(),DX.ravel()
        self.K = len(self.dy)
        self.start = m['cell'].to_numpy(dtype=np.int32)
        self.k = ((dy+self.r)*(2*self.r+1)+(dx+self.r)).astype(np.int32)
        self.k[(np.abs(dy)>self.r)|(np.abs(dx)>self.r)] = -1
        sy,sx = np.divmod(np.arange(c.nc),c.nx)
        ey = sy[:,None]+self.dy[None,:]
        ex = sx[:,None]+self.dx[None,:]
        is_inside = (ey>=0)&(ey<c.ny)&(ex>=0)&(ex<c.nx)
        tar = np.full((c.nc,self.K), -1, dtype=np.int32)
        dest = np.clip(ey,0,c.ny-1)*c.nx + np.clip(ex,0,c.nx-1)
        tar[is_inside & supported[dest]] = dest[is_inside & supported[dest]]
        violation = np.stack([-ey,ey-(c.ny-1),ex-(c.nx-1),-ex],axis=-1)
        edge = np.argmax(violation,axis=-1)
        tar[~is_inside] = c.nc+edge[~is_inside]
        tar[~supported,:] = -1
        self.dest = tar

    def histogram(self, weights=None):
        w = np.ones(len(self.start)) if weights is None else np.asarray(weights)
        ok = self.k>=0
        H = np.zeros((self.c.nc,self.K), dtype=float)
        np.add.at(H,(self.start[ok],self.k[ok]),w[ok])
        return H

    def counts(self,H):
        rows = np.broadcast_to(np.arange(self.c.nc)[:,None],self.dest.shape)
        good = (self.dest>=0)&(H>0)
        indices = rows[good]*self.c.ns+self.dest[good]
        v = np.bincount(indices,weights=H[good],minlength=self.c.ns**2)
        return v.reshape(self.c.ns,self.c.ns)

    def pooled(self, weights=None):
        H = self.histogram(weights)
        pooled = H.sum(axis=0)
        if pooled.sum()<=0:
            raise ValueError('No offsets for pooled baseline.')
        C = self.counts(np.broadcast_to(pooled/pooled.sum(),H.shape))
        return normalise(C,self.c)

    def smoothed(self,sigma,alpha,weights=None,P0=None):
        H = self.histogram(weights)
        if sigma>0:
            H = gaussian_filter(H.reshape(self.c.ny,self.c.nx,self.K),
                  sigma=(float(sigma),float(sigma),0.0),mode='constant',cval=0.0,
                  truncate=2.0).reshape(self.c.nc,self.K)
        C = self.counts(H)
        if alpha:
            prior = self.pooled(weights) if P0 is None else P0
            C[:self.c.nc] += float(alpha)*prior[:self.c.nc]*self.supported[:,None]
        return normalise(C,self.c)


def predicted_probabilities(P,h,case_cells):
    return np.linalg.matrix_power(P,int(h))[case_cells]


def cover(probs,truth,q):
    order=np.argsort(-probs,axis=1)
    cumul=np.cumsum(np.take_along_axis(probs,order,axis=1),axis=1)
    size=np.sum(cumul<q-1e-12,axis=1)+1
    rank=np.argmax(order==truth[:,None],axis=1)
    return (rank<size),size


def evaluate(P,cases,c,model,split_name,h,censored,unsupported_starts=0):
    if cases.empty:
        return None
    probs=predicted_probabilities(P,h,cases['cell'].to_numpy(dtype=int))
    truth=cases['true_state'].to_numpy(dtype=int)
    hit50,_=cover(probs,truth,0.5)
    hit90,size90=cover(probs,truth,0.9)
    mass=probs[:,:c.nc].sum(axis=1)
    within=~cases['true_exited'].to_numpy() & (mass>1e-12)
    lat_cent,lon_cent=c.cell_centres
    weighted_lat=(probs[:,:c.nc]*lat_cent).sum(axis=1)/np.maximum(mass,1e-12)
    weighted_lon=(probs[:,:c.nc]*lon_cent).sum(axis=1)/np.maximum(mass,1e-12)
    err=haversine(weighted_lat[within],weighted_lon[within],
                  cases['lat_end'].to_numpy()[within],cases['lon_end'].to_numpy()[within])
    baseline=haversine(cases['lat'].to_numpy()[within],cases['lon'].to_numpy()[within],
                       cases['lat_end'].to_numpy()[within],cases['lon_end'].to_numpy()[within])
    # Conditional spatial coverage excludes exit states and conditions on retention.
    conditional=probs[within,:c.nc] / mass[within,None]
    inside_truth=truth[within]
    spatial50,spatial_size50=cover(conditional,inside_truth,0.5)
    spatial90,spatial_size90=cover(conditional,inside_truth,0.9)
    fate_probs=np.column_stack([mass,probs[:,c.nc:]])
    fate_target=np.where(truth<c.nc,0,truth-c.nc+1)
    y=np.eye(5)[fate_target]
    brier=np.sum((fate_probs-y)**2,axis=1)
    true_probability=probs[np.arange(len(truth)),truth]
    drifter_n=cases['id'].nunique()
    return dict(split=split_name,model=model,horizon_days=int(h),cases=len(truth),
        distinct_drifters=int(drifter_n),censored_starts=int(censored),
        unsupported_starts=int(unsupported_starts),observed_first_exits=int((truth>=c.nc).sum()),
        coverage50_all_pct=100*hit50.mean(),coverage90_all_pct=100*hit90.mean(),
        median_states_in_90pct_set=float(np.median(size90)),
        coverage50_position_pct=100*spatial50.mean() if within.any() else np.nan,
        coverage90_position_pct=100*spatial90.mean() if within.any() else np.nan,
        median_cells_in_90pct_position_area=float(np.median(spatial_size90)) if within.any() else np.nan,
        conditional_position_cases=int(within.sum()),
        # For the persistence baseline, use the actual release coordinates rather
        # than the centre of the containing 1-degree grid cell.
        median_centre_error_km=(float(np.median(baseline)) if model=='M0a persistence'
                                else float(np.median(err))) if len(err) else np.nan,
        median_persistence_error_km=float(np.median(baseline)) if len(baseline) else np.nan,
        mean_log_score=float(np.log(np.maximum(true_probability,c.log_floor)).mean()),
        mean_fate_brier=float(np.mean(brier)))


def check_matrix(P,c):
    assert P.shape==(c.ns,c.ns)
    assert np.isfinite(P).all() and P.min()>=-1e-12
    np.testing.assert_allclose(P.sum(axis=1),1.0,atol=1e-8)
    np.testing.assert_allclose(P[c.nc:,c.nc:],np.eye(4),atol=1e-8)


def score_models(models,d,ids,c,split,supported=None):
    starts=candidate_starts(d,ids,c)
    metrics=[]
    for k in c.horizons:
        known,ncensored=predict_cases(d,starts,int(k),c)
        n_unsupported=0
        if supported is not None and not known.empty:
            keep=supported[known['cell'].to_numpy(dtype=int)]
            n_unsupported=int((~keep).sum())
            known=known.loc[keep].copy()
        for name,P in models.items():
            row=evaluate(P,known,c,name,split,int(k),ncensored,n_unsupported)
            if row:
                metrics.append(row)
    return pd.DataFrame(metrics)


def save_plots(out,metrics,c,P_m2,support):
    fig,axs=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    colors={'M0a persistence':'#808080','M0b displacement':'#555c70',
            'M1 empirical':'#2376bc','M2 smoothed':'#e38b26'}
    for name,group in metrics.groupby('model',sort=False):
        if name=='M0a persistence':
            # The persistence baseline has no predicted spatial cloud beyond its start.
            pass
        col=colors.get(name,'#555555')
        group=group.sort_values('horizon_days')
        axs[0].plot(group.horizon_days,group.median_centre_error_km,'o-',color=col,label=name)
        axs[1].plot(group.horizon_days,group.coverage90_all_pct,'o-',color=col,label=name)
        axs[2].plot(group.horizon_days,group.mean_log_score,'o-',color=col,label=name)
    axs[0].set_title('Conditional median location error')
    axs[0].set_ylabel('Median error (km)')
    axs[1].axhline(90,color='black',ls='--',lw=1,label='90% nominal')
    axs[1].set_ylim(0,100)
    axs[1].set_title('All-state 90% coverage')
    axs[1].set_ylabel('Observed coverage (%)')
    axs[2].set_title('Log score (higher is better)')
    axs[2].set_ylabel('Mean log probability')
    for ax in axs:
        ax.set_xlabel('Prediction horizon (days)')
        ax.grid(alpha=.2)
    axs[2].legend(fontsize=7,loc='best')
    fig.savefig(out/'fig_model_validation.png',dpi=230,bbox_inches='tight')
    plt.close(fig)

    # Prognostic maps are illustrative model outputs, not validation results.
    start=int(cell_of(np.array([c.release_lat]),np.array([c.release_lon]),c)[0])
    p=np.linalg.matrix_power(P_m2,30)[start]
    area=np.cos(np.deg2rad(c.cell_centres[0]))*support
    if area.sum()<=0:
        return
    u=np.zeros(c.ns)
    u[:c.nc]=area/area.sum()
    final=u @ np.linalg.matrix_power(P_m2,30)
    relative=np.divide(final[:c.nc]/max(final[:c.nc].sum(),1e-12),u[:c.nc],
                       out=np.full(c.nc,np.nan),where=u[:c.nc]>0)
    pl=p[:c.nc].copy()
    pl[(pl<=0)|~support]=np.nan
    relative[~support]=np.nan
    fig,axs=plt.subplots(1,2,figsize=(9.5,4.1),layout='constrained')
    ex=[c.lon_min,c.lon_max,c.lat_min,c.lat_max]
    im=axs[0].imshow(pl.reshape(c.ny,c.nx),origin='upper',extent=ex,
                     norm=LogNorm(vmin=1e-5,vmax=max(1e-4,np.nanmax(pl))),cmap='Blues')
    axs[0].plot(c.release_lon,c.release_lat,'k*',ms=9)
    axs[0].set_title('M2 release, day 30 (unvalidated map)')
    fig.colorbar(im,ax=axs[0],shrink=.75,label='Probability per cell')
    im2=axs[1].imshow(relative.reshape(c.ny,c.nx),origin='upper',extent=ex,
                      cmap='RdBu_r',norm=LogNorm(vmin=.1,vmax=10))
    axs[1].set_title('M2 relative density, day 30 (unvalidated map)')
    fig.colorbar(im2,ax=axs[1],shrink=.75,label='Relative density (>1 accumulates)')
    for ax in axs:
        ax.set_xlabel('Longitude (°E)');ax.set_ylabel('Latitude (°N)')
    fig.savefig(out/'fig_M2_release_density.png',dpi=230,bbox_inches='tight')
    plt.close(fig)


def bootstrap_scalar(m,support,c,sigma,alpha,nboot,seed):
    if nboot<=0:
        return pd.DataFrame()
    ids=m['id'].unique()
    if not len(ids):
        return pd.DataFrame()
    rowix=pd.Index(ids).get_indexer(m['id'])
    rng=np.random.default_rng(seed)
    off=OffsetModel(m,support,c)
    start=int(cell_of(np.array([c.release_lat]),np.array([c.release_lon]),c)[0])
    allres=[]
    for b in range(nboot):
        count=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids))
        w=count[rowix]
        P=off.smoothed(sigma,alpha,weights=w)
        p=np.linalg.matrix_power(P,30)[start]
        allres.append({'bootstrap':b,'release_still_R_day30':float(p[:c.nc].sum()),
                       **{f'exit_{e}_day30':float(p[c.nc+j]) for j,e in enumerate(EXITS)}})
    return pd.DataFrame(allres)


def export_operator(out,name,P,c,support,params):
    lat,lon=c.cell_centres
    np.savez_compressed(out/f'operator_{name}.npz',P=P,cell_lat=lat,cell_lon=lon,
                        supported=support,states=np.array([f'cell_{i}' for i in range(c.nc)]
                        +[f'exit_{e}' for e in EXITS]),
                        config=json.dumps({'region':[c.lat_min,c.lat_max,c.lon_min,c.lon_max],
                                           'cell_deg':c.cell_deg,'dt_days':1,
                                           'drifter_type':c.drifter_type,**params}))


def run_pipeline(input_path=None,output=None,quick=False,bootstrap=0,final_test=False,
                 drifter_type='drogued'):
    c=Config(drifter_type=drifter_type)
    path=find_input(input_path)
    out=Path(output).expanduser() if output else path.parent/'models_improved'
    out.mkdir(parents=True,exist_ok=True)
    print('Saving all results to:',out,flush=True)
    raw=load_prepare(path,c)
    d=daily_snapshots(raw,c)
    split=make_splits(d,out/'splits.csv',c)
    d['split']=d['id'].map(split)
    moves=observed_daily_moves(d,c)
    train=moves[moves['split']=='train']
    if c.drifter_type=='drogued':
        train=train[train['drogued']]
    elif c.drifter_type=='undrogued':
        train=train[~train['drogued']]
    train=train.copy()
    if train.empty:
        raise ValueError('No training transitions for selected drifter type.')
    supported=np.zeros(c.nc,dtype=bool)
    supported[np.unique(train['cell'].to_numpy(dtype=int))]=True
    starts= d[(d['split']=='val') & d['inR']]
    if c.drifter_type=='drogued': starts=starts[starts.drogued]
    if c.drifter_type=='undrogued': starts=starts[~starts.drogued]
    print(f'Training moves: {len(train):,}; training-support cells: {supported.sum():,} / {c.nc}',flush=True)
    off=OffsetModel(train,supported,c)
    P0a=np.eye(c.ns)
    P0b=off.pooled()
    P1=empirical(train,c)
    for P in (P0a,P0b,P1): check_matrix(P,c)

    sigmas=(0.0,0.75) if quick else c.sigmas
    alphas=(0.0,5.0) if quick else c.alphas
    cases={k:predict_cases(d,starts,k,c) for k in (7,30)}
    tune=[]
    for sigma in sigmas:
        for alpha in alphas:
            P=off.smoothed(sigma,alpha,P0=P0b)
            check_matrix(P,c)
            log_scores=[]
            for k,(t,_) in cases.items():
                if t.empty:continue
                t=t[supported[t['cell'].to_numpy(dtype=int)]]
                if t.empty:continue
                p=predicted_probabilities(P,k,t['cell'].to_numpy(dtype=int))
                log_scores.append(float(np.log(np.maximum(p[np.arange(len(t)),t['true_state'].to_numpy(dtype=int)],c.log_floor)).mean()))
            if log_scores:
                tune.append({'sigma':sigma,'alpha':alpha,'validation_mean_log_score':float(np.mean(log_scores))})
            print(f'  M2 sigma={sigma}, alpha={alpha}: {np.mean(log_scores) if log_scores else float("nan"):.4f}',flush=True)
    if not tune:
        raise ValueError('No evaluable validation cases at 7 or 30 days. Inspect data/splits.')
    tuning=pd.DataFrame(tune).sort_values('validation_mean_log_score',ascending=False)
    tuning.to_csv(out/'m2_tuning.csv',index=False)
    best=tuning.iloc[0]
    sigma,alpha=float(best.sigma),float(best.alpha)
    P2=off.smoothed(sigma,alpha,P0=P0b)
    check_matrix(P2,c)
    models={'M0a persistence':P0a,'M0b displacement':P0b,
            'M1 empirical':P1,'M2 smoothed':P2}
    val_ids=set(d.loc[d.split=='val','id'])
    validation=score_models(models,d,val_ids,c,'validation',supported=supported)
    validation.to_csv(out/'validation_metrics.csv',index=False)
    if validation.empty:
        raise ValueError('No validation observations available for the requested horizons.')
    print('\nM2 selected on validation: sigma=',sigma,'alpha=',alpha,flush=True)
    cols=['model','horizon_days','cases','coverage90_all_pct','coverage90_position_pct',
          'median_centre_error_km','mean_log_score','mean_fate_brier']
    print(validation[cols].round(3).to_string(index=False),flush=True)
    save_plots(out,validation,c,P2,supported)

    counts = train.groupby('cell')['id'].agg(['size','nunique']).reindex(range(c.nc)).fillna(0)
    lat,lon=c.cell_centres
    pd.DataFrame({'cell':np.arange(c.nc),'lat_centre':lat,'lon_centre':lon,
                  'training_moves':counts['size'].astype(int).to_numpy(),
                  'distinct_training_drifters':counts['nunique'].astype(int).to_numpy(),
                  'supported':supported,
                  'low_support':counts['nunique'].to_numpy()<c.minimum_unique_drifters}).to_csv(out/'cell_table.csv',index=False)
    export_operator(out,'M1',P1,c,supported,{})
    export_operator(out,'M2',P2,c,supported,{'sigma':sigma,'alpha':alpha})

    boot=bootstrap_scalar(train,supported,c,sigma,alpha,int(bootstrap),c.seed)
    if not boot.empty:
        boot.to_csv(out/'bootstrap_M2.csv',index=False)
        boot.drop(columns='bootstrap').quantile([.05,.50,.95]).to_csv(out/'bootstrap_M2_quantiles.csv')

    if final_test:
        # Refit after tuning: TEST and RECENT_TEST have not influenced selection.
        tv=moves[moves.split.isin(['train','val'])].copy()
        if c.drifter_type=='drogued': tv=tv[tv.drogued]
        elif c.drifter_type=='undrogued': tv=tv[~tv.drogued]
        supported_tv=np.zeros(c.nc,dtype=bool)
        supported_tv[tv.cell.unique().astype(int)]=True
        final_off=OffsetModel(tv,supported_tv,c)
        p0=final_off.pooled()
        final_models={'M0a persistence':P0a,'M0b displacement':p0,
                      'M1 empirical':empirical(tv,c),
                      'M2 smoothed':final_off.smoothed(sigma,alpha,P0=p0)}
        for s in ('test','recent_test'):
            ids=set(d.loc[d.split==s,'id'])
            met=score_models(final_models,d,ids,c,s,supported=supported_tv)
            met.to_csv(out/f'{s}_metrics.csv',index=False)
            print(f'\nFINAL {s} metrics:',len(met),'rows',flush=True)
    summary={'source':str(path),'output':str(out),'training_move_count':int(len(train)),
             'n_train_drifters':int(d.loc[d.split=='train','id'].nunique()),
             'n_validation_drifters':int(d.loc[d.split=='val','id'].nunique()),
             'best_M2_sigma':sigma,'best_M2_alpha':alpha,
             'tests_run':bool(final_test),'bootstrap_replicates':int(bootstrap),
             'notes':['Independent final test sets must be used only once after hyperparameters are frozen.',
                      'First exits are defined from available hourly observations within the buffered extract.',
                      'Unsupported cells use self-loops for matrix arithmetic; these are NOT validated forecasts.',
                      '90% all-state sets include four absorbing exits; position-only sets condition on retention.',
                      'Non-random censoring and correlated starts from the same drifter remain limitations.',
                      'Figures are model predictions, not independently validated physical material forecasts.']}
    (out/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nDone. Results in',out,flush=True)
    return validation,summary


def main():
    parser=argparse.ArgumentParser(description='BMC M0/M1/M2 validation, using existing GDP extract.')
    parser.add_argument('--input',default=None,help='Existing gdp_bmc_extract.parquet (auto-detected in Colab Drive)')
    parser.add_argument('--output',default=None,help='Results folder; defaults to models_improved next to input')
    parser.add_argument('--quick',action='store_true',help='Smaller M2 tuning grid for faster first run')
    parser.add_argument('--bootstrap',type=int,default=0,help='Optional number of whole-drifter bootstrap resamples')
    parser.add_argument('--final-test',action='store_true',help='ONLY after freezing all model choices')
    parser.add_argument('--drifter-type',choices=['drogued','undrogued','all'],default='drogued')
    args=parser.parse_args()
    run_pipeline(args.input,args.output,args.quick,args.bootstrap,args.final_test,args.drifter_type)


if __name__=='__main__':
    main()
