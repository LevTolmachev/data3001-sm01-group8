#!/usr/bin/env python3
"""
DATA3001 Group S01-G8 - M3 v2: support-adaptive grid and time-step transport model
================================================================================

Research purpose
  Develop an M3 transition model for the Brazil-Malvinas Confluence (BMC), and
  compare it fairly against the group's ORIGINAL, FROZEN M2 operator.
  This is a GitHub/local-Python workflow: no Colab, Google Drive mount or data
  download is performed by this script.

What this script does (in order)
  1. Loads the existing GDP extract and the ORIGINAL project assets:
       private_data/gdp_bmc_extract.parquet
       models/splits.csv
       models/operator_M2_smoothed_drogued.npz
     The input must contain id, time, lat, lon and drogued columns.

  2. Reuses the FROZEN drifter-level train / val / test / recent_test split.
     It NEVER randomly re-splits drifters. The M2 operator is never refitted.

  3. Builds candidate M3 transition operators from TRAINING drifters only:
       grid resolution: 1 degree or 2 degrees (default), plus 0.5 degrees (--full)
       transition lag: 1 or 2 days
       Gaussian spatial pooling width (sigma): 0.75 or 1.5 cells
       support-adaptive prior strength (kappa): 5 or 20 pseudo-observations
     Each model includes absorbing first-exit states: N, S, E and W.
     Track segments end at observation gaps or changes in drogue status.

  4. Tunes M3 on VALIDATION drifters, using mean log score at 14 and 30 days.
     Uses the SAME forecast origin, observed outcome and common 1-degree
     404-state scoring space for M2 and M3. Other horizons are diagnostics.
     When scores are close, preference is given to the simpler candidate.

  5. Evaluates the selected M3 against the frozen M2 on matched cases:
       - mean log score (higher is better)
       - all-state 90% coverage (closer to 90% is better)
       - within-region 90% position coverage
       - median positional error in km (lower is better)
       - five-state fate Brier score (lower is better)
     Uses whole-drifter paired bootstrap intervals for comparison.

  6. Saves a selected sparse M3 transition operator, its projection to the
     common 1-degree outcome space, validation tables, diagnostic JSON files,
     and a SHA-256 manifest for reproducible model selection.
     Includes a TRAINING-only Chapman-Kolmogorov time-step consistency check.

  7. FINAL TEST / RECENT TEST: runs only with a frozen selection.json and
     the explicit --confirm-final-test flag. Matching hashes of source code,
     GDP input, frozen split and M2 baseline are required. Never use the
     independent holdouts to choose or revise M3 hyperparameters.

How to run from a local clone of your GitHub repository
  Install dependencies (Python 3.11/3.12 recommended):
    python -m pip install -r requirements-m3-dev.txt

  Validate software without using real test drifters:
    python -m pytest -q tests/test_m3.py

  Development search (16 M3 configurations):
    python m3_grid_timestep.py tune --input private_data/gdp_bmc_extract.parquet --splits models/splits.csv --m2 models/operator_M2_smoothed_drogued.npz --out results/m3

  Full search, including 0.5-degree cells (24 configurations):
    python m3_grid_timestep.py tune --input private_data/gdp_bmc_extract.parquet --splits models/splits.csv --m2 models/operator_M2_smoothed_drogued.npz --out results/m3_full --full

  Plot and interpret the selected M3 vs M2 validation results:
    python compare_m2_m3.py --results results/m3_full --split validation

Main files written into --out during tuning
    candidate_ranking.csv           All M3 candidate validation scores
    selection.json                  Selected M3 settings and source/input hashes
    operator_M3_selected.npz        Selected sparse M3 transport operator
    M3_projection_to_1degree.npz    Map M3 states into the 404 reference states
    validation_comparison.csv       Matched M2/M3 evaluation summary
    validation_case_scores.csv      Individual scored validation origins
    paired_bootstrap_validation.csv Whole-drifter paired uncertainty estimates
    M3_training_diagnostics.json    Grid support and training diagnostics
    M3_CK_diagnostics.json          Time-step consistency check

Important research limitations
  - This code does not itself prove that M3 outperforms M2. Results require
    evaluation on the group's genuine historical GDP observations.
  - Original historical M2 CSV scores must NOT be compared directly with
    these scores: both models are re-scored here under a shared protocol.
  - When a 2-degree forecast is projected to 1-degree cells, subcell
    probabilities are allocated using training-only occupancy estimates;
    this does not add physical resolution to the 2-degree model.
  - This script implements aligned square grids, not hexagonal grids.
  - DO NOT commit raw GDP Parquet files or independent-test outputs to public
    GitHub without confirming the team's data/research-sharing policy.
  - After an M3 selection is frozen, ANY edit to this source file (including
    comments) changes its SHA-256 hash and blocks that selection's final test.
    Finish documentation BEFORE rerunning tune and freezing the model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter

EXITS = ("N", "S", "E", "W")
REF_LAT_MIN, REF_LAT_MAX = -50., -30.
REF_LON_MIN, REF_LON_MAX = -60., -40.
REF_N = 400
REF_NS = 404
M3_PIPELINE_VERSION = '2.1.0'
LOG_FLOOR = 1e-6   # Same as group's original step2_models_M0_M1_M2.py
SELECT_HORIZONS = (14, 30)  # divisible by both 1- and 2-day transitions
REPORT_HORIZONS = (14, 30, 60, 90)


@dataclass(frozen=True)
class Grid:
    degree: float

    def __post_init__(self):
        if self.degree not in (0.5, 1., 2.):
            raise ValueError('Only aligned 0.5°, 1°, and 2° square grids supported.')

    @property
    def nx(self):
        return int(round(20 / self.degree))

    @property
    def ny(self):
        return self.nx

    @property
    def nc(self):
        return self.nx * self.ny

    @property
    def ns(self):
        return self.nc + 4

    def indices(self, lat, lon):
        lat, lon = np.asarray(lat), np.asarray(lon)
        ix = np.floor((lon - REF_LON_MIN) / self.degree).astype(int)
        iy = np.floor((REF_LAT_MAX - lat) / self.degree).astype(int)
        return np.clip(iy, 0, self.ny-1) * self.nx + np.clip(ix, 0, self.nx-1)

    def centres(self):
        i = np.arange(self.nc)
        iy, ix = divmod(i, self.nx)
        return REF_LAT_MAX - (iy + .5) * self.degree, REF_LON_MIN + (ix+.5)*self.degree


@dataclass(frozen=True)
class Choice:
    degree: float
    lag: int
    sigma: float
    kappa: float

    @property
    def name(self):
        return f"M3_g{self.degree:g}_dt{self.lag}_s{self.sigma:g}_k{self.kappa:g}"


def sha256_file(path: Path):
    d = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            d.update(block)
    return d.hexdigest()


def inside(lat, lon):
    return ((np.asarray(lat) >= REF_LAT_MIN) & (np.asarray(lat) < REF_LAT_MAX)
            & (np.asarray(lon) >= REF_LON_MIN) & (np.asarray(lon) < REF_LON_MAX))


def exit_edge(lat, lon):
    lat, lon = np.asarray(lat), np.asarray(lon)
    violations = np.stack([lat - REF_LAT_MAX, REF_LAT_MIN - lat,
                           lon - REF_LON_MAX, REF_LON_MIN - lon], axis=-1)
    return np.argmax(violations, axis=-1)


def load_data(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing input {path}; use --input to locate existing GDP extract.')
    if path.suffix.lower() in ('.parquet', '.pq'):
        try:
            df = pd.read_parquet(path, columns=['id', 'time', 'lat', 'lon', 'drogued'])
        except ImportError as e:
            raise RuntimeError('Parquet engine missing. pip install -r requirements-m3.txt') from e
    elif path.suffix.lower() == '.csv':
        df = pd.read_csv(path, usecols=['id', 'time', 'lat', 'lon', 'drogued'])
    else:
        raise ValueError('Only CSV (for tests) and Parquet (real data) supported.')
    # Do not accidentally turn a missing identifier into the literal "nan".
    df = df.dropna(subset=['id','time','lat','lon','drogued']).copy()
    df['id'] = df['id'].astype(str)
    df = df.loc[df['id'].str.strip().ne('')].copy()
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce').dt.tz_convert(None).dt.round('h')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    def normalize_flag(v):
        if v in (True, 1, '1', 'True', 'true', 'TRUE'): return True
        if v in (False, 0, '0', 'False', 'false', 'FALSE'): return False
        raise ValueError(f'Invalid drogue value: {v!r}')
    df['drogued'] = df['drogued'].map(normalize_flag).astype(bool)
    df = df.dropna(subset=['id','time','lat','lon'])
    df = df.loc[df.lat.between(-90,90) & df.lon.between(-180,180)]
    df = df.drop_duplicates(['id','time'], keep='first').sort_values(['id','time']).reset_index(drop=True)
    if df.empty: raise ValueError('No valid GDP records.')
    # Separate tracks on any non-consecutive hour, ID change or drogue change.
    boundaries = (df.id.ne(df.id.shift()) |
                  df.time.diff().ne(pd.Timedelta(hours=1)) |
                  df.drogued.ne(df.drogued.shift()))
    df['segment'] = boundaries.cumsum().astype('int32')
    df['in_R'] = inside(df.lat.to_numpy(), df.lon.to_numpy())
    # Within a continuous segment, locate the FIRST subsequent outside-R record.
    # Scoring this exit first avoids false in-R truths for re-entering drifters.
    n = len(df)
    first_out = np.full(n, -1, dtype=np.int64)
    segment_ids = df.segment.to_numpy()
    outside = ~df.in_R.to_numpy()
    start = np.r_[0, np.flatnonzero(segment_ids[1:] != segment_ids[:-1])+1]
    stops = np.r_[start[1:], n]
    for lo, hi in zip(start, stops):
        idx = np.where(outside[lo:hi], np.arange(lo,hi), n)
        closest = np.minimum.accumulate(idx[::-1])[::-1]
        first_out[lo:hi] = np.where(closest<n, closest, -1)
    has = first_out>=0
    df['exit_time'] = pd.NaT
    df['exit_edge'] = -1
    if has.any():
        dest = df.iloc[first_out[has]]
        df.loc[has, 'exit_time'] = dest.time.to_numpy()
        df.loc[has, 'exit_edge'] = exit_edge(dest.lat.to_numpy(),dest.lon.to_numpy()).astype(int)
    return df


def load_splits(path: Path, df):
    if not path.is_file(): raise FileNotFoundError(f'Frozen splits missing: {path}')
    s = pd.read_csv(path, dtype={'id':str})
    if not {'id','split'}.issubset(s.columns): raise ValueError('splits needs id,split columns.')
    if s.id.duplicated().any(): raise ValueError('Duplicate IDs in splits.')
    allowed = {'train','val','test','recent_test'}
    if not set(s['split']).issubset(allowed): raise ValueError('Unexpected split label.')
    in_ids = set(df.loc[df.in_R & df.time.dt.hour.eq(0), 'id'].unique())
    # A loaded extract can contain buffer-only IDs and very short non-midnight visits.
    # Original Step 2 defined the frozen eligible IDs from midnight snapshots.
    missing = in_ids - set(s.id)
    if missing:
        raise ValueError(f'{len(missing)} in-region IDs missing from frozen splits. Wrong extract/splits?')
    if not {'train','val','test','recent_test'}.issubset(set(s['split'])):
        raise ValueError('All train/val/test/recent_test splits required.')
    return dict(zip(s.id, s['split']))


def load_baseline(path: Path):
    z = np.load(path, allow_pickle=False)
    if 'P' not in z.files: raise ValueError('Baseline NPZ lacks P.')
    P = np.asarray(z['P'], dtype=float)
    if P.shape != (REF_NS, REF_NS):
        raise ValueError('M2 must have the original 404x404 1° operator.')
    if not np.isfinite(P).all() or P.min() < -1e-9 or not np.allclose(P.sum(axis=1),1,atol=1e-7):
        raise ValueError('Invalid probabilities in frozen M2 operator.')
    if not np.allclose(P[REF_N:,REF_N:],np.eye(4),atol=1e-7):
        raise ValueError('M2 exit states must be absorbing.')
    cfg = json.loads(str(z['config'])) if 'config' in z.files else {}
    if cfg:
        if float(cfg.get('cell_deg',1)) != 1. or int(cfg.get('dt_days',1)) != 1:
            raise ValueError('M2 configuration differs from original frozen baseline.')
        if cfg.get('drogue_set','drogued') != 'drogued':
            raise ValueError('M2 comparison requires drogued operator.')
    ocean = np.asarray(z['ocean'], dtype=bool) if 'ocean' in z.files else np.ones(REF_N,bool)
    return sparse.csr_matrix(P), ocean


def start_rows(df, mapping, group):
    daily = df.loc[(df.time.dt.hour == 0) & (df.time.dt.minute == 0) & df.in_R & df.drogued,
                   ['id','segment','time','lat','lon','exit_time','exit_edge']].copy()
    daily['split'] = daily.id.map(mapping)
    return daily[daily.split.eq(group)].drop(columns='split').reset_index(drop=True)


def outcomes(df, starts, lag_days):
    """One set of EXACT-CLOCK outcomes, even after first exit; censor missing ends."""
    right = df.loc[(df.time.dt.hour==0) & (df.time.dt.minute==0),
                   ['id','segment','time','lat','lon']].rename(
                       columns={'time':'end_time', 'lat':'end_lat', 'lon':'end_lon'})
    left = starts.copy()
    left['end_time'] = left.time + pd.Timedelta(days=lag_days)
    merged = left.merge(right, on=['id','segment','end_time'],how='left',sort=False)
    ex = merged.exit_time.notna() & (merged.exit_time <= merged.end_time)
    observed = ex | merged.end_lat.notna()
    m = merged.loc[observed].copy()
    if m.empty:
        return m.assign(truth_exit=pd.Series(dtype='int8'))
    ex = m.exit_time.notna() & (m.exit_time <= m.end_time)
    m['truth_exit'] = np.where(ex,m.exit_edge.astype(int),-1).astype('int8')
    return m.reset_index(drop=True)


def training_counts(df, mapping, lag):
    st = start_rows(df,mapping,'train')
    mov = outcomes(df,st,lag)
    if len(mov)<10: raise ValueError(f'Too few valid training moves for lag {lag}: {len(mov)}')
    return mov


def ref_truth(cases):
    states = np.empty(len(cases),dtype=int)
    escaped = cases.truth_exit.to_numpy()>=0
    states[escaped] = REF_N + cases.loc[escaped,'truth_exit'].to_numpy(dtype=int)
    st = Grid(1.0).indices(cases.loc[~escaped,'end_lat'],cases.loc[~escaped,'end_lon'])
    states[~escaped] = st
    return states


def train_prior_weights(train, ocean):
    """Training-only intra-2° disaggregation weights on the common 1° grid."""
    one = Grid(1.)
    pts = one.indices(train.loc[train.truth_exit.lt(0),'end_lat'],train.loc[train.truth_exit.lt(0),'end_lon'])
    hist = np.bincount(pts,minlength=REF_N).astype(float)
    hist[~ocean] = 0
    return hist


def project_to_reference(g: Grid, ocean_ref: np.ndarray, weights_ref: np.ndarray):
    """Mass-preserving stochastic rebinning from M3 states to common M2 1° cells.

    0.5°: aggregate four small cells to each 1° target.
    1°: identity.
    2°: TRAIN-only within-parent occupancy to distribute mass; never uses val/test.
    """
    lat,lon=g.centres()
    parent = Grid(1.).indices(lat,lon)
    rows, cols, data=[],[],[]
    if g.degree <= 1.:
        rows=list(range(g.nc));cols=parent.tolist();data=[1.]*g.nc
    else:
        fine = Grid(1.)
        flat,flon=fine.centres()
        coarse_parent=g.indices(flat,flon)
        for i in range(g.nc):
            children = np.flatnonzero(coarse_parent==i)
            w = weights_ref[children].copy()
            if w.sum()<=0:
                w=ocean_ref[children].astype(float)
            if w.sum()<=0:
                w=np.ones(len(children))
            w/=w.sum()
            rows.extend([i]*len(children));cols.extend(children.tolist());data.extend(w.tolist())
    for e in range(4):
        rows.append(g.nc+e);cols.append(REF_N+e);data.append(1.)
    B=sparse.csr_matrix((data,(rows,cols)),shape=(g.ns,REF_NS))
    if not np.allclose(np.asarray(B.sum(axis=1)).ravel(),1):raise AssertionError('projection not stochastic')
    return B


def model_ocean_mask(g: Grid, ocean_ref):
    lat,lon=g.centres()
    if g.degree<=1.: return ocean_ref[Grid(1.).indices(lat,lon)]
    rlat,rlon=Grid(1.).centres()
    parent=g.indices(rlat,rlon)
    return np.bincount(parent,weights=ocean_ref.astype(int),minlength=g.nc)>0


def fit_m3(moves, g: Grid, choice: Choice, ocean_ref):
    """Estimate support-adaptive displacement model, retaining absorbing exits.

    Local histogram over integer grid displacements, local support measured by
    DISTINCT TRAINING DRIFTERS. Prior: 85% Gaussian-neighbour displacement +
    15% training-global displacement. W_i=min(unique_drifters_i, 100).
    P_i proportional to W_i * empirical + kappa * prior (after target rebinning).
    """
    ocean=model_ocean_mask(g,ocean_ref)
    src=g.indices(moves.lat,moves.lon)
    srcrow,srccol=divmod(src,g.nx)
    is_exit=moves.truth_exit.to_numpy()>=0
    nxtrow=np.zeros(len(moves),dtype=int)
    nxtcol=np.zeros(len(moves),dtype=int)
    insideend=~is_exit
    if insideend.any():
        # Destination may be inside unsampled cells; no test-set data required.
        valid_tgt=g.indices(moves.loc[insideend,'end_lat'],moves.loc[insideend,'end_lon'])
        nxtrow[insideend],nxtcol[insideend]=divmod(valid_tgt,g.nx)
    # For exits, approximate displacement by first out-of-R location cannot be
    # directly read from cases (only edge). Record exit via empirical state counts.
    dr=nxtrow-srcrow;dc=nxtcol-srccol
    dist=np.maximum(np.abs(dr[insideend]),np.abs(dc[insideend]))
    radius=max(1,min(15,int(np.quantile(dist,0.995)) if len(dist) else 1))
    w=2*radius+1
    K=w*w
    H=np.zeros((g.nc,K),dtype=np.float64)
    valid=insideend & (np.abs(dr)<=radius) & (np.abs(dc)<=radius)
    k=(dr[valid]+radius)*w+(dc[valid]+radius)
    np.add.at(H,(src[valid],k),1.)
    # Empirical rare long jumps and exits retained EXACTLY as state counts.
    extra_rows=src[~valid]
    extra_cols=np.full(len(extra_rows),0,dtype=int)
    extra_exit=is_exit[~valid]
    extra_cols[extra_exit]=g.nc+moves.truth_exit.to_numpy()[~valid][extra_exit].astype(int)
    extra_cols[~extra_exit]=g.indices(moves.loc[~valid,'end_lat'].to_numpy()[~extra_exit],
                                     moves.loc[~valid,'end_lon'].to_numpy()[~extra_exit])
    rare=sparse.coo_matrix((np.ones(len(extra_rows)),(extra_rows,extra_cols)),shape=(g.nc,g.ns)).tocsr()
    counts=np.bincount(src,minlength=g.nc).astype(float)
    # Drifter count (not observation count) for effective local confidence.
    dfr=pd.DataFrame({'id':moves.id.to_numpy(),'src':src}).drop_duplicates()
    n_unique=np.bincount(dfr.src.to_numpy(),minlength=g.nc).astype(float)
    support=np.minimum(n_unique,100.)
    Hn=gaussian_filter(H.reshape(g.ny,g.nx,K),sigma=(choice.sigma,choice.sigma,0),mode='constant',truncate=2.).reshape(g.nc,K)
    denom=Hn.sum(axis=1)
    global_hist=H.sum(axis=0)
    if global_hist.sum()==0: global_hist[radius*w+radius]=1.
    global_hist/=global_hist.sum()
    q=(.85*np.divide(Hn,denom[:,None],out=np.tile(global_hist,(g.nc,1)),where=denom[:,None]>0)
       +.15*global_hist[None,:])
    # Use local data for non-exit motions, a prior for the entire transition matrix.
    rr,cc,vv=[],[],[]
    dy,dx=np.divmod(np.arange(K),w)
    dy=dy-radius;dx=dx-radius
    for i in range(g.nc):
        if not ocean[i]:
            rr.append(i);cc.append(i);vv.append(1.)
            continue
        ri,ci=divmod(i,g.nx)
        yy=ri+dy;xx=ci+dx
        inn=(yy>=0)&(yy<g.ny)&(xx>=0)&(xx<g.nx)
        tgt=np.empty(K,dtype=int)
        tgt[inn]=yy[inn]*g.nx+xx[inn]
        # Outside: choose N/S/E/W by maximum overrun in grid cells.
        out=np.flatnonzero(~inn)
        if len(out):
            violations=np.stack([-yy[out],yy[out]-(g.ny-1),xx[out]-(g.nx-1),-xx[out]],axis=1)
            tgt[out]=g.nc+np.argmax(violations,axis=1)
        # Do not assign prior mass to land; renormalise after removing.
        acceptable=(tgt>=g.nc) | ocean[np.minimum(tgt,g.nc-1)]
        rowlocal=H[i] / max(counts[i],1.)
        # Local effective sample is tied to unique drifters; the fraction of
        # rare/exit counts is retained in extra row directly.
        p=(support[i]*rowlocal+choice.kappa*q[i])/(support[i]+choice.kappa)
        p[~acceptable]=0.
        valid_idx=np.flatnonzero(p>0)
        rr.extend([i]*len(valid_idx));cc.extend(tgt[valid_idx].tolist());vv.extend(p[valid_idx].tolist())
        # Rare jump/exit empirical probabilities contribute outside main H.
        # Use the same support scaling to avoid over-weighting densely sampled tracks.
        if rare.indptr[i+1]>rare.indptr[i]:
            cols=rare.indices[rare.indptr[i]:rare.indptr[i+1]]
            vals=rare.data[rare.indptr[i]:rare.indptr[i+1]]
            frac=support[i]/max(counts[i],1.)/(support[i]+choice.kappa)
            rr.extend([i]*len(cols));cc.extend(cols.tolist());vv.extend((vals*frac).tolist())
    for e in range(4):
        rr.append(g.nc+e);cc.append(g.nc+e);vv.append(1.)
    P=sparse.coo_matrix((vv,(rr,cc)),shape=(g.ns,g.ns)).tocsr()
    P.sum_duplicates();P.eliminate_zeros()
    totals=np.asarray(P.sum(axis=1)).ravel()
    empty=np.flatnonzero(totals<1e-12)
    if len(empty):
        P=P.tolil()
        for i in empty: P[i,i]=1.
        P=P.tocsr();totals=np.asarray(P.sum(axis=1)).ravel()
    P=sparse.diags(1./totals)@P
    if np.min(P.data)<0 or not np.allclose(np.asarray(P.sum(axis=1)).ravel(),1,atol=1e-10):
        raise AssertionError('Invalid M3 operator')
    diag={'training_moves':int(len(moves)),'cells_observed':int((counts>0).sum()),
          'cells_support_10':int((n_unique>=10).sum()),'max_offset_cells':radius,
          'rare_or_exit_moves':int((~valid).sum()), 'n_states':g.ns, 'nnz':P.nnz}
    return P, diag


def predict_probabilities(P, grid: Grid, cases, lag, B, horizons):
    """Propagate same release starts and score all candidates at 1° + four exits."""
    # Propagating via sparse P transpose retains output mass; at 0.5° several
    # thousand source cells may cost more time, hence full grid is opt-in.
    starts = np.unique(grid.indices(cases.lat,cases.lon))
    states = np.zeros((len(starts),grid.ns),dtype=float)
    states[np.arange(len(starts)),starts]=1.
    steps=sorted({int(h//lag) for h in horizons})
    if any(h%lag for h in horizons):
        raise ValueError('Forecast horizons must be multiples of transition lag.')
    result={}
    bysteps=set(steps)
    for t in range(1,max(steps)+1):
        states=(P.T@states.T).T
        if t in bysteps:
            comp=np.asarray(states@B)
            comp=np.clip(comp,0,1)
            comp/=np.maximum(comp.sum(axis=1,keepdims=True),1e-30)
            result[t*lag]=(starts,comp)
    return result


def _ranked_coverage(probs: np.ndarray, truth: np.ndarray, mass: float):
    """Shortest probability-set coverage; equal-probability ties are deterministic."""
    ranks = np.argsort(-probs, axis=1, kind='stable')
    sorted_p = np.take_along_axis(probs, ranks, axis=1)
    cumulative = np.cumsum(sorted_p, axis=1)
    counts = (cumulative < mass - 1e-12).sum(axis=1) + 1
    inverse = np.argsort(ranks, axis=1)
    covered = inverse[np.arange(len(truth)), truth] < counts
    return covered, counts


def score_cases(prob, states, cases, horizon, model):
    """Score the SAME complete cases for every model on the 1-degree outcome space.

    Two coverage definitions deliberately coexist:
    - all-state: 400 cells PLUS N/S/E/W; all cases, including exits;
    - position-conditional: normalised within-R cells, only outcomes in R.
    An ``area`` is NOT measured in km² here: it is a number of grid cells/states.
    """
    probs = np.clip(np.asarray(prob, dtype=float), 0, 1)
    if probs.shape != (len(cases), REF_NS):
        raise ValueError('Predicted shape does not match cases and 404 states')
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-7):
        raise AssertionError('Prediction mass not conserved')
    y = ref_truth(cases)
    logp = np.log(np.maximum(probs[np.arange(len(y)), y], LOG_FLOOR))
    hit50, n50 = _ranked_coverage(probs, y, .50)
    hit90, n90 = _ranked_coverage(probs, y, .90)

    # Probability of each FATE: in R, N, S, E, W.
    fate = np.concatenate([probs[:, :REF_N].sum(axis=1, keepdims=True),
                           probs[:, REF_N:]], axis=1)
    fate_truth = np.where(y < REF_N, 0, y - REF_N + 1)
    onehot = np.eye(5)[fate_truth]
    fate_brier = np.square(fate - onehot).sum(axis=1)
    escaped = y >= REF_N
    retention_pred = fate[:, 0]

    # Conditional within-region spatial calibration is NOT the same as
    # all-state calibration and is never evaluated on already-exited cases.
    pin = probs[:, :REF_N]
    inside_mass = pin.sum(axis=1)
    inside = (~escaped) & (inside_mass > 1e-12)
    hit90_position = np.full(len(y), np.nan)
    n90_position = np.full(len(y), np.nan)
    if inside.any():
        conditional = pin[inside] / inside_mass[inside, None]
        hh, nn = _ranked_coverage(conditional, y[inside], .90)
        hit90_position[inside] = hh.astype(float)
        n90_position[inside] = nn

    # Median distance error is conditional on ACTUALLY still being inside R.
    centre_lat, centre_lon = Grid(1.).centres()
    la = (pin * centre_lat).sum(axis=1) / np.maximum(inside_mass, 1e-12)
    lo = (pin * centre_lon).sum(axis=1) / np.maximum(inside_mass, 1e-12)
    exlat = cases.end_lat.to_numpy(dtype=float)
    exlon = cases.end_lon.to_numpy(dtype=float)
    distance = np.full(len(y), np.nan)
    if inside.any():
        a1, a2, b1, b2 = map(np.deg2rad,
                            (la[inside], exlat[inside], lo[inside], exlon[inside]))
        hav = (np.sin((a2-a1)/2)**2 + np.cos(a1)*np.cos(a2)*np.sin((b2-b1)/2)**2)
        distance[inside] = 6371.0088*2*np.arcsin(np.sqrt(np.clip(hav,0,1)))

    # Explicit start-time identifier makes pairing robust to file order.
    ids = cases.id.astype(str).to_numpy()
    time = pd.to_datetime(cases.time).astype(str).to_numpy()
    keys = np.char.add(np.char.add(ids, '|'), time)
    if len(np.unique(keys)) != len(keys):
        raise ValueError('Duplicate start IDs in evaluation cases; paired scores ambiguous.')
    row = {
        'model': model, 'horizon_days': int(horizon), 'cases': len(y),
        'drifters': int(cases.id.nunique()),
        'in_R_cases': int((~escaped).sum()), 'exited_cases': int(escaped.sum()),
        'mean_log_score': float(logp.mean()),
        'coverage50_pct': float(100*hit50.mean()),
        'coverage90_pct': float(100*hit90.mean()),
        'coverage90_position_pct': float(100*np.nanmean(hit90_position)) if inside.any() else np.nan,
        'median_area_cells': float(np.median(n90)),
        'median_area_states_50pct': float(np.median(n50)),
        'median_position_area_cells_90pct': float(np.nanmedian(n90_position)) if inside.any() else np.nan,
        'median_centre_error_km': float(np.nanmedian(distance)) if inside.any() else np.nan,
        'mean_fate_brier': float(fate_brier.mean()),
        'observed_retention_pct': float(100*(~escaped).mean()),
        'predicted_retention_pct': float(100*retention_pred.mean()),
    }
    raw = pd.DataFrame({
        'case_key': keys, 'id': ids, 'start_time': time, 'horizon_days': int(horizon),
        'model': model, 'log_score': logp, 'hit50': hit50.astype(int),
        'hit90': hit90.astype(int), 'hit90_position': hit90_position,
        'centre_error_km': distance, 'fate_brier': fate_brier,
        'observed_in_R': (~escaped).astype(int), 'predicted_in_R': retention_pred,
    })
    return row, raw


def evaluate_one(P,grid,gcases,lag,B,horizons,name):
    # Unique starts are propagated once per horizon, not once per observation.
    combined=pd.concat([gcases[h][['lat','lon']] for h in horizons],ignore_index=True)
    class Tmp: pass
    t=Tmp();t.lat=combined.lat;t.lon=combined.lon
    preds=predict_probabilities(P,grid,t,lag,B,horizons)
    rows,details=[],[]
    for h in horizons:
        cases=gcases[h]
        st,pr=preds[h]
        idx=np.searchsorted(st,grid.indices(cases.lat,cases.lon))
        rr,dd=score_cases(pr[idx],st,cases,h,name)
        rows.append(rr);details.append(dd)
    return rows,pd.concat(details,ignore_index=True)


def paired_bootstrap(raw, model_m3, horizons, repeats=300, seed=42):
    """Paired *drifter-cluster* bootstrap, same cases and prediction horizon.

    Includes log score, 90%-coverage, 5-state fate Brier and median positional
    error differences (M3 minus M2). Selection-set bands are descriptive ONLY.
    """
    if repeats < 20:
        raise ValueError('At least 20 bootstrap replicates are required')
    a = raw.loc[raw.model.eq('M2 frozen')]
    b = raw.loc[raw.model.eq(model_m3)]
    key = ['case_key', 'horizon_days']
    if a.duplicated(key).any() or b.duplicated(key).any():
        raise ValueError('Duplicate case keys: cannot pair models')
    out = []
    rng = np.random.default_rng(seed)
    for h in horizons:
        aa = a.loc[a.horizon_days.eq(h)].sort_values(key).reset_index(drop=True)
        bb = b.loc[b.horizon_days.eq(h)].sort_values(key).reset_index(drop=True)
        if len(aa) != len(bb) or not aa[key].equals(bb[key]) or not aa.id.equals(bb.id):
            raise ValueError('M2 and M3 were not scored on identical timestamped cases')
        if aa.empty:
            raise ValueError(f'No eligible cases at {h} days')
        ids, ind = np.unique(aa.id, return_inverse=True)
        n_ids = len(ids)
        m = len(aa)
        metrics = {
            'logscore': bb.log_score.to_numpy() - aa.log_score.to_numpy(),
            'coverage90_pp': (bb.hit90.to_numpy() - aa.hit90.to_numpy())*100.,
            'fate_brier': bb.fate_brier.to_numpy() - aa.fate_brier.to_numpy(),
        }
        groups = [np.flatnonzero(ind==i) for i in range(n_ids)]
        sum_n = np.bincount(ind, minlength=n_ids)
        sum_metrics = {name: np.bincount(ind,weights=v,minlength=n_ids)
                       for name,v in metrics.items()}
        distance_a = aa.centre_error_km.to_numpy(float)
        distance_b = bb.centre_error_km.to_numpy(float)
        both_valid = np.isfinite(distance_a) & np.isfinite(distance_b)
        observed_distance = (float(np.nanmedian(distance_b[both_valid])
                          - np.nanmedian(distance_a[both_valid])) if both_valid.any() else np.nan)
        trials = {name: [] for name in metrics}
        trials['median_distance_km'] = []
        for _ in range(repeats):
            sampled = rng.integers(0,n_ids,size=n_ids)
            mult = np.bincount(sampled,minlength=n_ids)
            total = mult@sum_n
            for name in metrics:
                trials[name].append(float((mult@sum_metrics[name])/total))
            if both_valid.any():
                indices_with_positions = [groups[i][both_valid[groups[i]]]
                                         for i in sampled if both_valid[groups[i]].any()]
                if indices_with_positions:
                    loc = np.concatenate(indices_with_positions)
                    trials['median_distance_km'].append(float(np.median(distance_b[loc]) -
                                                             np.median(distance_a[loc])))
        def interval(x):
            if not len(x): return (np.nan, np.nan)
            return float(np.quantile(x, .05)),float(np.quantile(x,.95))
        log_lo,log_hi = interval(trials['logscore'])
        cov_lo,cov_hi = interval(trials['coverage90_pp'])
        fate_lo,fate_hi = interval(trials['fate_brier'])
        dist_lo,dist_hi = interval(trials['median_distance_km'])
        out.append({
            'horizon_days': h, 'cases': m, 'unique_drifters': n_ids,
            'm3_minus_m2_logscore': float(metrics['logscore'].mean()),
            'paired_bootstrap_p05': log_lo, 'paired_bootstrap_p95': log_hi,
            'm3_minus_m2_coverage90_pp': float(metrics['coverage90_pp'].mean()),
            'coverage90_p05_pp': cov_lo, 'coverage90_p95_pp': cov_hi,
            'm3_minus_m2_fate_brier': float(metrics['fate_brier'].mean()),
            'fate_brier_p05': fate_lo, 'fate_brier_p95': fate_hi,
            'm3_minus_m2_median_error_km': observed_distance,
            'median_error_p05_km': dist_lo, 'median_error_p95_km': dist_hi,
            'note': 'bootstrap over whole drifters; validation-selected comparison is exploratory',
        })
    return pd.DataFrame(out)


def dump_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')

def chapman_kolmogorov_check(moves_1d, moves_2d, selected, ocean_ref):
    """TRAIN-only diagnostic: direct two-day P versus one-day P squared.

    High TV distance indicates poor Chapman-Kolmogorov consistency, but can
    also arise from sparse support, model smoothing and nonstationarity.
    Not a validation/independent-test score or proof of Markov property.
    """
    g=Grid(selected.degree)
    c1=Choice(selected.degree,1,selected.sigma,selected.kappa)
    c2=Choice(selected.degree,2,selected.sigma,selected.kappa)
    P1,_=fit_m3(moves_1d,g,c1,ocean_ref)
    P2,_=fit_m3(moves_2d,g,c2,ocean_ref)
    raw1=(P1@P1).tocsr()[:g.nc]
    raw2=P2.tocsr()[:g.nc]
    support=pd.DataFrame({'id':moves_1d.id.to_numpy(),
                          'cell':g.indices(moves_1d.lat,moves_1d.lon)}).drop_duplicates()
    count=np.bincount(support.cell.to_numpy(),minlength=g.nc)
    valid=np.flatnonzero(count>=10)
    if not len(valid):
        return {'eligible_cells':0,'warning':'No cells visited by >=10 training drifters'}
    tv=np.asarray(np.abs((raw1-raw2)[valid]).sum(axis=1)).ravel()*.5
    return {'lag_days_compared':'1+1 versus direct 2',
            'eligible_cells':int(len(valid)),
            'mean_total_variation_distance':float(tv.mean()),
            'median_total_variation_distance':float(np.median(tv)),
            'p90_total_variation_distance':float(np.quantile(tv,.90)),
            'interpretation':'Training-only CK consistency diagnostic, not an independent score'}




def baseline_projections(ocean):
    return project_to_reference(Grid(1.),ocean,np.ones(400))


def run(args):
    if args.command in ('test','recent-test') and not args.confirm_final_test:
        raise RuntimeError('Final holdouts are sealed. Add --confirm-final-test only after choices frozen.')
    if args.command in ('test','recent-test') and args.selection:
        lock=args.selection.parent / (args.command.replace('-', '_').upper() + '_USED.json')
        if lock.exists():
            raise RuntimeError(f'Holdout already evaluated with this frozen selection: {lock}')
    args.out.mkdir(parents=True,exist_ok=True)
    df=load_data(args.input)
    mapping=load_splits(args.splits,df)
    P2,ocean=load_baseline(args.m2)
    # Baseline source is checked by cryptographic hash when running held-out test.
    m2hash=sha256_file(args.m2)
    splitshash=sha256_file(args.splits)
    datahash=sha256_file(args.input)
    codehash=sha256_file(Path(__file__).resolve())
    if args.command=='tune':
        choices=[]
        degrees=(0.5,1.,2.) if args.full else (1.,2.)
        for degree in degrees:
            for lag in (1,2):
                for sigma in (0.75,1.5):
                    for kappa in (5.,20.):
                        choices.append(Choice(degree,lag,sigma,kappa))
        group='val'
    else:
        if not args.selection or not args.selection.is_file():
            raise FileNotFoundError('A frozen selection.json is required for final test.')
        sel=json.loads(args.selection.read_text())
        if any([sel['sha256_input']!=datahash,sel['sha256_splits']!=splitshash,
                sel['sha256_baseline_m2']!=m2hash,
                sel.get('sha256_m3_pipeline')!=codehash]):
            raise RuntimeError('Input, split or baseline differs from frozen validation selection.')
        choices=[Choice(**sel['selected_choice'])]
        group='test' if args.command=='test' else 'recent_test'
    print(f'Loaded {len(df):,} rows; train/val/test/recent: '
          f'{pd.Series(mapping).value_counts().to_dict()}',flush=True)
    horizons=SELECT_HORIZONS if args.command=='tune' else REPORT_HORIZONS
    samples={h:outcomes(df,start_rows(df,mapping,group),h) for h in horizons}
    for h,ca in samples.items():
        if len(ca)==0: raise ValueError(f'No eligible {group} outcomes at day {h}.')
    seltrain=train_prior_weights(training_counts(df,mapping,1),ocean)
    P2B=baseline_projections(ocean)
    m2rows,m2raw=evaluate_one(P2,Grid(1.),samples,1,P2B,horizons,'M2 frozen')
    allrows=[];records=[];diags={}
    # Reuse all moves and per-grid projection for candidates with identical lag/degree.
    train_cache={lag:training_counts(df,mapping,lag) for lag in {c.lag for c in choices}}
    for j,c in enumerate(choices,1):
        g=Grid(c.degree)
        print(f'[{j}/{len(choices)}] Fitting {c.name}',flush=True)
        P,diag=fit_m3(train_cache[c.lag],g,c,ocean)
        B=project_to_reference(g,ocean,seltrain)
        rows,raw=evaluate_one(P,g,samples,c.lag,B,horizons,c.name)
        score=float(np.mean([r['mean_log_score'] for r in rows if r['horizon_days'] in SELECT_HORIZONS]))
        allrows.append({'model':c.name,'degree':c.degree,'lag':c.lag,'sigma':c.sigma,
                        'kappa':c.kappa,'selection_logscore':score,
                        'training_moves':diag['training_moves'],
                        'observed_cells':diag['cells_observed'],
                        'cells_with_10_unique_drifters':diag['cells_support_10'],
                        'grid_cells':g.nc,
                        'logscore_improvement_vs_m2':score-np.mean([r['mean_log_score'] for r in m2rows])})
        records.append((c,rows,raw,diag))
        diags[c.name]=diag
    if args.command=='tune':
        rankings=pd.DataFrame(allrows).sort_values(['selection_logscore','degree','lag'],ascending=[False,True,True])
        rankings.to_csv(args.out/'candidate_ranking.csv',index=False)
        # Within 0.01 of best: prefer low-complexity reference grid / 1-day lag.
        close=rankings[rankings.selection_logscore>=rankings.selection_logscore.max()-.01]
        close=close.assign(prefer=(close.degree-1).abs()*5+(close.lag-1))
        winner=close.sort_values(['prefer','selection_logscore'],ascending=[True,False]).iloc[0]
        selected_choice=next(c for c,rows,raw,diag in records if c.name==winner['model'])
        selected=(selected_choice,*next((rows,raw,diag) for c,rows,raw,diag in records if c==selected_choice))
        m2primary=np.mean([r['mean_log_score'] for r in m2rows])
        gain=winner.selection_logscore-m2primary
        recommendation='M3 candidate' if gain>=.01 else 'M2 retained (M3 has no compelling validation gain)'
        sel={'selected_choice':asdict(selected_choice), 'selection_horizons_days':list(SELECT_HORIZONS),
             'selection_objective':'mean validation log score on 1-degree outcomes at 14 and 30 days',
             'selection_margin':.01, 'm3_minus_m2_validation_logscore':float(gain),
             'pipeline_version':M3_PIPELINE_VERSION,
             'sha256_m3_pipeline':codehash,
             'evaluation_protocol':'first hourly exit; common 1-degree bins; shared exact cases; log floor 1e-6',
             'm2_historical_metrics_are_not_directly_comparable':True,
             'recommendation':recommendation,'sha256_input':datahash,
             'sha256_splits':splitshash,'sha256_baseline_m2':m2hash,
             'data_rows':len(df),'validation_cases':{str(h):len(c) for h,c in samples.items()},
             'validation_only':True,
             'important_note':'M3 chosen using validation; differences are NOT independent test estimates.'}
        dump_json(args.out/'selection.json',sel)
        c,r,raw,diag=selected
        # Save M3 operator and projection so other group members can reproduce the forecasts.
        P,diag=fit_m3(train_cache[c.lag],Grid(c.degree),c,ocean)
        B=project_to_reference(Grid(c.degree),ocean,seltrain)
        sparse.save_npz(args.out/'operator_M3_selected.npz',P)
        sparse.save_npz(args.out/'M3_projection_to_1degree.npz',B)
        # Selection used only days 14 and 30. Add 60/90 validation diagnostics
        # AFTER freezing the parameter ranking; these horizons never change the winner.
        extended_horizons=REPORT_HORIZONS
        extra_horizons=[h for h in extended_horizons if h not in horizons]
        additional={h:outcomes(df,start_rows(df,mapping,'val'),h) for h in extra_horizons}
        additional={h:v for h,v in additional.items() if len(v)>0}
        extension={**samples,**additional}
        m2_extended,m2_cases_extended=evaluate_one(P2,Grid(1.),extension,1,P2B,sorted(extension),'M2 frozen')
        m3_extended,m3_cases_extended=evaluate_one(P,Grid(c.degree),extension,c.lag,B,sorted(extension),c.name)
        pd.DataFrame(m2_extended+m3_extended).to_csv(args.out/'validation_comparison.csv',index=False)
        paired_cases=pd.concat([m2_cases_extended,m3_cases_extended],ignore_index=True)
        paired_cases.to_csv(args.out/'validation_case_scores.csv',index=False)
        bands=paired_bootstrap(paired_cases,c.name,sorted(extension),repeats=args.bootstrap_reps,seed=args.seed)
        bands.to_csv(args.out/'paired_bootstrap_validation.csv',index=False)
        # A cautious provisional rule, not a claim of independent significance.
        band14_30=bands[bands.horizon_days.isin(SELECT_HORIZONS)]
        not_clearly_negative=bool((band14_30.paired_bootstrap_p05>0).all())
        m230=next(row for row in m2_extended if row['horizon_days']==30)
        m330=next(row for row in m3_extended if row['horizon_days']==30)
        coverage_loss=abs(m330['coverage90_pct']-90)-abs(m230['coverage90_pct']-90)
        if gain>=.01 and not_clearly_negative and coverage_loss<=2.0:
            recommendation='M3 promising on validation; still requires independent test'
        else:
            recommendation='Keep M2 as default pending independent evidence; M3 exploratory'
        sel['recommendation']=recommendation
        sel['validation_bootstrap_5pct_positive_at_selection_horizons']=not_clearly_negative
        sel['coverage90_distance_to_target_change_30d_pp']=float(coverage_loss)
        sel['bootstrap_reps']=args.bootstrap_reps
        sel['bootstrap_seed']=args.seed
        dump_json(args.out/'selection.json',sel)
        dump_json(args.out/'M3_training_diagnostics.json',diag)
        ck=chapman_kolmogorov_check(train_cache[1],train_cache[2],c,ocean)
        dump_json(args.out/'M3_CK_diagnostics.json',ck)
        print('\nSelected:',c.name,' / ',recommendation,' / gain=',round(gain,5),flush=True)
    else:
        c,rows,raw,diag=records[0]
        pd.DataFrame(m2rows+rows).to_csv(args.out/f'{group}_comparison.csv',index=False)
        pd.concat([m2raw,raw],ignore_index=True).to_csv(args.out/f'{group}_case_scores.csv',index=False)
        paired_bootstrap(pd.concat([m2raw,raw],ignore_index=True),c.name,horizons,repeats=args.bootstrap_reps,seed=args.seed).to_csv(args.out/f'{group}_paired_bootstrap.csv',index=False)
        final_log={'group':group,'selected_choice':asdict(c),
                     'sha256_input':datahash,'sha256_splits':splitshash,'sha256_baseline_m2':m2hash,
                     'sha256_m3_pipeline':codehash,'pipeline_version':M3_PIPELINE_VERSION,
                     'sealed_selection':str(args.selection),'test_was_explicitly_authorised':True}
        dump_json(args.out/'final_run_manifest.json',final_log)
        # Best-effort local safeguard against repeatedly using the same holdout
        # to tune decisions. Copy/pasting the selection or changing paths could
        # bypass it: the team must enforce an ethical single-use protocol.
        lock=args.selection.parent / (args.command.replace('-', '_').upper()+'_USED.json')
        dump_json(lock,final_log)
    print('Results:',args.out.resolve(),flush=True)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('command',choices=['tune','test','recent-test'])
    p.add_argument('--input',required=True,type=Path,help='Existing GDP .parquet (not committed to git)')
    p.add_argument('--splits',required=True,type=Path,help='Original, immutable models/splits.csv')
    p.add_argument('--m2',required=True,type=Path,help='Original frozen operator_M2_smoothed_drogued.npz')
    p.add_argument('--out',type=Path,default=Path('results/m3'))
    p.add_argument('--full',action='store_true',help='Also run compute-intensive 0.5° grid candidates')
    p.add_argument('--bootstrap-reps',type=int,default=300,help='Paired whole-drifter bootstrap resamples')
    p.add_argument('--seed',type=int,default=42,help='Bootstrap random seed')
    p.add_argument('--selection',type=Path,help='Previously frozen selection.json for test')
    p.add_argument('--confirm-final-test',action='store_true',help='Explicitly authorise opening holdout')
    return p.parse_args(argv)

if __name__=='__main__':
    try:run(parse_args())
    except Exception as e:
        print(f'M3 error: {type(e).__name__}: {e}',file=sys.stderr)
        raise
