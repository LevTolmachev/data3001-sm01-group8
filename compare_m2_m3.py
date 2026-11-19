#!/usr/bin/env python3
"""
DATA3001 Group S01-G8 - Compare ORIGINAL M2 vs validation-selected M3 (v2)
=============================================================================

Research purpose
  Present a transparent, reproducible comparison between the group's original
  frozen M2 transport operator and the new M3 model on the SAME held-out cases,
  prediction horizons and common 1-degree outcome space (400 cells + 4 exits).
  This is a GitHub/local-Python script; no Colab or Google Drive is needed.

What this script does (in order)
  1. Reads comparison tables written by m3_grid_timestep.py, for one split:
       validation : validation_comparison.csv / paired_bootstrap_validation.csv
       test       : test_comparison.csv / test_paired_bootstrap.csv
       recent_test: recent_test_comparison.csv / recent_test_paired_bootstrap.csv
     It NEVER loads raw GDP data, trains operators or selects M3 parameters.

  2. Checks that exactly two models are reported: the ORIGINAL 'M2 frozen'
     and one selected M3. Both must have identical horizons and matching
     case counts (and, when available, drifter and outcome counts).

  3. Produces four side-by-side performance plots over prediction horizon:
       mean log score               higher is better
       all-state 90% coverage       closer to 90% is better
       median position error (km)  lower is better
       five-state fate Brier score lower is better
     The five fate categories are: still in region, exit N, S, E and W.

  4. Plots M3-minus-M2 log-score gains with 90% paired whole-drifter
     bootstrap intervals when the upstream bootstrap file is available.
     On VALIDATION data, these intervals describe exploratory results;
     model-selection optimism has not been removed.

  5. Saves a candidate ranking plot (validation only), a CSV of model-score
     differences and a Markdown interpretation/limitations summary.

How to run from the GitHub repository root
  First execute m3_grid_timestep.py tune and inspect selection.json.

  Compare validation results:
    python compare_m2_m3.py --results results/m3_full --split validation

  Write figures to a separate directory (optional):
    python compare_m2_m3.py --results results/m3_full --out results/m3_charts --split validation

  Once the independent holdout was explicitly authorised and scored:
    python compare_m2_m3.py --results results/m3_final_test --split test

Files written into --out (defaults to --results)
    M2_vs_M3_metrics.png            Four main metrics over prediction horizons
    M3_minus_M2_logscore_CI.png      Paired log-score gain and uncertainty
    M3_candidate_ranking.png         Validation-only candidate search, if present
    M3_minus_M2_metrics.csv         Horizon-wise M3 minus M2 differences
    COMPARISON_README.md            Summary, definitions and cautions

How to interpret the results
  Positive M3-minus-M2 Log Score favours M3. Lower position error or lower
  Brier score favours that model. Calibration near the nominated 90% target
  is preferable to coverage that is merely very large.

Important scientific limitations
  - The comparator is the original saved M2 operator, NOT a newly fitted M2.
  - This script uses the shared M3 scoring protocol; original Step 2 CSV
    metrics must NOT be pasted into these comparisons.
  - Validation performance is exploratory because M3 was chosen there.
  - This script does not open independent test drifters; the upstream model
    script enforces final-test authorisation and input/source hash checks.
  - All-state coverage includes N/S/E/W. Position error is calculated only
    for in-region observed outcomes where the forecast has in-region mass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = (
    ('mean_log_score', 'Mean log score', 'Higher is better'),
    ('coverage90_pct', '90% all-state coverage (%)', 'Closer to 90% is better'),
    ('median_centre_error_km', 'Median position error (km)', 'Lower is better'),
    ('mean_fate_brier', '5-state fate Brier score', 'Lower is better'),
)
SPLIT_FILES = {
    'validation': ('validation_comparison.csv', 'paired_bootstrap_validation.csv'),
    'test': ('test_comparison.csv', 'test_paired_bootstrap.csv'),
    'recent_test': ('recent_test_comparison.csv', 'recent_test_paired_bootstrap.csv'),
}


def load_results(root: Path, split: str):
    if split not in SPLIT_FILES:
        raise ValueError(f'Unknown split {split!r}')
    comparisons, bands = SPLIT_FILES[split]
    comp = root / comparisons
    if not comp.is_file():
        raise FileNotFoundError(f'Comparison data missing: {comp}. Run m3_grid_timestep.py first.')
    df = pd.read_csv(comp)
    required = {'model', 'horizon_days', 'cases', 'drifters'} | {x[0] for x in METRICS}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing comparison columns: {sorted(required-set(df.columns))}')
    if df.duplicated(['model','horizon_days']).any():
        raise ValueError('Duplicate model + horizon records; invalid comparison')
    names = set(df.model.unique())
    if 'M2 frozen' not in names or len(names) != 2:
        raise ValueError('Expected exactly M2 frozen and one selected M3')
    m3name = next(n for n in names if n != 'M2 frozen')
    m2 = df.loc[df.model.eq('M2 frozen')].set_index('horizon_days').sort_index()
    m3 = df.loc[df.model.eq(m3name)].set_index('horizon_days').sort_index()
    if not m2.index.equals(m3.index):
        raise ValueError('M2 and M3 must share EXACT forecast horizons')
    for c in ('cases','drifters','in_R_cases','exited_cases'):
        if c in m2.columns and not m2[c].equals(m3[c]):
            raise ValueError(f'M2 and M3 not evaluated on the same number of {c}')
    bpath = root/bands
    interval = pd.read_csv(bpath) if bpath.is_file() else None
    if interval is not None:
        if interval.horizon_days.duplicated().any() or set(interval.horizon_days) != set(m2.index):
            raise ValueError(f'Bootstrap horizons do not match {comp.name}')
        interval = interval.set_index('horizon_days').loc[m2.index]
    return df, m2, m3, m3name, interval


def plot_summary(m2, m3, m3name, out, split):
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2), layout='constrained')
    axes = axes.ravel()
    for ax, (name, title, direction) in zip(axes, METRICS):
        ax.plot(m2.index, m2[name], '-s', color='#28648d', lw=2.7, ms=6, label='M2 frozen')
        ax.plot(m3.index, m3[name], '-D', color='#db7c2c', lw=2.7, ms=6, label='M3 selected')
        if name == 'coverage90_pct':
            ax.axhline(90, ls='--', color='#444444', lw=1, label='90% target')
        ax.set_title(title, loc='left', fontweight='bold', fontsize=16)
        ax.set_xlabel('Prediction horizon (days)')
        ax.set_ylabel(direction)
        ax.set_xticks(m2.index)
        ax.grid(axis='y', alpha=.22)
        ax.spines[['top', 'right']].set_visible(False)
        if name == 'coverage90_pct':
            ax.set_ylim(0, 100)
    axes[0].legend(frameon=False)
    fig.suptitle('Original M2 vs M3 • '+split.replace('_',' ').title() + ' • common 1° outcomes',
                 fontsize=19, fontweight='bold')
    fig.savefig(out/'M2_vs_M3_metrics.png', dpi=230, bbox_inches='tight')
    plt.close(fig)


def plot_differences(m2, m3, bands, out, split):
    h = m2.index.to_numpy(dtype=float)
    diff = (m3.mean_log_score-m2.mean_log_score).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 5.1), layout='constrained')
    ax.axhline(0, color='#444444', lw=1.25, ls='--')
    if bands is not None and {'paired_bootstrap_p05','paired_bootstrap_p95'}.issubset(bands.columns):
        lo = bands.paired_bootstrap_p05.to_numpy(dtype=float)
        hi = bands.paired_bootstrap_p95.to_numpy(dtype=float)
        yerr = np.stack([np.maximum(0, diff-lo),np.maximum(0,hi-diff)])
        ax.errorbar(h, diff, yerr=yerr, color='#c66829', lw=2.3, marker='D',
                    capsize=5, ms=7, label='M3 − M2; 90% drifter-cluster bootstrap interval')
    else:
        ax.plot(h, diff, color='#c66829', lw=2.3, marker='D', ms=7, label='M3 − M2')
    ax.set_xticks(h)
    ax.set_xlabel('Prediction horizon (days)')
    ax.set_ylabel('Mean log-score difference (higher favours M3)')
    ax.set_title('Paired probabilistic skill difference', loc='left', fontsize=18, weight='bold')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out/'M3_minus_M2_logscore_CI.png',dpi=230,bbox_inches='tight')
    plt.close(fig)


def plot_ranking(root, out):
    path = root/'candidate_ranking.csv'
    if not path.is_file(): return None
    d=pd.read_csv(path)
    needed={'model','logscore_improvement_vs_m2','selection_logscore'}
    if not needed.issubset(d.columns):
        raise ValueError(f'{path.name} missing {sorted(needed-set(d.columns))}')
    d=d.sort_values('selection_logscore',ascending=True).tail(25)
    fig,ax=plt.subplots(figsize=(10,max(5.3, len(d)*.31)),layout='constrained')
    vals=d.logscore_improvement_vs_m2.to_numpy()
    ax.barh(np.arange(len(d)),vals,color=['#27805e' if v>0 else '#bc5e52' for v in vals])
    ax.set_yticks(np.arange(len(d)), d.model, fontsize=9)
    ax.axvline(0,color='#222222',lw=1.3)
    ax.set_xlabel('Mean validation log-score advantage over M2 (days 14, 30)')
    ax.set_title('Validation-only candidate search (top 25)',loc='left',fontsize=17,weight='bold')
    fig.savefig(out/'M3_candidate_ranking.png', dpi=220,bbox_inches='tight')
    plt.close(fig)
    return path


def report_markdown(m2,m3,m3name,bands,split,out):
    lines = ['# M2 vs M3 • '+split.replace('_',' ').title(),'',
        '**Comparison protocol:** same frozen drifter split, identical cases, observed first exit, '
        '1° outcome bins plus four exit states (N/S/E/W), and log-score floor of 1e-6.','',
        '| Day | Cases | M2 log score | M3 log score | Δ log score | M2 coverage | M3 coverage | M2 error (km) | M3 error (km) |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for h in m2.index:
        a,b=m2.loc[h],m3.loc[h]
        fmt = lambda v: 'n/a' if pd.isna(v) else f'{v:.3f}'
        lines.append(f'| {h} | {int(a.cases)} | {fmt(a.mean_log_score)} | {fmt(b.mean_log_score)} | '
                     f'{fmt(b.mean_log_score-a.mean_log_score)} | {fmt(a.coverage90_pct)}% | {fmt(b.coverage90_pct)}% | '
                     f'{fmt(a.median_centre_error_km)} | {fmt(b.median_centre_error_km)} |')
    lines += ['',
         '**Direction:** log score higher is better; median in-region positional error lower is better; '
         'nominal 90% coverage should be close to 90%, not necessarily as high as possible.',
         '',
         '**Care:** all-state 90% coverage includes exits; distance error is evaluated only for '
         'cases observed in R with forecast in-R mass. Fate Brier is a 5-category (still/N/S/E/W) '
         'squared error, not the same metric as the original historical fate-validation CSV.',
         '',
         '**Original historical Step 2 CSV metrics must NOT be directly merged with these scores:** '
         'this pipeline uses first-hourly-exit censoring and common 1° evaluation.',
         '']
    if split=='validation':
        lines+=['**Evidence status:** model selection AND these reported scores used the validation split. '
                'Any apparent improvement is exploratory and can be optimistic. Final recommendation must '
                'await the independently opened test and recent-test splits.','']
    else:
        lines+=['**Evidence status:** held-out scores from a selected, frozen M3. '
                'Do not retune parameters in response to these test scores.','']
    if bands is not None:
        lines.append('## Drifter-cluster bootstrap • log-score gain (M3 − M2)')
        lines.append('')
        lines.append('| Days | Difference | 5% | 95% | Drifters |')
        lines.append('|---:|---:|---:|---:|---:|')
        for h,row in bands.iterrows():
            lines.append(f'| {h} | {row.m3_minus_m2_logscore:.4f} | '
                         f'{row.paired_bootstrap_p05:.4f} | {row.paired_bootstrap_p95:.4f} | {int(row.unique_drifters)} |')
        lines.append('')
    lines.append(f'Selected M3 identifier: `{m3name}`. Refer to `selection.json` for frozen configuration and hashes.')
    (out/'COMPARISON_README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(root: Path,out: Path,split: str):
    df,m2,m3,m3name,bands=load_results(root,split)
    out.mkdir(parents=True,exist_ok=True)
    deltas=pd.DataFrame({'horizon_days':m2.index,'model_m3':m3name,'cases':m2.cases.to_numpy(),
                         'delta_mean_log_score':m3.mean_log_score.to_numpy()-m2.mean_log_score.to_numpy(),
                         'delta_coverage90_pp':m3.coverage90_pct.to_numpy()-m2.coverage90_pct.to_numpy(),
                         'delta_median_position_error_km':m3.median_centre_error_km.to_numpy()-m2.median_centre_error_km.to_numpy(),
                         'delta_mean_fate_brier':m3.mean_fate_brier.to_numpy()-m2.mean_fate_brier.to_numpy()})
    deltas.to_csv(out/'M3_minus_M2_metrics.csv',index=False)
    plot_summary(m2,m3,m3name,out,split)
    plot_differences(m2,m3,bands,out,split)
    if split=='validation': plot_ranking(root,out)
    report_markdown(m2,m3,m3name,bands,split,out)
    print(f'Compared: M2 frozen vs {m3name} • {split} • {len(m2)} horizons')
    print('M3 − M2 (positive log score means M3 is better):')
    print(deltas.round(4).to_string(index=False))
    print('Wrote plots, deltas and COMPARISON_README.md to',out.resolve())
    return deltas


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--split',choices=tuple(SPLIT_FILES),default='validation',
                   help='Never accidentally select test data just because its CSV exists')
    args=p.parse_args(argv)
    run(args.results,args.out or args.results,args.split)


if __name__=='__main__':
    main()
