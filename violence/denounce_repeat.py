"""Held-out validation BH: police records per victim-year, among victims who recorded.

Polícia Civil de Minas Gerais, Belo Horizonte, 2010-2021, women victims of intimate
partner AMEACA / LESAO CORPORAL / VIAS DE FATO, keyed by `IDvítima`: **1.1627 occurrences
per victim-year +- 0.0248**, over 93,813 victim-years, with shape 0.8771 / 0.0959 / 0.0265
/ 0.0005 across the bands 1, 2, 3-6, 7+ (validation/pcmg_target.py, which recomputes all of
it from the microdata).

The model's `denounce` IS a police record by construction (it is what P_MPU_REQUESTED is
applied to), so the two constructs meet with no conversion between flow and stock and
none between temporal aggregations. The target is a Minas Gerais number, which makes Belo
Horizonte an out-of-sample test. Nothing here is a target: the parameters are the
calibrated ones.

In the model a woman above her own threshold_action acts on EVERY assault of that year,
and the only draw left is which act (ACTION_DENOUNCE_SHARE, routed entirely to denounce
once she has left). Her record count is therefore ~Binomial(assaults, share) conditional
on >= 1.

Usage (from project root):
    python violence/denounce_repeat.py -m "BELO HORIZONTE" -i 40 -s 144 -j 10

Writes output/denounce_repeat_<metro>.csv. Counts are POOLED across replicates before the
shares are formed, never averaged across runs.
"""
import argparse
import builtins
import os
import sys
from collections import Counter

sys.path.append(os.path.abspath('violence'))

import numpy as np
import pandas as pd

import model as M
import parallel
import settings
import generalization_aps as ga

_12_MONTHS = 12

# validation/pcmg_target.py, output/pcmg_target_BELO HORIZONTE.csv.
TARGET = 1.1627
TARGET_BAND = 0.0248
TARGET_SHAPE = {'1': 0.8771, '2': 0.0959, '3-6': 0.0265, '7+': 0.0005}

BANDS = [(1, 1, '1'), (2, 2, '2'), (3, 6, '3-6'), (7, 10 ** 6, '7+')]


def run_once(metro, steps):
    """Step a year at a time, collecting the per-woman denunciation counts at each boundary.

    The counters are annual and ga.reset_annual_counters clears them, so the read has to
    happen at the boundary and before the reset — the same constraint every harness here
    is under, and the reason it cannot be reconstructed from the final state.
    """
    home = M.Home(metro=metro)
    home.verbose = False
    stdout_print = builtins.print
    builtins.print = lambda *a, **k: None
    counts = Counter()
    attacked = denounced = 0
    try:
        for step in range(1, steps + 1):
            home.step()
            if step % _12_MONTHS:
                continue
            home.update_incarcerated()
            year = ga.action_year_counts(home)
            counts.update(year['denounce_counts'])
            attacked += year['attacked']
            denounced += year['denounced']
            ga.reset_annual_counters(home)
    finally:
        builtins.print = stdout_print
    return {'counts': dict(counts), 'attacked': attacked, 'denounced': denounced}


def _one_run(job):
    return run_once(*job)


def measure(rows, metro):
    """Pool the per-run histograms, then form the moment and the shape once."""
    pooled = Counter()
    for row in rows:
        pooled.update({int(k): v for k, v in row['counts'].items()})

    denouncing_woman_years = sum(pooled.values())
    records = sum(k * v for k, v in pooled.items())
    # Replicate-level means, kept only for the standard error: the reported moment is the
    # pooled ratio, but the spread across runs is what says whether it is resolved.
    per_run = [sum(int(k) * v for k, v in r['counts'].items()) / sum(r['counts'].values())
               for r in rows if r['counts']]

    out = {
        'metro': metro,
        'iterations': len(rows),
        'attacked_woman_years': sum(r['attacked'] for r in rows),
        'denouncing_woman_years': denouncing_woman_years,
        'records': records,
        'records_per_denouncing_woman_year': records / denouncing_woman_years,
        'se': float(np.std(per_run, ddof=1) / np.sqrt(len(per_run))) if len(per_run) > 1
              else float('nan'),
        'target': TARGET,
        'target_band': TARGET_BAND,
        'max_in_a_year': max(pooled) if pooled else 0,
        'ACTION_THRESHOLD_MIN': settings.ACTION_THRESHOLD_MIN,
        'ACTION_THRESHOLD_MAX': settings.ACTION_THRESHOLD_MAX,
        'ESCALATION_STEP': settings.ESCALATION_STEP,
    }
    for lo, hi, label in BANDS:
        share = sum(v for k, v in pooled.items() if lo <= k <= hi) / denouncing_woman_years
        out[f'share_{label}'] = share
        out[f'target_share_{label}'] = TARGET_SHAPE[label]
    return out


def _report(row):
    value = row['records_per_denouncing_woman_year']
    inside = abs(value - TARGET) <= TARGET_BAND
    print(f"\n denouncing woman-years           {row['denouncing_woman_years']}")
    print(f" records per denouncing year      {value:.4f}"
          f"  (se {row['se']:.4f})   vs {TARGET} +- {TARGET_BAND}"
          f"   {'IN BAND' if inside else 'OUT'}")
    print(' shape                            model    PCMG')
    for _, _, label in BANDS:
        print(f'   {label:<5}                        {row[f"share_{label}"]:.4f}'
              f'   {TARGET_SHAPE[label]:.4f}')
    print(f" most records by one woman in a year  {row['max_in_a_year']}", flush=True)


def run(metro, iterates, steps, workers=None, suffix=''):
    print(f'=== {metro}, {iterates} runs, {steps} steps ===', flush=True)
    done = [0]

    def tick(_i, _r, done=done):
        done[0] += 1
        print(f'  run {done[0]}/{iterates}', flush=True)

    rows = parallel.map_runs(_one_run, [(metro, steps)] * iterates,
                             workers=workers, on_result=tick)
    row = measure(rows, metro)
    _report(row)
    out = f'output/denounce_repeat_{metro}{suffix}.csv'
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f'\nDone -> {out}')
    return row


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--metro', default='BELO HORIZONTE')
    p.add_argument('-i', '--iterations', type=int, default=40)
    p.add_argument('-s', '--steps', type=int, default=144)
    p.add_argument('-j', '--workers', type=int, default=None)
    p.add_argument('-o', '--suffix', default='')
    a = p.parse_args()
    run(a.metro, a.iterations, a.steps, a.workers, a.suffix)
