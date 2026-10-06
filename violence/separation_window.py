"""Held-out separation stock, decomposed by how many years each woman was assaulted.

The separation decision is drawn in months in which she is assaulted, so the lifetime
share who ever left depends on how many such years a woman accumulates. This module runs
the same harness as separation_stock.py and records, per woman, the calendar years of the
run in which she was assaulted, the year of her first assault and whether she ever left
through the action stage. It reports the stock by number of assaulted years, next to what
a constant annual hazard (the calibrated flow, separations per assaulted woman-year) would
give over the same number of years.

Grouping by the total number of assaulted years is endogenous: leaving is followed by
post-separation contact and by new partners, both of which add assaulted years after the
break. The life table therefore counts each woman only up to the year she first left:
h_k is the share leaving in her k-th assaulted year among those who reached it without
having left. A constant hazard shows up as a flat h_k; heterogeneity in the action
threshold shows up as a falling one, and then a constant-hazard projection of the
lifetime stock overstates it.

Usage (from project root):
    python violence/separation_window.py -m "PORTO ALEGRE" -i 150 -j 8 -o _150rep

Writes output/separation_window_<metro><suffix>.csv (one row per woman),
..._summary.csv (stock by assaulted years) and ..._hazard.csv (the life table and its
projection).
"""
import argparse
import builtins
import os
import random
import sys

sys.path.append(os.path.abspath('violence'))

import numpy as np
import pandas as pd

import model as M
import parallel
import generalization_aps as ga
from agents import Woman

_12_MONTHS = 12
BASE_SEED = 20260923
DATASENADO_STOCK = 0.79


def run_once(metro, steps, run):
    seed = BASE_SEED + run
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    home = M.Home(metro=metro)
    home.verbose = False
    stdout_print = builtins.print
    builtins.print = lambda *a, **k: None
    seen = {}
    try:
        for step in range(1, steps + 1):
            home.step()
            year = (step - 1) // _12_MONTHS
            for agent in ga._get_agents(home):
                if not isinstance(agent, Woman):
                    continue
                rec = seen.setdefault(id(agent), {'agent': agent, 'last': 0, 'years': set(),
                                                  'break_year': None})
                if agent.assaults_lifetime > rec['last']:
                    rec['years'].add(year)
                    rec['last'] = agent.assaults_lifetime
                if rec['break_year'] is None and agent.ever_broke:
                    rec['break_year'] = year
            if step % _12_MONTHS:
                continue
            home.update_incarcerated()
            ga.reset_annual_counters(home)
    finally:
        builtins.print = stdout_print

    rows = []
    for rec in seen.values():
        agent = rec['agent']
        if agent.age <= 18 or agent.is_dead or not agent.ever_attacked or not rec['years']:
            continue
        rows.append({
            'run': run,
            'years_assaulted': len(rec['years']),
            'first_year': min(rec['years']),
            'years': ';'.join(str(y) for y in sorted(rec['years'])),
            'break_year': rec['break_year'],
            'threshold_action': agent.threshold_action,
            'assaults': agent.assaults_lifetime,
            'ever_broke': int(agent.ever_broke),
            'ex_partner': int(agent.ever_separated_from_aggressor),
        })
    return rows


def _one_run(job):
    return run_once(*job)


def summarise(df, hazard):
    by_k = df.groupby('years_assaulted').agg(
        women=('ever_broke', 'size'), share_ever_broke=('ever_broke', 'mean'))
    by_k['constant_hazard'] = 1 - (1 - hazard) ** by_k.index
    return by_k


def break_index(row):
    """Her assaulted year in which she first left (1-based), or None if she never did.

    The decision is drawn in months she is assaulted, so the break year is normally one of
    her assaulted years; if not, it is counted against the assaulted years up to it."""
    if pd.isna(row['break_year']):
        return None
    years = [int(y) for y in row['years'].split(';')]
    return max(1, sum(y <= row['break_year'] for y in years))


def life_table(df):
    """h_k: share leaving in her k-th assaulted year among those who reached it unbroken."""
    k_break = df.apply(break_index, axis=1)
    k_seen = np.where(k_break.isna(), df['years_assaulted'], k_break)
    rows = []
    for k in range(1, int(k_seen.max()) + 1):
        at_risk = int((k_seen >= k).sum())
        left = int((k_break == k).sum())
        rows.append({'k': k, 'at_risk': at_risk, 'left': left,
                     'hazard': left / at_risk if at_risk else np.nan})
    return pd.DataFrame(rows).set_index('k')


def years_to_stock(hazards, target=DATASENADO_STOCK, cap=60):
    """Assaulted years until 1 - prod(1 - h_k) reaches `target`, carrying the last h forward."""
    survive = 1.0
    for k in range(1, cap + 1):
        survive *= 1 - hazards[min(k, len(hazards)) - 1]
        if 1 - survive >= target:
            return k
    return None


def run(metro, iterates, steps, workers=None, hazard=0.161, suffix='', min_at_risk=200):
    jobs = [(metro, steps, i) for i in range(iterates)]
    results = parallel.map_runs(_one_run, jobs, workers=workers)
    df = pd.DataFrame([r for rows in results for r in rows])
    stem = f'output/separation_window_{metro}{suffix}'
    df.to_csv(f'{stem}.csv', index=False)

    by_k = summarise(df, hazard)
    by_k.to_csv(f'{stem}_summary.csv')
    print(by_k.to_string())

    table = life_table(df)
    # Heterogeneity check: the same life table within terciles of the action threshold.
    tercile = pd.qcut(df['threshold_action'], 3, labels=['low', 'mid', 'high'])
    for name, part in df.groupby(tercile, observed=True):
        table[f'hazard_{name}_tau_a'] = life_table(part)['hazard']
    table['cum_empirical'] = 1 - (1 - table['hazard']).cumprod()
    table['cum_constant'] = 1 - (1 - hazard) ** table.index
    table.to_csv(f'{stem}_hazard.csv')
    print('\n' + table.round(3).to_string())

    # Project only on cells with enough women at risk; the tail is too thin to read.
    firm = table.loc[table['at_risk'] >= min_at_risk, 'hazard'].tolist()
    print(f"\n assaulted years to reach {DATASENADO_STOCK}: constant {hazard} -> "
          f"{years_to_stock([hazard])}; empirical h_k (k <= {len(firm)}, last carried) -> "
          f"{years_to_stock(firm)}")
    per_run = df.groupby('run')['ever_broke'].mean()
    print(f" stock by run: mean {per_run.mean():.3f}  se {per_run.std(ddof=1) / len(per_run) ** .5:.4f}")
    print(f"\n women {len(df)}  share_ever_broke {df.ever_broke.mean():.3f}"
          f"  mean years assaulted {df.years_assaulted.mean():.2f}"
          f"  assaults per victim {df.assaults.mean():.2f}")
    print(f" constant hazard over each woman's own years "
          f"{(1 - (1 - hazard) ** df.years_assaulted).mean():.3f}")
    return df


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--metro', default='PORTO ALEGRE')
    p.add_argument('-i', '--iterations', type=int, default=10)
    p.add_argument('-s', '--steps', type=int, default=144)
    p.add_argument('-j', '--workers', type=int, default=None)
    p.add_argument('--hazard', type=float, default=0.161)
    p.add_argument('-o', '--suffix', default='')
    a = p.parse_args()
    run(a.metro, a.iterations, a.steps, a.workers, a.hazard, a.suffix)
