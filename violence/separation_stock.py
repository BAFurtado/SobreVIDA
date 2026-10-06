"""Held-out validation: among women ever assaulted, the share who ever left.

DataSenado 2025 reports a STOCK — 79% of the women who reported assault by a husband or
partner were ex-partners at interview. ACTION_THRESHOLD was calibrated against a FLOW (0.15
separations per assaulted woman-year), and the stock is never used to tune anything, so it
tests the specification rather than the fit. Nothing here is a target.

The reduction itself lives in generalization_aps.separation_stock_moment; this module only
runs replicates in parallel and pools them. Shares are re-derived from POOLED COUNTS, never
averaged across runs, so a replicate with few victims cannot weigh as much as one with many.

Usage (from project root):
    python violence/separation_stock.py -m "PORTO ALEGRE" -i 10 -j 10

Writes output/separation_stock_<metro>_pooled.csv.
"""
import argparse
import builtins
import os
import sys

sys.path.append(os.path.abspath('violence'))

import pandas as pd

import model as M
import parallel
import settings
import generalization_aps as ga

_12_MONTHS = 12

# DataSenado 2025, women assaulted by husband/partner who are ex-partners at interview.
TARGET = 0.79


def run_once(metro, steps):
    """Step a year at a time and reduce to the stock row inside the worker.

    The annual reset is not optional: Woman.peer_action_norm reads the counters it clears,
    so a harness that skips it measures a different model from the one ga.main() runs. It
    also empties schedule.dead_agents, which is why the `_all` variant below (the one that
    includes women killed during the run) only sees the final year's deaths. At ~0.2
    femicides per run that is immaterial, and the primary shares are on LIVING women.
    """
    home = M.Home(metro=metro)
    home.verbose = False
    stdout_print = builtins.print
    builtins.print = lambda *a, **k: None
    try:
        for step in range(1, steps + 1):
            home.step()
            if step % _12_MONTHS:
                continue
            home.update_incarcerated()
            ga.reset_annual_counters(home)
        row = ga.separation_stock_moment([home], metro, write=False)
    finally:
        builtins.print = stdout_print
    return row


def _one_run(job):
    return run_once(*job)


def measure(rows, metro):
    """Pool the per-run counts, then form the shares once."""
    df = pd.DataFrame(rows)
    # separation_stock_moment returns ever_broke_all only as a share; recover the count so
    # the pooled denominator is a sum of counts like every other field.
    ever_broke_all = (df.share_ever_broke_all * df.ever_attacked_all).round().sum()
    assaults = (df.assaults_per_victim * df.ever_attacked_living).round().sum()

    victims = df.ever_attacked_living.sum()
    victims_all = df.ever_attacked_all.sum()
    return {
        'metro': metro,
        'iterations': len(df),
        'ever_attacked_living': victims,
        'ever_broke_living': df.ever_broke_living.sum(),
        'share_ever_broke': df.ever_broke_living.sum() / victims,
        'ex_partner_living': df.ex_partner_living.sum(),
        'share_ex_partner': df.ex_partner_living.sum() / victims,
        'assaults_per_victim': assaults / victims,
        'ever_attacked_all': victims_all,
        'share_ever_broke_all': ever_broke_all / victims_all,
        'target': TARGET,
        'ACTION_THRESHOLD_MIN': settings.ACTION_THRESHOLD_MIN,
        'ACTION_THRESHOLD_MAX': settings.ACTION_THRESHOLD_MAX,
        'ESCALATION_STEP': settings.ESCALATION_STEP,
    }


def _report(row):
    print(f"\n victims (living, ever attacked)  {row['ever_attacked_living']}")
    print(f" assaults per victim (lifetime)   {row['assaults_per_victim']:.2f}"
          f"   (era 1,74 antes da escalada)")
    print(f" share_ever_broke   (strict)      {row['share_ever_broke']:.3f}"
          f"   vs target {row['target']}")
    print(f" share_ex_partner   (broad)       {row['share_ex_partner']:.3f}"
          f"   vs target {row['target']}")
    print(f" share_ever_broke_all (incl dead) {row['share_ever_broke_all']:.3f}", flush=True)


def run(metro, iterates, steps, workers=None):
    print(f'=== {metro}, {iterates} runs, {steps} steps ===', flush=True)
    done = [0]

    def tick(_i, _r, done=done):
        done[0] += 1
        print(f'  run {done[0]}/{iterates}', flush=True)

    rows = parallel.map_runs(_one_run, [(metro, steps)] * iterates,
                             workers=workers, on_result=tick)
    row = measure(rows, metro)
    _report(row)
    out = f'output/separation_stock_{metro}_pooled.csv'
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f'\nDone -> {out}')
    return row


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--metro', default='PORTO ALEGRE')
    p.add_argument('-i', '--iterations', type=int, default=10)
    p.add_argument('-s', '--steps', type=int, default=144)
    p.add_argument('-j', '--workers', type=int, default=None)
    a = p.parse_args()
    run(a.metro, a.iterations, a.steps, a.workers)
