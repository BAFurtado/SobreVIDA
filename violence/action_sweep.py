"""Sweep the victim action threshold and record the two victim-side annual moments.

The action channel has two parameters:

  ACTION_THRESHOLD_MIN/MAX  ->  how many assaulted women act at all
  ACTION_DENOUNCE_SHARE     ->  the split of those acts into denounce vs. separation

Stage 1 sweeps the threshold range at a fixed share, stage 2 sweeps the share. The targets
are annual denunciation in [0.142, 0.257] and annual separation 0.15, both with ASSAULTED
women as denominator.

The two targets do NOT sum to a target for act_rate. All three are "at least once in the
year" shares of the same women, and a woman who both denounces and leaves is counted in
both, so act_rate lies between the larger of the two rates (one act nested in the other)
and their sum (disjoint). The two rates are read against their targets; act_rate is the
monotone dial the threshold turns, not a moment.

The moments are ANNUAL and the decision is PER ASSAULT: a woman assaulted several times
in a year draws several times, so the per-draw probability that produces a given annual
rate depends on the assault frequency of the aggressor side. The sweep therefore has to
be re-run after any change to the aggressor side.

Usage (from project root), stage 1 then stage 2:
    python violence/action_sweep.py -m "PORTO ALEGRE" -i 2 -s 144
    python violence/action_sweep.py -m "PORTO ALEGRE" -i 2 -s 144 --stage share

Writes output/action_sweep_<stage>_<metro>.csv incrementally (one row per grid cell).
"""
import argparse
import builtins
import os
import sys

sys.path.append(os.path.abspath('violence'))

import numpy as np
import pandas as pd

import agents
import model as M
import parallel
import settings
import generalization_aps as ga

# Ranges to try, as (min, max) of the uniform draw on threshold_action. The propensity is
# a weighted average of three components in [0, 1] whose weights sum to 1, so it cannot
# exceed 1.
THRESHOLD_GRID = [
    (0.08, 0.40),
    (0.06, 0.32),
    (0.04, 0.26),
    (0.02, 0.20),
    (0.01, 0.15),
]

# Annual targets among assaulted women.
#
# DENUNCIATION is an INTERVAL, not a point, because no Brazilian survey reports
# P(report to police | IPV in the last 12 months) directly. FBSP Visível e Invisível 5th
# ed. (2025), Gráfico 14 p.41, gives attitudes after "a agressão mais grave dos últimos 12
# meses":
#   0.142  denounced at a Delegacia da Mulher  -> LOWER bound, since it omits the 0.103
#          who went to an ordinary delegacia, and a boletim there produces an MPU request
#          just as a DEAM one does (the model's `denounce` is a police record: that is what
#          P_MPU_REQUESTED = 0.653, from the POA police microdata, operates on).
#   0.257  "procurou órgão oficial", deduplicated -> UPPER bound, right channel but wrong
#          population: Gráfico 10 p.36 shows the perpetrator of that most serious
#          aggression is a partner or ex-partner in only about half of cases, the rest
#          being siblings, friends, stepparents, parents (0.052) and children (0.030).
# FBSP publishes no attitude-by-perpetrator cross-tab, so the IPV-specific rate cannot be
# recovered from the report. The model is required to land inside the interval.
TARGET_DENOUNCE_LO = 0.142
TARGET_DENOUNCE_HI = 0.257
TARGET_DENOUNCE = (TARGET_DENOUNCE_LO + TARGET_DENOUNCE_HI) / 2

# SEPARATION is 0.15 with a band of 0.03. It has no direct Brazilian source. DataSenado
# 2025 corroborates it indirectly: 79% of women assaulted by a husband/partner are
# ex-partners at interview. That is a STOCK where 0.15 is a FLOW, and a constant annual
# hazard h over T years of exposure gives cumulative 1-(1-h)^T, so h = 0.15 implies 0.80
# at T = 10. The stock is held out as a validation moment (separation_stock.py).
TARGET_BREAK = 0.15
TARGET_BAND = 0.03


# Stage 2 sweeps the share at a fixed threshold range. The share is the only dial that
# trades denunciation for separation. It is drawn per ACT while the targets are per-woman
# flags, so repeat actors dilute both rates and the share that hits the pair of targets
# need not equal their ratio.
SHARE_GRID = [0.40, 0.423, 0.44, 0.47]


def set_params(lo, hi, share=None):
    # Woman.__init__ reads settings.* at call time, but agents.py also does
    # `from settings import *`; set both so nothing depends on which binding wins.
    settings.ACTION_THRESHOLD_MIN, settings.ACTION_THRESHOLD_MAX = lo, hi
    agents.ACTION_THRESHOLD_MIN, agents.ACTION_THRESHOLD_MAX = lo, hi
    if share is not None:
        # trigger_call_help reads the bare module global, bound by `from settings import *`.
        settings.ACTION_DENOUNCE_SHARE = share
        agents.ACTION_DENOUNCE_SHARE = share


def run_once(metro, steps):
    """Step a model for `steps` months, accumulating the annual victim-side counts.

    Mirrors ga.main's year boundary — update_incarcerated, then reset the annual
    counters — but skips the AREAP/municipio aggregation, which this sweep does not use.
    The reset is not optional: Woman.peer_action_norm reads those counters.
    """
    home = M.Home(metro=metro)
    home.verbose = False
    years = []

    # model.step() prints the current date every step; silence it for the sweep.
    stdout_print = builtins.print
    builtins.print = lambda *a, **k: None
    try:
        for step in range(1, steps + 1):
            home.step()
            if step % 12 == 0:
                home.update_incarcerated()
                years.append(ga.action_year_counts(home))
                ga.reset_annual_counters(home)
    finally:
        builtins.print = stdout_print

    return years


def measure(all_years):
    """Pool woman-years across every year and iteration.

    Pooling rather than averaging annual rates weights each year by its number of
    assaulted women, which is what the empirical rates are: a share of victims, not a
    mean of yearly shares.
    """
    total = {k: sum(y[k] for y in all_years)
             for k in ('women', 'attacked', 'assaults', 'acted',
                       'denounced', 'broke', 'remained',
                       'action_events', 'denounce_events', 'break_events')}
    props = np.array([p for y in all_years for p in y['props']])
    thresholds = np.array([t for y in all_years for t in y['thresholds']])

    n = total['attacked']
    rate = (lambda x: x / n if n else float('nan'))

    row = {
        'ACTION_THRESHOLD_MIN': settings.ACTION_THRESHOLD_MIN,
        'ACTION_THRESHOLD_MAX': settings.ACTION_THRESHOLD_MAX,
        'ACTION_DENOUNCE_SHARE': settings.ACTION_DENOUNCE_SHARE,
        'ACTION_EXPERIENCE_REF': settings.ACTION_EXPERIENCE_REF,
        'woman_years': total['women'],
        'attacked_woman_years': n,
        'assaults_per_attacked_year': (total['assaults'] / n) if n else float('nan'),
        # The moment the threshold range governs.
        'act_rate': rate(total['acted']),
        # The two calibration targets.
        'denounce_rate': rate(total['denounced']),
        'break_rate': rate(total['broke']),
        'remain_rate': rate(total['remained']),
        # Composition, at the event level the parameter actually governs. It should come
        # back at ACTION_DENOUNCE_SHARE unless Break acts are being dropped.
        'achieved_denounce_share': (
            total['denounce_events'] / (total['denounce_events'] + total['break_events'])
            if (total['denounce_events'] + total['break_events']) else float('nan')),
        # Acts that crossed the threshold but ended nothing and reported nothing, because
        # the Break branch found no relationship. Should be ~0; if it is not, the woman-
        # level break_rate is biased down and the model.py relationship wiring is leaking.
        'acts_lost_to_no_relationship': (
            total['action_events'] - total['denounce_events'] - total['break_events']),
        'acts_per_acting_woman': (
            total['action_events'] / total['acted'] if total['acted'] else float('nan')),
        # Where the propensity sits relative to the bar, among assaulted women.
        'prop_p10': np.percentile(props, 10) if len(props) else float('nan'),
        'prop_p50': np.median(props) if len(props) else float('nan'),
        'prop_p90': np.percentile(props, 90) if len(props) else float('nan'),
        'threshold_p50': np.median(thresholds) if len(thresholds) else float('nan'),
    }
    return row


def cells(stage, share=None):
    """Grid cells as (lo, hi, share). Stage 1 walks the threshold at a fixed share;
    stage 2 walks the share at the threshold settings.py carries.

    `share` overrides the value in settings.py for the threshold stage, which is needed
    whenever the targets move: psi is fixed by the ratio of the two targets, so a new
    denunciation target implies a new psi before the level is located.
    """
    if stage == 'threshold':
        return [(lo, hi, share) for lo, hi in THRESHOLD_GRID]
    return [(settings.ACTION_THRESHOLD_MIN, settings.ACTION_THRESHOLD_MAX, s)
            for s in SHARE_GRID]


def _one_run(job):
    """One replicate, start to finish, inside a worker process.

    Sets the parameters here rather than inheriting them: pool workers are forked once and
    reused across cells, so a worker that trusted the settings it was forked with would
    run the first cell's parameters for the whole sweep.
    """
    lo, hi, share, metro, steps = job
    set_params(lo, hi, share)
    return run_once(metro, steps)


def run(metro, iterates, steps, stage='threshold', grid=None, suffix='', workers=None,
        share=None):
    out_path = f'output/action_sweep_{stage}_{metro}{suffix}.csv'
    rows = []
    for lo, hi, share in (grid if grid is not None else cells(stage, share)):
        set_params(lo, hi, share)
        print(f'\n=== cell threshold_action=U[{lo}, {hi}] '
              f'denounce_share={settings.ACTION_DENOUNCE_SHARE} ===', flush=True)

        jobs = [(lo, hi, share, metro, steps)] * iterates
        done = [0]

        def tick(_i, _r, done=done):
            done[0] += 1
            print(f'  run {done[0]}/{iterates}', flush=True)

        all_years = [y for years in parallel.map_runs(_one_run, jobs, workers=workers,
                                                      on_result=tick)
                     for y in years]

        row = measure(all_years)
        rows.append(row)
        print(f"  attacked woman-years = {row['attacked_woman_years']}"
              f"   assaults/year = {row['assaults_per_attacked_year']:.2f}", flush=True)
        d_ok = TARGET_DENOUNCE_LO <= row['denounce_rate'] <= TARGET_DENOUNCE_HI
        b_ok = abs(row['break_rate'] - TARGET_BREAK) <= TARGET_BAND
        print(f"  act rate      = {row['act_rate']:.3f}"
              f"   (admissible band {max(TARGET_DENOUNCE, TARGET_BREAK):.2f}"
              f"-{TARGET_DENOUNCE + TARGET_BREAK:.2f})", flush=True)
        print(f"  denounce rate = {row['denounce_rate']:.3f}"
              f"   [{TARGET_DENOUNCE_LO}-{TARGET_DENOUNCE_HI}] {'IN' if d_ok else 'out'}"
              f"   break rate = {row['break_rate']:.3f}"
              f"   [{TARGET_BREAK}+-{TARGET_BAND}] {'IN' if b_ok else 'out'}", flush=True)
        print(f"  propensity p10/p50/p90 = {row['prop_p10']:.3f} /"
              f" {row['prop_p50']:.3f} / {row['prop_p90']:.3f}"
              f"   vs threshold p50 {row['threshold_p50']:.3f}", flush=True)
        # Write incrementally so a long sweep is never lost.
        pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f'\nSweep done -> {out_path}')
    print('Pick the cell whose denounce_rate and break_rate sit closest to '
          f'{TARGET_DENOUNCE} and {TARGET_BREAK}, then refine ACTION_DENOUNCE_SHARE '
          'against the RATIO of the two (0.11 / 0.15 = 0.733). The share cannot be read '
          'off the ratio directly: it governs per-act draws while the targets are '
          'per-woman flags, and the overlap between them depends on how often a woman '
          'acts twice in a year (acts_per_acting_woman).')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--metro', default='PORTO ALEGRE')
    p.add_argument('-i', '--iterations', type=int, default=2)
    p.add_argument('-s', '--steps', type=int, default=144)
    p.add_argument('--stage', choices=['threshold', 'share'], default='threshold',
                   help='threshold: locate the tau_action range. '
                        'share: refine the composition at the tau_action in settings.py.')
    p.add_argument('-g', '--grid', type=float, nargs='+',
                   help='flat lo hi pairs overriding the stage grid, e.g. '
                        '-g 0.09 0.42 0.08 0.40 (threshold stage only)')
    p.add_argument('-o', '--suffix', default='',
                   help='appended to the output filename, to keep a coarse sweep')
    p.add_argument('-j', '--workers', type=int, default=None,
                   help='parallel replicates; 1 runs in this process (readable tracebacks)')
    p.add_argument('--share', type=float, default=None,
                   help='ACTION_DENOUNCE_SHARE to hold during the threshold stage, '
                        'overriding settings.py')
    a = p.parse_args()
    grid = None
    if a.grid:
        if len(a.grid) % 2:
            p.error('--grid takes lo hi pairs, so an even number of values')
        # a.share, not None: None means "leave settings.py alone", which would silently
        # run an explicit grid at the file's psi while --share was given on the line.
        grid = [(a.grid[k], a.grid[k + 1], a.share) for k in range(0, len(a.grid), 2)]
    run(a.metro, a.iterations, a.steps, a.stage, grid, a.suffix, a.workers, a.share)