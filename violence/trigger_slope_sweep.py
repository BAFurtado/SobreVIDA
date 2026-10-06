"""Locate TRIGGER_SLOPE: the softness of the violence trigger.

Unlike the other sweeps, this one is NOT a search for a single target. The slope trades two
things off against each other, and the cell has to be read on both at once:

  - PREVALENCE (target 0.267, FBSP 5th ed.). Any positive hazard floor compounds over 144
    steps, so a slope that is too soft leaks violence into the whole population.
  - The COVARIATE GRADIENT. A slope that is too sharp makes being violent depend on the
    lifetime peak of disposition, which is nearly homogeneous across men, so the
    socioeconomic gradient of disposition does not reach the outcome.

So the cell to pick is the softest one that still lets tau reach the prevalence target —
softest, because that is the one that carries the most of the agent's life into the outcome.

TRIGGER_SLOPE = 0 is included as the deterministic anchor: it reproduces a deterministic
trigger and gives the baseline both moments are read against.

Usage (from project root):
    python violence/trigger_slope_sweep.py -m "PORTO ALEGRE" -i 3 -s 144

Writes output/trigger_slope_sweep_<metro>.csv incrementally (one row per grid cell).
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath('violence'))

import pandas as pd

# prevalence_sweep carries the pieces every tau harness needs: run_once with the annual
# reset, the per-agent disposition tracker, and the measurement both sweeps report on.
import prevalence_sweep as ps
import agents
import settings

# Sharp to soft. 0.0 is the deterministic anchor; 0.08 is expected to be far too soft (a man
# permanently 0.1 below threshold would face a ~1 - (1 - sigma(-1.25))^144 lifetime hazard),
# and is kept in to show the leak rather than to be chosen.
SLOPE_GRID = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]


def set_slope(slope):
    # The mechanism reads the global bound in the `agents` module (imported by name), so
    # override there; also set `settings` so the row reports the value in force.
    agents.TRIGGER_SLOPE = slope
    settings.TRIGGER_SLOPE = slope


def run(metro, iterates, steps):
    out_path = f'output/trigger_slope_sweep_{metro}.csv'
    rows = []
    for slope in SLOPE_GRID:
        set_slope(slope)
        print(f'\n=== cell TRIGGER_SLOPE={slope} ===', flush=True)
        homes, men_rows = [], []
        for _ in range(iterates):
            home = ps.run_once(metro, steps)
            homes.append(home)
            men_rows += ps.collect(home)
        row = ps.measure(homes, men_rows)
        rows.append(row)
        print(f"  women ever attacked = {row['share_women_ever_attacked']:.3f}"
              f"   (target 0.267)", flush=True)
        print(f"  corr(violent, years_study) = {row['corr_violent_years_study']:+.3f}"
              f"   (upstream {row['corr_meandisp_years_study']:+.3f})", flush=True)
        print(f"  lottery agreement = {row['lottery_agreement']:.3f}", flush=True)
        # Write incrementally so a long sweep is never lost.
        pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'\nSweep done -> {out_path}')
    print('Pick the SOFTEST cell whose prevalence tau can still reach, not the cell whose '
          'prevalence is closest to target: tau is the dial for prevalence, the slope is '
          'the dial for how much of the agent lands in the outcome.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--metro', default='PORTO ALEGRE')
    p.add_argument('-i', '--iterations', type=int, default=3)
    p.add_argument('-s', '--steps', type=int, default=144)
    a = p.parse_args()
    run(a.metro, a.iterations, a.steps)