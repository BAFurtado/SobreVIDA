"""Sweep the free MPU structural parameters and record the emergent moments.

The moments (non-compliance, femicide-with-prior-MPU) are NOT tuned to: the sweep
varies MPU_THRESHOLD_BOOST x P_VIOL and reports where they land.
Directly-observed inputs (P_MPU_GRANTED, MPU_DURATION) are held fixed.

Usage (from project root):
    python violence/mpu_sweep.py -m "PORTO ALEGRE" -i 3 -s 144

Writes output/mpu_sweep_<metro>.csv incrementally (one row per grid cell).
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath('violence'))

import pandas as pd

import agents
import settings
import generalization_aps as ga

# Grid scaled to the GAP the boost has to close between a violent man's disposition and
# his threshold, not to the level of disposition. A boost wider than that gap puts the
# raised threshold out of reach and drives non-compliance to zero. Boost 0.0 is the
# no-deterrence anchor: the upper bound on non-compliance.
BOOST_GRID = [0.0, 0.02, 0.04, 0.06, 0.09, 0.13]
# P_VIOL governs what happens AFTER a violation (the couple separates), not the violation
# itself, so it is held at its default.
PVIOL_GRID = [0.40]


def set_params(boost, pviol):
    # The mechanism reads the globals bound in the `agents` module (from settings import *),
    # so override there; also set `settings` so mpu_summary reports the values in force.
    agents.MPU_THRESHOLD_BOOST = boost
    agents.P_VIOL = pviol
    settings.MPU_THRESHOLD_BOOST = boost
    settings.P_VIOL = pviol


def run(metro, iterates, steps):
    out_path = f'output/mpu_sweep_{metro}.csv'
    rows = []
    for boost in BOOST_GRID:
        for pviol in PVIOL_GRID:
            set_params(boost, pviol)
            print(f'\n=== cell boost={boost} P_VIOL={pviol} ===', flush=True)
            m = ga.main(metro=metro, iterates=iterates, steps=steps)
            rows.append(m)
            # Write incrementally so a long sweep is never lost.
            pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'\nSweep done -> {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--metro', default='PORTO ALEGRE')
    p.add_argument('-i', '--iterations', type=int, default=3)
    p.add_argument('-s', '--steps', type=int, default=144)
    a = p.parse_args()
    run(a.metro, a.iterations, a.steps)
