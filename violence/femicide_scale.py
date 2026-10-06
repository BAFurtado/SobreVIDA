"""Locate FEMICIDE_BASE analytically instead of by simulation.

The moment is nearly unobservable at the model's scale: the target rate (Anuario FBSP
2023, T27, Rio Grande do Sul — 1.72 femicides per 100k women in 2021, 1.97 in 2022)
against ~820 adult women over 12 years predicts ~0.18 events per run. Searching for the
parameter by counting deaths would need hundreds of replicates.

It does not have to be searched. `trigger_femicide` draws against `n_prior *
FEMICIDE_BASE`, linear in the parameter, so the expected number of femicides in a run is

    E[femicides] = FEMICIDE_BASE x SUM(n_prior over every assault)

and that sum is a high-count quantity — thousands of assaults per run — which ten
replicates pin down tightly. Equating it to the target gives the parameter in one pass:

    FEMICIDE_BASE = (rate / 1e5) x adult woman-years / SUM(n_prior)

The linearity is exact in the draw and only approximate in the sum, because a woman
killed early stops accumulating assaults. At the rates in play (<1 death per run) that
feedback is negligible; --confirm re-runs at the located value and reports the realised
rate, which is the test of that assumption.

Usage (from project root):
    python violence/femicide_scale.py -m "PORTO ALEGRE" -i 10 -j 10
    python violence/femicide_scale.py -m "PORTO ALEGRE" -i 10 -j 10 --confirm 2e-5

Writes output/femicide_scale_<metro>.csv.
"""
import argparse
import builtins
import os
import sys

sys.path.append(os.path.abspath('violence'))

import numpy as np
import pandas as pd

import model as M
import parallel
import settings
import generalization_aps as ga
from agents import Men, Woman

_12_MONTHS = 12

# Anuario Brasileiro de Seguranca Publica 2023, Tabela 27, Rio Grande do Sul, per 100k
# women of ALL ages — the published denominator, not this model's.
TARGET_RATES = {2021: 1.72, 2022: 1.97}

# THE PUBLISHED RATE CANNOT BE COMPARED TO THIS MODEL'S DENOMINATOR WITHOUT THESE FOUR
# FACTORS.
# The synthetic population is 1,000 couples: no children, no never-partnered women, and
# 99.4% of its woman-years are exposed to a partner or an ex-partner. The Anuario's
# denominator is every woman in Rio Grande do Sul. So the numerator is restricted to the
# deaths this model's mechanism can produce, and the denominator to the population it
# actually represents:
#
#   numerator    0.798  femicides by a partner (60.7%) or ex-partner (19.1%)
#                       — Anuario 2025, Grafico 47, Brazil 2024
#   numerator    0.955  victims aged 18+ (excludes 0.7% aged 0-11, 3.8% aged 12-17)
#                       — Anuario 2025, Grafico 43, Brazil 2024
#   denominator  0.741  adult share of the female population of RS
#                       — the model's own input, input/2010/num_people_age_gender_AP.csv
#   denominator  0.807  share of 18+ women who live in or have lived in a union
#                       — IBGE, Census 2022, Nupcialidade e Familia, Grafico 1
#
# The two numerator restrictions nearly cancel the two denominator ones: net 1.274, i.e.
# targets of 2.19 and 2.51 per 100k in the model's denominator.
DENOMINATOR_FACTORS = {'partner_or_ex': 0.798, 'adult_victim': 0.955,
                       'adult_women': 0.741, 'ever_in_union': 0.807}
TARGET_FACTOR = ((DENOMINATOR_FACTORS['partner_or_ex'] * DENOMINATOR_FACTORS['adult_victim'])
                 / (DENOMINATOR_FACTORS['adult_women'] * DENOMINATOR_FACTORS['ever_in_union']))

_base_trigger_femicide = Men.trigger_femicide


def _accumulate(self, victim):
    """Add this assault's n_prior to the run total, then draw as usual.

    Wraps rather than replaces the draw so the run stays the model's own run: the deaths
    it produces still remove women from the population, which is exactly the feedback the
    analytic step assumes away and --confirm checks.
    """
    self.model.sum_prior += victim.n_received_attacks
    return _base_trigger_femicide(self, victim)


Men.trigger_femicide = _accumulate


def _all_agents(home):
    return list(home.schedule.agents_t) + list(home.schedule.dead_agents)


def run_once(metro, steps, femicide_base):
    """Step a year at a time, counting women at each boundary and assaults throughout.

    The head count is taken BEFORE ga.reset_annual_counters, which empties
    `schedule.dead_agents` — a woman killed during the year is only visible here, and she
    was exposed for that year.

    `femicide_base` is passed to the constructor because Home takes it as a keyword whose
    default was bound to settings.FEMICIDE_BASE at import: assigning on the settings
    module, which is how every other axis is swept, would leave the run on the import-time
    value.
    """
    home = M.Home(metro=metro, femicide_base=femicide_base)
    home.verbose = False
    home.sum_prior = 0
    stdout_print = builtins.print
    builtins.print = lambda *a, **k: None

    adult_woman_years = 0
    all_woman_years = 0
    try:
        for step in range(1, steps + 1):
            home.step()
            if step % _12_MONTHS:
                continue
            home.update_incarcerated()
            for agent in _all_agents(home):
                if not isinstance(agent, Woman):
                    continue
                all_woman_years += 1
                # Same denominator the prevalence and persistence harnesses use, so the
                # rate below is read on the population this model actually represents.
                if agent.age > 18:
                    adult_woman_years += 1
            ga.reset_annual_counters(home)

        assaults = sum(a.n_attacks for a in _all_agents(home) if isinstance(a, Men))
        stats = {
            'sum_prior': home.sum_prior,
            'assaults': assaults,
            'adult_woman_years': adult_woman_years,
            'all_woman_years': all_woman_years,
            'femicides': home.femicide_count,
            'femicides_post_sep': home.femicide_post_separation,
            'femicides_ever_mpu': home.femicide_with_ever_mpu,
            'femicides_rel_mpu': home.femicide_with_relationship_mpu,
        }
    finally:
        builtins.print = stdout_print
    return stats


def _one_run(job):
    metro, steps, femicide_base = job
    return run_once(metro, steps, femicide_base)


def measure(stats, femicide_base):
    """Turn the per-run totals into the implied parameter, one value per target rate."""
    df = pd.DataFrame(stats)
    sum_prior = df.sum_prior.mean()
    awy = df.adult_woman_years.mean()

    row = {
        'FEMICIDE_BASE_run': femicide_base,
        'runs': len(df),
        'assaults_per_run': df.assaults.mean(),
        'sum_prior_per_run': sum_prior,
        'sum_prior_cv': df.sum_prior.std(ddof=1) / sum_prior if len(df) > 1 else np.nan,
        'adult_woman_years_per_run': awy,
        'all_woman_years_per_run': df.all_woman_years.mean(),
        # What the run actually produced, against what the parameter in force predicts.
        # A gap between these two is the feedback the analytic step assumes away.
        'femicides_observed': df.femicides.sum(),
        'femicides_expected': femicide_base * sum_prior * len(df),
        'rate_observed': df.femicides.sum() / df.adult_woman_years.sum() * 1e5,
    }
    row['target_factor'] = TARGET_FACTOR
    for year, rate in TARGET_RATES.items():
        # The target expressed in the model's own denominator, never the published rate.
        target = rate * TARGET_FACTOR
        row[f'target_rate_{year}'] = target
        row[f'implied_base_{year}'] = (target / 1e5) * awy / sum_prior
        row[f'expected_femicides_{year}'] = (target / 1e5) * awy
    return row


def _report(row):
    print(f"\n assaults per run                 {row['assaults_per_run']:.0f}")
    print(f" SUM n_prior per run              {row['sum_prior_per_run']:.0f}"
          f"   (cv across runs {row['sum_prior_cv']:.3f})")
    print(f" adult woman-years per run        {row['adult_woman_years_per_run']:.0f}"
          f"   (all ages {row['all_woman_years_per_run']:.0f})")
    print(f" femicides observed / expected    {row['femicides_observed']:.0f}"
          f" / {row['femicides_expected']:.2f}"
          f"   at FEMICIDE_BASE = {row['FEMICIDE_BASE_run']:.2e}")
    lo, hi = (row[f'target_rate_{y}'] for y in sorted(TARGET_RATES))
    print(f" rate observed                    {row['rate_observed']:.2f} per 100k"
          f"   (target {lo:.2f}-{hi:.2f} in THIS denominator;"
          f" published {min(TARGET_RATES.values())}-{max(TARGET_RATES.values())}"
          f" x {row['target_factor']:.3f})")
    for year, rate in TARGET_RATES.items():
        print(f" implied FEMICIDE_BASE ({year}, {rate * TARGET_FACTOR:.2f})   "
              f"{row[f'implied_base_{year}']:.3e}"
              f"   -> {row[f'expected_femicides_{year}']:.2f} events/run")


def run(metro, iterates, steps, workers=None, suffix='', femicide_base=None):
    in_force = femicide_base if femicide_base is not None else settings.FEMICIDE_BASE
    print(f'=== {metro}, {iterates} runs, FEMICIDE_BASE = {in_force:.2e} ===', flush=True)

    jobs = [(metro, steps, in_force)] * iterates
    done = [0]

    def tick(_i, _r, done=done):
        done[0] += 1
        print(f'  run {done[0]}/{iterates}', flush=True)

    stats = parallel.map_runs(_one_run, jobs, workers=workers, on_result=tick)

    row = measure(stats, in_force)
    _report(row)
    out_path = f'output/femicide_scale_{metro}{suffix}.csv'
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f'\nDone -> {out_path}')
    return row


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--metro', default='PORTO ALEGRE')
    p.add_argument('-i', '--iterations', type=int, default=10)
    p.add_argument('-s', '--steps', type=int, default=144)
    p.add_argument('-j', '--workers', type=int, default=None)
    p.add_argument('-o', '--suffix', default='')
    p.add_argument('--confirm', type=float, default=None,
                   help='run at this FEMICIDE_BASE instead of the one in settings.py')
    a = p.parse_args()
    run(a.metro, a.iterations, a.steps, a.workers, a.suffix, a.confirm)