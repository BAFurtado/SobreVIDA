"""Sweep the aggressor threshold and record the base-prevalence moments.

The share of men who ever offend, propagated through partner turnover, sets the share of
women ever attacked — the substrate every downstream moment (denunciation, MPU grants,
femicide composition) rests on. Target 0.267 (FBSP 5th ed.); see settings.THRESHOLD_MIN.

The sweep is over the CENTRE of the tau draw at a FIXED WIDTH, because prevalence
identifies only the centre: it pins where the draw sits relative to the disposition
distribution, not how wide it is. The width is fixed by argument (0.15, see settings)
and held fixed here.

Measures, per grid cell:
  - share of women attacked at least once (lifetime prevalence) — the moment being fitted
  - the covariate gradient of the outcome, which is what the narrow width buys and the
    reason to read every prevalence cell on two numbers rather than one
  - share of men ever violent and ever holding an MPU, dispersion of disposition, its
    lifetime peak, and the accumulated stress stock

Usage (from project root):
    python violence/prevalence_sweep.py -i 5 -s 144

Writes output/prevalence_sweep_<metro>.csv incrementally (one row per grid cell).
"""
import argparse
import builtins
import os
import sys

sys.path.append(os.path.abspath('violence'))

import pandas as pd

import agents as agents_mod
import model as M
import parallel
import settings
import generalization_aps as ga
from agents import Person, Men

# Width of the uniform tau draw, held fixed across the sweep — see the module docstring.
TAU_WIDTH = 0.15

# Centres to try. Under the stochastic trigger the response is steep: at width 0.15 a 0.01
# step in the centre moves prevalence by roughly 0.045, so the grid is fine and short
# rather than wide. It brackets the target 0.267 on both sides.
CENTRE_GRID = [1.000, 1.005, 1.010, 1.015, 1.020]

_base_calculate_disposition = Person.calculate_disposition


def _track(self):
    """Accumulate each agent's disposition series alongside the normal update.

    Both the peak and the mean are needed: the peak is what a deterministic trigger keys
    on, the mean is the summary that actually carries the agent's covariates.
    """
    _base_calculate_disposition(self)
    d = self.disposition
    if not hasattr(self, '_disp_n'):
        self._disp_n = 0
        self._disp_sum = 0.0
        self.max_disposition = -1.0
        # Tau as drawn at creation: receive_mpu mutates self.threshold later, and only for
        # men who were already violent, which would bias any comparison against it.
        self._tau0 = self.threshold
    self._disp_n += 1
    self._disp_sum += d
    if d > self.max_disposition:
        self.max_disposition = d


Person.calculate_disposition = _track


def set_tau(centre, width=TAU_WIDTH):
    settings.THRESHOLD_MIN = centre - width / 2
    settings.THRESHOLD_MAX = centre + width / 2


def set_boost(boost):
    """Set the MPU threshold boost, when the sweep is over deterrence rather than tau.

    The mechanism reads the global bound in `agents` by `from settings import *`, so
    setting only `settings` would leave the model running the imported value; set both.
    """
    if boost is None:
        return
    settings.MPU_THRESHOLD_BOOST = boost
    agents_mod.MPU_THRESHOLD_BOOST = boost


def run_once(metro, steps):
    home = M.Home(metro=metro)
    home.verbose = False
    # model.step() prints the current date every step; silence it for the sweep.
    stdout_print = builtins.print
    builtins.print = lambda *a, **k: None
    try:
        for step in range(1, steps + 1):
            home.step()
            if step % 12 == 0:
                home.update_incarcerated()
                # Not optional: Woman.peer_action_norm reads denounce/break_times/
                # remain_times as within-year quantities. Without the reset they
                # accumulate over the whole run, the peer norm drifts up, and this
                # harness calibrates against a victim side more active than the one
                # ga.main() runs.
                ga.reset_annual_counters(home)
    finally:
        builtins.print = stdout_print
    return home


def collect(home):
    """One record per adult man, dead agents included.

    Restricting to survivors would drop exactly the men whose lives ran long enough for a
    lifetime maximum to accumulate.
    """
    rows = []
    for agent in list(home.schedule.agents_t) + list(home.schedule.dead_agents):
        if not isinstance(agent, Men) or not hasattr(agent, '_disp_n'):
            continue
        if agent.age <= 18 or agent._disp_n < 12:
            continue
        rows.append({
            'peak': agent.max_disposition,
            'mean': agent._disp_sum / agent._disp_n,
            'tau': agent._tau0,
            'violent': 1 if agent.n_attacks > 0 else 0,
            'n_attacks': agent.n_attacks,
            'ever_mpu': 1 if getattr(agent, 'ever_had_mpu', False) else 0,
            'years_study': agent.years_study,
            'age': agent.age,
            'stock': agent.accumulated_stress,
            'societal': agent.societal,
        })
    return rows


def run_stats(home):
    """Reduce one finished Home to the scalars measure() needs.

    Called in the worker process: a Home holds mesa schedule references that do not
    pickle, so nothing but plain numbers may cross back to the parent.
    """
    # Adults only, on both sides. The survey this moment is calibrated against (FBSP
    # Visível e Invisível, 5th ed.) reports lifetime intimate-partner prevalence among
    # women aged 16 or more; children are instantiated as Men/Woman too, so counting every
    # female agent would put girls in the denominator and bias the share down.
    women = [a for a in list(home.schedule.agents_t) + list(home.schedule.dead_agents)
             if getattr(a, 'gender', None) == 'female' and a.age > 18]

    return {
        'women': len(women),
        # Lifetime, so it must read the never-reset flag: got_attacked is cleared
        # every year by the reset above, and would report only the final year's victims.
        'attacked': sum(1 for w in women if getattr(w, 'ever_attacked', False)),
        'mpu_granted': home.mpu_granted_count,
        'mpu_violated': home.mpu_grant_violated_count,
        'femicides': home.femicide_count,
    }


def measure(stats, men_rows):
    """Pool the per-run scalars and the per-man records of one grid cell."""
    total = {k: sum(s[k] for s in stats)
             for k in ('women', 'attacked', 'mpu_granted', 'mpu_violated', 'femicides')}
    women_n, attacked = total['women'], total['attacked']

    df = pd.DataFrame(men_rows)
    # A model of the outcome that ignores the agent's entire history. Agreement near 1
    # means the trigger is a lottery on the tau draw and the mechanism upstream is wasted.
    lottery = (df.tau < df.peak.median()).astype(int)

    granted, violated = total['mpu_granted'], total['mpu_violated']

    return {
        'THRESHOLD_MIN': settings.THRESHOLD_MIN,
        'THRESHOLD_MAX': settings.THRESHOLD_MAX,
        'centre': (settings.THRESHOLD_MIN + settings.THRESHOLD_MAX) / 2,
        'width': settings.THRESHOLD_MAX - settings.THRESHOLD_MIN,
        'TRIGGER_SLOPE': settings.TRIGGER_SLOPE,
        'MPU_THRESHOLD_BOOST': settings.MPU_THRESHOLD_BOOST,
        'W_MACHISM': settings.W_MACHISM,
        'STRESS_DECAY': settings.STRESS_DECAY,
        'iterations': len(stats),
        'women': women_n,
        'men': len(df),
        # The moment being fitted (target 0.267)
        'share_women_ever_attacked': attacked / women_n if women_n else float('nan'),
        'share_men_ever_violent': df.violent.mean(),
        'attacks_per_violent_man': (df.n_attacks.sum() / df.violent.sum()
                                    if df.violent.sum() else float('nan')),
        'share_men_ever_mpu': df.ever_mpu.mean(),
        # Does the outcome see the agent's life, or only their tau draw?
        'corr_violent_years_study': df.violent.corr(df.years_study),
        'corr_violent_age': df.violent.corr(df.age),
        'corr_violent_mean_disp': df.violent.corr(df['mean']),
        'corr_violent_tau': df.violent.corr(df.tau),
        # Reference: the gradient available upstream, which the trigger may or may not
        # pass on. Dichotomisation attenuates it by ~0.76 at this base rate.
        'corr_meandisp_years_study': df['mean'].corr(df.years_study),
        'lottery_agreement': (lottery == df.violent).mean(),
        'peak_disposition_p50': df.peak.median(),
        'peak_disposition_p90': df.peak.quantile(0.9),
        'peak_sd': df.peak.std(),
        'mean_sd': df['mean'].std(),
        'stock_p50': df.stock.median(),
        'stock_sd': df.stock.std(),
        'societal_p50': df.societal.median(),
        'societal_sd': df.societal.std(),
        'mpu_granted': granted,
        'non_compliance_rate': (violated / granted) if granted else float('nan'),
        'femicides': total['femicides'],
    }


def _one_run(job):
    """One replicate, start to finish, inside a worker process.

    Sets tau here rather than inheriting it: pool workers are forked once and reused
    across cells, so a worker that trusted the settings it was forked with would run the
    first cell's parameters for the whole sweep.
    """
    centre, width, boost, metro, steps = job
    set_tau(centre, width)
    set_boost(boost)
    home = run_once(metro, steps)
    return collect(home), run_stats(home)


def run(metro, iterates, steps, centres=CENTRE_GRID, width=TAU_WIDTH, suffix='',
        workers=None, boosts=None):
    """Sweep the tau centre, or — when `boosts` is given — the MPU deterrence boost.

    The two share one harness because measure() already reports both the prevalence
    moment and the MPU non-compliance moment off the same run: sweeping the boost needs a
    parameter to vary, not a second collector. When boosts is given the tau centre is held
    at whichever single value `centres` carries.
    """
    out_path = f'output/prevalence_sweep_{metro}{suffix}.csv'
    rows = []
    cells = ([(centres[0], b) for b in boosts] if boosts
             else [(c, None) for c in centres])
    for centre, boost in cells:
        set_tau(centre, width)
        set_boost(boost)
        print(f'\n=== cell tau=U[{settings.THRESHOLD_MIN:.4f}, '
              f'{settings.THRESHOLD_MAX:.4f}] (centre {centre}, width {width})'
              f' boost={settings.MPU_THRESHOLD_BOOST} ===', flush=True)
        jobs = [(centre, width, boost, metro, steps)] * iterates
        done = [0]

        def tick(_i, _r, done=done):
            done[0] += 1
            print(f'  run {done[0]}/{iterates}', flush=True)

        results = parallel.map_runs(_one_run, jobs, workers=workers, on_result=tick)
        men_rows = [r for rows_, _ in results for r in rows_]
        stats = [s for _, s in results]
        row = measure(stats, men_rows)
        rows.append(row)
        print(f"  women ever attacked = {row['share_women_ever_attacked']:.3f}"
              f"   (target 0.267)", flush=True)
        print(f"  corr(violent, years_study) = {row['corr_violent_years_study']:+.3f}"
              f"   (upstream {row['corr_meandisp_years_study']:+.3f})", flush=True)
        print(f"  attacks per violent man = {row['attacks_per_violent_man']:.2f}"
              f"   men ever MPU = {row['share_men_ever_mpu']:.3f}", flush=True)
        print(f"  MPU grants = {row['mpu_granted']}"
              f"   non-compliance = {row['non_compliance_rate']:.3f}   (target 0.232)"
              f"   femicides = {row['femicides']}", flush=True)
        # Write incrementally so a long sweep is never lost.
        pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'\nSweep done -> {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--metro', default='PORTO ALEGRE')
    p.add_argument('-i', '--iterations', type=int, default=2)
    p.add_argument('-s', '--steps', type=int, default=144)
    p.add_argument('-c', '--centres', type=float, nargs='+', default=CENTRE_GRID)
    p.add_argument('-w', '--width', type=float, default=TAU_WIDTH)
    p.add_argument('-o', '--suffix', default='',
                   help='appended to the output filename, to keep a baseline sweep')
    p.add_argument('-j', '--workers', type=int, default=None,
                   help='parallel replicates; 1 runs in this process (readable tracebacks)')
    p.add_argument('-b', '--boosts', type=float, nargs='+',
                   help='sweep MPU_THRESHOLD_BOOST instead of the tau centre, holding tau '
                        'at the single value given by --centres')
    a = p.parse_args()
    run(a.metro, a.iterations, a.steps, a.centres, a.width, a.suffix, a.workers, a.boosts)