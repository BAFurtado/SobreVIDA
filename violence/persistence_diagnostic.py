"""Where does repeated victimisation get lost — inside the year, or between years?

FBSP Visível e Invisível 5th ed. p.38 gives, for the MOST SERIOUS violence of the last 12 months alone,
51.9% one or two times, 35.0% three to six, 10.7% seven to ten and 2.4% more than ten —
a mean of ~3.55 assaults per victim per YEAR (>=2.58 taking the floor of each band), and a
lower bound twice over, since it counts only the most serious form and excludes the 44.8%
who could not say how often.

The lifetime figure factors exactly:

    assaults per victim (lifetime) = years with any assault x assaults per violent year

so the two candidate failures are separable and this harness separates them. The first
factor is CONTINUITY between years, the second INTENSITY within the year, and only the
second is what the FBSP band table measures.

The aggressor side is measured on the same runs, because a violent man's assaults are
months spent above his own threshold: with TRIGGER_SLOPE = 0.02 the hazard is ~1 within a
few hundredths of the threshold, so `months above threshold` and `assaults` should agree
unless something is taking him back down.

Usage (from project root):
    python violence/persistence_diagnostic.py -m "PORTO ALEGRE" -i 5 -s 144 -j 5

Writes output/persistence_diagnostic_<metro>.csv (one row of summary moments) and
output/persistence_victim_years_<metro>.csv (the woman-year microdata behind it).
"""
import argparse
import builtins
import os
import sys
from itertools import product

sys.path.append(os.path.abspath('violence'))

import numpy as np
import pandas as pd

import agents as agents_mod
import model as M
import parallel
import settings
import generalization_aps as ga
from agents import Men, Person, Woman

_12_MONTHS = 12

# The Classification weights at sixteen digits, so the four-digit values in settings can
# be tested against them.
LEGACY_CLASSIFICATION = (0.4247638278222363, 0.24510988891416424, 0.03534022141538644)

_base_calculate_disposition = Person.calculate_disposition


def _track(self):
    """Count, per man, the months his disposition sits above his own threshold.

    Placed on calculate_disposition because it runs every step for every agent, whereas
    `act` runs only while he has a partner or an ex-partner tie — and the gap between
    those two counts is itself part of the answer. Read after the update, i.e. on the same
    value violence_hazard will see this step.
    """
    _base_calculate_disposition(self)
    if not isinstance(self, Men):
        return
    if self.disposition > self.threshold:
        self._months_above_tau = getattr(self, '_months_above_tau', 0) + 1
    if self.is_monitored:
        self._ever_monitored = True


Person.calculate_disposition = _track


def _all_agents(home):
    return list(home.schedule.agents_t) + list(home.schedule.dead_agents)


def run_once(metro, steps):
    """Step the model a year at a time, snapshotting every woman at each boundary.

    The snapshot is taken BEFORE ga.reset_annual_counters, which clears the annual
    counters and — the reason this cannot be reconstructed afterwards — empties
    `schedule.dead_agents`, so a woman killed during the year is only visible here.
    """
    home = M.Home(metro=metro)
    home.verbose = False
    stdout_print = builtins.print
    builtins.print = lambda *a, **k: None

    # unique_id -> cumulative assaults at the end of the previous year, so the per-year
    # count is a difference of a counter that is never reset.
    prev_cum = {}
    rows = []
    try:
        for step in range(1, steps + 1):
            home.step()
            if step % _12_MONTHS:
                continue
            year = step // _12_MONTHS
            home.update_incarcerated()
            for agent in _all_agents(home):
                if not isinstance(agent, Woman) or agent.age <= 18:
                    continue
                cum = agent.assaults_lifetime
                wid = agent.unique_id
                rows.append({
                    'wid': wid,
                    'year': year,
                    'assaults': cum - prev_cum.get(wid, 0),
                    # Exposure: a woman with neither partner nor ex-partner tie cannot be
                    # assaulted at all in this model, so she is not at risk this year.
                    'exposed': int(agent.spouse is not None or agent.ex_partner is not None),
                    'dead': int(agent.is_dead),
                })
                prev_cum[wid] = cum
            ga.reset_annual_counters(home)

        men = []
        for agent in _all_agents(home):
            if not isinstance(agent, Men) or agent.age <= 18:
                continue
            if agent.n_attacks:
                men.append({
                    'n_attacks': agent.n_attacks,
                    'months_above': getattr(agent, '_months_above_tau', 0),
                    'ever_mpu': int(agent.ever_had_mpu),
                    'ever_monitored': int(getattr(agent, '_ever_monitored', False)),
                    'ever_arrested': int(agent.times_arrested > 0),
                })
        # Lifetime prevalence on the SAME denominator prevalence_sweep.run_stats uses —
        # adult women alive at the end plus those who died in the final year, the annual
        # reset having cleared the earlier ones. Carried here so the collateral damage of
        # an escalation step on the calibrated moment is visible in the same table.
        women = [a for a in _all_agents(home)
                 if getattr(a, 'gender', None) == 'female' and a.age > 18]
        stats = {
            'women': len(women),
            'attacked': sum(1 for w in women if getattr(w, 'ever_attacked', False)),
            'mpu_granted': home.mpu_granted_count,
            'mpu_violated': home.mpu_grant_violated_count,
            'mpu_violated_post_sep': home.mpu_grant_violated_post_sep,
            'mpu_violated_in_rel': home.mpu_grant_violated_in_rel,
            'breach_delays': home.mpu_breach_delay_months,
            'femicides': home.femicide_count,
            'post_sep_contacts': home.post_sep_contact_count,
            'post_sep_assaults': home.post_sep_assault_count,
            'arrests': home.arrest_count,
        }
    finally:
        builtins.print = stdout_print

    return rows, men, stats


def _apply(overrides):
    """Set a cell's parameters on both modules that can be holding them.

    `agents` pulls most settings in by name at import time, so assigning on `settings`
    alone would leave the model running the imported value — the escalation settings are
    the exception, read through the `settings` module at call time. Writing both is
    harmless where only one is live, and is the only thing that makes a sweep axis
    actually bite.
    """
    for name, value in overrides.items():
        # A dotted name is a field of a settings dataclass (Classification.HIGH). Both
        # modules hold the same class object, so one assignment reaches agents too.
        if '.' in name:
            holder, field = name.split('.')
            setattr(getattr(settings, holder), field, value)
            continue
        setattr(settings, name, value)
        if hasattr(agents_mod, name):
            setattr(agents_mod, name, value)


def _one_run(job):
    """One replicate in a worker process, with this cell's parameters set locally.

    Workers are forked once and reused across cells, so the parameters are set here and
    not inherited.
    """
    metro, steps, overrides = job
    _apply(overrides)
    return run_once(metro, steps)


def measure(vy, men, stats):
    """Reduce the woman-year panel to the moments that separate the two failures."""
    victims = vy.groupby('wid').assaults.sum()
    victims = victims[victims > 0]
    n_victims = len(victims)

    violent_years = vy[vy.assaults > 0]
    # Years each victim was alive and at risk, so that "years with violence" is read
    # against exposure rather than against the full 12-year window.
    exposed_years = vy[vy.exposed == 1].groupby('wid').size()

    # Continuity: of the victim-years followed by another year of the same woman's life,
    # how many of those next years also carry an assault. This is the transition
    # probability the between-year story is about.
    nxt = vy.sort_values(['wid', 'year']).copy()
    nxt['next_assaults'] = nxt.groupby('wid').assaults.shift(-1)
    paired = nxt.dropna(subset=['next_assaults'])
    after_violent = paired[paired.assaults > 0]
    after_calm = paired[paired.assaults == 0]

    # FBSP p.38 bands, on the same denominator the survey uses: victims of the year.
    bands = pd.cut(violent_years.assaults, [0, 2, 6, 10, np.inf],
                   labels=['1-2', '3-6', '7-10', '>10'])
    band_share = bands.value_counts(normalize=True).reindex(['1-2', '3-6', '7-10', '>10'])

    dfm = pd.DataFrame(men)
    tot = {k: sum(s[k] for s in stats)
           for k in stats[0] if not isinstance(stats[0][k], list)}
    delays = np.array([d for s in stats for d in s['breach_delays']])

    out = {
        'ESCALATION_STEP': settings.ESCALATION_STEP,
        'ESCALATION_MAX': settings.ESCALATION_MAX,
        'PATROL_COVERAGE': settings.PATROL_COVERAGE,
        'POST_SEP_CONTACT_BASE': settings.POST_SEP_CONTACT_BASE,
        'MPU_MONITORING_LEVEL': settings.MPU_MONITORING_LEVEL,
        'MPU_THRESHOLD_BOOST': settings.MPU_THRESHOLD_BOOST,
        'women_years': len(vy),
        'victims': n_victims,
        'violent_woman_years': len(violent_years),
        # The lifetime number, reproduced here so the factorisation below
        # can be checked against it.
        'assaults_per_victim_lifetime': victims.mean(),
        # FACTOR 1 — continuity between years.
        'violent_years_per_victim': len(violent_years) / n_victims,
        'exposed_years_per_victim': exposed_years.reindex(victims.index).mean(),
        'share_exposed_years_violent': (len(violent_years)
                                        / exposed_years.reindex(victims.index).sum()),
        # FACTOR 2 — intensity within the year. This is the FBSP construct: >= 2.58.
        'assaults_per_violent_year': violent_years.assaults.mean(),
        'p_violent_year_after_violent': (after_violent.next_assaults > 0).mean(),
        'p_violent_year_after_calm': (after_calm.next_assaults > 0).mean(),
        'band_1_2': band_share['1-2'],
        'band_3_6': band_share['3-6'],
        'band_7_10': band_share['7-10'],
        'band_gt10': band_share['>10'],
        # Aggressor side, on the same runs.
        'violent_men': len(dfm),
        'attacks_per_violent_man': dfm.n_attacks.mean(),
        'months_above_tau_per_violent_man': dfm.months_above.mean(),
        'share_violent_men_ever_mpu': dfm.ever_mpu.mean(),
        'share_violent_men_ever_monitored': dfm.ever_monitored.mean(),
        'share_violent_men_ever_arrested': dfm.ever_arrested.mean(),
        # Collateral: escalation is downstream of the trigger, so it must not be read
        # without the moment the trigger was calibrated against (target 0.267).
        'share_women_ever_attacked': tot['attacked'] / tot['women'],
        'mpu_granted': tot['mpu_granted'],
        'non_compliance_rate': (tot['mpu_violated'] / tot['mpu_granted']
                                if tot['mpu_granted'] else float('nan')),
        # Same denominator, split by the conduct that breached the order first.
        'non_compliance_post_sep': (tot['mpu_violated_post_sep'] / tot['mpu_granted']
                                    if tot['mpu_granted'] else float('nan')),
        'non_compliance_in_rel': (tot['mpu_violated_in_rel'] / tot['mpu_granted']
                                  if tot['mpu_granted'] else float('nan')),
        # Timing of the breach within the order. If deterrence needs several months to
        # bite and the breaches land in month 1, strengthening it cannot move the rate.
        'breach_delay_median': np.median(delays) if len(delays) else float('nan'),
        'breach_share_first_month': (delays <= 1).mean() if len(delays) else float('nan'),
        'breach_share_first_quarter': (delays <= 3).mean() if len(delays) else float('nan'),
        'femicides': tot['femicides'],
        # Interruption channel: arrests drawn on breaches, and the share of granted orders
        # that produced one. The second is what the NEV source reports directly (0.040).
        'arrests': tot['arrests'],
        'arrests_per_grant': (tot['arrests'] / tot['mpu_granted']
                              if tot['mpu_granted'] else float('nan')),
        'P_ARREST_ON_BREACH': settings.P_ARREST_ON_BREACH,
        'ARREST_DURATION': settings.ARREST_DURATION,
        'post_sep_contacts': tot['post_sep_contacts'],
        'post_sep_assaults': tot['post_sep_assaults'],
        'THRESHOLD_MIN': settings.THRESHOLD_MIN,
        'THRESHOLD_MAX': settings.THRESHOLD_MAX,
        'TRIGGER_SLOPE': settings.TRIGGER_SLOPE,
        'ACTION_THRESHOLD_MIN': settings.ACTION_THRESHOLD_MIN,
        'ACTION_THRESHOLD_MAX': settings.ACTION_THRESHOLD_MAX,
    }
    return out


def _report(row):
    print(f"\n victims                          {row['victims']}")
    print(f" assaults per victim (lifetime)   {row['assaults_per_victim_lifetime']:.2f}")
    print(f"   = violent years per victim     {row['violent_years_per_victim']:.2f}"
          f"   (of {row['exposed_years_per_victim']:.1f} exposed years)")
    print(f"   x assaults per violent year    {row['assaults_per_violent_year']:.2f}"
          f"   (FBSP p.38: >= 2.58)")
    print(f" P(violent year | violent year-1) {row['p_violent_year_after_violent']:.3f}"
          f"   | calm year-1 {row['p_violent_year_after_calm']:.3f}")
    print(f" FBSP bands  1-2 {row['band_1_2']:.3f} (0.519) | 3-6 {row['band_3_6']:.3f}"
          f" (0.350) | 7-10 {row['band_7_10']:.3f} (0.107) | >10 {row['band_gt10']:.3f}"
          f" (0.024)")
    print(f" attacks per violent man          {row['attacks_per_violent_man']:.2f}"
          f"   months above tau {row['months_above_tau_per_violent_man']:.2f}")
    print(f" violent men ever monitored       "
          f"{row['share_violent_men_ever_monitored']:.3f}"
          f"   ever MPU {row['share_violent_men_ever_mpu']:.3f}")
    print(f" women ever attacked              "
          f"{row['share_women_ever_attacked']:.3f}   (calibrated at 0.267)")
    print(f" MPU grants {row['mpu_granted']}   non-compliance "
          f"{row['non_compliance_rate']:.3f}   (target 0.21 +/-0.04; post-sep "
          f"{row['non_compliance_post_sep']:.3f} + in-rel "
          f"{row['non_compliance_in_rel']:.3f})")
    print(f" arrests {row['arrests']}   per grant {row['arrests_per_grant']:.3f}"
          f" (NEV 0.040)   violent men ever arrested "
          f"{row['share_violent_men_ever_arrested']:.3f}")
    print(f" breach delay median {row['breach_delay_median']:.1f} months"
          f"   within 1st month {row['breach_share_first_month']:.3f}"
          f"   within 1st quarter {row['breach_share_first_quarter']:.3f}")
    print(f" femicides {row['femicides']}"
          f"   post-sep contacts {row['post_sep_contacts']}"
          f" / assaults {row['post_sep_assaults']}", flush=True)


def run(metro, iterates, steps, workers=None, suffix='', axes=None, keep_panel=True):
    """Run the cartesian product of the swept axes.

    `axes` maps a settings name to the values it takes; an empty mapping runs one cell
    with whatever settings.py carries, which is how the baseline row is produced without
    a special case.
    """
    axes = {k: v for k, v in (axes or {}).items() if v}
    names = list(axes)
    out_path = f'output/persistence_diagnostic_{metro}{suffix}.csv'
    rows = []
    for combo in product(*(axes[n] for n in names)) if names else [()]:
        overrides = dict(zip(names, combo))
        print(f'\n=== cell {overrides or "baseline (settings.py as written)"} ===',
              flush=True)
        jobs = [(metro, steps, overrides)] * iterates
        done = [0]

        def tick(_i, _r, done=done):
            done[0] += 1
            print(f'  run {done[0]}/{iterates}', flush=True)

        results = parallel.map_runs(_one_run, jobs, workers=workers, on_result=tick)

        # Woman ids repeat across replicates; offset them so a groupby cannot merge
        # two different women from two different runs.
        frames, men, stats = [], [], []
        for k, (vrows, men_rows, run_stats) in enumerate(results):
            df = pd.DataFrame(vrows)
            df['wid'] = df.wid + k * 10_000_000
            df['run'] = k
            frames.append(df)
            men.extend(men_rows)
            stats.append(run_stats)
        vy = pd.concat(frames, ignore_index=True)

        # The parameters measure() reads off the settings module are the parent's,
        # not the worker's, so write this cell's values back before reducing.
        _apply(overrides)
        row = measure(vy, men, stats)
        rows.append(row)
        _report(row)
        if keep_panel:
            vy.to_csv(f'output/persistence_victim_years_{metro}{suffix}.csv',
                      index=False)
        pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'\nDone -> {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--metro', default='PORTO ALEGRE')
    p.add_argument('-i', '--iterations', type=int, default=3)
    p.add_argument('-s', '--steps', type=int, default=144)
    p.add_argument('-j', '--workers', type=int, default=None)
    p.add_argument('-o', '--suffix', default='')
    p.add_argument('-e', '--escalations', type=float, nargs='+',
                   help='ESCALATION_STEP values to sweep')
    p.add_argument('-x', '--maxes', type=float, nargs='+',
                   help='ESCALATION_MAX values to sweep (cap on the erosion per tie)')
    p.add_argument('-p', '--patrols', type=float, nargs='+',
                   help='PATROL_COVERAGE values to sweep (share of granted orders the '
                        'patrol reaches; 0 leaves every order unenforced)')
    p.add_argument('-c', '--contacts', type=float, nargs='+',
                   help='POST_SEP_CONTACT_BASE values to sweep')
    p.add_argument('-M', '--monitoring', type=float, nargs='+',
                   help='MPU_MONITORING_LEVEL values to sweep')
    p.add_argument('-B', '--boosts', type=float, nargs='+',
                   help='MPU_THRESHOLD_BOOST values to sweep')
    p.add_argument('-a', '--arrests', type=float, nargs='+',
                   help='P_ARREST_ON_BREACH values to sweep (custody on a breach; 0 '
                        'removes the interruption channel)')
    p.add_argument('-d', '--durations', type=float, nargs='+',
                   help='ARREST_DURATION values to sweep (months in custody)')
    p.add_argument('-L', '--legacy-classification', action='store_true',
                   help='run the inherited 16-decimal Classification weights instead of '
                        'the 4-digit ones settings.py carries, as the sensitivity test '
                        'of the rounding (settings.py Classification)')
    a = p.parse_args()
    legacy = {'Classification.HIGH': [LEGACY_CLASSIFICATION[0]],
              'Classification.MEDIUM': [LEGACY_CLASSIFICATION[1]],
              'Classification.LOW': [LEGACY_CLASSIFICATION[2]]} if \
        a.legacy_classification else {}
    run(a.metro, a.iterations, a.steps, a.workers, a.suffix,
        axes={**legacy,
              'ESCALATION_STEP': a.escalations,
              'ESCALATION_MAX': a.maxes,
              'PATROL_COVERAGE': a.patrols,
              'POST_SEP_CONTACT_BASE': a.contacts,
              'MPU_MONITORING_LEVEL': a.monitoring,
              'MPU_THRESHOLD_BOOST': a.boosts,
              'P_ARREST_ON_BREACH': a.arrests,
              'ARREST_DURATION': a.durations})