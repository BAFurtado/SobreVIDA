"""Policy arms: what the two deterrence channels are worth, city-wide and by AREAP.

Everything upstream of this file is calibration — one parameter against one moment. This
one holds the calibrated vector fixed and switches CHANNELS off, which is the only
question the model was built to answer and the one `sec:results` needs.

Three levers, and they are NESTED, not independent. The order is the judicial act; the patrol and the arrest are
what enforce it. No order, no patrol and no custody. So:

  MPU      — P_MPU_REQUESTED (POA police microdata) x P_MPU_GRANTED (CNJ 2025) decide
             whether an order exists at all, and MPU_THRESHOLD_BOOST is what the order
             alone is worth.
  PATROL   — PATROL_COVERAGE is the share of GRANTED orders the Patrulha Maria da Penha
             actually reaches, 0.75 from IPEA 2019 p.45. Covered orders get the visits
             (MPU_MONITORING_LEVEL); uncovered ones exist on paper only.
  ARREST   — P_ARREST_ON_BREACH is custody in flagrante when the order is breached, 0.17
             from NEV/Piracicaba. It is the only channel that INTERRUPTS a violent spell
             rather than lowering a hazard.

Eight arms:

  baseline           calibrated model as settings.py carries it
  no_mpu             P_MPU_REQUESTED = 0 — the order never exists
  inert_mpu          the order is granted and counted, but neither raises the threshold,
                     nor monitors, nor leads to custody (boost, monitoring, arrest = 0)
  no_arrest          P_ARREST_ON_BREACH = 0 — a breach carries no custody
  no_patrol          PATROL_COVERAGE = 0 — every order exists on paper, none is visited
  full_patrol        PATROL_COVERAGE = 1 — the policy question with a budget behind it
  no_mpu_no_patrol   no order and therefore no enforcement, the floor of the design
  more_reporting     a lower action threshold for women: more of them report (upstream)

`no_mpu` and `inert_mpu` separate two things the aggregate rate confounds: the order
EXISTING (a breach still ends the tie through P_VIOL, and it is still counted) from the
order DOING anything. `inert_mpu` therefore switches off all three consequences of the
order, custody included: an order that can still put him in jail is not inert.
P_MPU_GRANTED = 0 would be observationally equivalent to `no_mpu`: `mpu_requested_count`
is instrumentation only, and nothing in the model conditions on a request having been
made and refused.

FEMICIDE IS NOT COUNTED, IT IS PREDICTED. At the calibrated scale a run produces ~0.2
deaths, so a difference between arms measured by counting deaths would be noise. The
hazard is linear in `n_prior`, so E[femicides] = FEMICIDE_BASE x SUM(n_prior), a
thousands-strong count that ten replicates pin down — the same device femicide_scale.py
used to locate the parameter, here accumulated PER AREAP so the map is not built out of
zeros.

SENSITIVITY (--sens): the same arm table, once per value of any settings parameter, with
the deltas paired inside each value and the seed block shared across values. Meant to be
swept WIDER than the calibrated window, out to values the calibration itself rejects — the
point is to show whether the policy conclusions survive there, not to refine the parameter.
See run() for why one-at-a-time needs that caveat in this model.

Usage (from project root):
    python violence/scenarios.py -m "PORTO ALEGRE" -i 10 -s 144 -j 10
    python violence/scenarios.py -m "BELO HORIZONTE" -i 10 -s 144 -j 10
    python violence/scenarios.py -m "PORTO ALEGRE" -i 2 -s 12 -j 2 -a baseline no_mpu
    python violence/scenarios.py -m "PORTO ALEGRE" -i 10 -s 144 -j 10 \
        --sens ESCALATION_STEP 0.010 0.013 0.020 0.025 -o _sens

BELO HORIZONTE runs the parameters calibrated on PORTO ALEGRE. That is a claim of
TRANSFERENCE, not of fit, and only the demographic substrate
(input/2010 per AREAP) differs between the two cities.

Writes output/scenarios_<metro>.csv          one row per arm, plus deltas against baseline
       output/scenarios_areap_<metro>.csv    one row per arm x AREAP, pooled over years
       output/scenarios_wage_quartile_<metro>.csv  the same, pooled into wage quartiles
       output/scenarios_panel_<metro>.csv    arm x AREAP x year, the raw panel
"""
import argparse
import builtins
import os
import random
import sys

sys.path.append(os.path.abspath('violence'))

import numpy as np
import pandas as pd

import agents as agents_mod
import model as M
import parallel
import settings
import generalization_aps as ga
from agents import Men, Woman
from data import edataframe, fill_data

_12_MONTHS = 12

# Anuario Brasileiro de Seguranca Publica 2023, T27, Rio Grande do Sul, carried here so
# the femicide column can be read against the target that fixed FEMICIDE_BASE. The rates
# are already expressed in the model's own denominator (adult women in a union or ex-union
# — see femicide_scale.py for the four correction factors), which is what the panel
# counts.
TARGET_FEMICIDE_RATE = (2.19, 2.51)

# AREAP-level average family wage, the same 2010 census input the population is built
# from. It groups the areas for the spatial contrast: an AREAP holds ~12 of the model's
# agents, so a per-area rate is nearly all noise, while a wage quartile pools ~40 areas
# and ~500 agents. Wage is EXOGENOUS to the arms — grouping by a baseline OUTCOME instead
# would make the deltas mean-revert by construction.
AREAP_WAGE = pd.read_csv('violence/input/2010/average_variance_family_wages.csv', sep=';')
WAGE_GROUPS = 4

ARMS = {
    'baseline': {},
    'no_mpu': {'P_MPU_REQUESTED': 0.0},
    'inert_mpu': {'MPU_THRESHOLD_BOOST': 0.0, 'MPU_MONITORING_LEVEL': 0.0,
                  'P_ARREST_ON_BREACH': 0.0},
    'no_arrest': {'P_ARREST_ON_BREACH': 0.0},
    'no_patrol': {'PATROL_COVERAGE': 0.0},
    'full_patrol': {'PATROL_COVERAGE': 1.0},
    'no_mpu_no_patrol': {'P_MPU_REQUESTED': 0.0, 'PATROL_COVERAGE': 0.0},
    # UPSTREAM arm, and the only one that moves the mouth of the funnel. Every other arm
    # acts downstream of the report; this one asks what a city gets by making women report
    # more, and `more_reporting` x `full_patrol` is the upstream/downstream comparison.
    # The calibrated threshold stays U[0.13, 0.65] in settings.py and is overridden here
    # at run time, as full_patrol overrides coverage. U[0.095, 0.475] lands denounce_rate
    # on the CEILING of the FBSP band, 0.257 (Visível e Invisível 5th ed. p.41): 0.253 at
    # 8 replicates of action_sweep.py --stage threshold --share 0.571. The arm is "as much
    # reporting as the survey evidence will support", not an arbitrary increase.
    'more_reporting': {'ACTION_THRESHOLD_MIN': 0.095, 'ACTION_THRESHOLD_MAX': 0.475},
}

_base_trigger_femicide = Men.trigger_femicide
_base_apply_monitoring = Men.apply_monitoring


def _mark_monitored(self, *args, **kwargs):
    """Record that this man was under monitoring at least once.

    `apply_monitoring` is the single entry point, and the flag survives the spell
    expiring, which `is_monitored` does not. The only caller is `receive_mpu` under a
    patrolled order, so this equals "ever patrolled".
    """
    self._ever_monitored = True
    return _base_apply_monitoring(self, *args, **kwargs)


def _accumulate(self, victim):
    """Add this assault's `n_prior` to the run total and to the victim's AREAP.

    Wraps rather than replaces the draw, so the run remains the model's own: the deaths it
    does produce still remove women from the population. That feedback is what the
    analytic step assumes away, and at <1 death per run it is negligible.
    """
    n_prior = victim.n_received_attacks
    self.model.sum_prior += n_prior
    self.model.sum_prior_areap[victim.address] = (
        self.model.sum_prior_areap.get(victim.address, 0) + n_prior
    )
    return _base_trigger_femicide(self, victim)


Men.trigger_femicide = _accumulate
Men.apply_monitoring = _mark_monitored


def _all_agents(home):
    return list(home.schedule.agents_t) + list(home.schedule.dead_agents)


def _areap_register(home, year):
    """Aggregate the annual counters by AREAP, one record per AREAP present this year.

    `ga.year_register` does this and the municipality level too, but the municipality
    lookup filters a dataframe once per agent per year and dominates the run time. Only
    the AREAP level is wanted here, so the lookup is dropped rather than paid for.

    `Females` is adult women, attacked or not: it is the population at risk, and summing
    it over the years of a run gives the adult woman-years that every rate below divides
    by. Note that this is a HEAD COUNT at each year boundary, which is why it must be
    taken before ga.reset_annual_counters empties `schedule.dead_agents` — a woman killed
    during the year was exposed during it.
    """
    areaps = {}
    for agent in _all_agents(home):
        if agent.address not in areaps:
            areaps[agent.address] = edataframe(agent)
        else:
            fill_data(agent, areaps)
    return [dict(AREAP=addr, year=year, **counts) for addr, counts in areaps.items()]


def run_once(job):
    """One replicate: step a year at a time, snapshotting AREAP counters at each boundary.

    Workers are forked once and reused across arms, so this arm's parameters are set here
    and never inherited.

    COMMON RANDOM NUMBERS. `seed` pairs replicate k of every arm: same synthetic
    population, same partner market, same draw sequence up to the point the arms diverge.
    Without it each arm is an independent sample and every delta carries the baseline's
    own noise. Paired, the comparison is within-replicate and the noise mostly
    cancels; `_paired_stats` is what reads it that way. Note this seeds the same two
    generators parallel._seed_worker does, and for the same reason: mesa takes each
    Model's own seed from the stdlib global.
    """
    metro, steps, overrides, seed, families = job
    _apply(overrides)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))

    home = M.Home(metro=metro, femicide_base=settings.FEMICIDE_BASE,
                  initial_relationships=families)
    home.verbose = False
    home.sum_prior = 0
    home.sum_prior_areap = {}
    stdout_print = builtins.print
    builtins.print = lambda *a, **k: None

    panel = []
    # Victim-side annual moments, accumulated over the years of the run. The denominator
    # is women ASSAULTED in the year, not all women, which is the construct the FBSP
    # attitude question uses and is not recoverable from the AREAP table.
    annual = {'attacked': 0, 'denounced': 0, 'broke': 0, 'assaults': 0}
    try:
        for step in range(1, steps + 1):
            home.step()
            if step % _12_MONTHS:
                continue
            home.update_incarcerated()
            counts = ga.action_year_counts(home)
            for key in annual:
                annual[key] += counts[key]
            panel.extend(_areap_register(home, step // _12_MONTHS))
            ga.reset_annual_counters(home)

        # Lifetime prevalence on the denominator every other harness uses: adult women
        # alive at the end, plus those who died in the final year — the annual reset
        # having cleared the earlier dead.
        women = [a for a in _all_agents(home) if isinstance(a, Woman) and a.age > 18]
        men = [a for a in _all_agents(home) if isinstance(a, Men) and a.age > 18]
        violent = [m for m in men if m.n_attacks]
        stats = {
            'women': len(women),
            'ever_attacked': sum(1 for w in women if w.ever_attacked),
            'violent_men': len(violent),
            # Realised reach of the deterrence channels among the men who assault.
            # Monitoring only ever comes from a patrolled order, so
            # `monitored` <= `ever_mpu` by construction — the gap between the two is the
            # share of orders that exist on paper and are never visited.
            'violent_men_monitored': sum(1 for m in violent
                                         if getattr(m, '_ever_monitored', False)),
            'violent_men_ever_mpu': sum(1 for m in violent if m.ever_had_mpu),
            'mpu_requested': home.mpu_requested_count,
            'mpu_granted': home.mpu_granted_count,
            'mpu_patrolled': home.mpu_patrolled_count,
            'mpu_violated': home.mpu_grant_violated_count,
            # Custody drawn on breaches: the only channel that interrupts a spell.
            'arrests': home.arrest_count,
            'post_sep_contacts': home.post_sep_contact_count,
            'post_sep_assaults': home.post_sep_assault_count,
            'sum_prior': home.sum_prior,
            'femicides': home.femicide_count,
            # Annual, victim-denominated. `assaults` here is the within-year count summed
            # over years, NOT sum(m.n_attacks): the latter loses the men cleared from
            # `dead_agents` at earlier year boundaries.
            **annual,
        }
    finally:
        builtins.print = stdout_print

    return panel, stats, home.sum_prior_areap


def _apply(overrides):
    """Set an arm's parameters on both modules that can be holding them.

    `agents` pulls most settings in by name at import time, so assigning on `settings`
    alone would leave the model running the imported value. Writing both is harmless
    where only one is live, and is the only thing that makes an arm actually bite.
    """
    for name, value in overrides.items():
        setattr(settings, name, value)
        if hasattr(agents_mod, name):
            setattr(agents_mod, name, value)


def _restore(baseline):
    _apply(baseline)


# Outcomes for which the between-replicate standard error is carried. Without one, an arm
# difference cannot be told from run-to-run noise, and at ten replicates the MPU arms sit
# right at that line while the patrol arms are ten times past it.
SE_OUTCOMES = ['prevalence', 'assaults_per_woman_year', 'denounce_rate',
               'separation_rate', 'non_compliance_rate', 'femicide_rate_expected']


def _per_replicate(panel, stats, femicide_base):
    """The same outcomes formed one replicate at a time, for their dispersion.

    A replicate is the unit of independence here — within one, the women are coupled
    through the same population and the same partner market — so the standard error of an
    arm's mean is the between-replicate sd over sqrt(n), and nothing finer is honest.
    """
    df = pd.DataFrame(stats)
    df['woman_years'] = panel.groupby('run').Females.sum().to_numpy()
    grants = df.mpu_granted.replace(0, np.nan)
    attacked = df.attacked.replace(0, np.nan)
    return pd.DataFrame({
        'prevalence': df.ever_attacked / df.women,
        'assaults_per_woman_year': df.assaults / df.woman_years,
        'denounce_rate': df.denounced / attacked,
        'separation_rate': df.broke / attacked,
        'non_compliance_rate': df.mpu_violated / grants,
        'femicide_rate_expected': femicide_base * df.sum_prior / df.woman_years * 1e5,
    })


def paired_deltas(arm_reps, base_reps, suffix=''):
    """Within-replicate differences against a reference arm, replicate by replicate.

    This is the payoff of the common random numbers: replicate k of an arm and replicate k
    of the reference share a population and a draw sequence, so their difference removes
    the run-to-run variation both inherit. The standard error of that difference is
    usually far smaller than the pooled error of two independent means, and it cannot
    inherit a baseline that drew low.

    Returns, per outcome, the mean paired difference and the standard error OF THAT
    DIFFERENCE. Replicates where either side is undefined (an arm with no orders has no
    non-compliance rate) drop out pairwise rather than silently misaligning.

    `suffix` names the reference when it is NOT the baseline, giving
    `paired_d_<outcome>_vs_<arm>`. The baseline contrast stays unsuffixed.
    """
    out = {}
    for col in SE_OUTCOMES:
        d = (arm_reps[col] - base_reps[col]).dropna()
        n = len(d)
        out[f'paired_d_{col}{suffix}'] = d.mean() if n else float('nan')
        out[f'paired_se_{col}{suffix}'] = (d.std(ddof=1) / np.sqrt(n)
                                           if n > 1 else float('nan'))
    return out


# Paired contrasts between two ARMS, on top of every arm's contrast against the baseline.
#
# The baseline-only design cannot answer the question
# the protective order actually poses. `no_mpu` removes the order entirely; `inert_mpu`
# leaves the order standing but strips everything that enforces it (threshold boost,
# monitoring, arrest on breach). The difference BETWEEN THEM is the value of the order as a
# document — its existence net of its execution — and it sits between two arms, not between
# an arm and the baseline. Read against the baseline instead, `inert_mpu` lands somewhere
# between it and `no_mpu` without separating from either.
#
# Common random numbers make the paired reading legitimate: the two arms already share
# replicate seeds, so replicate k differs only in the channel under test.
#
# (arm, reference): the delta is arm MINUS reference.
ARM_CONTRASTS = [('inert_mpu', 'no_mpu')]


def measure(arm, panel, stats, steps, femicide_base):
    """Reduce the replicates of one arm to a row of city-wide outcomes.

    Counts are pooled across replicates BEFORE forming any ratio — never a mean of
    ratios, which would weight a small replicate like a large one. The standard errors are
    the one exception, and they are computed from the per-replicate ratios by definition.
    """
    tot = {k: sum(s[k] for s in stats) for k in stats[0]}
    reps = _per_replicate(panel, stats, femicide_base)
    ses = {f'se_{c}': reps[c].std(ddof=1) / np.sqrt(reps[c].notna().sum())
           for c in SE_OUTCOMES}
    # Adult woman-years: the head counts summed over year boundaries and replicates. This
    # is the denominator of every population rate below, and of the femicide rate, so it
    # has to be the AREAP table's own `Females` rather than an end-of-run head count.
    woman_years = panel['Females'].sum()
    expected_femicides = femicide_base * tot['sum_prior']

    return {
        'arm': arm,
        'runs': len(stats),
        'steps': steps,
        # Parameters in force, so the row describes itself.
        'P_MPU_REQUESTED': settings.P_MPU_REQUESTED,
        'P_MPU_GRANTED': settings.P_MPU_GRANTED,
        'MPU_THRESHOLD_BOOST': settings.MPU_THRESHOLD_BOOST,
        'MPU_MONITORING_LEVEL': settings.MPU_MONITORING_LEVEL,
        'PATROL_COVERAGE': settings.PATROL_COVERAGE,
        'adult_woman_years': woman_years,
        # --- outcomes on the whole adult female population -----------------------
        'prevalence': tot['ever_attacked'] / tot['women'],
        'assaults': tot['assaults'],
        'assaults_per_woman_year': tot['assaults'] / woman_years,
        'assaults_per_victim': (tot['assaults'] / tot['ever_attacked']
                                if tot['ever_attacked'] else float('nan')),
        # --- outcomes among women assaulted in the year --------------------------
        'denounce_rate': (tot['denounced'] / tot['attacked']
                          if tot['attacked'] else float('nan')),
        'separation_rate': (tot['broke'] / tot['attacked']
                            if tot['attacked'] else float('nan')),
        'assaults_per_violent_year': (tot['assaults'] / tot['attacked']
                                      if tot['attacked'] else float('nan')),
        # --- the two channels ----------------------------------------------------
        'mpu_granted': tot['mpu_granted'],
        'mpu_patrolled': tot['mpu_patrolled'],
        'share_orders_patrolled': (tot['mpu_patrolled'] / tot['mpu_granted']
                                   if tot['mpu_granted'] else float('nan')),
        'non_compliance_rate': (tot['mpu_violated'] / tot['mpu_granted']
                                if tot['mpu_granted'] else float('nan')),
        # Custody per granted order — the quantity the NEV source reports directly
        # (46 arrests over 1,163 orders forwarded to the patrol, i.e. 0.040).
        'arrests': tot['arrests'],
        'arrests_per_grant': (tot['arrests'] / tot['mpu_granted']
                              if tot['mpu_granted'] else float('nan')),
        'share_violent_men_monitored': (tot['violent_men_monitored'] / tot['violent_men']
                                        if tot['violent_men'] else float('nan')),
        'share_violent_men_ever_mpu': (tot['violent_men_ever_mpu'] / tot['violent_men']
                                       if tot['violent_men'] else float('nan')),
        'post_sep_contacts': tot['post_sep_contacts'],
        'post_sep_assaults': tot['post_sep_assaults'],
        # --- femicide, predicted rather than counted -----------------------------
        'sum_prior': tot['sum_prior'],
        'femicides_observed': tot['femicides'],
        'femicides_expected': expected_femicides,
        'femicide_rate_expected': expected_femicides / woman_years * 1e5,
        'femicide_rate_observed': tot['femicides'] / woman_years * 1e5,
        **ses,
    }


def measure_areap(arm, panel, sum_prior_areap, femicide_base):
    """Same outcomes at AREAP level, pooled over years and replicates.

    An AREAP is a census area, so its counts are small and its rates are noisy: this table
    is for the spatial CONTRAST between arms, not for reading any single area. `Females`
    is again the adult woman-years at risk, and an area that never held an adult woman
    yields NaN rather than a division by zero.
    """
    grp = panel.groupby('AREAP', as_index=False).agg(
        adult_woman_years=('Females', 'sum'),
        attacked_woman_years=('Attacked', 'sum'),
        assaults=('Number of attacks', 'sum'),
        denounce=('Denounce', 'sum'),
        breaks=('Break', 'sum'),
        femicides_observed=('Femicide', 'sum'),
    )
    grp['arm'] = arm
    grp['sum_prior'] = grp.AREAP.map(sum_prior_areap).fillna(0)

    wy = grp.adult_woman_years.replace(0, pd.NA)
    grp['assaults_per_woman_year'] = grp.assaults / wy
    grp['share_woman_years_attacked'] = grp.attacked_woman_years / wy
    grp['denounce_per_woman_year'] = grp.denounce / wy
    grp['separation_per_woman_year'] = grp.breaks / wy
    grp['femicides_expected'] = femicide_base * grp.sum_prior
    grp['femicide_rate_expected'] = grp.femicides_expected / wy * 1e5
    return grp


def wage_quartiles(areaps):
    """Map each AREAP of this metro to a wage quartile, cut once for all arms.

    Cut on the areas the model actually instantiates rather than on the national file, so
    the quartiles describe the city; cut once and reused, so an arm cannot change the
    grouping it is being read through. Areas missing from the census file (none expected)
    fall out as NaN rather than into an arbitrary group.
    """
    wage = AREAP_WAGE.assign(AREAP=AREAP_WAGE.AREAP.astype(str))
    here = wage[wage.AREAP.isin({str(a) for a in areaps})]
    labels = [f'Q{i + 1}' for i in range(WAGE_GROUPS)]
    q = pd.qcut(here.avg_wage, WAGE_GROUPS, labels=labels)
    return dict(zip(here.AREAP, q)), dict(zip(here.AREAP, here.avg_wage))


def measure_wage_groups(areap, groups, wages):
    """Pool the AREAP table into wage quartiles, per arm.

    Counts are summed and the ratios formed afterwards, so a large area weighs more than
    a small one — a mean of the per-area rates would give a 6-woman AREAP the same say as
    a 60-woman one.
    """
    df = areap.copy()
    df['AREAP'] = df.AREAP.astype(str)
    df['wage_quartile'] = df.AREAP.map(groups)
    df['avg_wage'] = df.AREAP.map(wages)
    df = df.dropna(subset=['wage_quartile'])

    grp = df.groupby(['arm', 'wage_quartile'], as_index=False, observed=True).agg(
        areaps=('AREAP', 'nunique'),
        avg_wage=('avg_wage', 'mean'),
        adult_woman_years=('adult_woman_years', 'sum'),
        attacked_woman_years=('attacked_woman_years', 'sum'),
        assaults=('assaults', 'sum'),
        denounce=('denounce', 'sum'),
        breaks=('breaks', 'sum'),
        sum_prior=('sum_prior', 'sum'),
        femicides_expected=('femicides_expected', 'sum'),
    )
    wy = grp.adult_woman_years.replace(0, pd.NA)
    grp['assaults_per_woman_year'] = grp.assaults / wy
    grp['share_woman_years_attacked'] = grp.attacked_woman_years / wy
    grp['denounce_per_woman_year'] = grp.denounce / wy
    grp['separation_per_woman_year'] = grp.breaks / wy
    grp['femicide_rate_expected'] = grp.femicides_expected / wy * 1e5
    return grp


def measure_municipality(areap, iterates, years, min_women=100.0):
    """Pool the AREAP table into MUNICIPALITIES, per arm.

    Municipality is the 7-digit prefix of the AREAP code (model.community_violence keys
    the homicide series by exactly this prefix). This
    is a re-aggregation of the same panel `measure_wage_groups` reads, grouped by the
    municipal prefix instead of by wage quartile — no model mechanism changes.

    Why municipality and not AREAP: an AREAP holds ~6-12 of the model's women, so a per-area
    rate is nearly all noise; a municipality pools its AREAPs into a cell with enough women
    to resolve an arm delta. Counts are summed and ratios formed afterwards
    so a large municipality weighs more than a small one, exactly as in the wage reducer.

    `min_women` flags cells the run cannot support: municipalities below the threshold get
    `has_power=False` so the report and any map can pool them into a single "insufficient
    support" line instead of drawing noise. It is read as WOMEN LIVING IN
    THE CELL IN A TYPICAL YEAR OF ONE REPLICATE, which is what sets the noise on that
    cell's rate. The column it is tested against is pooled over arms, replicates and years,
    so it has to be divided back down by all three: pooling 40 replicates of 7 arms
    multiplies every cell by 280 and would pass a municipality holding two women.
    """
    df = areap.copy()
    df['AREAP'] = df.AREAP.astype(str)
    df['municipality'] = df.AREAP.str[:7]

    grp = df.groupby(['arm', 'municipality'], as_index=False, observed=True).agg(
        areaps=('AREAP', 'nunique'),
        adult_woman_years=('adult_woman_years', 'sum'),
        attacked_woman_years=('attacked_woman_years', 'sum'),
        assaults=('assaults', 'sum'),
        denounce=('denounce', 'sum'),
        breaks=('breaks', 'sum'),
        sum_prior=('sum_prior', 'sum'),
        femicides_expected=('femicides_expected', 'sum'),
    )
    wy = grp.adult_woman_years.replace(0, pd.NA)
    grp['assaults_per_woman_year'] = grp.assaults / wy
    grp['share_woman_years_attacked'] = grp.attacked_woman_years / wy
    grp['denounce_per_woman_year'] = grp.denounce / wy
    grp['separation_per_woman_year'] = grp.breaks / wy
    grp['femicide_rate_expected'] = grp.femicides_expected / wy * 1e5
    # Statistical-power flag. The pooled woman-years are divided back down to the women
    # this cell holds in one year of one arm of one replicate, which is the sample the
    # cell's rate is actually formed on. Pooled ACROSS arms so a cell is either reportable
    # in every arm or in none — the delta needs the pair present. Below the threshold goes
    # to the "insufficient support" bucket.
    per_mun = grp.groupby('municipality').adult_woman_years.sum()
    per_replicate_year = per_mun / (grp.arm.nunique() * iterates * years)
    grp['women_per_replicate_year'] = grp.municipality.map(per_replicate_year)
    grp['has_power'] = grp.municipality.map(per_replicate_year >= min_women)
    return grp


def _report_municipality(grp, top=12):
    """Print the municipal contrast as deltas against baseline, biggest cells first."""
    base = grp[grp.arm == 'baseline'].set_index('municipality')
    if base.empty:
        return
    powered = base[base.has_power].sort_values('adult_woman_years', ascending=False)
    muns = list(powered.index[:top])
    if not muns:
        return
    w = 20
    print('\n --- by MUNICIPALITY (largest cells), assaults per adult woman-year ---')
    print(f"   {'arm':<18}" + ''.join(f'{m:>{w}}' for m in muns))
    for arm in grp.arm.unique():
        row = grp[grp.arm == arm].set_index('municipality')
        cells = []
        for m in muns:
            v = row.assaults_per_woman_year.get(m, float('nan'))
            d = '' if arm == 'baseline' else f' ({v - base.assaults_per_woman_year[m]:+.3f})'
            cells.append(f'{v:.3f}{d}')
        print(f"   {arm:<18}" + ''.join(f'{c:>{w}}' for c in cells))
    print(f"   {'n AREAPs':<18}"
          + ''.join(f'{int(base.areaps[m]):>{w}}' for m in muns))
    print(f"   {'adult woman-yrs':<18}"
          + ''.join(f'{base.adult_woman_years[m]:>{w}.0f}' for m in muns))
    # The pooled count above flatters the cell by the replicate x arm factor; this is the
    # sample the rate is really formed on, and the number the power threshold reads.
    print(f"   {'women / repl-yr':<18}"
          + ''.join(f'{base.women_per_replicate_year[m]:>{w}.0f}' for m in muns))
    n_drop = int((~base.has_power).sum())
    if n_drop:
        print(f"   ({n_drop} municipalities below power threshold pooled out)", flush=True)


def _report_groups(grp):
    """Print the spatial contrast as deltas against the baseline arm, quartile by quartile."""
    base = grp[grp.arm == 'baseline'].set_index('wage_quartile')
    if base.empty:
        return
    w = 18
    print('\n --- by AREAP wage quartile (Q1 poorest), assaults per adult woman-year ---')
    print(f"   {'arm':<18}" + ''.join(f'{q:>{w}}' for q in base.index))
    for arm in grp.arm.unique():
        row = grp[grp.arm == arm].set_index('wage_quartile')
        cells = []
        for q in base.index:
            v = row.assaults_per_woman_year.get(q, float('nan'))
            d = '' if arm == 'baseline' else f' ({v - base.assaults_per_woman_year[q]:+.3f})'
            cells.append(f'{v:.3f}{d}')
        print(f"   {arm:<18}" + ''.join(f'{c:>{w}}' for c in cells))
    print(f"   {'n AREAPs':<18}"
          + ''.join(f'{int(base.areaps[q]):>{w}}' for q in base.index))
    print(f"   {'mean wage (R$)':<18}"
          + ''.join(f'{base.avg_wage[q]:>{w}.0f}' for q in base.index), flush=True)


def _report_contrast(arm, ref, row, ref_row):
    """Print one arm-to-arm paired contrast, with the z* that decides whether it separated.

    The z* is the whole point here and the reason the block prints its own `n`. This
    contrast exists to answer a question that can come back NEGATIVE and still publish —
    "the two components are not separable in the violence outcome at this n" — so the
    reader has to be able to see how much resolution was bought.
    """
    print(f'\n --- paired contrast: {arm} MINUS {ref} '
          f'(within-replicate, common random numbers) ---')
    for col in SE_OUTCOMES:
        d = row.get(f'paired_d_{col}_vs_{ref}', float('nan'))
        se = row.get(f'paired_se_{col}_vs_{ref}', float('nan'))
        if d != d:
            continue
        z = f'{d / se:+.1f}' if se == se and se else '  n/a'
        mark = ' *' if se == se and se and abs(d / se) >= 2 else '  '
        print(f'   {col:<28} {d:+.4f}  se {se:.4f}   z*={z}{mark}'
              f'   ({ref_row.get(col, float("nan")):.3f} -> {row.get(col, float("nan")):.3f})')
    print('   * separates at |z*| >= 2; anything below is a null AT THIS n, not a zero.',
          flush=True)


def _report(row, base=None):
    def delta(key, fmt='.3f'):
        """The difference against baseline, and how many standard errors of it that is.

        The z is the point of the column: at ten replicates a 5% shift in the assault rate
        is not resolved, and printing only the percentage invites reading it as an effect.

        Prefers the PAIRED standard error, which is the one the common random numbers buy
        — replicate k against replicate k, so the arms' shared run-to-run variation
        cancels instead of being added in quadrature. `z*` marks the paired reading; a
        bare `z` is the unpaired fallback, for a row that has no pairing (a caller that
        skipped the baseline arm).
        """
        if base is None or not np.isfinite(row[key]) or not np.isfinite(base[key]):
            return ''
        d = row[key] - base[key]
        pct = d / base[key] * 100 if base[key] else float('nan')
        z = ''
        pse = row.get(f'paired_se_{key}')
        if pse is not None and np.isfinite(pse) and pse > 0:
            z = f'  z*={row[f"paired_d_{key}"] / pse:+.1f}'
        else:
            se_key = f'se_{key}'
            if se_key in row and se_key in base:
                pooled = np.hypot(row[se_key], base[se_key])
                if np.isfinite(pooled) and pooled > 0:
                    z = f'  z={d / pooled:+.1f}'
        return f'   [{d:+{fmt}}  {pct:+.1f}%{z}]'

    print(f"\n --- {row['arm']}  ({row['runs']} runs) ---")
    print(f" prevalence (lifetime)            {row['prevalence']:.3f}"
          f"{delta('prevalence')}")
    print(f" assaults per adult woman-year    {row['assaults_per_woman_year']:.3f}"
          f"{delta('assaults_per_woman_year')}")
    print(f" assaults per victim (lifetime)   {row['assaults_per_victim']:.2f}"
          f"{delta('assaults_per_victim', '.2f')}")
    print(f" assaults per violent year        {row['assaults_per_violent_year']:.2f}"
          f"{delta('assaults_per_violent_year', '.2f')}   (FBSP 2.58-3.55)")
    print(f" denounce | separation (of victims)  {row['denounce_rate']:.3f}"
          f" | {row['separation_rate']:.3f}"
          f"{delta('denounce_rate')}")
    print(f" MPU granted {row['mpu_granted']:>5}   patrolled "
          f"{row['mpu_patrolled']:>5} ({row['share_orders_patrolled']:.2f})"
          f"   non-compliance {row['non_compliance_rate']:.3f}"
          f"{delta('non_compliance_rate')}")
    print(f" arrests on breach {row['arrests']:>5}   per grant "
          f"{row['arrests_per_grant']:.3f}   (NEV Piracicaba 0.040)")
    print(f" violent men: ever MPU            "
          f"{row['share_violent_men_ever_mpu']:.3f}"
          f"   ever monitored {row['share_violent_men_monitored']:.3f}")
    print(f" post-sep contacts / assaults     {row['post_sep_contacts']:>5}"
          f" / {row['post_sep_assaults']}")
    print(f" femicide rate EXPECTED           "
          f"{row['femicide_rate_expected']:.2f} per 100k"
          f"{delta('femicide_rate_expected', '.2f')}"
          f"   (target {TARGET_FEMICIDE_RATE[0]}-{TARGET_FEMICIDE_RATE[1]};"
          f" observed {row['femicide_rate_observed']:.2f} on"
          f" {row['femicides_observed']:.0f} deaths)", flush=True)


# Outcomes the delta table carries. Kept short on purpose: a delta on every column
# invites reading noise as effect in the ones the design cannot resolve.
DELTA_COLUMNS = [
    'prevalence', 'assaults_per_woman_year', 'assaults_per_victim',
    'assaults_per_violent_year', 'denounce_rate', 'separation_rate',
    'non_compliance_rate', 'share_violent_men_monitored',
    'share_violent_men_ever_mpu', 'femicide_rate_expected',
]


def run(metro, iterates, steps, workers=None, suffix='', arms=None, keep_panel=True,
        base_seed=20260808, sens=None, families=1000):
    """Run every arm, optionally once per value of a SENSITIVITY parameter.

    `sens` is (name, values). The parameter is set underneath every arm and the whole arm
    table is produced once per value, with the paired deltas computed WITHIN each value:
    an arm is always read against the baseline that shares its parameter, never against a
    baseline that sits somewhere else in parameter space. The seed block is the same
    across values, so the comparison BETWEEN values is paired too.

    The axis is swept WIDER than the calibrated window (for ESCALATION_STEP the window is
    three thousandths wide, see settings.py), out to values where the calibration itself
    fails, to show whether the policy conclusions move. Inside the window the differences
    are replicate noise.

    The model has a feedback loop between her action channel and his assault frequency,
    so moving one
    parameter with the others held fixed visits cells the model would never be CALIBRATED
    at. This axis reports the calibration moments in the same rows for exactly that reason
    — read which targets each value misses before reading its deltas.
    """
    arms = arms or list(ARMS)
    # Baseline first whatever the caller asked for: every paired delta is against it, and
    # a paired delta needs the baseline's replicates already in hand.
    if 'baseline' in arms:
        arms = ['baseline'] + [a for a in arms if a != 'baseline']
    sens_name, sens_values = sens if sens else (None, [None])
    # The values settings.py carries on entry, so each arm starts from the calibrated
    # vector rather than from whatever the previous arm left behind.
    touched = {k for arm in arms for k in ARMS[arm]}
    if sens_name:
        touched.add(sens_name)
    frozen = {k: getattr(settings, k) for k in touched}

    # Common random numbers: replicate k gets the same seed in every arm. Fixed rather
    # than drawn, so the whole table reproduces.
    seeds = [base_seed + k for k in range(iterates)]

    rows, areap_frames, wage_frames, mun_frames, panel_frames = [], [], [], [], []
    for sens_value in sens_values:
        if sens_name:
            print(f'\n########## {sens_name} = {sens_value} ##########', flush=True)
        base_row, base_reps = None, None
        # Every arm's per-replicate frame and its row, kept so ARM_CONTRASTS can pair two
        # arms after the loop. Held per sens_value: a contrast never crosses values, for
        # the same reason the baseline delta does not.
        arm_reps, arm_rows = {}, {}
        value_areaps = []
        for arm in arms:
            # The arm wins where the two touch the same parameter: an arm is a claim about
            # a channel, the sensitivity value is only the substrate it runs on.
            overrides = ({sens_name: sens_value, **ARMS[arm]} if sens_name
                         else dict(ARMS[arm]))
            print(f'\n=== {metro} / arm {arm}: {overrides or "settings.py as written"} ===',
                  flush=True)
            _restore(frozen)
            _apply(overrides)
            femicide_base = settings.FEMICIDE_BASE

            jobs = [(metro, steps, overrides, s, families) for s in seeds]
            done = [0]

            def tick(_i, _r, done=done):
                done[0] += 1
                print(f'  run {done[0]}/{iterates}', flush=True)

            results = parallel.map_runs(run_once, jobs, workers=workers, on_result=tick)

            panels, stats, sum_prior_areap = [], [], {}
            for k, (prows, run_stats, spa) in enumerate(results):
                df = pd.DataFrame(prows)
                df['run'] = k
                panels.append(df)
                stats.append(run_stats)
                for addr, n in spa.items():
                    sum_prior_areap[addr] = sum_prior_areap.get(addr, 0) + n
            panel = pd.concat(panels, ignore_index=True)
            panel['arm'] = arm

            # measure() reads the parameters off the settings module, which in the parent
            # is this arm's only because _apply ran above — the workers set their own.
            row = measure(arm, panel, stats, steps, femicide_base)
            reps = _per_replicate(panel, stats, femicide_base)
            if arm == 'baseline':
                base_row, base_reps = row, reps
            elif base_reps is not None:
                row.update(paired_deltas(reps, base_reps))
            arm_reps[arm], arm_rows[arm] = reps, row

            areap = measure_areap(arm, panel, sum_prior_areap, femicide_base)
            if sens_name:
                row['sens_param'] = sens_name
                row['sens_value'] = sens_value
                panel['sens_value'] = sens_value
                areap['sens_value'] = sens_value
            rows.append(row)
            _report(row, base_row if arm != 'baseline' else None)

            areap_frames.append(areap)
            value_areaps.append(areap)
            if keep_panel:
                panel_frames.append(panel)

        # Unpaired deltas, against THIS value's baseline. Written here rather than over the
        # whole table so a sensitivity run never reads an arm against a baseline that ran
        # at another value of the swept parameter.
        if base_row is not None:
            for r in rows[-len(arms):]:
                for col in DELTA_COLUMNS:
                    r[f'd_{col}'] = r[col] - base_row[col]

        # Arm-to-arm paired contrasts. Only pairs whose BOTH arms ran in this call are
        # formed; asking for a contrast without running its reference is a silent no-op
        # rather than a crash, so `-a baseline no_mpu` still works.
        for arm, ref in ARM_CONTRASTS:
            if arm in arm_reps and ref in arm_reps:
                arm_rows[arm].update(
                    paired_deltas(arm_reps[arm], arm_reps[ref], suffix=f'_vs_{ref}'))
                _report_contrast(arm, ref, arm_rows[arm], arm_rows[ref])

        # Wage quartiles are pooled within the value, for the same reason.
        value_areap = pd.concat(value_areaps, ignore_index=True)
        groups, wages = wage_quartiles(value_areap.AREAP.unique())
        wage_table = measure_wage_groups(value_areap, groups, wages)
        if sens_name:
            wage_table['sens_value'] = sens_value
        wage_frames.append(wage_table)
        _report_groups(wage_table)

        # Same panel, grouped by municipality (AREAP[:7]) instead of wage quartile.
        # Re-aggregation only; no model change.
        mun_table = measure_municipality(value_areap, iterates, steps // _12_MONTHS)
        if sens_name:
            mun_table['sens_value'] = sens_value
        mun_frames.append(mun_table)
        _report_municipality(mun_table)

    _restore(frozen)

    out = pd.DataFrame(rows)
    arm_path = f'output/scenarios_{metro}{suffix}.csv'
    out.to_csv(arm_path, index=False)

    areap = pd.concat(areap_frames, ignore_index=True)
    areap_path = f'output/scenarios_areap_{metro}{suffix}.csv'
    areap.to_csv(areap_path, index=False)

    wage_table = pd.concat(wage_frames, ignore_index=True)
    wage_path = f'output/scenarios_wage_quartile_{metro}{suffix}.csv'
    wage_table.to_csv(wage_path, index=False)

    mun_table = pd.concat(mun_frames, ignore_index=True)
    mun_path = f'output/scenarios_municipality_{metro}{suffix}.csv'
    mun_table.to_csv(mun_path, index=False)

    if keep_panel:
        pd.concat(panel_frames, ignore_index=True).to_csv(
            f'output/scenarios_panel_{metro}{suffix}.csv', index=False)

    print(f'\nDone -> {arm_path}\n     -> {areap_path}\n     -> {wage_path}'
          f'\n     -> {mun_path}')
    return out, areap, wage_table, mun_table


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--metro', default='PORTO ALEGRE')
    p.add_argument('-i', '--iterations', type=int, default=10)
    p.add_argument('-s', '--steps', type=int, default=144)
    p.add_argument('-j', '--workers', type=int, default=None)
    p.add_argument('-o', '--suffix', default='')
    p.add_argument('-a', '--arms', nargs='+', default=None, choices=list(ARMS),
                   help='subset of arms to run; baseline first if deltas are wanted')
    p.add_argument('--no-panel', action='store_true',
                   help='skip the arm x AREAP x year panel file')
    p.add_argument('-f', '--families', type=int, default=1000,
                   help='number of couples (relationships) instantiated per run; agents '
                        '= 2x this. Default 1000 (~2k agents). Raise (e.g. 5000) to give '
                        'the medium municipalities enough women per cell for the municipal '
                        'table. Setup cost grows ~quadratically with it.')
    p.add_argument('--base-seed', type=int, default=20260808,
                   help='first common-random-number seed; replicate k of every arm runs '
                        'on base_seed + k, so the arms are paired and the table reproduces')
    p.add_argument('--sens', nargs='+', metavar=('PARAM', 'VALUE'),
                   help='sensitivity axis: a settings name followed by the values to '
                        'sweep, e.g. --sens ESCALATION_STEP 0.010 0.013 0.020 0.025. '
                        'Every arm is run once per value, deltas are paired within the '
                        'value, and the calibration moments come out in the same rows so '
                        'the reader can see which targets each value misses.')
    a = p.parse_args()
    sens = None
    if a.sens:
        if len(a.sens) < 2:
            p.error('--sens takes a parameter name followed by at least one value')
        name = a.sens[0]
        if not hasattr(settings, name):
            p.error(f'--sens: {name} is not a name in settings.py')
        sens = (name, [float(v) for v in a.sens[1:]])
    run(a.metro, a.iterations, a.steps, a.workers, a.suffix, a.arms,
        keep_panel=not a.no_panel, base_seed=a.base_seed, sens=sens,
        families=a.families)