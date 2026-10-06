"""Model constants.

Every constant the model reads is defined here and nowhere else. Each carries its
unit and its provenance, in one of four kinds:

  data        taken from a cited source and never adjusted to improve fit
  calibrated  fitted to one empirical moment (three parameters only)
  argument    fixed by a stated argument rather than by a moment
  assumption  set by the modeller; varied in the sensitivity analysis where noted

Parameter sweeps override these by assignment. `scenarios._apply` writes both this
module and `agents`, because `agents` binds several of them by name at import.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Protective orders (medidas protetivas de urgência, MPU)
# ---------------------------------------------------------------------------

# Multiplicative reduction applied to disposition and accumulated stress at each step
# of a monitoring spell. Default for apply_monitoring; orders use
# MPU_MONITORING_LEVEL below.
MONITORING_LEVEL = 0.2

# Share of granted orders the Patrulha Maria da Penha actually reaches. Every granted
# order is forwarded to it and no visit occurs without one, so coverage is a property
# of the order rather than a channel beside it. A covered order carries the visits; an
# uncovered one carries only the threshold boost.
# Assumption, from a single-site practitioner estimate: about a quarter of visits do
# not start or are interrupted for want of a usable address (IPEA 2019, pp. 45, 56).
# An optimistic ceiling, so the scenario arms bracket it at 0.0 and 1.0.
PATROL_COVERAGE = 0.75

# A denunciation reaches an active order in two stages, with different denominators.
# Chained, 0.653 x 0.655 = 0.428 orders per denunciation.
# Data: Polícia Civil do Rio Grande do Sul, Divisão de Planejamento e Coordenação,
# "Ocorrências policiais de violência doméstica em Porto Alegre de 2012 a 2023"
# (freedom-of-information request 44.026). Share of threat and bodily-harm occurrences
# with an adult female victim that carry a request. A period average, not a steady
# state: it rises from 0.450 in 2012 to 0.688 in 2020.
P_MPU_REQUESTED = 0.653
# Data: CNJ (2025), 619k granted out of 945k requested.
P_MPU_GRANTED = 0.655

# Months an order runs. Assumption: the Maria da Penha Law sets no fixed term, and CNJ
# data put the critical window in the first six months.
MPU_DURATION = 12

# Reduction applied at each step of a visited order, stronger than MONITORING_LEVEL.
# Assumption, and reported as NOT IDENTIFIED: the non-compliance moment is flat in it
# over the whole useful range, because 91-94% of breached orders are breached in the
# first month, before the channel runs once (Men.step calls act() before
# update_monitoring).
MPU_MONITORING_LEVEL = 0.30

# Increase in the aggressor's inhibition threshold while an order stands. Breaching an
# order has been a crime in its own right since Lei 13.641/2018 (art. 24-A), and this
# is the expected sanction that carries.
# CALIBRATED against the non-compliance moment, 0.21 +/- 0.04 breaches per granted
# order, and against nothing else. The response is steep and monotone, so the moment
# identifies it sharply.
# The value exceeds ESCALATION_STEP: an order must more than undo the inhibition that
# the assault prompting it had eroded. The two are offsets on the same threshold, in
# opposite directions, which is what makes them commensurable.
MPU_THRESHOLD_BOOST = 0.09

# Probability that a breach ends the relationship: the couple separates and the man
# detaches from the household. This is the removal from the home the order itself
# provides for, and is distinct from custody (P_ARREST_ON_BREACH). Assumption.
# Drawn only on the in-relationship conduct, in Men.step.
P_VIOL = 0.40

# Probability that a breach ends in custody, drawn in Men.register_mpu_violation. This
# is the only channel in the model that interrupts a violent spell mid-course: while
# detained the man leaves schedule.agents_t and does not act.
# Data (a floor): NEV/USP, Patrulha Maria da Penha of Piracicaba/SP, 2017-2019 —
# 46 arrests in flagrante annualised over 91 recorded breaches. It counts arrests by
# the municipal guard only, so it is an order-of-magnitude anchor, not a fine target.
P_ARREST_ON_BREACH = 0.17

# Months in custody. Argument: the lower bound of the statutory term for art. 24-A
# (Lei 13.641/2018, "detenção, de 3 meses a 2 anos").
ARREST_DURATION = 3

# ---------------------------------------------------------------------------
# Post-separation window
# ---------------------------------------------------------------------------

# Months a relationship that has held violence persists after cohabitation ends. The
# relationship, in this model, ends when this window closes, not when the couple stops
# living together.
# Data: Campos & Colman (2024), who locate the risk between three months and one year
# after separation, with a history of prior violence in every case.
POST_SEP_WINDOW = 12

# Probability of the ex-partner making contact in the first month after separation,
# decaying linearly to zero at the end of the window. Contact is drawn before violence,
# because it is the approach that art. 24-A defines as the breach.
# Assumption, and NOT IDENTIFIED by the calibration moments: decomposing the
# non-compliance rate by the conduct that breached the order first leaves this channel
# contributing 0.02 to 0.05 of it, so the moment cannot locate the parameter. Varied in
# the sensitivity analysis.
POST_SEP_CONTACT_BASE = 0.15

# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------

# Fraction by which a victim drawing Remain reduces her partner's accumulated stress.
# Represents the conciliatory phase of the cycle of violence; the reduction is partial,
# and stress rebuilds over subsequent steps. Assumption.
REMAIN_CALM_FACTOR = 0.05

# Ceiling on years of study in the synthetic population (input/generator.py draws
# 0-17). Turns years of study into a schooling deficit normalised to [0, 1], so the
# schooling term stays on the same scale as the other components of the inflow.
MAX_YEARS_STUDY = 17

# Number of additive components in the stress inflow. The inflow is their MEAN, so it
# stays on a [0, ~1] scale instead of growing with how many terms the sum carries.
STRESS_COMPONENTS = 10

# Fraction of the accumulated stock forgotten on a relief step. Forgetting is
# proportional to the stock rather than subtractive, so the stock cannot go negative
# and each agent settles at an equilibrium proportional to its own inflow,
# p * s / [(1 - p) * STRESS_DECAY]. Assumption.
STRESS_DECAY = 0.35

# Gender constant in the stress inflow: men carry GENDER_STRESS and women its
# complement, so 0.270 on men and 0.730 on women.
# The term is constant within gender, so it produces no heterogeneity between agents:
# it is a level shift, and the threshold draw absorbs level shifts by construction.
# Only the position of tau within the realised distribution is identified. It is not
# exactly a level shift, because the partial sum is multiplied by (1 + LOW) when the
# spouse is black, so agents at that branch absorb it unevenly.
GENDER_STRESS = 0.270482864567824

# ---------------------------------------------------------------------------
# Disposition and the violence trigger
# ---------------------------------------------------------------------------

# Learning rate of the Rescorla-Wagner update, used for both the aggressor's
# disposition and the victim's propensity to act. Sets the memory of the behavioural
# state. Assumption.
RESCORLA_WAGNER_ALPHA = 0.25

# Inhibition threshold, drawn U[THRESHOLD_MIN, THRESHOLD_MAX] once per agent and fixed
# thereafter, except for the escalation and protective-order offsets.
# CALIBRATED (centre) against lifetime prevalence of physical intimate-partner
# violence, 0.267 +/- 0.02 (FBSP, Visível e Invisível 5th ed., p. 17).
# The WIDTH is fixed by ARGUMENT, because prevalence identifies only the position of
# tau relative to the disposition distribution and says nothing about its dispersion.
# The width governs a trade-off: narrow, and who becomes violent is read almost
# entirely off the socioeconomic covariates; wide, and unexplained variation between
# men swamps them. 0.15 holds both, at the cost of passing under half of the available
# schooling gradient through to the outcome.
THRESHOLD_MIN = 0.933
THRESHOLD_MAX = 1.083

# Scale of the logistic violence hazard (Men.violence_hazard): the per-step probability
# of assaulting is logistic in the gap `disposition - threshold`, with this as its
# scale. At gap 0 the probability is 0.5.
# TRIGGER_SLOPE = 0 recovers a deterministic `disposition > threshold` comparison
# exactly, so the deterministic rule is the limiting case of this one.
# The value is a choice about the LIFETIME scale, not the monthly one: any positive
# floor compounds over 144 steps, so a man permanently below his threshold still faces
# a substantial lifetime probability of ever offending.
TRIGGER_SLOPE = 0.02

# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

# Amount by which each assault lowers the aggressor's own threshold: having crossed it
# makes crossing it again easier. The mirror image of MPU_THRESHOLD_BOOST, tracked as a
# separate offset so the two compose and unwind independently.
# Scoped to the relationship, not to the man: released with the ex-partner tie, so it
# survives the breakup and not the next relationship.
# CALIBRATED against assaults per victim-year, target interval [2.58, 3.55] (FBSP,
# p. 38). The identified region is a WINDOW, not a point: the intensity floor requires
# at least ~0.012 and the non-compliance ceiling at most ~0.015, and inside that window
# the moments are indistinguishable at the replicate budget used. Varied in the
# sensitivity analysis.
ESCALATION_STEP = 0.013

# Cap on cumulative erosion within one relationship. ARGUMENT: the width of the tau
# draw, so a violent relationship can carry a man across the whole spread of inhibition
# present in the population and no further. Without a cap he offends every month for as
# long as the relationship lasts.
ESCALATION_MAX = 0.15

# ---------------------------------------------------------------------------
# Victim action channel
# ---------------------------------------------------------------------------

# Weights of the three inputs to the propensity to act. They sum to 1, so the target
# lies in [0, 1] and the threshold range below reads on the same scale.
PER_ACTION_EXPERIENCE = 0.5   # assaults received in the current relationship
PER_ACTION_NETWORK = 0.3      # share of female peers who denounced or left this year
PER_ACTION_AUTONOMY = 0.2     # economic autonomy (working)

# Assaults in the current relationship at which the experience component saturates.
# Sets how fast repeated victimisation alone can carry a woman to her threshold,
# independently of peers and autonomy. Assumption.
ACTION_EXPERIENCE_REF = 12

# Threshold on the propensity to act, drawn U[ACTION_THRESHOLD_MIN,
# ACTION_THRESHOLD_MAX] once per woman. Above it she denounces or leaves; below it she
# remains.
# CALIBRATED against two annual moments, both on a denominator of ASSAULTED women:
#   reporting   interval [0.142, 0.257]  (FBSP p. 41; see below on why an interval)
#   separation  0.15 +/- 0.03            (DataSenado 2025)
# Reporting is an interval because no Brazilian survey reports P(report to police |
# IPV in the last 12 months). The published floor omits women who used an ordinary
# rather than a women's police station, whose report produces an order request just the
# same; the published ceiling is the right channel over the wrong population, since
# only about half of the aggressions it counts were committed by a partner.
# One free parameter against two moments, so the channel is OVER-IDENTIFIED and the
# pair is a specification test rather than a fit.
# The channel does not separate from the escalation block: the decision is drawn per
# assault while both targets are annual and per woman, so the per-assault probability
# that reproduces a given annual rate depends on how often women are assaulted, which
# is an output of the aggressor side. Any change upstream invalidates this one.
ACTION_THRESHOLD_MIN = 0.13
ACTION_THRESHOLD_MAX = 0.65

# Composition of the act, conditional on acting: the share who denounce rather than
# leave. ARGUMENT, not calibrated: 0.20 / (0.20 + 0.15) is the share implied by the
# ratio of the two annual targets, at the midpoint of the reporting interval.
ACTION_DENOUNCE_SHARE = 0.571

# ---------------------------------------------------------------------------
# Channel ceilings
# ---------------------------------------------------------------------------

# Weight of the partner's agitation in a man's disposition (the dyadic channel).
# Katerndahl et al. (2022) report from daily-diary data that her anger maximised his
# violence, so her disposition enters his as a fourth channel. The smallest of the
# four: a modifier, not a driver.
# Narrower than the source in two respects: there the effect is signed by the specific
# emotion, and here disposition is one scalar with one positive weight; and assaults
# received do not feed her agitation, so the coupling runs one way only.
PER_DYADIC = 0.05

# Share of the societal channel carried by the sexism index, the remainder by community
# violence. Each is normalised to [0, 1] before mixing, which is what makes this weight
# interpretable rather than fixed by the units the two arrive in.
# ARGUMENT (the midpoint): lifetime prevalence does not identify it. What it governs is
# where societal heterogeneity lives — community violence varies between municipalities,
# the sexism index within them — which makes it a question for the AREAP comparison and
# for sensitivity analysis, not for the threshold calibration.
W_MACHISM = 0.5

# Homicide rate per 100k inhabitants at which the community-violence component
# saturates. Municipality-years in the Porto Alegre metropolis over 2010-2021 run 0 to
# 119 per 100k with a median of 25, so the clip binds only on the rare extremes.
COMMUNITY_RATE_REF = 100.0


@dataclass
class Calibration:
    """Normalisation bounds and channel ceilings for the disposition target.

    Each channel is min-max normalised into [0, PER_*] and clipped, which is what
    keeps the network term from feeding disposition back into itself without bound.
    The ceilings are ordered: the individual channel dominates by an order of
    magnitude, and the societal, network and dyadic channels modulate it in that
    order, so the environment cannot by itself carry a man across his threshold.
    """

    # Bounds of the accumulated stress stock. A Porto Alegre run spans 0.47 (p10) to
    # 4.53 (p99), so this puts the median man mid-channel.
    MAX_INDIVIDUAL = 2.9575303190923434
    # Zero because proportional forgetting keeps the stock non-negative. A negative
    # bound would add a constant floor to every agent's disposition and compress the
    # differences between agents.
    MIN_INDIVIDUAL = 0.0
    # Attainable bounds of machismo_index() for an adult: the extreme age band summed
    # with the extreme schooling band of Perception_Woman. Analytic, not calibrated,
    # so the normalised index spans [0, 1] instead of clipping the youngest men to the
    # ceiling.
    MAX_MACHISM = 1.4875067375886525
    MIN_MACHISM = 0.7852467852467853
    # Channel ceilings.
    PER_INDIVIDUAL = 0.9854038846495861
    PER_NETWORK = 0.21694284808583042
    PER_MACHISM = 0.1370784698773831


# ---------------------------------------------------------------------------
# Femicide
# ---------------------------------------------------------------------------

# Per-assault increment to the probability that an assault is lethal. The probability
# scales linearly with the count of assaults already received in the current
# relationship, so the first assault already carries this value.
# Coupled to the reporting and separation rates, since both determine how long a
# relationship stays exposed to escalation.
# CALIBRATED analytically: the probability is linear in the prior count, so the
# expected number of femicides is this value times the summed prior counts over all
# assaults.
# Target: Anuário Brasileiro de Segurança Pública 2023, Table 27, Rio Grande do Sul,
# converted to the model's denominator by four factors (share of femicides committed by
# a partner or ex-partner; share of victims aged 18+; adult share of the female
# population; share of adults who live or have lived in a union). The net factor is
# 1.27, because the two numerator restrictions nearly cancel the two denominator ones.
# Target on the model's denominator: [2.19, 2.51] per 100k.
# The Porto Alegre occurrence file cannot serve this moment: it holds 7 completed
# femicides against 273 attempted, because a death becomes a homicide inquiry and
# leaves the domestic-violence occurrence system.
FEMICIDE_BASE = 0.000021

# Years a man is removed from the population after a femicide.
YEARS_IN_PRISON = 1.0391120019342974

# ---------------------------------------------------------------------------
# Demography and the working population
# ---------------------------------------------------------------------------

# Life expectancy by gender, the peak of the natural-death hazard.
LIFE_ESTIMATE = {'male': 72,
                 'female': 79}

# Divisor of the natural-death probability. Mortality over the 144-step horizon is low
# and the population is close to closed.
MODEL_SCALE = 628.7331959075738

# Rate of the Poisson draw governing re-pairing of single agents.
AVERAGE_TIME_RELATIONSHIP = 0.48322676286672483

# Share of the initial population in work, and the monthly probability of that status
# changing. Wage moves by at most PCT_CHANGE_WAGE when it does.
IS_WORKING = 0.5477346995234711
CHANCE_CHANGE_WORKING_STATUS = 0.02164878838398909
PCT_CHANGE_WAGE = 0.03004104843800275


@dataclass
class Classification:
    """Qualitative severities applied to the components of the stress inflow.

    Expert-informed classifications from the VIDA model (Mori Madeira, Alves Furtado
    & Dill, JASSS 24(4):1, 2021). They are not calibrated: the substantive content is
    the ordering HIGH > MEDIUM > LOW, not the values.
    """

    HIGH: float = 0.4248
    MEDIUM: float = 0.2451
    LOW: float = 0.03534


@dataclass
class Weight:
    """Peer network: size and the range of the uniform link weights.

    SIZE_INFLUENCE is the number of same-gender peers each agent is assigned, without
    replacement, at initialisation. Networks are not rewired thereafter. Assumption.
    """

    SIZE: int = 24
    SIZE_INFLUENCE: int = 18
    MIN: int = 0.0005863221558162167
    MAX: int = 0.03222007689257757


@dataclass
class WorkModality:
    """Work modality of employed agents. Informal work contributes more stress."""

    FORMAL: float = 1.1584679809282126
    INFORMAL: float = 0.15910777131556136


@dataclass
class HouseLocation:
    """Household location, from the census."""

    URBAN: int = 2.0118052378109845
    RURAL: int = 0.7647811390744345


@dataclass
class RacialIssueSpouse:
    """Differential in partnership rates: a black woman pairs with probability
    HAVE_SPOUSE rather than with certainty."""

    NOT_HAVE_SPOUSE: float = 0.0924418085788956
    HAVE_SPOUSE: float = 1.0 - NOT_HAVE_SPOUSE


@dataclass
class Perception_Woman:
    """Bracket scores from which machismo_index() is assembled, by age band and by
    schooling band, on two sub-scales.

    Data: Fundação Perseu Abramo survey of Brazilian attitudes towards gender roles
    (Venturi & Godinho, 2013). The index describes the aggressor's norms, so only Men
    carry it. The schooling brackets define no 9-10 band, so incomplete secondary is
    scored with the lower bracket.
    """

    AGE = {
        'MAX COMMON': 104,
        'MAX CRITICAL': 27,
        'BETWEEN 15 24 COMMON': 38,
        'BETWEEN 15 24 CRITICAL': 14,
        'BETWEEN 25 44 COMMON': 38,
        'BETWEEN 25 44 CRITICAL': 6,
        'MORE OR EQUAL 45 COMMON': 28,
        'MORE OR EQUAL 45 CRITICAL': 7
    }
    STUDY = {
        'MAX COMMON': 74,
        'MAX CRITICAL': 15,
        'LESS THAN 4 COMMON': 17,
        'LESS THAN 4 CRITICAL': 5,
        'BETWEEN 5 8 COMMON': 20,
        'BETWEEN 5 8 CRITICAL': 5,
        'BETWEEN 11 14 COMMON': 18,
        'BETWEEN 11 14 CRITICAL': 5,
        'GREATER OR EQUAL 15 COMMON': 19,
        'GREATER OR EQUAL 15 CRITICAL': 0
    }