import math
from datetime import date

import numpy as np
import pandas as pd
from mesa.agent import Agent
from mesa.model import Model
from scipy.stats import poisson
from collections import defaultdict

# Calibration targets are read as module attributes at call time so a sweep can
# override them after import.
import settings
from settings import Classification, \
    WorkModality, \
    HouseLocation, \
    RacialIssueSpouse, \
    Perception_Woman, \
    LIFE_ESTIMATE, \
    MODEL_SCALE, \
    Calibration, \
    MONITORING_LEVEL, \
    RESCORLA_WAGNER_ALPHA, \
    REMAIN_CALM_FACTOR, \
    MPU_DURATION, \
    MPU_MONITORING_LEVEL, \
    MPU_THRESHOLD_BOOST, \
    TRIGGER_SLOPE, \
    P_MPU_REQUESTED, \
    P_MPU_GRANTED, \
    P_VIOL, \
    PATROL_COVERAGE, \
    MAX_YEARS_STUDY, \
    ACTION_DENOUNCE_SHARE, \
    ACTION_EXPERIENCE_REF, \
    PER_ACTION_AUTONOMY, \
    PER_ACTION_EXPERIENCE, \
    PER_ACTION_NETWORK, \
    PER_DYADIC, \
    POST_SEP_WINDOW, \
    POST_SEP_CONTACT_BASE


class Person(Agent):
    """
    A person who lives within a family, incurs in chance of suffering violence.

    """

    def __init__(self, unique_id: int, model: Model, pos, gender='male', age=25, color='negra',
                 years_study=6, is_working=False, house_location=HouseLocation.URBAN,
                 wage=0, reserve_wage=.5, address=None, category='person', denounce=False, protection=False,
                 condemnation=False, call_rate=1, threshold=0, disposition=0, birthmonth=None,
                 relationship=None):

        super().__init__(unique_id, model)

        # Current relationship
        self.relationship = relationship
        # Death flag
        self.is_dead = False
        # Birth month
        self.birth_month: int = birthmonth
        # Count of attacks committed (lifetime, used as aggressor metric)
        self.n_attacks: int = 0
        # Count of attacks received in current relationship (reset on breakup; used for femicide probability)
        self.n_received_attacks: int = 0
        # Stress memory
        self.femicide_victim = False
        self.spouse_had_mpu = False  # instrumentation: victim's aggressor held an MPU (any relationship)
        self.accumulated_stress = 0
        # Starting the variables to Poisson arrival process
        self.call_rate = call_rate  # lambda (number of calls to update_stress in a interval of time)
        # Agent Zero: the disposition level above which the agent acts.
        self.threshold = threshold
        self.disposition = disposition
        # Number of past relationships, starting at 0.
        self.number_of_past_relationships = 0
        # Post-separation tie: set by desassign_spouse when the ending relationship held
        # violence, cleared by release_ex_partner once the window closes.
        self.ex_partner = None
        self.months_since_separation = 0
        self.pos = pos
        self.gender = gender
        self.age = age
        self.color = color
        self.years_study = years_study
        self.is_working = is_working
        # Work modality, for the employed only.
        self.work_modality = np.random.choice([WorkModality.FORMAL, WorkModality.INFORMAL]) \
            if self.is_working else None
        self.reserve_wage = reserve_wage
        self.wage = wage
        self.address = address
        # Set during simulation
        self.spouse = None
        self.got_attacked = 0
        self.assaulted = 0
        # Location of the agent residence (urban or rural)
        self.house_location = house_location
        self.family = None
        self.num_members_family = 1
        # Societal factor, set by update_societal.
        self.societal = 0
        self.category = category
        # Peer network weights.
        self.weight_vector = dict()
        # Measures of dissuasion
        self.denounce = denounce
        self.protection = protection
        self.condemnation = condemnation
        self.community_violence = 0

        self.stress_activated = True

        self.remain_times = 0
        self.break_times = 0

        self.violence_prevalence = defaultdict(int)
        self.month = 0
    
    def step(self):
        """
        A model step.
        """
        # An accumulated probability function is used to obtain the probability of n 
        # (value generated randomly with np.random.poisson) events or less occurring.
        # Probability of death by natural cause
        death_age = self.death_by_age(estimate=LIFE_ESTIMATE['male'] if self.gender == 'male'
        else LIFE_ESTIMATE['female'], age=self.age)

        if np.random.random() < death_age:
            self.is_dead = True
            return

        trigger_per = poisson.cdf(self.call_rate, np.random.poisson(self.call_rate))
        # Using the probability above to active or not update_stress function
        if np.random.random() < trigger_per:
            # Increases stress
            self.update_stress()
        else:
            # Decreases stress
            self.update_stress(decreases=True)

        # Calculate societal
        self.update_societal()
        # Calculate disposition
        self.calculate_disposition()
        # Acting on the disposition is subclass business: aggression belongs to Men,
        # responding to an assault belongs to Woman.
        self.act()

        # Go to the next month
        self.month = (self.month + 1) % 12

    def act(self):
        """Turn disposition into behaviour. No-op on the base class."""

    def has_adult_partner(self):
        return self.age > 18 and self.spouse is not None

    def sexism_index(self):
        """Sexist-norm component of the societal channel, on its own scale.

        Zero on the base class: the index is built from Perception_Woman brackets and
        describes the aggressor's norms, so only Men carry it.
        """
        return 0.0

    def partner_component(self):
        """Contribution of the partner's state to this agent's disposition. Zero here."""
        return 0.0

    def step_change(self):
        # How conditions that cause stress change?
        if self.model.random.random() < self.model.chance_changing_working_status:
            self.is_working = not self.is_working
            if not self.is_working:
                self.wage = 0
            else:
                self.reserve_wage *= self.model.random.uniform(-self.model.pct_change_wage,
                                                               self.model.pct_change_wage)
                self.wage = self.reserve_wage

    # Probability of death by age 
    @staticmethod
    def death_by_age(estimate, age):
        difference = abs(estimate - age)
        death_per = (1 - math.log(difference + 1) / math.log(101))
        return max(0, min(death_per, 1)) / MODEL_SCALE  

    def restore_inhibition(self):
        """Undo whatever the escalation channel took off the threshold. No-op here.

        Defined on Person because the two events that end a tie — release_ex_partner and
        the start of a new relationship — are Person business, while the offset itself
        belongs to Men.
        """

    def assign_spouse(self, agent):
        # A new tie starts from the inhibition each was drawn with: the erosion of
        # escalation is a property of the relationship it happened in, and a man who
        # re-partners before his ex-partner window closes would otherwise carry it over.
        self.restore_inhibition()
        agent.restore_inhibition()
        # Checks if the woman is black and associates a probability of not being in a relationship
        if self.color == 'preta':
            if np.random.random() < RacialIssueSpouse.HAVE_SPOUSE:
                self.spouse = agent
                agent.spouse = self
                self.number_of_past_relationships += 1
                return self.spouse
        else:
            self.spouse = agent
            agent.spouse = self
            self.number_of_past_relationships += 1
            return self.spouse

    # Ending a relationship
    def desassign_spouse(self):
        if self.spouse:
            partner = self.spouse
            # A relationship that held violence leaves an ex-partner tie behind; one that
            # did not end clean. The tie is what carries post-separation risk.
            if self.n_received_attacks or partner.n_received_attacks:
                for a, b in ((self, partner), (partner, self)):
                    a.ex_partner = b
                    a.months_since_separation = 0
                    # Lifetime: a relationship that held violence has ended, by whatever
                    # route. This is the DataSenado construct ("is an ex-partner now"),
                    # which `ever_broke` misses whenever she was not the one who left.
                    if hasattr(a, 'ever_separated_from_aggressor'):
                        a.ever_separated_from_aggressor = True
            else:
                # n_received_attacks feeds the femicide hazard, and Campos & Colman find a
                # prior-violence history in 100% of post-separation femicides — so it is
                # cleared only when the tie itself is cleared, never at the breakup that
                # precedes the risk.
                partner.n_received_attacks = 0
                self.n_received_attacks = 0
                # "MPU within the current relationship" stops applying with the tie, not
                # with the cohabitation: the order was granted against this man for this
                # woman and normally outlives their living together.
                for p in (self, partner):
                    if hasattr(p, 'mpu_this_relationship'):
                        p.mpu_this_relationship = False
                    p.restore_inhibition()
            partner.spouse = None
            self.spouse = None

    def release_ex_partner(self):
        """Drop the post-separation tie on both sides and clear what it was keeping alive."""
        ex = self.ex_partner
        for p in (self, ex):
            if p is None:
                continue
            p.ex_partner = None
            p.months_since_separation = 0
            p.n_received_attacks = 0
            if hasattr(p, 'mpu_this_relationship'):
                p.mpu_this_relationship = False
            # The eroded inhibition dies with the tie it was eroded in — the post-
            # separation window is the last stretch where it still applies.
            p.restore_inhibition()

    # societal factor update function
    def update_societal(self):
        """Societal channel in [0, 1]: sexist norms and local violence context.

        Each component is normalised on its own scale BEFORE being mixed, so the balance
        between them is set by settings.W_MACHISM and not by the units they arrive in —
        the sexism index runs on [0.785, 1.488] while community violence is a homicide rate
        per 100k inhabitants. Both settings are read at call time so a sweep can override
        them after import.
        """
        # Sexist factor (societal), normalised over the attainable range of the index.
        machism = self.sexism_index()
        machism_norm = self.normalize_(machism,
                                       Calibration.MAX_MACHISM,
                                       Calibration.MIN_MACHISM,
                                       1.0)
        # Spatial violence, normalised against the reference homicide rate.
        community_norm = min(1.0, self.community_violence / settings.COMMUNITY_RATE_REF)

        w = settings.W_MACHISM
        self.societal = w * machism_norm + (1 - w) * community_norm


    # Including stress on societal
    def update_stress(self, decreases=False):
        # Although all adults stress is updated, only 'males' are aggressors
        # Before updating stress, check whether working and income conditions have changed
        self.step_change()

        # Relief step: forgetting proportional to the accumulated stock. The equilibrium
        # is proportional to the agent's own stress inflow, and the stock stays
        # non-negative and bounded.
        if decreases:
            self.accumulated_stress -= settings.STRESS_DECAY * self.accumulated_stress
            return

        stress_level = 0
        # Update stress based on gender, wage level, hours at home, family size and history of violence
        # Schooling protects: stress rises with the schooling DEFICIT, so more years of
        # study contribute less stress. The deficit is normalised to [0, 1] so this term
        # stays on the same scale as the others in the sum. Below 6 years of study the
        # deficit weighs more (HIGH) than at or above 6 (LOW).
        schooling_deficit = max(0.0, (MAX_YEARS_STUDY - self.years_study) / MAX_YEARS_STUDY)
        if self.years_study >= 6:
            stress_level += schooling_deficit * (1 + Classification.LOW)
        else:
            stress_level += schooling_deficit * (1 + Classification.HIGH)
        # Gender differential, constant within gender: men carry GENDER_STRESS and women
        # its complement, i.e. 0.270 on men against 0.730 on women. Constant within
        # gender means a level shift, which the threshold draw largely absorbs; see the
        # note on GENDER_STRESS in settings.py.
        stress_level += self.model.gender_stress if self.gender == 'male' else (1 - self.model.gender_stress)
        # Salary
        stress_level += (1 - self.wage) * Classification.HIGH
        # Neighborhood quality: wage influences neighborhood_quality and house_size
        stress_level += -self.family.family_wage * Classification.MEDIUM
        # House size: higher the wage or smaller the wage per person, less contribution to stress
        stress_level += 1 - (self.family.family_wage / self.num_members_family) * Classification.MEDIUM
        # Type of work (if the woman works)
        if self.work_modality:
            stress_level += (1 - self.work_modality) * Classification.MEDIUM
        # The greater the number of people in the family, the greater the probability of aggression by the man
        # and more contribution to stress
        stress_level += self.num_members_family * Classification.HIGH
        # Adding more age ranges (the older the woman is, the lower the increase in stress)
        # Younger, more tolerant, although more chance of suffering violence
        if 18 < self.age < 29:
            stress_level += 1 * Classification.HIGH
        elif 28 < self.age < 60:
            stress_level += 1 * Classification.MEDIUM
        elif self.age >= 60:
            stress_level += 1 * Classification.LOW
        # Ethnicity influences victimization, likelihood increases when spouse is black
        # This stress indicator will only update for married males
        if self.spouse is not None:
            if self.spouse.color == 'preta':
                stress_level *= (1 + Classification.LOW)
            # Relevance of women participating in the labor market
        if self.is_working:
            # Increases -- level Classification.HIGH -- when spouse WORKS
            stress_level -= (1 * Classification.LOW)
        # History of assault
        # This stress indicator updates only for those attackers who already have a history
        stress_level += self.assaulted * Classification.LOW

        # Mean contribution across the stress components, so the inflow stays on the same
        # [0, ~1] scale as each individual term instead of growing with their number.
        stress_level /= settings.STRESS_COMPONENTS
        # Adding new stress to memory
        self.accumulated_stress += stress_level

    # Signed, mean-based peer influence on disposition (both amplifying and damping).
    def network_influence(self):
        """Two-directional peer influence, bounded-friendly.

        Uses the MEAN of weighted peer dispositions (not the sum) so the term does not
        scale with network size — the sum was the driver of the disposition runaway.

        The base effect is contagion/damping toward the peer mean *relative to the
        agent's own disposition*: peers with higher disposition pull it up, peers with
        lower disposition pull it down (this is what makes it work "both ways").

        Prosocial peer norms (denouncing / leaving) do NOT enter here. They raise a
        woman's propensity to ACT, which is a separate stock — see
        Woman.calculate_action_disposition and Woman.peer_action_norm.

        Returns a raw signed value; calculate_disposition bounds it via tanh.
        """
        if not self.weight_vector:
            return 0.0

        peer_mean = np.mean([agent.disposition * self.weight_vector[agent]
                             for agent in self.weight_vector.keys()])

        # Contagion (positive) / damping (negative) toward the peer mean.
        return peer_mean - self.disposition

    @staticmethod
    def normalize_(value, maxv, minv, per):
        # Min-max scale into [0, per], CLIPPED. The clipping is what keeps out-of-range
        # inputs from producing out-of-range outputs, which would let the network term
        # feed disposition back into itself without bound.
        scaled = (value - minv) / (maxv - minv)
        scaled = min(1.0, max(0.0, scaled))
        return scaled * per

    # Disposition, by Rescorla-Wagner update toward a target.
    def calculate_disposition(self) -> None:
        # The stress stock and the societal factor stay on their own scales; only their
        # contribution to disposition is rescaled.
        individual = self.normalize_(
            self.accumulated_stress,
            Calibration.MAX_INDIVIDUAL,
            Calibration.MIN_INDIVIDUAL,
            Calibration.PER_INDIVIDUAL
        )

        # update_societal already returns a [0, 1] value, so only the channel ceiling
        # is applied here.
        societal = Calibration.PER_MACHISM * self.societal

        # Target value.
        target_disposition = individual + societal + self.partner_component()

        if self.weight_vector:
            # Bounded, signed network term in [-PER_NETWORK, +PER_NETWORK] (tanh squashing),
            # so peers can both raise and damp disposition without it ever running away.
            network = Calibration.PER_NETWORK * np.tanh(self.network_influence())
            target_disposition += network

        # Rescorla-Wagner update.
        self.disposition += RESCORLA_WAGNER_ALPHA * (target_disposition - self.disposition)







class Relationship(Agent):
    """
    A relationship that provides the environment and contain agents who might become victims or aggressors
    """

    def __init__(self, unique_id: int, model: Model,
                 pos, init_date: date, estimated_duration: date):
        super().__init__(unique_id, model)
        self.pos = pos
        self.context_stress = 0
        self.members = dict()
        self.family_wage = 0
        self.address = None
        # Setting the start date of the relationship
        self.init_date = init_date
        # Setting the estimated duration of the relationship
        self.estimated_duration = estimated_duration
        self.it_ended = False

    def add_agent(self, agent: Person):
        self.members[agent.unique_id] = agent
        agent.family = self
        if self.address is None:
            self.address = agent.address



class Men(Person):
    def __init__(self, unique_id: int, model: Model, pos, gender='male', age=25, color='negra',
                 years_study=6, is_working=False, house_location=HouseLocation.URBAN,
                 wage=0, reserve_wage=.5, address=None, category='person', denounce=False, protection=False,
                 condemnation=False, call_rate=1, threshold=0, disposition=0, birthmonth=None,
                 relationship=None):

        super().__init__(
            unique_id, model, pos, gender, age, color,
            years_study, is_working, house_location,
            wage, reserve_wage, address, category, denounce,
            protection, condemnation, call_rate, threshold,
            disposition, birthmonth, relationship
        )

        # Monitoring system
        self.is_monitored = False
        self.monitoring_time = 0
        self.monitoring_level = MONITORING_LEVEL
        self.violence_recidivism_risk = 0.0

        # Escalation: how much his own violence has taken off the threshold within the
        # current tie. Kept apart from threshold_mpu_boost so the two offsets compose and
        # unwind independently — an order can be granted and expire while the erosion
        # stands, and the erosion is released with the tie while the order still runs.
        self.threshold_escalation = 0.0

        # Protective order (medida protetiva de urgência).
        self.has_mpu = False
        self.threshold_mpu_boost = 0.0  # how much the threshold was raised; undone at expiry
        self.monitoring_duration = 0    # total months of the current grant
        self.mpu_time = 0               # months left on the order, independent of the visits
        self.mpu_patrolled = False      # the order was forwarded to a patrol that reached him
        self.mpu_violation_count = 0    # breaches of an active order
        self.times_arrested = 0         # arrests in flagrante for breaching
        # Instrumentation flags (measurement only, do not affect behaviour)
        self.ever_had_mpu = False           # ever received an MPU (any relationship)
        self.mpu_this_relationship = False  # received an MPU in the current relationship (reset on breakup)
        self._grant_violated = False        # current grant already counted as violated (avoid double count)

    def act(self):
        if self.has_adult_partner():
            self.trigger_violence()
        elif self.ex_partner is not None:
            self.pursue_ex_partner()

    def pursue_ex_partner(self):
        """Approach the woman he separated from, and the assault that may follow it.

        Contact is drawn first and violence second, because that is the order the art. 24-A
        reports describe: the violation the statute defines is the approach or the contact,
        with threat as its most frequent companion and bodily harm further down. Drawing it
        this way puts the existing deterrence channels — the threshold boost and the
        monitoring decay — on the escalation rather than only on cohabiting violence.

        A man who re-partners stops pursuing: `act` only reaches here when he has no
        current partner. That is a simplification, not a finding.
        """
        ex = self.ex_partner
        if ex is None or ex.is_dead or self.is_dead:
            self.release_ex_partner()
            return

        self.months_since_separation += 1
        ex.months_since_separation = self.months_since_separation
        if self.months_since_separation > POST_SEP_WINDOW:
            self.release_ex_partner()
            return

        # Linear decay to zero at the end of the window: the risk Campos & Colman describe
        # sits within the first year, heaviest early.
        decay = 1.0 - (self.months_since_separation - 1) / POST_SEP_WINDOW
        if np.random.random() >= POST_SEP_CONTACT_BASE * decay:
            return

        self.model.post_sep_contact_count += 1
        # The contact itself is the breach, whether or not it turns physical. An arrest on
        # that contact is what stops it from turning physical — the interruption bites here
        # precisely because the approach is drawn before the violence.
        if self.has_mpu and self.register_mpu_violation(post_separation=True):
            return

        if np.random.random() < self.violence_hazard():
            self.assault(ex, post_separation=True)

    def erode_inhibition(self):
        """Lower the threshold by one escalation step, up to the cap for this tie.

        Mirror image of receive_mpu's boost: the order raises the barrier, having crossed
        it lowers it. See settings.ESCALATION_STEP for why the model needs the act to
        leave a trace at all.
        """
        step = min(settings.ESCALATION_STEP,
                   settings.ESCALATION_MAX - self.threshold_escalation)
        if step <= 0:
            return
        self.threshold_escalation += step
        self.threshold -= step

    def restore_inhibition(self):
        if not self.threshold_escalation:
            return
        self.threshold += self.threshold_escalation
        self.threshold_escalation = 0.0

    def register_mpu_violation(self, post_separation=False):
        """Count a breach of an active order, and draw the arrest that may follow it.

        The channel of the FIRST breach is recorded too, since the two conducts are not
        the same thing: approach or contact is what art. 24-A itself defines, whereas a
        new assault under a still-cohabiting order is the model's other path in.

        Returns True if the breach ended in custody, so the caller can stop: an arrested
        man commits nothing further this month. This is the model's only channel that
        interrupts a violent spell — see settings.P_ARREST_ON_BREACH.
        """
        arrested = np.random.random() < settings.P_ARREST_ON_BREACH
        if arrested:
            self.times_arrested += 1
            self.model.detain(self)
        self.mpu_violation_count += 1
        if not self._grant_violated:
            self._grant_violated = True
            self.model.mpu_grant_violated_count += 1
            # monitoring_time counts down from monitoring_duration, so the difference is
            # how long the order had been standing when it was first breached.
            self.model.mpu_breach_delay_months.append(
                self.monitoring_duration - self.monitoring_time)
            if post_separation:
                self.model.mpu_grant_violated_post_sep += 1
            else:
                self.model.mpu_grant_violated_in_rel += 1
        return arrested

    def sexism_index(self):
        return self.machismo_index()

    def partner_component(self):
        """Dyadic channel: the partner's agitation raises this man's disposition.

        Katerndahl reports the woman's own agitated state as a risk factor for the
        partner's aggression, so her disposition enters his as a fourth channel. Her
        disposition lives on the same scale as his, so it is normalised against the sum
        of the channel ceilings before being weighted.
        """
        if self.spouse is None:
            # Separation does not switch this channel off; it changes what fills it. Campos
            # & Colman give the motive in every post-separation case they read as
            # possessiveness and non-acceptance of the ending, so the channel is carried at
            # full weight from the breakup and decays over the same window as the pursuit.
            if self.ex_partner is None:
                return 0.0
            decay = 1.0 - min(self.months_since_separation, POST_SEP_WINDOW) / POST_SEP_WINDOW
            return PER_DYADIC * decay
        ceiling = (Calibration.PER_INDIVIDUAL + Calibration.PER_NETWORK
                   + Calibration.PER_MACHISM)
        return PER_DYADIC * min(1.0, max(0.0, self.spouse.disposition / ceiling))

    def machismo_index(self) -> float:
        indice = 0

        def normalize(value, min_val, max_val):
            return (value - min_val) / (max_val - min_val)

        # Scores by age band.
        age_factors = {
            'BETWEEN 15 24 COMMON': Perception_Woman.AGE['BETWEEN 15 24 COMMON'],
            'BETWEEN 25 44 COMMON': Perception_Woman.AGE['BETWEEN 25 44 COMMON'],
            'MORE OR EQUAL 45 COMMON': Perception_Woman.AGE['MORE OR EQUAL 45 COMMON'],
            'MAX COMMON': Perception_Woman.AGE['MAX COMMON'],
            'BETWEEN 15 24 CRITICAL': Perception_Woman.AGE['BETWEEN 15 24 CRITICAL'],
            'BETWEEN 25 44 CRITICAL': Perception_Woman.AGE['BETWEEN 25 44 CRITICAL'],
            'MORE OR EQUAL 45 CRITICAL': Perception_Woman.AGE['MORE OR EQUAL 45 CRITICAL'],
            'MAX CRITICAL': Perception_Woman.AGE['MAX CRITICAL']
        }

        # Scores by schooling band.
        study_factors = {
            'LESS THAN 4 COMMON': Perception_Woman.STUDY['LESS THAN 4 COMMON'],
            'BETWEEN 5 8 COMMON': Perception_Woman.STUDY['BETWEEN 5 8 COMMON'],
            'BETWEEN 11 14 COMMON': Perception_Woman.STUDY['BETWEEN 11 14 COMMON'],
            'GREATER OR EQUAL 15 COMMON': Perception_Woman.STUDY['GREATER OR EQUAL 15 COMMON'],
            'MAX COMMON': Perception_Woman.STUDY['MAX COMMON'],
            'LESS THAN 4 CRITICAL': Perception_Woman.STUDY['LESS THAN 4 CRITICAL'],
            'BETWEEN 5 8 CRITICAL': Perception_Woman.STUDY['BETWEEN 5 8 CRITICAL'],
            'BETWEEN 11 14 CRITICAL': Perception_Woman.STUDY['BETWEEN 11 14 CRITICAL'],
            'GREATER OR EQUAL 15 CRITICAL': Perception_Woman.STUDY['GREATER OR EQUAL 15 CRITICAL'],
            'MAX CRITICAL': Perception_Woman.STUDY['MAX CRITICAL']
        }

        # Age factor, each sub-scale normalised by its maximum.
        if 15 <= self.age <= 24:
            indice += normalize(age_factors['BETWEEN 15 24 COMMON'], 0, age_factors['MAX COMMON']) + \
                      normalize(age_factors['BETWEEN 15 24 CRITICAL'], 0, age_factors['MAX CRITICAL'])
        elif 25 <= self.age <= 44:
            indice += normalize(age_factors['BETWEEN 25 44 COMMON'], 0, age_factors['MAX COMMON']) + \
                      normalize(age_factors['BETWEEN 25 44 CRITICAL'], 0, age_factors['MAX CRITICAL'])
        elif self.age >= 45:
            indice += normalize(age_factors['MORE OR EQUAL 45 COMMON'], 0, age_factors['MAX COMMON']) + \
                      normalize(age_factors['MORE OR EQUAL 45 CRITICAL'], 0, age_factors['MAX CRITICAL'])
            
        # Years study factor    
        if self.years_study <= 4:
            indice += normalize(study_factors['LESS THAN 4 COMMON'], 0, study_factors['MAX COMMON']) + \
                      normalize(study_factors['LESS THAN 4 CRITICAL'], 0, study_factors['MAX CRITICAL'])
        # Covers 5-10 because the survey brackets in Perception_Woman.STUDY define no
        # 9-10 band: incomplete secondary is scored with the lower bracket rather than
        # with completed secondary.
        elif 5 <= self.years_study <= 10:
            indice += normalize(study_factors['BETWEEN 5 8 COMMON'], 0, study_factors['MAX COMMON']) + \
                      normalize(study_factors['BETWEEN 5 8 CRITICAL'], 0, study_factors['MAX CRITICAL'])
        elif 11 <= self.years_study <= 14:
            indice += normalize(study_factors['BETWEEN 11 14 COMMON'], 0, study_factors['MAX COMMON']) + \
                      normalize(study_factors['BETWEEN 11 14 CRITICAL'], 0, study_factors['MAX CRITICAL'])
        elif self.years_study >= 15:
            indice += normalize(study_factors['GREATER OR EQUAL 15 COMMON'], 0, study_factors['MAX COMMON']) + \
                      normalize(study_factors['GREATER OR EQUAL 15 CRITICAL'], 0, study_factors['MAX CRITICAL'])
        # Returned on its own scale — [0.785, 1.488] for an adult, lower below age 15 where
        # no age band applies. Calibration.MAX_MACHISM / MIN_MACHISM are those bounds and
        # update_societal normalises against them.
        return indice

    def trigger_femicide(self, victim):
        # Probability scales linearly with the count of prior attacks in this tie, clipped
        # to [0, 1] — the only ceiling the quantity admits, since it is a probability and no
        # source pins a hazard maximum. The clip is inert at any plausible FEMICIDE_BASE
        # (see settings.py); it is there so the draw stays well defined, not to bind.
        n_prior = victim.n_received_attacks
        prob = min(n_prior * self.model.femicide_base, 1.0)
        return np.random.random() < prob

    # Trigger violence according to disposition value
    def violence_hazard(self):
        """Per-step probability of assaulting, logistic in the gap `disposition - threshold`.

        Replaces a deterministic `disposition > threshold` comparison. Evaluated every step
        over a 144-month run, that comparison made violence a function of the LIFETIME
        MAXIMUM of disposition — and maxima of long noisy series are near-homogeneous, so
        the schooling gradient carried by disposition (-0.46 on its mean) reached the
        outcome as -0.07, and a draw on threshold alone classified 93% of men correctly.

        A smooth hazard makes the whole trajectory count rather than only its highest
        point: a man who sits high for years differs from one who spiked once.

        TRIGGER_SLOPE -> 0 recovers the deterministic trigger exactly, so the previous
        behaviour is the limiting case of this one and the two are comparable.
        """
        gap = self.disposition - self.threshold
        if TRIGGER_SLOPE <= 0:
            return 1.0 if gap > 0 else 0.0
        # Clipped to keep exp() from overflowing on agents far from their threshold; at
        # |z| = 60 the probability is already 0 or 1 to machine precision.
        z = np.clip(gap / TRIGGER_SLOPE, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def trigger_violence(self):
        """
        Uses disposition and threshold in probability of becoming violent
        """
        if np.random.random() < self.violence_hazard():
            self.assault(self.spouse)
        else:
            self.spouse.violence_prevalence[self.month] = 0

    def assault(self, victim, post_separation=False):
        """Record an assault on `victim` and draw the femicide that may follow.

        Shared by the in-relationship path and the post-separation one, so that leaving
        does not silently change what an assault is.
        """
        # Record the act on both sides, before the lethality draw.
        self.n_attacks += 1
        self.assaulted += 1
        victim.n_received_attacks += 1
        victim.got_attacked += 1
        victim.ever_attacked = True
        victim.assaults_lifetime += 1
        victim.violence_prevalence[self.month] = 1
        # Flags the victim's action stage for this step. She clears it once she has
        # responded, so an assault landing after her own step is answered next month
        # rather than lost.
        victim.attacked_this_step = True
        if post_separation:
            self.model.post_sep_assault_count += 1

        if self.trigger_femicide(victim):
            victim.is_dead = True
            victim.femicide_victim = True
            # Instrumentation: femicide composition by aggressor MPU history.
            self.model.femicide_count += 1
            if post_separation:
                self.model.femicide_post_separation += 1
            if self.ever_had_mpu:
                self.model.femicide_with_ever_mpu += 1
                victim.spouse_had_mpu = True
            if self.mpu_this_relationship:
                self.model.femicide_with_relationship_mpu += 1

        # The act leaves a trace on the man who committed it: what held him back holds him
        # back less next month. Applied after the femicide draw, so the first assault is
        # not also the escalated one.
        self.erode_inhibition()

        # First time offender get registered in the system and changes category into an Aggressor and a Victim.
        # self.assaulted reads 1 on the first assault, having been incremented above.
        if self.assaulted == 1:
            self.category = 'aggressor'
            victim.category = 'victim'

    def apply_monitoring(self, duration=12, level=MONITORING_LEVEL):
        self.is_monitored = True
        self.monitoring_time = duration
        self.monitoring_level = level

    def receive_mpu(self, duration=None, level=None, boost=None):
        """Grant an order against the aggressor: raise his threshold and, if the order is
        patrolled, start monitoring.

        TWO EFFECTS, AND THEY HAVE DIFFERENT INSTITUTIONS BEHIND THEM. The judicial act
        itself raises the threshold: since Lei 13.641/2018 breaching the order is a crime
        in its own right (art. 24-A), which is expected sanction and belongs on the
        inhibition side. The periodic visits are a separate body — the Patrulha Maria da
        Penha, to which every granted measure is forwarded (IPEA 2019, p.56) — and they
        are what `reduce_disposition` represents. Coverage is not universal, so an order
        can exist without ever being visited, and PATROL_COVERAGE is that fraction.

        Defaults read the settings globals at call time (not definition time) so a
        parameter sweep can override MPU_DURATION / MPU_MONITORING_LEVEL /
        MPU_THRESHOLD_BOOST by reassigning them on this module.
        """
        if duration is None:
            duration = MPU_DURATION
        if level is None:
            level = MPU_MONITORING_LEVEL
        if boost is None:
            boost = MPU_THRESHOLD_BOOST
        if self.has_mpu:
            return  # already under an order; orders do not stack
        self.has_mpu = True
        self.threshold_mpu_boost = boost
        self.monitoring_duration = duration
        # The order runs on its own clock, decremented by update_mpu, separate from
        # monitoring_time: an unpatrolled order is never monitored and must still expire.
        self.mpu_time = duration
        self.threshold += boost  # a higher threshold is harder to cross
        self.mpu_patrolled = np.random.random() < PATROL_COVERAGE
        if self.mpu_patrolled:
            self.apply_monitoring(duration=duration, level=level)
        # Instrumentation: count this grant (order-level) and mark aggressor state.
        self.ever_had_mpu = True
        self.mpu_this_relationship = True
        self._grant_violated = False
        self.model.mpu_granted_count += 1
        self.model.mpu_patrolled_count += int(self.mpu_patrolled)

    def update_mpu(self):
        """Run down the order's own clock and undo its threshold boost when it expires.

        Separate from update_monitoring because the order and the visits are separate
        institutions with separate coverage: an unpatrolled order still expires.
        """
        if not self.has_mpu:
            return
        self.mpu_time -= 1
        if self.mpu_time <= 0:
            self.threshold -= self.threshold_mpu_boost
            self.threshold_mpu_boost = 0.0
            self.monitoring_duration = 0
            self.mpu_time = 0
            self.has_mpu = False

    def update_monitoring(self):


        if not self.is_monitored:
            return


        self.monitoring_time -= 1 # Decrease remaining monitoring time

        self.reduce_disposition() # Apply behavioral effect of monitoring


        self.violence_recidivism_risk = (
            self.assaulted / (1 + self.monitoring_time + 1)
        )

        # Remove monitoring when time expires. The order's own expiry is update_mpu's job.
        if self.monitoring_time <= 0:
            self.is_monitored = False
            self.monitoring_level = MONITORING_LEVEL

    def reduce_disposition(self):
        reduction_factor = (1 - self.monitoring_level)

        # Reduce aggression tendency
        self.disposition *= reduction_factor

        # Reduce accumulated stress
        self.accumulated_stress *= reduction_factor

    # Monitoring is reachable only through a granted order: every granted measure is
    # forwarded to the Patrulha Maria da Penha and no visit occurs without one (IPEA
    # 2019, p. 56). The entry point is receive_mpu, gated by PATROL_COVERAGE.

    def step(self):
        # In custody after an arrest in flagrante: he does not act, and nothing else of
        # his runs either, monitoring included, so custody acts through interruption
        # only. Leaving the schedule is not enough on its own: schedule.step_breed
        # iterates agents_by_breed, not agents_t.
        if self in self.model.detained:
            return

        assaulted_before = self.assaulted
        super().step()

        # Breach while still cohabiting: an assault committed under an active order. The
        # post-separation case is recorded in pursue_ex_partner, where the conduct art.
        # 24-A defines is the contact itself; the gate on `spouse` keeps the two regimes
        # apart without double counting.
        if self.has_mpu and self.spouse is not None and self.assaulted > assaulted_before:
            self.register_mpu_violation()
            if np.random.random() < P_VIOL:
                if self.relationship is not None:
                    self.relationship.it_ended = True
                self.desassign_spouse()

        # Run down the order, then the visits it may or may not have brought with it.
        self.update_mpu()
        self.update_monitoring()


class Woman(Person):
    """A woman carries the victim side of the Agent Zero apparatus.

    `disposition` (inherited) is affect/agitation and drives no action of her own; it
    reaches the partner through Men.partner_component. `disposition_action` is a second
    stock — her propensity to act on an assault — compared against a threshold of her own.
    """

    def __init__(self, unique_id: int, model: Model, pos, gender='female', age=25, color='negra',
                 years_study=6, is_working=False, house_location=HouseLocation.URBAN,
                 wage=0, reserve_wage=.5, address=None, category='person', denounce=False,
                 protection=False, condemnation=False, call_rate=1, threshold=0, disposition=0,
                 birthmonth=None, relationship=None):

        super().__init__(
            unique_id, model, pos, gender, age, color,
            years_study, is_working, house_location,
            wage, reserve_wage, address, category, denounce,
            protection, condemnation, call_rate, threshold,
            disposition, birthmonth, relationship
        )

        # Agent Zero, victim side.
        self.disposition_action = 0.0
        self.threshold_action = np.random.uniform(settings.ACTION_THRESHOLD_MIN,
                                                  settings.ACTION_THRESHOLD_MAX)
        # Set by the aggressor's trigger_violence; cleared once she has responded.
        self.attacked_this_step = False
        # Lifetime flag, never reset. `got_attacked` is an ANNUAL count — anything asking
        # "was she ever assaulted" must read this, or it silently measures the last year
        # only in any harness that resets the annual counters.
        self.ever_attacked = False
        # Lifetime flag, never reset: she ended a relationship through the action stage,
        # i.e. in response to an assault. `break_times` is ANNUAL, so the accumulated
        # separation moment held out for validation cannot be read from it.
        self.ever_broke = False
        # Lifetime, never reset: a relationship in which she was assaulted has ended, by
        # any route — she left, he left, or partner rotation. Broader than `ever_broke`,
        # which counts only the acts she took herself.
        self.ever_separated_from_aggressor = False
        # Assaults received over the whole run, never reset. `n_received_attacks` is reset
        # on breakup and `got_attacked` annually, so neither counts how many times she
        # reached the action stage in her life — which is what the accumulated separation
        # share is a function of.
        self.assaults_lifetime = 0
        # Times she crossed threshold_action this year. Counted before the composition
        # draw, so it isolates what the threshold governs (whether she acts) from what
        # ACTION_DENOUNCE_SHARE governs (which act) — and it does not lose the acts whose
        # Break branch finds no relationship to end. Reset annually.
        self.action_times = 0
        # Denunciation EVENTS this year. `denounce` is a flag ("denounced at all this
        # year") read by peer_action_norm and by the output tables; this is its
        # event-level counterpart, the twin of break_times, so that
        # action_times - denounce_times - break_times is exactly the acts whose Break
        # branch found no relationship to end.
        self.denounce_times = 0

    def peer_action_norm(self):
        """Share of female peers who acted (denounced or left) rather than remained.

        Reads the annual counters, which year_register resets, so the norm is the norm of
        the current year and does not mix time scales.
        """
        if not self.weight_vector:
            return 0.0
        peers = list(self.weight_vector)
        denounced = sum(1 for a in peers if a.denounce)
        breaks = sum(a.break_times for a in peers)
        remains = sum(a.remain_times for a in peers)
        events = breaks + remains
        leaving_norm = breaks / events if events else 0.0
        return 0.5 * (denounced / len(peers)) + 0.5 * leaving_norm

    def calculate_action_disposition(self):
        """Rescorla-Wagner update of the propensity to act, on the same form as disposition.

        The three components are the victim-side counterparts of the aggressor's channels:
        accumulated victimisation (affective), the peer action norm (social), and economic
        autonomy (deliberative). Weights sum to 1, so the target lives in [0, 1].

        Assaults received deliberately do NOT feed `disposition` (agitation) in this
        version: that would close the escalation loop through Men.partner_component, and
        the loop gain would then be entangled with the prevalence calibration.
        """
        experience = min(self.n_received_attacks / ACTION_EXPERIENCE_REF, 1.0)
        autonomy = 1.0 if self.is_working else 0.0

        target = (PER_ACTION_EXPERIENCE * experience
                  + PER_ACTION_NETWORK * self.peer_action_norm()
                  + PER_ACTION_AUTONOMY * autonomy)

        self.disposition_action += RESCORLA_WAGNER_ALPHA * (target - self.disposition_action)

    def act(self):
        self.calculate_action_disposition()
        # She responds to an assault by a current partner or by the man she left: the
        # post-separation assault is the one protective orders exist for, so it cannot be
        # the one she has no stage to answer.
        if self.model.dissuasion and (self.has_adult_partner() or self.ex_partner is not None):
            self.trigger_call_help()

    def trigger_call_help(self):
        """Victim decision stage: Denounce, Break or Remain, following an assault.

        Gated on an assault in THIS step, matching the paper. The previous gate read the
        cumulative `got_attacked` counter, which year_register happens to reset annually —
        so whether a woman drew an action every month or only after an assault depended on
        which harness was running the model.
        """
        if not self.attacked_this_step:
            return
        self.attacked_this_step = False

        # Agent Zero: below her own threshold she stays. Remain is the default state,
        # not the residual of two small probabilities.
        if self.disposition_action <= self.threshold_action:
            self.remain_times += 1
            # Honeymoon/calm phase: victim staying temporarily reduces
            # the abuser's accumulated stress (Walker 1979 cycle of violence).
            # Does not reset fully — tension rebuilds in subsequent steps.
            if self.spouse is not None:
                self.spouse.accumulated_stress *= (1 - REMAIN_CALM_FACTOR)
            return

        # She acts. Which of the two acts is set by ACTION_DENOUNCE_SHARE, anchored on
        # the ratio of the two target rates. Leaving is on the menu only while there is
        # something to leave: once she has gone, the whole propensity routes to the
        # formal channel.
        self.action_times += 1
        can_leave = self.spouse is not None and self.relationship is not None
        if can_leave and np.random.random() >= ACTION_DENOUNCE_SHARE:
            self.relationship.it_ended = True
            self.break_times += 1
            self.ever_broke = True
            return

        self.denounce = True
        self.denounce_times += 1
        # Two stages, not one: the denunciation may or may not carry a request for an
        # MPU, and the request may or may not be granted. P_MPU_GRANTED is conditional
        # on a request having been made (CNJ counts concedidas/solicitadas), so a
        # denunciation reaches an order with probability P_MPU_REQUESTED x P_MPU_GRANTED.
        # The order is sought against whoever assaulted her — the partner she still has or
        # the one she left.
        aggressor = self.spouse if isinstance(self.spouse, Men) else self.ex_partner
        if isinstance(aggressor, Men) and np.random.random() < P_MPU_REQUESTED:
            self.model.mpu_requested_count += 1
            if np.random.random() < P_MPU_GRANTED:
                aggressor.receive_mpu()
