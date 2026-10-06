"""
The model: population, relationships, scheduling and data collection.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from scipy.stats import poisson

import settings
from agents import Person, Men, Woman, Relationship
from homicide import violence_per_year
from input import generator
from input.generator import metropolis
from schedule import RandomActivationByBreed
from settings import AVERAGE_TIME_RELATIONSHIP
from settings import Weight, YEARS_IN_PRISON
from settings import GENDER_STRESS, FEMICIDE_BASE, IS_WORKING, CHANCE_CHANGE_WORKING_STATUS, PCT_CHANGE_WAGE


class Home(Model):
    """
    A Home Violence Simulation Model
    """

    verbose = True  # Print-monitoring
    description = 'A model for simulating the victim aggressor interaction.'

    def __init__(self, height=40, width=40,
                 initial_relationships=1000,
                 metro='BRASILIA',
                 gender_stress=GENDER_STRESS,
                 femicide_base=FEMICIDE_BASE,
                 is_working_pct=IS_WORKING,
                 chance_changing_working_status=CHANCE_CHANGE_WORKING_STATUS,
                 pct_change_wage=PCT_CHANGE_WAGE,
                 dissuasion=True,
                 data_year=2010,
                 steps=50
                 ):
        """
        A violence model of Brazilian metropolis

        """
        super().__init__()

        # Register ##################################################################################################
        self.females = 0
        self.incarcerated = dict()
        # Custody after arrest in flagrante for breaching an order, in MONTHS. Separate
        # from `incarcerated`, which follows a femicide and is counted in YEARS.
        self.detained = dict()
        self.arrest_count = 0

        # MPU instrumentation (measurement only — read out of the existing mechanism).
        # Order-level: how many MPUs were requested, granted, and violated at least once.
        self.mpu_requested_count = 0
        self.mpu_granted_count = 0
        # Of the granted orders, how many the Patrulha Maria da Penha actually reached.
        # The visits are a separate institution from the order, with its own coverage.
        self.mpu_patrolled_count = 0
        self.mpu_grant_violated_count = 0
        # Which conduct breached the order first, on the same order-level denominator:
        # post-separation contact (art. 24-A's own conduct) or a new assault while still
        # living together. The two sum to mpu_grant_violated_count.
        self.mpu_grant_violated_post_sep = 0
        self.mpu_grant_violated_in_rel = 0
        # Months elapsed between a grant and its first breach, one entry per breached
        # order. Deterrence that only bites after a lag cannot be strengthened into a
        # lower rate if the breaches all land before the lag runs out.
        self.mpu_breach_delay_months = []
        # Femicide composition: total femicides and how many had an aggressor with a prior MPU.
        self.femicide_count = 0
        self.femicide_with_ever_mpu = 0          # aggressor ever held an MPU (any relationship)
        self.femicide_with_relationship_mpu = 0  # aggressor held an MPU within the current relationship
        # Post-separation channel: approaches made after the couple split, the ones that
        # turned physical, and the femicides that followed.
        self.post_sep_contact_count = 0
        self.post_sep_assault_count = 0
        self.femicide_post_separation = 0
        self.steps = steps
        self.steps_count = 0
        self.areap_data = {}
        self.violence_indice = violence_per_year
        self.single_agents = set()

        # Set parameters
        self.height = height
        self.width = width

        self.initial_relationships = initial_relationships
        self.metro = metro

        self.gender_stress = gender_stress
        self.femicide_base = femicide_base
        self.is_working_pct = is_working_pct
        self.chance_changing_working_status = chance_changing_working_status
        self.pct_change_wage = pct_change_wage

        self.dissuasion = dissuasion
        self.data_year = data_year
        self.neighborhood_stress = dict()

        # Setting a model datetime
        self.current_date = date(year=self.data_year, month=1, day=1)

        self.schedule = RandomActivationByBreed(self)
        self.grid = MultiGrid(self.height, self.width, torus=True)
        # Model reporters. How they work: lambda function passes model and other parameters.
        # Then another function may iterate over each 'agent in model.schedule.agents'
        # Then server is going to collect info using the keys in model_reporters dictionary
        model_reporters = {
            "Denounce": lambda m: self.count_type_citizens(m, 'denounce'),
            "Got attacked": lambda m: self.count_type_citizens(m, 'got_attacked'),
            "Females": lambda m: self.count_type_citizens(m, 'female'),
            "Stress": lambda m: self.count_stress(m)}
        self.datacollector = DataCollector(model_reporters=model_reporters)

        # Create people ---------------------------------------------------
        # Parameters to choose metropolitan region
        # More details available at input/population
        params = dict()
        params['PROCESSING_ACPS'] = [self.metro]
        params['MEMBERS_PER_FAMILY'] = 2  # Changing for the couple in a relationship
        params['INITIAL_FAMILIES'] = self.initial_relationships
        params['DATA_YEAR'] = self.data_year
        # Denominator for the community-violence rate (see community_violence).
        self.mun_population = generator.municipal_population(params)
        people, families = generator.main(params=params)
        n = sum([len(f) for f in families])
        working = np.random.choice([True, False], n, p=[self.is_working_pct, 1 - self.is_working_pct])
        i = 0
        for fam in families:
            # 1. Create a relationship.
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            # Create a relationship
            relationship = Relationship(self.next_id(), self, (x, y),
                                        init_date=date(year=self.data_year, month=1,
                                                       day=np.random.randint(1, 29)),
                                        # Estimated relationship duration included on object creation
                                        estimated_duration=self.current_date + timedelta(
                                            365 * self.relationship_duration())
                                        )
            
            # Add relationship to grid
            self.grid.place_agent(relationship, (x, y))
            self.schedule.add(relationship)
            # Marriage procedures
            engaged = ['male_adult', 'female_adult']
            to_marry = list()
            # Add people to relationship's
            for each in fam:
                person = people.loc[each]
                # Males carry the MPU/monitoring behaviour; females carry the victim
                # decision stage and its own disposition/threshold pair.
                agent_cls = Men if person.gender == 'male' else Woman
                doe = agent_cls(self.next_id(), self, (x, y),
                             # From IBGE's sampling
                             gender=person.gender,
                             age=person.age,
                             color=person.cor,
                             threshold=np.random.uniform(settings.THRESHOLD_MIN,
                                                         settings.THRESHOLD_MAX),
                             address=person.AREAP,
                             years_study=person.years_study,
                             reserve_wage=person.wage,
                             birthmonth=np.random.randint(1, 12),
                             # End of sampling
                             is_working=working[i])
                # Marrying
                if len(engaged) > 0:
                    if person.category in engaged:
                        engaged.remove(person.category)
                        to_marry.append(doe)
                # Change position in the random list previously sampled
                i += 1
                self.grid.place_agent(doe, (x, y))
                self.schedule.add(doe)
                relationship.add_agent(doe)
                # Same attribute the re-pairing branch below sets, so that Break and the
                # removal provision can find a relationship to end.
                doe.relationship = relationship

            # 2. Marry the couple
            if len(to_marry) == 2:
                to_marry[0].assign_spouse(to_marry[1])

            # 3. Update number of family members
            for member in relationship.members.values():
                member.num_members_family = len(relationship.members)
            

        self.running = True
        self.datacollector.collect(self)
        self.update_neighborhood_stress()

    def update_incarcerated(self):
        for agent in list(self.incarcerated.keys()):
            self.incarcerated[agent] -= 1
            if not self.incarcerated[agent]:
                agent.accumulated_stress = 0
                self.schedule.agents_t.add(agent)
                del self.incarcerated[agent]

    def detain(self, agent, months=None):
        """Take an agent into custody for `months`, starting next step.

        Custody is what interrupts a violent spell: while detained he draws no assault and
        no post-separation approach, and none of his own clocks run — the order's included,
        so an arrest suspends the measure rather than shortening it. Men.step is what
        enforces that; leaving agents_t only keeps him out of re-partnering and the
        neighbourhood updates, since schedule.step_breed iterates agents_by_breed.

        A second arrest while already detained restarts the count rather than stacking, the
        same convention receive_mpu uses for a second order.
        """
        if agent in self.incarcerated:
            return
        # Read at call time, so a sweep can override it on the settings module.
        self.detained[agent] = settings.ARREST_DURATION if months is None else months
        self.schedule.agents_t.discard(agent)
        self.arrest_count += 1

    def update_detained(self):
        """Run custody down by one month and put the agent back on the schedule.

        Called at the top of step(), before the agents act, so an agent detained during
        month t is absent for the whole of t+1..t+`months` and acts again at t+months+1.
        Releasing at zero rather than decrementing past it is what makes that count exact.
        """
        for agent, months_left in list(self.detained.items()):
            if months_left <= 0:
                del self.detained[agent]
                if not agent.is_dead:
                    self.schedule.agents_t.add(agent)
                continue
            self.detained[agent] = months_left - 1

    def _in_random_order(self, agents):
        """A reproducible random ordering of a set of agents.

        Sets of objects iterate in hash order, which for agents is derived from their
        memory address and therefore differs between processes and between runs. The
        loops that iterate them draw random numbers, so the iteration order fixes the RNG
        sequence. Sorting by unique_id makes it deterministic, and shuffling with the
        model's own seeded generator removes the link to creation order, which follows
        the AREAP the agent was generated in.
        """
        return self.random.sample(sorted(agents, key=lambda a: a.unique_id), len(agents))

    def update_agents(self):
        # Don't remove 'list()' that copies set()
        for agent in self._in_random_order(self.schedule.agents_t):
            
            # Verify if agent died            
            if agent.gender == 'female' and agent.femicide_victim:
                self.incarcerated[agent.spouse] = YEARS_IN_PRISON
                self.schedule.agents_t.discard(agent.spouse)

            if agent.is_dead:
                self.schedule.dead_agents.add(agent)
                self.schedule.agents_t.discard(agent)
                continue

            if not agent.spouse:
                self.single_agents.add(agent)

            # If current agent is single, use the poisson probability to relocate it to a new pair, if possible.
            trigger_per = poisson.cdf(AVERAGE_TIME_RELATIONSHIP, np.random.poisson(AVERAGE_TIME_RELATIONSHIP))
            if np.random.choice([True, False], p=[trigger_per, 1 - trigger_per]):
                if not agent.spouse:
                    x = self.random.randrange(self.width)
                    y = self.random.randrange(self.height)

                    # Picks the FIRST eligible partner, so the order must be random
                    # and reproducible.
                    for agent_partner in self._in_random_order(self.single_agents):
                        if agent_partner.gender != agent.gender:
                            relationship = Relationship(self.next_id(), self, (x, y),
                                                        init_date=date(year=self.data_year, month=1,
                                                                       day=np.random.randint(1, 29)),
                                                        estimated_duration=self.current_date + timedelta(
                                                            365 * self.relationship_duration()))
                            agent.assign_spouse(agent_partner)
                            relationship.add_agent(agent)
                            relationship.add_agent(agent_partner)
                            agent_partner.relationship = agent.relationship = relationship
                            self.grid.place_agent(relationship, (x, y))
                            self.schedule.relationships.add(relationship)
                            self.single_agents.discard(agent)
                            self.single_agents.discard(agent_partner)
                            break

            # Increases age according to individual birth month.
            if agent.birth_month == self.current_date.month:
                agent.age += 1

            # Update spatial violence considering current year.
            agent.community_violence = self.community_violence(agent.address)
            
        for key in self.areap_data.keys():
            self.areap_data[key]['Females'] = 0
            self.areap_data[key]['Denounce'] = 0

    def update_relationships(self):
        for relation in list(self.schedule.relationships):
            # Verify if relationship ended
            if relation.it_ended:
                for member in relation.members.values():
                    member.desassign_spouse()
                self.schedule.relationships.discard(relation)
                continue
            for member in relation.members.values():
                # Verify if a member in a relationship is dead
                if member.is_dead:
                    member.desassign_spouse()
                    self.schedule.relationships.discard(relation)
                    break

    @staticmethod
    def relationship_duration(average_finish=AVERAGE_TIME_RELATIONSHIP):
        # Average rate of relationship breakups over a one-year period.
        # If at least one event occurs, the current relationship has ended.
        events_count = 0
        while not events_count:
            events_count = np.random.poisson(average_finish)
        if events_count:
            # A sample of an exponential representing the duration of the relationship,
            # in years, is taken.
            return np.random.exponential(scale=1 / events_count)
            # Taking the inverse of the average rate as the lambda
            # of the exponential distribution
        return average_finish

    def step(self):
        print('Model current date:', self.current_date)
        # Release whoever finished custody before the agents act this month.
        self.update_detained()
        # The model runs the agent.step() randomly
        self.schedule.step()

        # collect data
        self.datacollector.collect(self)

        # Stop when no Person is left.
        if self.schedule.get_breed_count(Person) == 0:
            self.running = False

        # Update agents
        self.update_agents()

        # Update relationships
        self.update_relationships()

        # Go to the next month
        self.current_date += timedelta(days=31)
        self.current_date -= timedelta(days=self.current_date.day - 1)

    def attr_weight(self, agent, agents_t):
        # Select peers of the same gender and not the agent itself
        same_gender_peers = [agent_w for agent_w in agents_t
                             if agent_w.gender == agent.gender and agent_w != agent]

        agent.weight_vector = dict(
            zip(
                np.random.choice(same_gender_peers, size=Weight.SIZE_INFLUENCE, replace=False),
                np.random.uniform(Weight.MIN, Weight.MAX, size=Weight.SIZE_INFLUENCE)
            )
        )

    # Calculate community violence
    def community_violence(self, address) -> float:
        """Homicides per 100k inhabitants in the agent's municipality, current year.

        homicide.py stores absolute counts, which track municipal population (Spearman .96
        across the Porto Alegre metropolis), so the count is divided by the census
        population of the municipality.
        """
        municipe = int(str(address)[:7])
        homicides = self.violence_indice[municipe][self.current_date.year]
        return 1e5 * homicides / self.mun_population[municipe]

    # Community
    def update_neighborhood_stress(self):
        # attr_weight draws the peer network out of this list with np.random.choice, so
        # the list is put in a reproducible random order (see _in_random_order).
        agents_t = self._in_random_order(self.schedule.agents_t)
        for agent in agents_t:
            same_gender = [a for a in agents_t if a.gender == agent.gender and a != agent]
            if len(same_gender) >= Weight.SIZE_INFLUENCE:
                self.attr_weight(agent, agents_t)

    @staticmethod
    def count_type_citizens(model: Model, condition):
        """
        Helper method to count agents by Type.
        """
        func_set = {
            'denounce': lambda agent: agent.denounce,
            'got_attacked': lambda agent: agent.got_attacked,
            'female': lambda agent: 1 if agent.gender == 'female' and agent.age > 18 else 0
        }
        return np.sum([func_set[condition](agent) if isinstance(agent, Person) else 0
                       for agent in model.schedule.agents_t])

    @staticmethod
    def count_stress(model):
        """
        Return the mean of the context_stress
        """
        return np.mean([agent.context_stress for agent in model.schedule.agents
                        if isinstance(agent, Relationship)]) if model.schedule.agents.__len__() else 0

    def run_model(self, step_count=200):

        if self.verbose:
            print('Initial number of people: ', self.schedule.get_breed_count(Person))

        # Steps are not being set here, but on superclass. Changes should be made in the step function above!
        for _ in range(step_count):
            self.step()

        if self.verbose:
            print('')
            print('Final number People: ', self.schedule.get_breed_count(Person))


def generate_output():
    # Running for all metropolis
    output = pd.DataFrame(columns=['metropolis', 'stress'])
    for metro in metropolis:
        home = Home(metro=metro)
        for _ in range(10):
            home.step()
        model_df = home.datacollector.get_model_vars_dataframe()
        output.loc[output.shape[0]] = [metro, model_df.loc[9, 'Stress']]
    output.to_csv('input/output.csv', sep=';', index=False)


if __name__ == '__main__':
    h = Home()
    for j in range(10):
        h.step()
    m_df = h.datacollector.get_model_vars_dataframe()
