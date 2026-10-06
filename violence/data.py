import argparse
import pandas as pd
import model
from agents import Woman

# General variables 
mun_codes = pd.read_excel('violence/input/mun_cods.ods')

# Function to return a formatted dataframe with final results
def fdataframe(data_dict, year, label='Município'):
    return pd.DataFrame([
        (year, key,
         data_dict[key]['Attacked'],
         data_dict[key]['Denounce'],
         data_dict[key]['Females'],
         data_dict[key]['Femicide'],
         data_dict[key]['Number of attacks'],
         data_dict[key]['Break'],
         data_dict[key]['Remain'])
        for key in data_dict
    ],
        columns=['Year', label, 'Attacked', 'Denounce', 'Females',
                 'Femicide', 'Number of attacks', 'Break', 'Remain']
    )

def is_adult_woman(agent):
    """Denominator of the rates in _compute_rates: adult women, attacked or not.

    Children are instantiated as Men/Woman too, so gender alone would put girls in the
    denominator of rates about intimate-partner violence.
    """
    return isinstance(agent, Woman) and agent.age > 18

# Function to return a formatted dataframe events count
def edataframe(agent):
    return {'Attacked': 1 if agent.got_attacked else 0,
            'Denounce': 1 if agent.denounce else 0,
            'Females': 1 if is_adult_woman(agent) else 0,
            'Femicide': 1 if agent.femicide_victim else 0,
            'Number of attacks': agent.got_attacked,
            'Break': agent.break_times,
            'Remain': agent.remain_times}

# Function to fill final data columns
def fill_data(agent, data, idx=None, mun=None):
    indx = mun.values[0] if idx else agent.address
    # Outside the got_attacked branch: the column is the population at risk, not a count
    # of events.
    if is_adult_woman(agent):
        data[indx]['Females'] += 1
    if agent.got_attacked:
        data[indx]['Attacked'] += 1
        data[indx]['Number of attacks'] += agent.got_attacked
        if agent.denounce:
            data[indx]['Denounce'] += 1
        if agent.femicide_victim:
            data[indx]['Femicide'] += 1
        if agent.break_times:
            data[indx]['Break'] += agent.break_times
        if agent.remain_times:
            data[indx]['Remain'] += agent.remain_times
