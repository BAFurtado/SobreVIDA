import argparse
import sys
import os
sys.path.append(os.path.abspath('violence'))
import generalization_aps

parser = argparse.ArgumentParser(prog='SobreVIDA', description='A simulation model of domestic violence')
parser.add_argument('-i', '--iterations',
                    dest='iterations',
                    help='Number of models that will be created',
                    default=5)
parser.add_argument('-s', '--steps',
                    dest='steps',
                    help='Number of model steps',
                    default=144)

parser.add_argument('-m', '--metro',
                    type=str,
                    default="PORTO ALEGRE",
                    help='Metropolis to run the model\nPossible values: ("BELO HORIZONTE", "BRASILIA", "CAMPO GRANDE", "FORTALEZA", "GOIANIA", "PORTO ALEGRE", "RECIFE", "RIO DE JANEIRO", "SALVADOR", "SAO PAULO", "VITORIA")')

arguments = dict(parser.parse_args()._get_kwargs())

generalization_aps.main(metro=arguments['metro'], iterates=int(arguments['iterations']), steps=int(arguments['steps']))
    