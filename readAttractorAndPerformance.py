from train_force import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str)
args = parser.parse_args()
print(args.d)
dmg_params, dmg_x = read_data_variable_size(args.d + '_500_training_steps_1.0%_removed',prefix='train',dir='')
print('Attractor: ', dmg_params['attractor'])
print('Performance: ', dmg_params['pct_correct'])