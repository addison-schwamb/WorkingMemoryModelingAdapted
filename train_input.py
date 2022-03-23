"""
main file to train/test RNN and perform posthoc tests on WashU cluster

written in Python 3.8.3
@ Elham
"""

#input string format: "{\"g\":<val>,\"pg\":<val>,\"fb_var\":<val>,\"input_var\":<val>,\"n_train\":<val>,\"encoding\":<val>,\"seed\":<val>,\"init_dist\":\"<dist>\"}"

import argparse
import json
import sys
import os
from SPM_task import *
from train_force import *
from posthoc_tests import *
from damage_network import *
dir = ''

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str)
args = parser.parse_args()
kwargs= json.loads(args.d)

def set_all_parameters( g, pg, fb_var, input_var,  n_train, encoding, seed, init_dist, train_input, pct_rmv, inhibitory = True, activation='tanh', isFORCE = False):
    params = dict()

    net_params = dict()
    net_params['d_input'] = 2
    net_params['d_output'] = 1
    net_params['tau'] = 1
    net_params['dt'] = 0.1
    net_params['g'] = g
    net_params['pg'] = pg
    net_params['N'] = 1000
    net_params['fb_var'] = fb_var
    net_params['input_var'] = input_var
    params['network'] = net_params

    task_params = dict()
    t_intervals = dict()
    t_intervals['fixate_on'], t_intervals['fixate_off'] = 0, 0
    t_intervals['cue_on'], t_intervals['cue_off'] = 0, 0
    t_intervals['stim_on'], t_intervals['stim_off'] = 10, 5
    t_intervals['delay_task'] = 0
    t_intervals['response'] = 5
    task_params['time_intervals'] = t_intervals
    task_params['t_trial'] = sum(t_intervals.values()) + t_intervals['stim_on'] + t_intervals['stim_off']
    task_params['output_encoding'] = encoding  # how 0, 1, 2 are encoded
    task_params['keep_perms'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    task_params['n_digits'] = 9
    params['task'] = task_params

    train_params = dict()
    train_params['update_step'] = 2  # update steps of FORCE
    train_params['alpha_w'] = 1.
    train_params['alpha_d'] = 1.
    train_params['n_train'] = n_train  # training steps
    train_params['n_train_ext'] = 0
    train_params['n_test'] = 20   # test steps
    train_params['init_dist'] = init_dist
    train_params['activation'] = activation
    train_params['FORCE'] = isFORCE
    train_params['epsilon'] = [0.005, 0.01, 0.05, 0.1]
    params['train'] = train_params
    
    damage_params = dict()
    damage_params['pct_rmv'] = pct_rmv
    damage_params['inhibitory'] = inhibitory
    params['damage'] = damage_params

    other_params = dict()
    other_params['name'] = str(n_train) + '_training_steps'
    print('name is = ',other_params['name']  )
    #str(task_params['output_encoding']) + '_g' + str(net_params['g']) + '_' +
    #  str(train_params['n_train']+ train_params['n_train_ext'])+ 'Gauss_S' + 'FORCE'
    other_params['n_plot'] = 10
    other_params['seed'] = seed  #default is 0
    other_params['train_input'] = train_input
    params['msc'] = other_params

    return params


def get_digits_reps():
    with open('allDigCNNMNIST', 'rb') as f:
        z_mean, z_log_var, z_sample = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)

    y_test, x_test = np.array(y_test), x_test.reshape([x_test.shape[0], 28, 28])

    return y_test, z_sample


params = set_all_parameters(**kwargs)
labels, digits_rep = get_digits_reps()
task_prs = params['task']
train_prs = params['train']
net_prs = params['network']
damage_prs = params['damage']
msc_prs = params['msc']
task = sum_task_experiment(task_prs['n_digits'], train_prs['n_train'], train_prs['n_train_ext'], train_prs['n_test'], task_prs['time_intervals'],
                           net_prs['dt'], task_prs['output_encoding'], task_prs['keep_perms'] , digits_rep, labels, msc_prs['seed'])


exp_mat, target_mat, dummy_mat, input_digits, output_digits = task.experiment()

if not msc_prs['train_input']:
    print('Training single network with FORCE Reinforce\n')
	single_net = Network({'network':net_prs,'train':train_prs}, msc_prs['seed'])
    x_train, params = train(params, exp_mat, target_mat, dummy_mat, input_digits, dist=train_prs['init_dist'])
    x_ICs, r_ICs, internal_x = test(params, x_train, exp_mat, target_mat, dummy_mat, input_digits)
    msc_prs['name'] = 'single_network_' + msc_prs['name']
    params['msc'] = msc_prs

elif msc_prs['train_input']:
    int_params, int_x_train = read_data_variable_size('intact_net_500', prefix='train', dir=dir)
    print('Training input to damaged network with FORCE Reinforce\n')
    #x_train, params = train_input()
    #x_ICs, r_ICs, internal_x = test_input()

if damage_prs['pct_rmv'] > 0:
    model_prs = params['model']
    damage_prs = params['damage']
    JT = model_prs['JT']
    JT = remove_neurons(JT,damage_prs['pct_rmv'],damage_prs['inhibitory'])
    model_prs['JT'] = JT
    params['model'] = model_prs
    msc_prs['name'] = msc_prs['name'] + '_' + str(damage_prs['pct_rmv']*100) + '%_'
    if damage_prs['inhibitory']: msc_prs['name'] = msc_prs['name'] + 'inhibitory_removed'
    else: msc_prs['name'] = msc_prs['name'] = msc_prs['name'] + 'excitatory_removed'
    params['msc'] = msc_prs
    

save_data_variable_size(params, x_train, name=msc_prs['name'], prefix='train', dir=dir)


error_ratio = error_rate(params, x_ICs, digits_rep, labels)






















