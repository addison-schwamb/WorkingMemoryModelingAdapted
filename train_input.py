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
from Network import *
from train_force import *
from posthoc_tests import *
from damage_network import *


def set_all_parameters( g, pg, fb_var, input_var,  n_train, encoding, seed, init_dist, input, pct_rmv, net_id = None, activation='tanh', isFORCE = False):
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
    net_params['pct_rmv'] = pct_rmv
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
    task_params['counter'] = 0
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
    

    other_params = dict()
    other_params['name'] = str(g) + '_' + str(pg) + '_' + str(fb_var) + '_' + str(seed) + '_' + str(n_train) + '_training_steps'
    print('name is = ',other_params['name']  )
    #str(task_params['output_encoding']) + '_g' + str(net_params['g']) + '_' +
    #  str(train_params['n_train']+ train_params['n_train_ext'])+ 'Gauss_S' + 'FORCE'
    other_params['n_plot'] = 10
    other_params['seed'] = seed  #default is 0
    other_params['input'] = input
    other_params['net_id'] = net_id
    params['msc'] = other_params

    return params


def get_digits_reps():
    with open('allDigCNNMNIST', 'rb') as f:
        z_mean, z_log_var, z_sample = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)

    y_test, x_test = np.array(y_test), x_test.reshape([x_test.shape[0], 28, 28])

    return y_test, z_sample

def main(d):
    dir = ''

    #parser = argparse.ArgumentParser()
    #parser.add_argument('-d', type=str)
    #args = parser.parse_args()
    #kwargs= json.loads(args.d)
    kwargs = json.loads(d)

    params = set_all_parameters(**kwargs)
    labels, digits_rep = get_digits_reps()
    task_prs = params['task']
    train_prs = params['train']
    net_prs = params['network']
    msc_prs = params['msc']
    task = sum_task_experiment(task_prs['n_digits'], train_prs['n_train'], train_prs['n_train_ext'], train_prs['n_test'], task_prs['time_intervals'],
                               net_prs['dt'], task_prs['output_encoding'], task_prs['keep_perms'] , digits_rep, labels, msc_prs['seed'])


    exp_mat, target_mat, dummy_mat, input_digits, output_digits = task.experiment()

    if not msc_prs['input']:
        print('Training single network with FORCE Reinforce\n')
        net_input_params = {**net_prs, **train_prs}
        single_net = Network(net_input_params, msc_prs['seed'])
        single_net, task_prs = train(single_net, task_prs, exp_mat, target_mat, dummy_mat, input_digits, dist=train_prs['init_dist'])
        single_net, pre_pct_correct, x_ICs, r_ICs, internal_x = test(single_net, task_prs, exp_mat, target_mat, dummy_mat, input_digits)
        single_net.params['pct_correct'] = []
        single_net.params['pct_correct'].append(pre_pct_correct)
        ph_params = set_posthoc_params(x_ICs, r_ICs)
    
        trajectories, unique_z_mean, unique_zd_mean, pre_dmg_att = attractor_type(single_net, task_prs, ph_params, digits_rep, labels, 0)
        single_net.params['attractor'] = []
        single_net.params['attractor'].append(pre_dmg_att)
        print(pre_dmg_att)
        single_net.save_network(name=msc_prs['name'], prefix='train', dir=dir)
        #print('Percent Correct: ', str(pre_pct_correct*100), '%')

    elif msc_prs['input']:
        dmg_params, dmg_x = read_data_variable_size(msc_prs['net_id']+'_500_training_steps_1.0%_removed', prefix='train', dir=dir)
        net_input_params = {**net_prs, **train_prs}
        dmg_net = Network(net_input_params, msc_prs['seed'])
        dmg_net.params = dmg_params
        dmg_net.x = dmg_x
        net_input_params['d_input'] += 1
        ext_net = Network(net_input_params, msc_prs['seed'])
        
        print('training helper network to input to damaged network')
        ext_net, task_prs = train_ext_net(ext_net, dmg_net, task_prs, exp_mat, target_mat, dummy_mat, input_digits, dist=train_prs['init_dist'])
        ext_net, input_pct_correct, x_ICs, r_ICs, external_x = test_ext_net(ext_net, dmg_net, task_prs, exp_mat, target_mat, dummy_mat, input_digits)
        print('Percent Correct: ', str(input_pct_correct*100), '%')
        msc_prs['name'] = msc_prs['name'] + '_damaged_' + msc_prs['net_id']
        params['msc'] = msc_prs
        
        ext_net.params['pct_correct'] = []
        ext_net.params['pct_correct'].append(input_pct_correct)
        #dmg_net, pct_correct, x_ICs, r_ICs, internal_x = test(dmg_net, task_prs, exp_mat, target_mat, dummy_mat, input_digits)
        ph_params = set_posthoc_params(x_ICs, r_ICs)
        
        trajectories, unique_z_mean, unique_zd_mean, helper_att = attractor_type(ext_net, task_prs, ph_params, digits_rep, labels, 0)
        trajectories, unique_z_mean, unique_zd_mean, input_dmg_att = attractor_type(ext_net, task_prs, ph_params, digits_rep, labels, 1, dmg_net)
        ext_net.params['attractor'] = []
        ext_net.params['attractor'].append(helper_att)
        ext_net.params['attractor'].append(input_dmg_att)
        print(helper_att)
        print(input_dmg_att)
        ext_net.save_network(name=msc_prs['name'], prefix='fb_helper', dir=dir)

    


    if net_prs['pct_rmv'] > 0:
        single_net.remove_neurons()
        msc_prs['name'] = msc_prs['name'] + '_' + str(net_prs['pct_rmv']*100) + '%_removed'
        params['msc'] = msc_prs

        single_net, post_pct_correct, x_ICs, r_ICs, internal_x = test(single_net, task_prs, exp_mat, target_mat, dummy_mat, input_digits)
        single_net.params['pct_correct'].append(post_pct_correct)
        
        ph_params = set_posthoc_params(x_ICs, r_ICs)

        trajectories, unique_z_mean, unique_zd_mean, post_dmg_att = attractor_type(single_net, task_prs, ph_params, digits_rep, labels, 0)
        single_net.params['attractor'].append(post_dmg_att)
        print(post_dmg_att)
        single_net.save_network(name=msc_prs['name'], prefix='train', dir=dir)
    
    

    
    if msc_prs['input']:
        return input_dmg_att, input_pct_correct
    else:
        return pre_dmg_att, post_dmg_att, pre_pct_correct, post_pct_correct

if __name__ == "__main__":
    print(sys.argv[2])
    main(sys.argv[2])