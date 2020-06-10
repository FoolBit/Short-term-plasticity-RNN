# 用来把训好的模型得到的数据保存成.mat
import scipy.io as sio
import numpy as np
from parameters import *
from sklearn import svm
import time
import pickle
import stimulus
import os
import copy
import matplotlib.pyplot as plt
from itertools import product
from scipy import signal
from scipy.optimize import curve_fit


def run_model(x, h_init_org, syn_x_init_org, syn_u_init_org, weights, suppress_activity = None):

    """ Simulate the RNN """

    # copying data to ensure nothing gets changed upstream
    h_init = copy.copy(h_init_org)
    syn_x_init = copy.copy(syn_x_init_org)
    syn_u_init = copy.copy(syn_u_init_org)

    network_weights = {k:v for k,v in weights.items()}

    if par['EI']:
        network_weights['w_rnn'] = par['EI_matrix'] @ np.maximum(0,network_weights['w_rnn'])
        network_weights['w_in'] = np.maximum(0,network_weights['w_in'])
        network_weights['w_out'] = np.maximum(0,network_weights['w_out'])

    h, syn_x, syn_u = \
        rnn_cell_loop(x, h_init, syn_x_init, syn_u_init, network_weights, suppress_activity)

    # Network output
    y = [h0 @ network_weights['w_out'] + weights['b_out'] for h0 in h]

    syn_x   = np.stack(syn_x)
    syn_u   = np.stack(syn_u)
    h       = np.stack(h)
    y       = np.stack(y)

    return y, h, syn_x, syn_u


def rnn_cell_loop(x_unstacked, h, syn_x, syn_u, weights, suppress_activity):

    h_hist = []
    syn_x_hist = []
    syn_u_hist = []

    # Loop through the neural inputs to the RNN
    for t, rnn_input in enumerate(x_unstacked):

        if suppress_activity is not None:
            h, syn_x, syn_u = rnn_cell(np.squeeze(rnn_input), h, syn_x, syn_u, weights, suppress_activity[t])
        else:
            h, syn_x, syn_u = rnn_cell(np.squeeze(rnn_input), h, syn_x, syn_u, weights, 1)
        print('h ', h.shape, ' syn_x ', syn_x.shape, ' syn_u ', syn_u.shape)

        h_hist.append(h)
        syn_x_hist.append(syn_x)
        syn_u_hist.append(syn_u)

    return h_hist, syn_x_hist, syn_u_hist

def rnn_cell(rnn_input, h, syn_x, syn_u, weights, suppress_activity):

    # Update the synaptic plasticity paramaters
    if par['synapse_config'] is not None:
        # implement both synaptic short term facilitation and depression
        syn_x_new = syn_x + (par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h)*par['dynamic_synapse']
        syn_u_new = syn_u + (par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h)*par['dynamic_synapse']
        syn_x_new = np.minimum(1, np.maximum(0, syn_x_new))
        syn_u_new = np.minimum(1, np.maximum(0, syn_u_new))
        h_post = syn_u_new*syn_x_new*h

    else:
        # no synaptic plasticity
        h_post = h


    # Update the hidden state
    h = np.maximum(0, h*(1-par['alpha_neuron'])
                   + par['alpha_neuron']*(rnn_input @ weights['w_in']
                   + h_post @ weights['w_rnn'] + weights['b_rnn'])
                   + np.random.normal(0, par['noise_rnn'],size = h.shape))

    h *= suppress_activity

    if par['synapse_config'] is None:
        syn_x_new = np.ones_like(h)
        syn_u_new = np.ones_like(h)

    return h, syn_x_new, syn_u_new


data_dir = './savedir/'
filename = data_dir + 'WMnew.pkl' # 保存的模型文件
results = pickle.load(open(filename, 'rb'))

update_parameters(results['parameters'])
stim = stimulus.Stimulus()

# generate trials with match probability at 50%
trial_info = stim.generate_trial(test_mode = True)
input_data = np.squeeze(np.split(trial_info['neural_input'], par['num_time_steps'], axis=0))

h_init = results['weights']['h']

y, h, syn_x, syn_u = run_model(input_data, h_init, \
    results['parameters']['syn_x_init'], results['parameters']['syn_u_init'], results['weights'])

syn_efficacy = syn_x*syn_u

'''
# generate trials with random sample and test stimuli, used for decoding
trial_info_decode = stim.generate_trial(test_mode = True)
input_data = np.squeeze(np.split(trial_info_decode['neural_input'], par['num_time_steps'], axis=0))
_, h_decode, syn_x_decode, syn_u_decode = run_model(input_data, h_init, \
    results['parameters']['syn_x_init'], results['parameters']['syn_u_init'], results['weights'])
'''

sio.savemat('WMnew.mat',
            {'y':y,
             'h':h,
             'syn_x':syn_x,
             'syn_u':syn_u,
             'syn_efficacy':syn_efficacy,
             'stim1':trial_info['sample'][:,0],
             'stim2':trial_info['sample'][:,1],
             'cue':trial_info['rule'],
             })