#!/usr/bin/env python2
#
# coding: utf-8
#
# Plot pseudo-hierarhical Bayesian model simulation (only rely on individual
# MCMC chains)
#

from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
if not '--show' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import scipy

import pints.io
import pints.plot
import plot_hbm_func as plot_func

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

# About files
file_name = 'syn-101'
temperature = 25.0
fakedatanoise = 11.0  # roughly what the recordings are, 10-12 pA
cell = 0

basename = './out-mcmc/syn-101'
load_name = '%s/c' % (basename)  # if any prefix in all files
saveas = './figs/'
n_non_model_param = 1
variable_names = [r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
                  r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', r'$g_{Kr}$', 'noise']

n_chains = 3
n_samples = 200000
warm_up = int(n_samples * 3. / 4.)
thinning = 1

# Load chains
chain_file = '%s%s-chain.csv' % (load_name, cell)
chains_load = pints.io.load_samples(chain_file, n_chains)

# Thinning and discard warm up
chains = []
for i in range(n_chains):
    chains_load[i] = chains_load[i][warm_up:n_samples:thinning, :]
    chains_load[i][:, :-1*n_non_model_param] = transform_to_model_param(
            chains_load[i][:, :-1*n_non_model_param])
    c = np.zeros(chains_load[i].shape)
    c[:, :-2] = chains_load[i][:, 1:-1]
    c[:, -2] = chains_load[i][:, 0]
    c[:, -1] = chains_load[i][:, -1]
    chains.append(c)

# Load CMAES and true parameters
fitted = np.loadtxt('./out/syn-101/solution-%s.txt' % cell)
true = np.loadtxt('./out/syn-101-true/solution-%s.txt' % cell)
true = np.append(true, fakedatanoise)

_, axes = plot_func.trace_fold(chains, ref_parameters=None,
        n_percentiles=None)

axes = plot_func.change_labels_trace_fold(axes, variable_names)

# Plot CMAES and true parameters
for i in range(len(variable_names)):
    ai, aj = int(i / 2), i % 2 * 2
    if i == len(variable_names) - 2:
        axes[ai, aj].axvline(fitted[0], ls='-', c='r')
        axes[ai, aj].axvline(true[0], ls='--', c='k')
        axes[ai, aj + 1].axhline(fitted[0], ls='-', c='r')
        axes[ai, aj + 1].axhline(true[0], ls='--', c='k')
    elif i == len(variable_names) - 1:
        axes[ai, aj].axvline(true[-1], ls='--', c='k')
        axes[ai, aj + 1].axhline(true[-1], ls='--', c='k')
    else:
        axes[ai, aj].axvline(fitted[i + 1], ls='-', c='r')
        axes[ai, aj].axvline(true[i + 1], ls='--', c='k')
        axes[ai, aj + 1].axhline(fitted[i + 1], ls='-', c='r')
        axes[ai, aj + 1].axhline(true[i + 1], ls='--', c='k')

plt.tight_layout(pad=0.1, w_pad=0.005, h_pad=0.005,
                 rect=(0.001, 0.001, 0.999, 0.999))
plt.savefig('%smcmc-trace-%s.png' % (saveas, cell), bbox_iches='tight', dpi=300)
plt.close('all')

