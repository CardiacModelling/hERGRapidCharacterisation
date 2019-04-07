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
import glob

import pints.plot
import plot_hbm_func as plot_func

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param


# About files
qc_dir = '.'
file_list = ['herg25oc1']
temperatures = [25.0]
fit_seed = '542811797'

# Control fitting seed --> OR DONT
# control_seed = np.random.randint(0, 2**30)
control_seed = int(fit_seed)
np.random.seed(control_seed)

saveas = 'figs/pseudoHBM-syn-101/'
n_non_model_param = 1
which_hyper_func = 1
variable_names = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
                  r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']

if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))

try:
    nexp = int(sys.argv[1])
except IndexError:
    nexp = 100


# Load result
file_prefix = './out-mcmc/syn-101-testnexp/syn-101-pseudohbm-lognorm'
simple_chain_final = np.loadtxt('%s-mean-nexp-%s.txt' % (file_prefix, nexp))
with open('%s-cov-nexp-%s.pkl' % (file_prefix, nexp), 'rb') as f:
    simple_cov_final = pickle.load(f)

# Load exp param
param_exp = []
path_to_exp = './out/syn-101-mcmcmean'
for i in range(nexp):
    file_exp = '%s/solution-%s.txt' % (path_to_exp, i)
    p = np.loadtxt(file_exp)
    param_exp.append(p)
param_exp = np.array(param_exp)
exp_transform_parameters = transform_from_model_param(param_exp.T).T
_, n_parameters = exp_transform_parameters.shape

# Covariance matrice
simple_cor_final = np.zeros(simple_cov_final.shape)
for i, s in enumerate(simple_cov_final):
    D = np.sqrt(np.diag(s))
    c = s / D / D[:, None]
    simple_cor_final[i, :, :] = c[:, :]

# this is for synthetic data
path2mean = '../room-temperature-only/kylie-room-temperature/' \
            + 'last-solution_C5.txt'
mean = np.loadtxt(path2mean)
# Change conductance unit nS->pS (new parameter use V, but here mV)
mean[0] = mean[0] * 1e3
mean = transform_from_model_param(mean)
cov_seed = 101
covariance = np.loadtxt('./out/cov-%s.txt' % cov_seed)
stddev = np.sqrt(np.diag(covariance))

ref_hyper = np.array([mean, stddev])
ref_variable_corr = [mean, covariance]

#
# Plot
#

print('Plotting cov...')
fig, axes = plot_func.plot_correlation_and_variable_covariance(
        simple_chain_final[::200],
        simple_cov_final[::200],
        simple_cor_final, corr=True,
        ref_parameters=ref_variable_corr,
        )

axes = plot_func.change_labels_correlation_and_variable_covariance(
        axes,
        variable_names
        )

# Plot individual parameters to lower triangle
for i in range(n_parameters):
    for j in range(n_parameters):
        if i == j:
            # Diagonal: plot histogram
            axes[i, j].hist(exp_transform_parameters[:, i], bins=20,
                            color='#7f7f7f', density=True, alpha=0.7,
                            zorder=-1)

        elif i > j:
            # Lower-left: plot scatters
            px_e = exp_transform_parameters[:, j]
            py_e = exp_transform_parameters[:, i]
            axes[i, j].scatter(px_e, py_e, c='#7f7f7f', alpha=0.7, zorder=-1)

        else:
            continue

if '--show' in sys.argv:
    plt.show()
else:
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0,
                     rect=(0.001, 0.001, 0.97, 0.965))
    plt.savefig('%scov-plot-nexp-%s.png' % (saveas, nexp), bbox_iches='tight',
            pad_inches=0, dpi=100)

# Add boxes for Michael
for i in range(0, n_parameters):
    plot_func.addbox(axes, (i, 0), color='#d9d9d9', alpha=0.75)
for j in range(1, n_parameters):
    plot_func.addbox(axes, (0, j), color='#d9d9d9', alpha=0.75)
for i in range(1, 5):
    for j in range(1, 5):
        plot_func.addbox(axes, (i, j), color='#fdb462', alpha=0.35)
for i in range(5, n_parameters):
    for j in range(5, n_parameters):
        plot_func.addbox(axes, (i, j), color='#ccebc5', alpha=0.75)

if '--show' in sys.argv:
    plt.show()
else:
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0,
                     rect=(0.001, 0.001, 0.97, 0.965))
    plt.savefig('%scov-plot-cbox2-nexp-%s.png' % (saveas, nexp),
            bbox_iches='tight', pad_inches=0, dpi=100)

plt.close('all')
