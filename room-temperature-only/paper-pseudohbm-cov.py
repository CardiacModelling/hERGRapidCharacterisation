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
plot_voltage_artefact = True

# Control fitting seed --> OR DONT
# control_seed = np.random.randint(0, 2**30)
control_seed = int(fit_seed)
np.random.seed(control_seed)

saveas = 'figs/paper/'
saveaslr = 'figs/paper-low-res/'
n_non_model_param = 1
which_hyper_func = 1
variable_names = [r'$\ln(g_{Kr})$', r'$\ln(p_1)$', r'$\ln(p_2)$',
        r'$\ln(p_3)$', r'$\ln(p_4)$', r'$\ln(p_5)$', r'$\ln(p_6)$',
        r'$\ln(p_7)$', r'$\ln(p_8)$', 'noise']

if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))

if not os.path.isdir(os.path.dirname(saveaslr)):
    os.makedirs(os.path.dirname(saveaslr))


# Load result
file_prefix = './out-mcmc/herg25oc1-pseudohbm-lognorm'
simple_chain_final = np.loadtxt('%s-mean.txt' % file_prefix)
with open('%s-cov.pkl' % file_prefix, 'rb') as f:
    simple_cov_final = pickle.load(f)

# Load exp param
param_exp = []
path_to_exp = './out/herg25oc1-mcmcmean'
files_exp = glob.glob(path_to_exp + '/*.txt')
for file_exp in files_exp:
    p = np.loadtxt(file_exp)
    param_exp.append(p)
param_exp = np.array(param_exp)
exp_transform_parameters = transform_from_model_param(param_exp.T).T
nexp, n_parameters = exp_transform_parameters.shape

# Covariance matrice
simple_cor_final = np.zeros(simple_cov_final.shape)
for i, s in enumerate(simple_cov_final):
    D = np.sqrt(np.diag(s))
    c = s / D / D[:, None]
    simple_cor_final[i, :, :] = c[:, :]


#
# Plot
#

print('Plotting cov...')
fig, axes = plot_func.plot_correlation_and_variable_covariance(
        simple_chain_final[::200],
        simple_cov_final[::200],
        simple_cor_final, corr=True,
        # ref_parameters=[simple_transform_mean, simple_transform_cov],
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

# Voltage artefact
if plot_voltage_artefact:

    # Load syn param from voltage-artefact
    param_syn = []
    path_to_syn = './out/herg25oc1-fakedata-voltageoffset'
    files_syn = glob.glob(path_to_syn + '/*.txt')
    for file_syn in files_syn:
        p = np.loadtxt(file_syn)
        param_syn.append(p)
    param_syn = np.array(param_syn)
    n_param = param_syn.shape[1]
    param_syn = np.log(param_syn)

    # Plot
    for i in range(n_param):
        for j in range(n_param):
            if i > j:
                # Lower-left: plot scatters
                px_s = param_syn[:, j]
                py_s = param_syn[:, i]
                axes[i, j].scatter(px_s, py_s, c='#d62728',
                        label='Syn. voltage offset')


if '--show' in sys.argv:
    plt.show()
else:
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0,
                     rect=(0.001, 0.001, 0.97, 0.965))
    plt.savefig('%scov-plot.png'%saveaslr, bbox_iches='tight', pad_inches=0,
                dpi=100)
    plt.savefig('%scov-plot.png'%saveas, bbox_iches='tight', pad_inches=0,
                dpi=300)
    # plt.savefig('%scov-plot.pdf'%saveas, format='pdf', bbox_inches='tight')

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
    plt.savefig('%scov-plot-cbox2.png'%saveaslr, bbox_iches='tight',
                pad_inches=0, dpi=100)
    plt.savefig('%scov-plot-cbox2.png'%saveas, bbox_iches='tight',
                pad_inches=0, dpi=300)
    plt.savefig('%scov-plot-cbox2.pdf'%saveas, bbox_iches='tight',
                format='pdf')

plt.close('all')
