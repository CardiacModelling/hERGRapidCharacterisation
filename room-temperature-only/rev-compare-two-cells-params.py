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
from plot_hbm_func import plot_cov_ellipse

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

def normal1D(x, mu, sigma):
    # normal distribution
    output = 1/(sigma * np.sqrt(2 * np.pi)) * \
             np.exp( - (x - mu)**2 / (2 * sigma**2) )
    return output


# About files
qc_dir = '.'
file_list = ['herg25oc1']
temperatures = [25.0]
fit_seed = '542811797'

# Control fitting seed --> OR DONT
# control_seed = np.random.randint(0, 2**30)
control_seed = int(fit_seed)
print('Using seed: ', control_seed)
np.random.seed(control_seed)

basename = 'out-mcmc/mcmc-herg25oc1'
load_name = '%s/' % (basename)  # if any prefix in all files
saveas = 'figs/paper/'
n_non_model_param = 1
which_hyper_func = 1
variable_names = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
                  r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']

if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))

cells = ['B20', 'C17']


# Load result
file_prefix = './out-mcmc/herg25oc1-pseudohbm-lognorm'
simple_chain_final = np.loadtxt('%s-mean.txt' % file_prefix)
with open('%s-cov.pkl' % file_prefix, 'rb') as f:
    simple_cov_final = pickle.load(f)

# Covariance matrice
simple_cor_final = np.zeros(simple_cov_final.shape)
for i, s in enumerate(simple_cov_final):
    D = np.sqrt(np.diag(s))
    c = s / D / D[:, None]
    simple_cor_final[i, :, :] = c[:, :]

sample_mean = simple_chain_final[::1000]
sample_cov = simple_cov_final[::1000]

#
# Simple mean and cov values from individual MCMC results
#
exp_chains = []
print('Loading results...')
for i_file, (file_name, temperature) in enumerate(zip(file_list,
                                                      temperatures)):
    # Load QC
    selectedwell = cells

    for i_cell, cell in enumerate(selectedwell):
        print('Loading %s' % cell)

        # Load fitting result
        chain_file = '%s%s-chain.csv' % (load_name, cell)
        exp_chains.append(pints.io.load_samples(chain_file, 1)[0])

n_samples = len(exp_chains[0])
warm_up = int(n_samples * 3. / 4.)
thinning = 1
# Thinning and discard warm up
exp_chains_final = np.asarray(exp_chains)[:, warm_up::thinning, :]
exp_transform_parameters = np.mean(exp_chains_final, axis=1)
# Remove non model parameters
exp_transform_parameters = exp_transform_parameters[:, :-1*n_non_model_param]

nexp, n_parameters = exp_transform_parameters.shape

simple_transform_mean = np.mean(exp_transform_parameters, axis=0)
simple_transform_cov = np.cov(np.asarray(exp_transform_parameters).T)

thinning2 = 2

fig, axes = plt.subplots(n_parameters, n_parameters,
        figsize=(3 * n_parameters, 3 * n_parameters))
for i in range(n_parameters):
    i_min = np.min(exp_chains_final[:, :, i])
    i_max = np.max(exp_chains_final[:, :, i])
    i_range = i_max - i_min
    i_min -= 0.1 * i_range
    i_max += 0.1 * i_range
    for j in range(n_parameters):
        j_min = np.min(exp_chains_final[:, :, j])
        j_max = np.max(exp_chains_final[:, :, j])
        j_range = j_max - j_min
        j_min -= 0.1 * j_range
        j_max += 0.1 * j_range

        ax = axes[i, j]

        if i == j:
            # Diagonal: Plot a 1d histogram
            for i_cell, chain_i in enumerate(exp_chains_final):
                ax.hist(chain_i[::thinning2 , i], bins=12, color='C%s' % (i_cell + 4))
                ax.axvline(exp_transform_parameters[i_cell, i], c='C%s' % (i_cell + 4), label='cell %s mean' % cells[i_cell], ls='--', lw=1.5)

            # 2 sigma covers up 95.5%
            xmin = np.min(sample_mean[:, i]) \
                   - 2.5 * np.max(np.sqrt(sample_cov[:, i, i]))
            xmax = np.max(sample_mean[:, i]) \
                   + 2.5 * np.max(np.sqrt(sample_cov[:, i, i]))
            xx = np.linspace(xmin, xmax, 100)
            ax.set_xlim(xmin, xmax)
            
            ax_marginal = ax.twinx()
            for m, s in zip(sample_mean[:, i], sample_cov[:, i, i]):
                ax_marginal.plot(xx, normal1D(xx, m, np.sqrt(s)), c='C2', alpha=0.1)
            if i == 0:
                ax_marginal.set_ylabel('Probability density', 
                                       color='C2',
                                       fontsize=16)
        elif i < j:
            # Upper right: No plot
            ax.axis('off')
        else:
            # Lower left: Plot a 2d histogram
            for i_cell, chain_i in enumerate(exp_chains_final):
                ax.scatter(chain_i[::thinning2, j], chain_i[::thinning2, i], color='C%s' % (i_cell + 4), alpha=0.01, s=10, linewidths=0)
                ax.axhline(exp_transform_parameters[i_cell, i], c='C%s' % (i_cell + 4), label='cell %s mean' % cells[i_cell], ls='--', lw=1.5)
                ax.axvline(exp_transform_parameters[i_cell, j], c='C%s' % (i_cell + 4), ls='--', lw=1.5)

            # 2 sigma covers up 95.5%
            xmin = np.min(sample_mean[:, j]) \
                   - 2.5 * np.max(np.sqrt(sample_cov[:, j, j]))
            xmax = np.max(sample_mean[:, j]) \
                   + 2.5 * np.max(np.sqrt(sample_cov[:, j, j]))
            ymin = np.min(sample_mean[:, i]) \
                   - 2.5 * np.max(np.sqrt(sample_cov[:, i, i]))
            ymax = np.max(sample_mean[:, i]) \
                   + 2.5 * np.max(np.sqrt(sample_cov[:, i, i]))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            
            for m, s in zip(sample_mean, sample_cov):
                # for xj, yi
                mu = np.array([m[j], m[i]])
                cov = np.array([[ s[j, j], s[j, i] ], 
                                [ s[i, j], s[i, i] ]])
                xx, yy = plot_cov_ellipse(mu, cov)
                ax.plot(xx, yy, c='C0', alpha=0.1)  #003366

        # Customise tick labels
        if j > 0:
            # Only show y tick labels for the first column
            ax.set_yticklabels([])
        if i < n_parameters - 1:
            # Only show x tick labels for the last row
            ax.set_xticklabels([])

    # Add labels for subplots at the edges
    if i > 0:
        axes[i, 0].set_ylabel(variable_names[i], fontsize=32)
    else:
        axes[i, 0].set_ylabel('Frequency', fontsize=32)
    axes[i, 0].tick_params('y', labelsize=26)

    axes[-1, i].set_xlabel(variable_names[i], fontsize=32)
    axes[-1, i].tick_params('x', labelsize=26, rotation=30)

axes[1, 0].legend(fontsize=32, loc="lower left", bbox_to_anchor=(1.3, 1.15),
                  bbox_transform=axes[1, 0].transAxes)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# Add boxes for Michael
for i in range(1, n_parameters):
    plot_func.addbox(axes, (i, 0), color='#d9d9d9', alpha=0.75)
for i in range(1, 5):
    for j in range(1, 5):
        if i > j:
            plot_func.addbox(axes, (i, j), color='#fdb462', alpha=0.35)
# Maybe 3 colours
for i in range(5, n_parameters):
    for j in range(5, n_parameters):
        if i > j:
            plot_func.addbox(axes, (i, j), color='#ccebc5', alpha=0.75)

plt.savefig('%srev-compare-%s_%s-v-%s-param.png' % (saveas, file_name, \
            cells[0], cells[1]), bbox_inch='tight', pad_inches=0, dpi=100)

plt.close()
