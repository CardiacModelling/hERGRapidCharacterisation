#!/usr/bin/env python2
#
# coding: utf-8
#
# Plot results from Bayesian Hierarchical Model simulation
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

import pints.io
import pints.plot
import plot_hbm_func as plot_func

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param


# About files
basename = 'out-mcmc/test-HBM'
load_name = '%s/' % (basename)  # if any prefix in all files
saveas = 'figs/test-HBM/'
n_exp = 20
thinning = 1  # Usually did thinning already...
which_hyper_func = 1
variable_names = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
                  r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']

if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))


# About model


# Load results
print('Loading results...')
chain = pints.io.load_samples('%schain.csv' % load_name)
exp_chains = pints.io.load_samples('%sexp_chain.csv' % load_name, n_exp)
with open('%scov_chain.pkl' % load_name, 'rb') as f:
    cov_chain = pickle.load(f)


# Process results
n_samples = 35000 # len(chain)
# drop first half of chain and thinning
chain_final = chain[(n_samples // 3):n_samples:thinning, :]
exp_chains_final = np.array(exp_chains)[:,
                                        (n_samples // 3):n_samples:thinning,
                                        :]
cov_chain_final = cov_chain[(n_samples // 3):n_samples:thinning, :, :]
assert(len(chain_final) == len(cov_chain_final))

hyper_mean = np.mean(chain_final, axis=0)
hyper_cov = np.mean(cov_chain_final, axis=0)


# Hyperparameters mean values ($\mu_i$)
chain_param = chain_final[:, :]

# Hyperparameters standard deviation values ($s_i$)
chain_stddev = np.sqrt(cov_chain_final.diagonal(0, 1, 2))
assert(len(chain_param) == len(chain_stddev))

# Individual experiments ($p_{i,j}$)
exp_chains_param = []
for exp_chain in exp_chains_final:
    exp_chain_tmp = np.copy(exp_chain)
    exp_chain_tmp[:, :-1] = transform_to_model_param(exp_chain[:, :-1])
    exp_chains_param.append(exp_chain_tmp)


# Some quick plots
pints.plot.trace(exp_chains_param, n_percentiles=99.9)
plt.savefig('%stmp-expchains.png' % saveas, bbox_iches='tight')
plt.close('all')
pints.plot.trace([chain_param], n_percentiles=99.9)
plt.savefig('%stmp-hypermean.png' % saveas, bbox_iches='tight')
plt.close('all')


# individual exp: trace
n_percentiles = 99.9

# plot inferred hyperparameters on top
hyper_func_opt = ['normal', 'log']
hyper_func = hyper_func_opt[which_hyper_func]

# plot inferred hyperparameters on top but no noise
# reference of covariance matrix
if False:
    # this is for synthetic data
    mean = list(m.unit_hypercube_to_param_intervals(model.original()))
    stddev = [u / 5.0 for u in mean]
    covariance = np.outer(np.array(stddev), np.array(stddev))
    import sklearn.datasets
    # rho = sklearn.datasets.make_spd_matrix(len(stddev), random_state=1)
    rho = sklearn.datasets.make_sparse_spd_matrix(len(stddev), alpha=0.5,
            norm_diag=True, random_state=1)
    covariance = rho * covariance
    ref_hyper = np.array([mean, stddev])
    ref_variable_corr = [mean, covariance]
else:
    # this is good for real data
    rho = covariance = np.zeros((9,9))
    ref_hyper = None
    ref_variable_corr = None

print('Plotting fig4-5...')
_, axes = plot_func.plot_posterior_predictive_distribution(
        [chain_param, chain_stddev],
        exp_chains_param,
        hyper_func=hyper_func,
        fold=True, ref_hyper=ref_hyper,
        n_percentiles=n_percentiles,
        normalise=True,
        )
axes = plot_func.change_labels_histogram_fold(axes, variable_names)
xlim = [[(1e4, 6.5e4), (1e-8, 0.3), (40, 150)],
        [(1e-8, 0.05), (35, 65), (60, 230)],
        [(5, 30), (5, 18), (20, 35)]]
for i in range(3):
    for j in range(3):
        axes[i][j].set_xlim(xlim[i][j])
# axes[0][0].legend([])  # suppress legend
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('%shbm-plot.png' % saveas, bbox_iches='tight')
    # plt.savefig('%s-fig4-5.pdf'%saveas, format='pdf', bbox_inches='tight')
plt.close('all')


# Covariance matrice
