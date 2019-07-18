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
saveas = 'figs/pseudoHBM-manualv2selected-herg25oc1/'
saveaslr = 'figs/pseudoHBM-manualv2selected-herg25oc1-low-res/'
n_non_model_param = 1
which_hyper_func = 1
variable_names = [r'$g_{Kr}$ $[pS]$', r'$p_1$ $[s^{-1}]$', r'$p_2$ $[V^{-1}]$',
        r'$p_3$ $[s^{-1}]$', r'$p_4$ $[V^{-1}]$', r'$p_5$ $[s^{-1}]$',
        r'$p_6$ $[V^{-1}]$', r'$p_7$ $[s^{-1}]$', r'$p_8$ $[V^{-1}]$', 'noise']

if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))

if not os.path.isdir(os.path.dirname(saveaslr)):
    os.makedirs(os.path.dirname(saveaslr))

usechain2 = [
    'C02',
    'C14',
    'E02',
    'G18',
    'H04',
    'H12',
    'H18',
    'K14',
    'M06',
    'O24',
    'P01',
    'P14'
]
usechain3 = [
    'D13',
    'E18',
    'H16',
    # 'K03',  #?
    'M11',
    'O16',
    'P03',
]

#
# Simple mean and cov values from individual MCMC results
#
exp_chains = []
print('Loading results...')
for i_file, (file_name, temperature) in enumerate(zip(file_list,
                                                      temperatures)):
    # Load QC
    selectedfile = '%s/manualv2selected-%s.txt' % (qc_dir, file_name)
    selectedwell = []
    with open(selectedfile, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                selectedwell.append(l.split()[0])

    for i_cell, cell in enumerate(selectedwell):
        print('Loading %s' % cell)

        # Load fitting result
        chain_file = '%s%s-chain.csv' % \
                (load_name, cell)
        if cell in usechain2:
            exp_chains.append(pints.io.load_samples(chain_file, 2)[1])
            continue
        elif cell in usechain3:
            exp_chains.append(pints.io.load_samples(chain_file, 3)[2])
            continue
        else:
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

#
# Save mcmc means and simple log mean
#
if not os.path.isdir('%s/%s' % ('./out', file_name + '-mcmcmean')):
    os.makedirs('%s/%s' % ('./out', file_name + '-mcmcmean'))
for i_cell, cell in enumerate(selectedwell):
    with open('%s/%s/%s-solution-%s.txt' % ('./out', file_name + '-mcmcmean',\
                file_name + '-staircaseramp-' + cell, fit_seed), 'w') as f:
        for k, x in enumerate(transform_to_model_param(
                              exp_transform_parameters[i_cell])):
            f.write(pints.strfloat(x) + '\n')
with open('%s/%s-mcmcsimplemean-%s.txt' % ('./out', file_name + \
            '-staircaseramp', fit_seed), 'w') as f:
    for k, x in enumerate(transform_to_model_param(simple_transform_mean)):
        f.write(pints.strfloat(x) + '\n')


# k_0: 0 = not imposing mean, no prior;
#      nexp = prior as strong as our experiments
# nu_0: 1 = no prior
#       nexp = scaling this gives prior of $\Sigma$ to be certain at Gamma_0
# mu_0: mean of the prior
# Gamma_0: covariance of the prior

k_0 = 0
nu_0 = 1
# mean of the transformed parameters
mu_0 = np.mean(exp_transform_parameters, axis=0)
assert(len(mu_0) == n_parameters)
estimated_cov = (np.std(exp_transform_parameters, axis=0)) ** 2
Gamma_0 = 1.0e-3 * np.diag(estimated_cov)
Gamma_0[0, 0] = estimated_cov[0] * nu_0  # Let g_Kr varies
assert(len(Gamma_0) == len(mu_0))


#
# Pseudo HBM using individual MCMC results
#
n_hbm_samples = (n_samples - warm_up)
thinning_hbm = 1
warm_up_hbm = int(n_hbm_samples // thinning_hbm // 2)
chain = np.zeros((n_hbm_samples / thinning_hbm, n_parameters))
cov_chain = np.zeros((n_hbm_samples / thinning_hbm,
                      n_parameters, n_parameters))
assert(n_hbm_samples <= exp_chains_final.shape[1])

for sample in range(n_hbm_samples):
    xs = exp_chains_final[:, sample, :-1*n_non_model_param]  # transformed
    xhat = np.mean(xs, axis=0)
    C = np.zeros((len(xhat), len(xhat)))
    for x in xs:
        C += np.outer(x - xhat, x - xhat)
    k = k_0 + nexp
    nu = nu_0 + nexp
    mu = (k_0 * mu_0 + nexp * xhat) / k
    tmp = xhat - mu_0
    Gamma = Gamma_0 + C + (k_0 * nexp) / k * np.outer(tmp, tmp)

    covariance_sample = scipy.stats.invwishart.rvs(df=nu, scale=Gamma)
    means_sample = scipy.stats.multivariate_normal.rvs(
            mean=mu, cov=covariance_sample / k)

    # thinning, store every (thinning)th sample
    if ((sample % thinning_hbm) == (thinning_hbm - 1)) and (sample != 0):
        # store sample to chain
        chain[sample/thinning_hbm - 1, :] = means_sample
        # store cov matrix to chain
        cov_chain[sample/thinning_hbm - 1, :, :] = covariance_sample[:, :]

# Discard warm up
simple_chain_final = chain[warm_up_hbm:-1, :]  # last one seems zero...
simple_cov_final = cov_chain[warm_up_hbm:-1, :, :]
# Covariance matrice
simple_cor_final = np.zeros(simple_cov_final.shape)
for i, s in enumerate(simple_cov_final):
    D = np.sqrt(np.diag(s))
    c = s / D / D[:, None]
    simple_cor_final[i, :, :] = c[:, :]

# Save it
np.savetxt('./out-mcmc/herg25oc1-pseudohbm-lognorm-mean.txt',
           simple_chain_final)
with open('./out-mcmc/herg25oc1-pseudohbm-lognorm-cov.pkl', 'wb') as f:
    pickle.dump(simple_cov_final, f)


#
# Plot
#

# Hyperparameters mean values ($\mu_i$)
chain_param = simple_chain_final[:, :]

# Hyperparameters standard deviation values ($s_i$)
chain_stddev = np.sqrt(simple_cov_final.diagonal(0, 1, 2))
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
    # this is good ref for real data
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

''' # set common xlim
xlim = [[(1e4, 6.5e4), (1e-8, 0.3), (40, 150)],
        [(1e-8, 0.05), (35, 65), (60, 230)],
        [(5, 30), (5, 18), (20, 35)]]
for i in range(3):
    for j in range(3):
        axes[i][j].set_xlim(xlim[i][j])
'''

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('%shbm-plot.png' % saveaslr, bbox_iches='tight', dpi=100)
    plt.savefig('%shbm-plot.png' % saveas, bbox_iches='tight', dpi=300)
    plt.savefig('%shbm-plot.pdf' % saveas, format='pdf', bbox_inches='tight')
plt.close('all')


# NOTE: Below function/plot has been moved to `paper-pseudohbm-cov.py`
sys.exit()


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
        plot_func.addbox(axes, (i, j), color='#fdb462', alpha=0.35)

if '--show' in sys.argv:
    plt.show()
else:
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0,
                     rect=(0.001, 0.001, 0.97, 0.965))
    plt.savefig('%scov-plot-cbox.png'%saveaslr, bbox_iches='tight',
                pad_inches=0, dpi=100)
    plt.savefig('%scov-plot-cbox.png'%saveas, bbox_iches='tight',
                pad_inches=0, dpi=300)
    plt.savefig('%scov-plot-cbox.pdf'%saveas, bbox_iches='tight',
                format='pdf')

# Maybe two colours...
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
