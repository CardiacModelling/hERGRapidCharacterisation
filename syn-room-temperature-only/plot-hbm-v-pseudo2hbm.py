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
fit_seed = '542811797'

# Control fitting seed --> OR DONT
# control_seed = np.random.randint(0, 2**30)
control_seed = int(fit_seed)
print('Using seed: ', control_seed)
np.random.seed(control_seed)

basename = 'out/syn-101-mcmcmean'
load_name = '%s/solution' % (basename)  # if any prefix in all files
saveas = 'figs/pseudoHBM-syn-101/'
n_non_model_param = 1
which_hyper_func = 1
variable_names = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
                  r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']

nexp = 31
testnexp = nexp - 1
n_samples = 200000
warm_up = int(n_samples * 3. / 4.)

usechain2 = usechain3 = []

if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))

#
# Simple mean and cov values from individual MCMC results
#
all_exp_parameters = []
all_exp_transform_parameters = []
print('Loading results...')
for i in range(nexp):
    print('Loading %s' % i)

    # Load fitting result
    param_file = '%s-%s.txt' % (load_name, i)
    p = np.loadtxt(param_file)
    all_exp_parameters.append(p)
    all_exp_transform_parameters.append(transform_from_model_param(p))

# Remove non model parameters
all_exp_parameters = np.asarray(all_exp_parameters)
all_exp_transform_parameters = np.asarray(all_exp_transform_parameters)

assert(nexp == all_exp_transform_parameters.shape[0])
n_parameters = all_exp_transform_parameters.shape[1]

simple_transform_mean = np.mean(all_exp_transform_parameters, axis=0)
simple_transform_cov = np.cov(np.asarray(all_exp_transform_parameters).T)


exp_transform_parameters = all_exp_transform_parameters[:testnexp, :]

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

# Pseudo HBM using individual MCMC results

n_hbm_samples = (n_samples - warm_up)
thinning_hbm = 2
warm_up_hbm = int(n_hbm_samples // thinning_hbm // 2)
chain = np.zeros((n_hbm_samples / thinning_hbm, n_parameters))
cov_chain = np.zeros((n_hbm_samples / thinning_hbm,
                      n_parameters, n_parameters))

xs = np.copy(exp_transform_parameters[:, :])  # transformed
xhat = np.mean(xs, axis=0)
C = np.zeros((len(xhat), len(xhat)))
for x in xs:
    C += np.outer(x - xhat, x - xhat)
k = k_0 + nexp
nu = nu_0 + nexp
mu = (k_0 * mu_0 + nexp * xhat) / k
tmp = xhat - mu_0
Gamma = Gamma_0 + C + (k_0 * nexp) / k * np.outer(tmp, tmp)

for sample in range(n_hbm_samples):

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


#
# Plot
#

# Hyperparameters mean values ($\mu_i$)
chain_param = simple_chain_final[:, :]

# Hyperparameters standard deviation values ($s_i$)
chain_stddev = np.sqrt(simple_cov_final.diagonal(0, 1, 2))
assert(len(chain_param) == len(chain_stddev))


# individual exp: trace
n_percentiles = 99.9

# plot inferred hyperparameters on top
hyper_func_opt = ['normal', 'log']
hyper_func = hyper_func_opt[which_hyper_func]

# plot inferred hyperparameters on top but no noise
# reference of covariance matrix
if False:
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
else:
    # this is good ref for real data
    rho = covariance = np.zeros((9,9))
    ref_hyper = None
    ref_variable_corr = None



print('Loading full HBM results...')
basename = 'out-mcmc/syn-101-testhbm'
load_name = '%s/' % (basename)  # if any prefix in all files
n_exp = 30
thinning = 1  # Usually did thinning already...
chain = pints.io.load_samples('%schain.csv' % load_name)
exp_chains = pints.io.load_samples('%sexp_chain.csv' % load_name, n_exp)
with open('%scov_chain.pkl' % load_name, 'rb') as f:
    cov_chain = pickle.load(f)

# Process results
n_samples = 10000 # len(chain)
# drop first half of chain and thinning
chain_final = chain[(n_samples // 3):n_samples:thinning, :]
exp_chains_final = np.array(exp_chains)[:,
                                        (n_samples // 3):n_samples:thinning,
                                        :]
cov_chain_final = cov_chain[(n_samples // 3):n_samples:thinning, :, :]
assert(len(chain_final) == len(cov_chain_final))

# Individual experiments ($p_{i,j}$)
exp_chains_param = []
for exp_chain in exp_chains_final:
    exp_chain_tmp = np.copy(exp_chain)
    exp_chain_tmp[:, :-1] = transform_to_model_param(exp_chain[:, :-1])
    exp_chains_param.append(exp_chain_tmp)

print('Plotting fig4-5...')
fig, axes = plot_func.plot_posterior_predictive_distribution(
        [chain_param, chain_stddev],
        exp_chains_param,
        hyper_func=hyper_func,
        fold=True, ref_hyper=ref_hyper,
        n_percentiles=n_percentiles,
        normalise=True,
        )


# Hyperparameters mean values ($\mu_i$)
chain_param = chain_final[:, :]

# Hyperparameters standard deviation values ($s_i$)
chain_stddev = np.sqrt(cov_chain_final.diagonal(0, 1, 2))
assert(len(chain_param) == len(chain_stddev))

_, axes = plot_func.plot_posterior_predictive_distribution(
        [chain_param, chain_stddev],
        exp_chains_param,
        hyper_func=hyper_func,
        fold=True, ref_hyper=None,
        n_percentiles=99.9,
        normalise=True,
        mode=2,
        fig=fig[0], axes=axes, axes2=fig[1]
        )

axes = plot_func.change_labels_histogram_fold(axes, variable_names)
axes[0, 0].ticklabel_format(axis='x', style='sci', scilimits=(0, 1))


plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0,
                 rect=(0.015, 0.001, 0.999, 0.999))
plt.savefig('%shbm-v-pseudo2hbm-plot-nexp-%s.png' % (saveas, testnexp),
        bbox_iches='tight', dpi=300)
plt.close('all')


