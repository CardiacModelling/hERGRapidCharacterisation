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

basename = 'out-mcmc/syn-101'
load_name = '%s/c' % (basename)  # if any prefix in all files
saveas = 'figs/pseudoHBM-syn-101/'
saveas_data = './out-mcmc/syn-101-testnexp'
n_non_model_param = 1
which_hyper_func = 1
variable_names = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
                  r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']

nexp = 125
n_samples = 200000

usechain2 = usechain3 = []

if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))

if not os.path.isdir(saveas_data):
    os.makedirs(saveas_data)

#
# Simple mean and cov values from individual MCMC results
#
exp_chains = []
print('Loading results...')
for i in range(nexp):
    print('Loading %s' % i)

    # Load fitting result
    chain_file = '%s%s-chain.csv' % \
            (load_name, i)
    if i in usechain2:
        exp_chains.append(pints.io.load_samples(chain_file,
                2)[1][:n_samples, :])
        continue
    elif i in usechain3:
        exp_chains.append(pints.io.load_samples(chain_file,
                3)[2][:n_samples, :])
        continue
    else:
        exp_chains.append(pints.io.load_samples(chain_file,
                1)[0][:n_samples, :])

assert(n_samples == len(exp_chains[0]))
warm_up = int(n_samples * 3. / 4.)
thinning = 1
# Thinning and discard warm up
all_exp_chains_final = np.asarray(exp_chains)[:, warm_up::thinning, :]
all_exp_transform_parameters = np.mean(all_exp_chains_final, axis=1)
# Remove non model parameters
all_exp_transform_parameters = all_exp_transform_parameters[:, :-1*n_non_model_param]

assert(nexp == all_exp_transform_parameters.shape[0])
n_parameters = all_exp_transform_parameters.shape[1]

simple_transform_mean = np.mean(all_exp_transform_parameters, axis=0)
simple_transform_cov = np.cov(np.asarray(all_exp_transform_parameters).T)

#
# Save mcmc means and simple log mean
#
if not os.path.isdir('%s/%s' % ('./out', file_name + '-mcmcmean')):
    os.makedirs('%s/%s' % ('./out', file_name + '-mcmcmean'))
for i in range(nexp):
    with open('%s/%s/solution-%s.txt' % ('./out', file_name + '-mcmcmean',\
                i), 'w') as f:
        for k, x in enumerate(transform_to_model_param(
                              all_exp_transform_parameters[i])):
            f.write(pints.strfloat(x) + '\n')
with open('%s/%s-mcmcsimplemean.txt' % ('./out', file_name), 'w') as f:
    for k, x in enumerate(transform_to_model_param(simple_transform_mean)):
        f.write(pints.strfloat(x) + '\n')


#
# Run pseudo HBM and test nexp effect
#

testnexps = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 125]

for testnexp in testnexps:

    exp_transform_parameters = all_exp_transform_parameters[:testnexp, :]
    exp_chains_final = all_exp_chains_final[:testnexp, :, :]

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
    np.savetxt('%s/%s-pseudohbm-lognorm-mean-nexp-%s.txt' \
            % (saveas_data, file_name, testnexp), simple_chain_final)
    with open('%s/%s-pseudohbm-lognorm-cov-nexp-%s.pkl' \
            % (saveas_data, file_name, testnexp), 'wb') as f:
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
    plt.savefig('%sexpchains-nexp-%s.png' % (saveas, testnexp),
            bbox_iches='tight')
    plt.close('all')
    pints.plot.trace([chain_param], n_percentiles=99.9)
    plt.savefig('%shypermean-nexp-%s.png' % (saveas, testnexp),
            bbox_iches='tight')
    plt.close('all')


    # individual exp: trace
    n_percentiles = 99.9

    # plot inferred hyperparameters on top
    hyper_func_opt = ['normal', 'log']
    hyper_func = hyper_func_opt[which_hyper_func]

    # plot inferred hyperparameters on top but no noise
    # reference of covariance matrix
    if True:
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
    plt.savefig('%shbm-plot-nexp-%s.png' % (saveas, testnexp),
            bbox_iches='tight', dpi=300)
    plt.close('all')


