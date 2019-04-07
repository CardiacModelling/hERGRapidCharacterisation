#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import pickle

import pints.io
import pints.plot

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

savedir = './out-mcmc'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

file_dir = './out'
file_list = [ # (file_name1, file_name2, ...)
        ['herg25oc1'],
        ['herg27oc1'],
        ['herg30oc1'],
        ['herg33oc1'],
        ['herg37oc3'],
        ]
temperatures = np.array([25.0, 27.0, 30.0, 33.0, 37.0])
temperatures += 273.15  # in K
assert(len(file_list) == len(temperatures))
fit_seed = 542811797
withfcap = False

#
# Define some parameters and labels
#
labels = [r'$g$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
          r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$']

# Loop through files
for i_temperature, (file_names, temperature) in enumerate(zip(file_list,
    temperatures)):

    save_name = file_names[0][:-1]

    exp_parameters = []
    exp_transform_parameters = []
    for file_name in file_names:
        ## Get fitting results
        files_dir = os.path.realpath(os.path.join(file_dir, file_name))
        searchwfcap = '-fcap' if withfcap else ''
        selectedfile = './manualselection/manualselected-%s.txt' % (file_name)
        selectedwell = []
        with open(selectedfile, 'r') as f:
            for l in f:
                if not l.startswith('#'):
                    selectedwell.append(l.split()[0])
        print('Running', file_name)

        for cell in selectedwell:
            param_file = '%s/%s-staircaseramp-%s-solution%s-%s.txt' \
                    % (files_dir, file_name, cell, searchwfcap, fit_seed)

            try:
                p = np.loadtxt(param_file)
            except FileNotFoundError:
                print('No cell %s %s fitting result' % (file_name, cell))
                continue
            exp_parameters.append(p)
            exp_transform_parameters.append(transform_from_model_param(p))
    exp_parameters = np.asarray(exp_parameters)
    exp_transform_parameters = np.asarray(exp_transform_parameters)

    nexp, n_parameters = exp_transform_parameters.shape

    # Control fitting seed --> OR DONT
    # fit_seed = np.random.randint(0, 2**30)
    print('Fit seed: ', fit_seed)
    np.random.seed(fit_seed)

    # k_0: 0 = not imposing mean, no prior;
    #      nexp = prior as strong as our experiments
    # nu_0: 1 = no prior
    #       nexp = scaling this gives prior of $\Sigma$ to be certain at
    #              Gamma_0
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

    n_hbm_samples = 40000
    thinning_hbm = 2
    warm_up_hbm = int(n_hbm_samples // thinning_hbm // 2)
    chain = np.zeros((int(n_hbm_samples // thinning_hbm), n_parameters))
    cov_chain = np.zeros((int(n_hbm_samples // thinning_hbm),
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
            chain[int(sample // thinning_hbm), :] = means_sample
            # store cov matrix to chain
            cov_chain[int(sample // thinning_hbm), :, :] = \
                    covariance_sample[:, :]

    # Discard warm up
    simple_chain_final = chain[warm_up_hbm:, :]  # last one seems zero...
    simple_cov_final = cov_chain[warm_up_hbm:, :, :]
    # Covariance matrice
    simple_cor_final = np.zeros(simple_cov_final.shape)
    for i, s in enumerate(simple_cov_final):
        D = np.sqrt(np.diag(s))
        c = s / D / D[:, None]
        simple_cor_final[i, :, :] = c[:, :]

    # Save it
    np.savetxt('%s/%s-pseudo2hbm-lognorm-mean.txt' \
            % (savedir, save_name), simple_chain_final)
    with open('%s/%s-pseudo2hbm-lognorm-cov.pkl' \
            % (savedir, save_name), 'wb') as f:
        pickle.dump(simple_cov_final, f)

    # Some quick plot
    pints.plot.trace([simple_chain_final], n_percentiles=99.9)
    plt.savefig('%s/%s-hypermean.png' % (savedir, save_name),
            bbox_iches='tight')
    plt.close('all')
