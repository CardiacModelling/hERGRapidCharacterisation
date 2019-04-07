#!/usr/bin/env python2
import sys
sys.path.append('../lib')
import os
import numpy as np
import scipy
import matplotlib
if not '--show' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

import pints.io
import pints.plot
import plot_hbm_func as plot_func

files_dir = './out'
qc_dir = '.'
file_list = ['herg25oc1']
temperatures = [25.0]
fit_seed = '542811797'

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

#
# Simple mean and cov values from fitting results
#
exp_parameters = []
exp_transform_parameters = []
for i_file, (file_name, temperature) in enumerate(zip(file_list,
                                                      temperatures)):
    # Load QC
    selectedfile = '%s/manualv2selected-%s.txt' % (qc_dir, file_name)
    selectedwell = []
    with open(selectedfile, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                selectedwell.append(l.split()[0])

    for i_cell, cell in enumerate(selectedwell[:20]):
        # Load fitting result
        param_file = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
                (files_dir, file_name, file_name, cell, fit_seed)
        x0 = np.loadtxt(param_file)
        transform_x0 = transform_from_model_param(x0)
        exp_parameters.append(x0)
        exp_transform_parameters.append(transform_x0)

n_parameters = len(exp_parameters[0])
nexp = len(exp_parameters)

simple_mean = np.mean(exp_parameters, axis=0)
simple_transform_mean = np.mean(exp_transform_parameters, axis=0)
simple_cov = np.cov(np.asarray(exp_parameters).T)
simple_transform_cov = np.cov(np.asarray(exp_transform_parameters).T)

# k_0: 0 = not imposing mean, no prior;
#      nexp = prior as strong as our experiments
# nu_0: 1 = no prior
#       nexp = scaling this gives prior of $\Sigma$ to be certain at Gamma_0
# mu_0: mean of the prior
# Gamma_0: covariance of the prior

k_0 = 0
nu_0 = 1
mu_0 = transform_from_model_param(np.array(exp_parameters))
mu_0 = np.mean(mu_0, axis=0)  # mean of the transformed parameters
assert(len(mu_0) == n_parameters)
estimated_cov = (np.std(
                    transform_from_model_param(
                        np.array(exp_parameters)
                    ), axis=0)
                )**2
Gamma_0 = 1.0e-3 * np.diag(estimated_cov)
Gamma_0[0, 0] = estimated_cov[0] * nu_0  # Let g_Kr varies
assert(len(Gamma_0) == len(mu_0))

#
# MCMC using fitted results
#
n_samples = 20000
thinning = 2
chain = np.zeros((n_samples / thinning, n_parameters))
cov_chain = np.zeros((n_samples / thinning, n_parameters, n_parameters))

xhat = np.copy(simple_transform_mean)
C = np.zeros((len(xhat), len(xhat)))
for x in np.asarray(exp_transform_parameters):
    C += np.outer(x - xhat, x - xhat)
k = k_0 + nexp
nu = nu_0 + nexp
mu = (k_0 * mu_0 + nexp * xhat) / k
tmp = xhat - mu_0
Gamma = Gamma_0 + C + (k_0 * nexp) / k * np.outer(tmp, tmp)
for sample in range(n_samples):
    covariance_sample = scipy.stats.invwishart.rvs(df=nu, scale=Gamma)
    means_sample = scipy.stats.multivariate_normal.rvs(
            mean=mu, cov=covariance_sample / k)

    # thinning, store every (thinning)th sample
    if ((sample % thinning) == (thinning - 1)) and (sample != 0):
        # store sample to chain
        chain[sample/thinning - 1, :] = means_sample
        # store cov matrix to chain
        cov_chain[sample/thinning - 1, :, :] = covariance_sample[:, :]

n_samples = n_samples // thinning
simple_chain_final = chain[(n_samples // 2):n_samples, :]
simple_cov_final = cov_chain[(n_samples // 2):n_samples, :, :]
# plot correlation histograms
simple_cor_final = np.zeros(simple_cov_final.shape)
for i, s in enumerate(simple_cov_final):
    D = np.sqrt(np.diag(s))
    c = s / D / D[:, None]
    simple_cor_final[i, :, :] = c[:, :]


#
# Compare with HBM results
#
# About files
basename = 'out-mcmc/test-HBM'
load_name = '%s/' % (basename)  # if any prefix in all files
saveas = 'figs/test-simple-mean/'
n_exp = 20
thinning = 1  # Usually did thinning already...
which_hyper_func = 1
variable_names = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
                  r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']

if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))

# Load results
print('Loading results...')
chain = pints.io.load_samples('%schain.csv' % load_name)
exp_chains = pints.io.load_samples('%sexp_chain.csv' % load_name, n_exp)
with open('%scov_chain.pkl' % load_name, 'rb') as f:
    cov_chain = pickle.load(f)


# Process results
n_samples = 26000 # len(chain)
# drop first half of chain and thinning
chain_final = chain[(n_samples // 2):n_samples:thinning, :]
exp_chains_final = np.array(exp_chains)[:,
                                        (n_samples // 2):n_samples:thinning,
                                        :]
cov_chain_final = cov_chain[(n_samples // 2):n_samples:thinning, :, :]
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
_, axes = pints.plot.trace([chain_param], n_percentiles=99.9)
for i, ax in enumerate(axes):
    ax[0].axvline(simple_transform_mean[i], color='k', ls='--', label='simple mean')
axes[0, 0].legend()
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

print('Plotting hbm plot...')
_, axes = plot_func.plot_posterior_predictive_distribution(
        [chain_param, chain_stddev],
        exp_chains_param,
        hyper_func=hyper_func,
        fold=True, ref_hyper=ref_hyper,
        n_percentiles=n_percentiles,
        normalise=True,
        )
axes = plot_func.change_labels_histogram_fold(axes, variable_names)
# axes[0][0].legend([])  # suppress legend
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('%shbm-plot.png' % saveas, bbox_iches='tight')
    # plt.savefig('%shbm-plot.pdf'%saveas, format='pdf', bbox_inches='tight')
plt.close('all')


# plot correlation histograms
cor_chain_final = np.zeros(cov_chain_final.shape)
for i, s in enumerate(cov_chain_final):
    D = np.sqrt(np.diag(s))
    c = s / D / D[:, None]
    cor_chain_final[i, :, :] = c[:, :]


print('Plotting cov...')
fig, axes = plot_func.plot_correlation_and_variable_covariance(
        chain_param[::50],
        cov_chain_final[::50],
        cor_chain_final, corr=True,
        # ref_parameters=[simple_transform_mean, simple_transform_cov]
        )
fig, axes = plot_func.plot_correlation_and_variable_covariance(
        simple_chain_final[::10],
        simple_cov_final[::10],
        simple_cor_final, corr=True,
        ref_parameters=[simple_transform_mean, simple_transform_cov],
        fig=fig, axes=axes, colours=['b', 'k', 'r']
        )
axes = plot_func.change_labels_correlation_and_variable_covariance(
        axes,
        variable_names
        )
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('%scov-plot.png'%saveas, bbox_iches='tight', pad_inches=0)
    # plt.savefig('%scov-plot.pdf'%saveas, format='pdf', bbox_inches='tight')
plt.close('all')
