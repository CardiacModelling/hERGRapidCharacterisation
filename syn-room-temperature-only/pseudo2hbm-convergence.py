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
import seaborn as sns


# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param


# About files
file_name = 'syn-101'
temperature = 25.0
fit_seed = '542811797'


saveas = 'figs/tophbm-testnexp'
n_non_model_param = 1
variable_names = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
                  r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']

nexp = 20000

if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))


path2mean = '../room-temperature-only/kylie-room-temperature/' \
            + 'last-solution_C5.txt'
true_mean = np.loadtxt(path2mean)
# Change conductance unit nS->pS (new parameter use V, but here mV)
true_mean[0] = true_mean[0] * 1e3
true_mean = transform_from_model_param(true_mean)
cov_seed = 101
true_cov = np.loadtxt('./out/cov-%s.txt' % cov_seed)
true_cor = np.loadtxt('./out/corr-%s.txt' % cov_seed)
true_std = np.sqrt(np.diag(true_cov))

all_transform_parameters = []

for i in range(nexp):
    np.random.seed(i)
    seed = np.random.randint(0, 2**30)
    np.random.seed(seed)

    all_transform_parameters.append(
            np.random.multivariate_normal(true_mean, true_cov))

all_transform_parameters = np.asarray(all_transform_parameters)
n_parameters = len(true_mean)


#
# Run pseudo HBM and test nexp effect
#

# Control fitting seed --> OR DONT
# control_seed = np.random.randint(0, 2**30)
control_seed = int(fit_seed)
print('Using seed: ', control_seed)
np.random.seed(control_seed)

testnexps = np.arange(20, 1120, 100)
testnexps = [20, 120, 520, 1020, 5020, 10020]
testnexps = [int(i) for i in np.logspace(np.log10(20), np.log10(20000), 10)]

err_mean_y = []
err_mean_x = []
err_std_y = []
err_std_x = []
err_cor_y = []
err_cor_x = []
mean_err_mean = []
mean_err_std = []
mean_err_cor = []

for testnexp in testnexps:

    transform_parameters = all_transform_parameters[:testnexp, :]

    # k_0: 0 = not imposing mean, no prior;
    #      nexp = prior as strong as our experiments
    # nu_0: 1 = no prior
    #       nexp = scaling this gives prior of $\Sigma$ to be certain at Gamma_0
    # mu_0: mean of the prior
    # Gamma_0: covariance of the prior

    k_0 = 0
    nu_0 = 1
    # mean of the transformed parameters
    mu_0 = np.mean(transform_parameters, axis=0)
    assert(len(mu_0) == n_parameters)
    estimated_cov = (np.std(transform_parameters, axis=0)) ** 2
    Gamma_0 = 1.0e-3 * np.diag(estimated_cov)
    Gamma_0[0, 0] = estimated_cov[0] * nu_0  # Let g_Kr varies
    assert(len(Gamma_0) == len(mu_0))

    # Pseudo HBM using individual MCMC results

    n_hbm_samples = 10000
    thinning_hbm = 1
    warm_up_hbm = int(n_hbm_samples // thinning_hbm // 2)
    chain = np.zeros((n_hbm_samples / thinning_hbm, n_parameters))
    cov_chain = np.zeros((n_hbm_samples / thinning_hbm,
                          n_parameters, n_parameters))

    xs = np.copy(transform_parameters)  # transformed
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
    mean = chain[warm_up_hbm:-1, :]  # last one seems zero...
    cov = cov_chain[warm_up_hbm:-1, :, :]
    # Covariance matrice
    std = np.zeros((cov.shape[0], cov.shape[1]))
    cor = np.zeros(cov.shape)
    for i, s in enumerate(cov):
        D = np.sqrt(np.diag(s))
        std[i, :] = D[:]
        c = s / D / D[:, None]
        cor[i, :, :] = c[:, :]


    err_mean = (mean - true_mean) / np.abs(true_mean)
    err_std = (std - true_std) / true_std
    err_cor = cor - true_cor

    err_mean_y.extend(np.sqrt(np.mean(err_mean ** 2, axis=1)))
    err_mean_x.extend(len(err_mean) * [testnexp])
    err_std_y.extend(np.sqrt(np.mean(err_std ** 2, axis=1)))
    err_std_x.extend(len(err_std) * [testnexp])
    err_cor_y.extend(np.sqrt(np.mean(np.mean(err_cor ** 2, axis=2), axis=1)))
    err_cor_x.extend(len(err_cor) * [testnexp])

    mean_err_mean.append(np.sqrt(np.mean(np.mean(err_mean, axis=0) ** 2)))
    mean_err_std.append(np.sqrt(np.mean(np.mean(err_std, axis=0) ** 2)))
    mean_err_cor.append(np.sqrt(np.mean(np.mean(err_cor, axis=0) ** 2)))


# Mean
plt.figure(figsize=(12, 6))
sns.violinplot(x=err_mean_x, y=err_mean_y, zorder=1)
plt.plot(mean_err_mean, ls='', marker='x', ms=12, c='r', zorder=2,
        label='posterior mean')

plt.ylabel(r'RMSPE of mean', fontsize=32)
plt.xlabel(r'$N_{exp}$', fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('%s-mean-violin.png' % saveas, bbox_inches='tight', dpi=300)
plt.savefig('%s-mean-violin.pdf' % saveas, format='pdf',
        bbox_inches='tight')
plt.close()


plt.figure(figsize=(12, 6))
plt.plot(np.log(testnexps), np.log(mean_err_mean), ls='', marker='x', ms=12,
        c='r', label='posterior mean')

m, c, r, _, _ = scipy.stats.linregress(np.log(testnexps), np.log(mean_err_mean))
plt.plot(np.log(testnexps), m * np.log(testnexps) + c, ls='--',
            label=r'$R^2=$%.3f, slope=%.3f' % (r ** 2, m))
plt.legend(fontsize=24)
plt.ylabel(r'ln(RMSPE of mean)', fontsize=32)
plt.xlabel(r'ln($N_{exp}$)', fontsize=32)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('%s-mean-lnln.png' % saveas, bbox_inches='tight')
plt.close()


# Std
plt.figure(figsize=(12, 6))
sns.violinplot(x=err_std_x, y=err_std_y, zorder=1)
plt.plot(mean_err_std, ls='', marker='x', ms=12, c='r', zorder=2,
        label='posterior mean')

plt.ylabel(r'RMSPE of std', fontsize=32)
plt.xlabel(r'$N_{exp}$', fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('%s-std-violin.png' % saveas, bbox_inches='tight', dpi=300)
plt.savefig('%s-std-violin.pdf' % saveas, format='pdf',
        bbox_inches='tight')
plt.close()


plt.figure(figsize=(12, 6))
plt.plot(np.log(testnexps), np.log(mean_err_std), ls='', marker='x', ms=12,
        c='r', label='posterior mean')

m, c, r, _, _ = scipy.stats.linregress(np.log(testnexps), np.log(mean_err_std))
plt.plot(np.log(testnexps), m * np.log(testnexps) + c, ls='--',
            label=r'$R^2=$%.3f, slope=%.3f' % (r ** 2, m))
plt.legend(fontsize=24)
plt.ylabel(r'ln(RMSPE of std)', fontsize=32)
plt.xlabel(r'ln($N_{exp}$)', fontsize=32)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('%s-std-lnln.png' % saveas, bbox_inches='tight')
plt.close()


# Cov
plt.figure(figsize=(12, 6))
sns.violinplot(x=err_cor_x, y=err_cor_y, zorder=1)
plt.plot(mean_err_cor, ls='', marker='x', ms=12, c='r', zorder=2,
        label='posterior mean')

plt.ylabel(r'RMSE of correlation', fontsize=32)
plt.xlabel(r'$N_{exp}$', fontsize=32)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('%s-cor-violin.png' % saveas, bbox_inches='tight', dpi=300)
plt.savefig('%s-cor-violin.pdf' % saveas, format='pdf',
        bbox_inches='tight')
plt.close()


plt.figure(figsize=(12, 6))
plt.plot(np.log(testnexps), np.log(mean_err_cor), ls='', marker='x', ms=12,
        mew=2, c='r', label='posterior mean')

m, c, r, _, _ = scipy.stats.linregress(np.log(testnexps), np.log(mean_err_cor))
print('RMSE(corr)|Ne=124: ', np.exp(m * np.log(124) + c))
plt.plot(np.log(testnexps), m * np.log(testnexps) + c, ls='--', lw=2,
            label=r'$R^2=$%.3f, slope=%.3f' % (r ** 2, m))
plt.axvline(np.log(124), c='#7f7f7f')
plt.axhline(m * np.log(124) + c, c='#7f7f7f')
plt.legend(fontsize=24)
plt.ylabel(r'$\ln$(RMSE of correlation)', fontsize=32)
plt.xlabel(r'$\ln$($N_{exp}$)', fontsize=32)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('%s-cor-lnln.pdf' % saveas, format='pdf',
        bbox_inches='tight')
plt.savefig('%s-cor-lnln.png' % saveas, bbox_inches='tight')
plt.close()

