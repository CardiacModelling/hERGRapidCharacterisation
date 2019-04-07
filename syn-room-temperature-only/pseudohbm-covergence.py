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
import seaborn as sns


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

load_data = './out-mcmc/syn-101-testnexp'
saveas = './figs/testnexp'
n_non_model_param = 1
which_hyper_func = 1
variable_names = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
                  r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']

nexp = 125
testnexps = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 125]

if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))


# True values
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


#
# Save mcmc means and simple log mean
#
for i in range(nexp):
    p = np.loadtxt('%s/%s/solution-%s.txt' % ('./out', \
            file_name + '-mcmcmean', i))


#
# Run pseudo HBM and test nexp effect
#


err_mean_y = []
err_mean_x = []
err_std_y = []
err_std_x = []
err_cor_y = []
err_cor_x = []

for testnexp in testnexps:

    mean = np.loadtxt('%s/%s-pseudohbm-lognorm-mean-nexp-%s.txt' \
            % (load_data, file_name, testnexp))
    with open('%s/%s-pseudohbm-lognorm-cov-nexp-%s.pkl' \
            % (load_data, file_name, testnexp), 'rb') as f:
        cov = pickle.load(f)

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


# Mean
plt.figure(figsize=(12, 6))
sns.violinplot(x=err_mean_x, y=err_mean_y, zorder=1)

plt.ylabel(r'RMSPE of mean', fontsize=32)
plt.xlabel(r'$N_{exp}$', fontsize=32)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('%s-mean-violin.png' % saveas, bbox_inches='tight', dpi=300)
plt.savefig('%s-mean-violin.pdf' % saveas, format='pdf',
        bbox_inches='tight')


# Std
plt.figure(figsize=(12, 6))
sns.violinplot(x=err_std_x, y=err_std_y, zorder=1)

plt.ylabel(r'RMSPE of std', fontsize=32)
plt.xlabel(r'$N_{exp}$', fontsize=32)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('%s-std-violin.png' % saveas, bbox_inches='tight', dpi=300)
plt.savefig('%s-std-violin.pdf' % saveas, format='pdf',
        bbox_inches='tight')


# Cov
plt.figure(figsize=(12, 6))
sns.violinplot(x=err_cor_x, y=err_cor_y, zorder=1)

plt.ylabel(r'RMSE of correlation', fontsize=32)
plt.xlabel(r'$N_{exp}$', fontsize=32)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('%s-cor-violin.png' % saveas, bbox_inches='tight', dpi=300)
plt.savefig('%s-cor-violin.pdf' % saveas, format='pdf',
        bbox_inches='tight')
