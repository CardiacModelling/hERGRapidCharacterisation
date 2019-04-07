#!/usr/bin/env python
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from scipy import stats

from hbmdistribution import PosteriorPredictiveLogNormal

# Fix seed
np.random.seed(101)

savedir = './figs/pseudoHBM-manualv2selected-herg25oc1-check'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

file_dir = './out'
qc_dir = '.'
file_list = [
        'herg25oc1',
        ]
temperatures = np.array([25.0])
temperatures += 273.15  # in K
fit_seed = '542811797'

param_name = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
              r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']


# Load parameters
obtained_parameters = []
for i_file, (file_name, temperature) in enumerate(zip(file_list,
                                                      temperatures)):
    # Load QC
    selectedfile = '%s/manualv2selected-%s.txt' % (qc_dir, file_name)
    selectedwell = []
    with open(selectedfile, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                selectedwell.append(l.split()[0])

    for cell in selectedwell:
        # MCMC samples mean
        param_file = '%s/%s-mcmcmean/%s-staircaseramp-%s-solution-%s.txt' % \
                (file_dir, file_name, file_name, cell, fit_seed)
        obtained_parameters.append(np.loadtxt(param_file))
obtained_parameters = np.asarray(obtained_parameters)
n_exp = obtained_parameters.shape[0]

# Pseudohbm Mean/Covariance
mean = np.loadtxt('./out-mcmc/herg25oc1-pseudohbm-lognorm-mean.txt')
with open('./out-mcmc/herg25oc1-pseudohbm-lognorm-cov.pkl', 'rb') as f:
    cov = pickle.load(f)

# Further thinning
mean = mean[::100, :]
cov = cov[::100, :, :]

# Create distribution
ppln = PosteriorPredictiveLogNormal(mean, cov)


# Do QQ-plot and PP-plot
# see https://stats.stackexchange.com/a/350545

# QQ-plot
figqq, axesqq = plt.subplots(3, 3, figsize=(10, 9))
axesqq[1, 0].set_ylabel('Sample\nquantiles', fontsize=16)
axesqq[-1, 1].set_xlabel('Theoretical\nquantiles', fontsize=16)
axesqq[0, 0].ticklabel_format(axis='both', style='sci', scilimits=(0, 1))

# PP-plot
figpp, axespp = plt.subplots(3, 3, figsize=(10, 9))
axespp[1, 0].set_ylabel('Empirical CDF', fontsize=16)
axespp[-1, 1].set_xlabel('Theoretical CDF', fontsize=16)

# PP-plot 2
figpp2, axespp2 = plt.subplots(1, 1, figsize=(7, 7))
axespp2.set_ylabel('Empirical CDF', fontsize=16)
axespp2.set_xlabel('Theoretical CDF', fontsize=16)
axespp2.set_xlim([-0.1, 1.1])
axespp2.set_ylim([-0.1, 1.1])
axespp2.tick_params(axis='both', labelsize=14)

for i_p in range(obtained_parameters.shape[1]):

    ai, aj = i_p / 3, i_p % 3
    ps = obtained_parameters[:, i_p]

    # Title
    axesqq[ai, aj].text(0.975, 0.025, param_name[i_p], fontsize=20,
                        ha='right', va='bottom',
                        transform=axesqq[ai, aj].transAxes)
    axespp[ai, aj].text(0.975, 0.025, param_name[i_p], fontsize=20,
                        ha='right', va='bottom',
                        transform=axespp[ai, aj].transAxes)

    # Sort parameters
    sortarg = np.argsort(ps)
    p_i = np.empty(n_exp, dtype=np.float)
    theoretical_q = np.empty(n_exp, dtype=np.float)
    for i, s in enumerate(sortarg):
        p_i[s] = (2. * (i + 1) - 1.) / (2. * n_exp)
        theoretical_q[s] = ppln.evaluate_marginal1d_ppf(i_p, p_i[s],
                guess=obtained_parameters[s, i_p])

    # Linear regression QQ
    m, c, r, _, _ = stats.linregress(theoretical_q, ps)
    vmin = np.min(theoretical_q)
    vmax = np.max(theoretical_q)
    vrange = vmax - vmin
    vmin = vmin - vrange * 0.1
    vmax = vmax + vrange * 0.1
    v = np.linspace(vmin, vmax, 21)

    # Plot QQ
    axesqq[ai, aj].scatter(theoretical_q, ps, marker='x')
    axesqq[ai, aj].plot(v, m * v + c, ls=':', c='C1',
            label=r'$R^2=$%.4f' % r ** 2)
    axesqq[ai, aj].plot(v, v, ls='--', c='#7f7f7f')
    axesqq[ai, aj].set_xlim([vmin, vmax])
    axesqq[ai, aj].set_ylim([vmin, vmax])
    axesqq[ai, aj].legend(loc=2, fontsize=14)

    # CDF
    x = ps[sortarg]
    ppx = ppln.evaluate_marginal1d_cdf(i_p, x)
    ppy = p_i[sortarg]

    # Linear regression PP
    m, c, r, _, _ = stats.linregress(ppx, ppy)
    v = np.linspace(-0.1, 1.1, 21)

    # Plot PP
    axespp[ai, aj].scatter(ppx, ppy, marker='x')
    axespp[ai, aj].plot(v, m * v + c, ls=':', c='C1',
            label=r'$R^2=$%.4f' % r ** 2)
    axespp[ai, aj].plot(v, v, ls='--', c='#7f7f7f')
    axespp[ai, aj].set_xlim([-0.1, 1.1])
    axespp[ai, aj].set_ylim([-0.1, 1.1])
    axespp[ai, aj].legend(loc=2, fontsize=14)
    if ai < 2:
        axespp[ai, aj].set_xticks([])
    if aj > 0:
        axespp[ai, aj].set_yticks([])

    # Plot PP 2
    axespp2.scatter(ppx, ppy, marker='x', c='C' + str(i_p),
            label=param_name[i_p] + r', $R^2=$%.4f' % r ** 2)
    axespp2.plot(v, m * v + c, ls='--', c='C' + str(i_p))

# Final adjustment
figqq.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
figpp.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
figpp2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
axespp2.legend(fontsize=14)

figqq.savefig('%s/QQ.pdf' % savedir, format='pdf')
figpp.savefig('%s/PP.pdf' % savedir, format='pdf')
figpp2.savefig('%s/PP2.pdf' % savedir, format='pdf')

plt.close('all')

