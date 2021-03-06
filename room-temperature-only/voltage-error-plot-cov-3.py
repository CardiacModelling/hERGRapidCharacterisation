#!/usr/bin/env python2
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import pickle

import sys
sys.path.append('../lib')
import plot_hbm_func as plot_func

saveas = 'figs/voltageoffset/'
n_non_model_param = 1
param_name = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
              r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']
plotKylie = False

if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))

# Mean/Covariance
simple_chain_final = np.loadtxt(
        './out-mcmc/herg25oc1-pseudohbm-lognorm-mean.txt')
with open('./out-mcmc/herg25oc1-pseudohbm-lognorm-cov.pkl', 'rb') as f:
    simple_cov_final = pickle.load(f)
simple_chain_final = simple_chain_final[::250]
simple_cov_final = simple_cov_final[::250]


# Load exp param
param_exp = []
path_to_exp = './out/herg25oc1-mcmcmean'
files_exp = glob.glob(path_to_exp + '/*.txt')
for file_exp in files_exp:
    p = np.loadtxt(file_exp)
    param_exp.append(p)
param_exp = np.array(param_exp)

# Load Kylie's param
param_kylie = []
path_to_kylies = './kylie-room-temperature'
files_kylie = glob.glob(path_to_kylies + '/*')
for file_kylie in files_kylie:
    p = np.loadtxt(file_kylie)
    # Change conductance unit nS->pS (new parameter use V, but here mV)
    p[0] = p[0] * 1e3
    param_kylie.append(p)
param_kylie = np.array(param_kylie)

# Load syn param from voltage-artefact
param_syn = []
path_to_syn = './out/herg25oc1-fakedata-voltageoffset'
files_syn = glob.glob(path_to_syn + '/*.txt')
for file_syn in files_syn:
    p = np.loadtxt(file_syn)
    param_syn.append(p)
param_syn = np.array(param_syn)

# Some checks and def var
assert(param_syn.shape[1] == param_exp.shape[1])
assert(param_kylie.shape[1] == param_exp.shape[1])
n_param = param_exp.shape[1]

# Change things to log
param_exp = np.log(param_exp)
param_kylie = np.log(param_kylie)
param_syn = np.log(param_syn)

# Plot the params!
fig_size = (3 * n_param, 3 * n_param)
# fig_size = (12, 12)
fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)

for i in range(n_param):
    for j in range(n_param):
        if i == j:
            # Diagonal: no plot
            # axes[i, j].axis('off')
            axes[i, j].set_xticklabels([])
            axes[i, j].set_yticklabels([])
            axes[i, j].tick_params(axis='both', which='both', bottom=False,
                                   top=False, left=False, right=False, 
                                   labelleft=False, labelbottom=False)

        elif i < j:
            # Top-right: no plot
            axes[i, j].axis('off')

        else:
            # Lower-left: plot scatters
            px_e = param_exp[:, j]
            py_e = param_exp[:, i]
            axes[i, j].scatter(px_e, py_e, c='#7f7f7f', alpha=0.5,
                    label='Data')

            px_s = param_syn[:, j]
            py_s = param_syn[:, i]
            axes[i, j].scatter(px_s, py_s, c='#d62728',
                    label='Syn. voltage offset')

            xmin = min(np.min(px_e), np.min(px_s))
            xmax = max(np.max(px_e), np.max(px_s))
            ymin = min(np.min(py_e), np.min(py_s))
            ymax = max(np.max(py_e), np.max(py_s))

            if plotKylie:
                px_k = param_kylie[:, j]
                py_k = param_kylie[:, i]
                axes[i, j].scatter(px_k, py_k, c='k',
                                   label='Beattie et al. 2018')
                xmin = min(xmin, np.min(px_k))
                xmax = max(xmax, np.max(px_k))
                ymin = min(ymin, np.min(py_k))
                ymax = max(ymax, np.max(py_k))

            # 2 sigma covers up 95.5%
            xmin = min(xmin, np.min(simple_chain_final[:, j]) \
                   - 2.5 * np.max(np.sqrt(simple_cov_final[:, j, j])))
            xmax = max(xmax, np.max(simple_chain_final[:, j]) \
                   + 2.5 * np.max(np.sqrt(simple_cov_final[:, j, j])))
            ymin = min(ymin, np.min(simple_chain_final[:, i]) \
                   - 2.5 * np.max(np.sqrt(simple_cov_final[:, i, i])))
            ymax = max(ymax, np.max(simple_chain_final[:, i]) \
                   + 2.5 * np.max(np.sqrt(simple_cov_final[:, i, i])))

            axes[i, j].set_xlim(xmin, xmax)
            axes[i, j].set_ylim(ymin, ymax)
            
            for ims, (m, s) in enumerate(zip(simple_chain_final,
                                           simple_cov_final)):
                # for xj, yi
                mu = np.array([m[j], m[i]])
                cov = np.array([[ s[j, j], s[j, i] ], 
                                [ s[i, j], s[i, i] ]])
                xx, yy = plot_func.plot_cov_ellipse(mu, cov)
                if ims == 0:
                    axes[i, j].plot(xx, yy, c='#1f77b4', alpha=0.2)
                else:
                    axes[i, j].plot(xx, yy, c='#1f77b4', alpha=0.2)


        # Set tick labels
        if i < n_param - 1 and i >= j:
            # Only show x tick labels for the last row
            axes[i, j].set_xticklabels([])
        if j > 0 and i >= j:
            # Only show y tick labels for the first column
            axes[i, j].set_yticklabels([])

    # Set axis labels and ticks
    if i > 0:
        axes[i, 0].set_ylabel(param_name[i], fontsize=32)
        axes[i, 0].tick_params('y', labelsize=26)
    if i < n_param - 1:
        axes[-1, i].set_xlabel(param_name[i], fontsize=32)
        axes[-1, i].tick_params('x', labelsize=26, rotation=30)


axes[1, 0].legend(fontsize=32, loc="lower left", bbox_to_anchor=(1.15, 1.15),
                  bbox_transform=axes[1, 0].transAxes)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('%scov-plot-3.png' % saveas, bbox_inch='tight', dpi=300)


# Add boxes for Michael
import sys
sys.path.append('../lib')
import plot_hbm_func as plot_func
for i in range(1, n_param):
    plot_func.addbox(axes, (i, 0), color='#d9d9d9', alpha=0.75)
for i in range(1, 5):
    for j in range(1, 5):
        if i > j:
            plot_func.addbox(axes, (i, j), color='#fdb462', alpha=0.35)
# Maybe 3 colours
for i in range(5, n_param):
    for j in range(5, n_param):
        if i > j:
            plot_func.addbox(axes, (i, j), color='#ccebc5', alpha=0.75)

plt.savefig('%scov-plot-3-cbox2.png' % saveas, bbox_inch='tight', dpi=100)
# plt.savefig('%scov-plot-3-cbox2.pdf' % saveas, format='pdf',
#         bbox_inch='tight')

plt.close()
