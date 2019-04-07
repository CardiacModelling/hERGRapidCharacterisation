#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import string

WELL_ID = [l+str(i).zfill(2)
           for l in string.ascii_uppercase[:16]
           for i in range(1,25)]

# Fix seed
np.random.seed(101)

savedir = './figs'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

file_dir = './out'
file_list = [
        'herg25oc1',
        ]
temperatures = np.array([25.0])
temperatures += 273.15  # in K
fit_seed = '542811797'
withfcap = False

file_name = file_list[0]
temperature = temperatures[0]

sweep1 = file_name
sweep2 = file_name + '-sweep2'

param_name = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
              r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']


#
# Create figure
#
fig, axes = plt.subplots(3, 3, figsize=(10, 9))
axes[0, 0].ticklabel_format(axis='both', style='sci', scilimits=(0, 1))
axes[1, 0].set_ylabel(r'$\Delta$Parameter', fontsize=24)
axes[-1, 1].set_xlabel(r'$\Delta E_K$', fontsize=24)


#
# Get selected cells
#
selectedfile = './manualv2selected-%s.txt' % (file_name)
selectedwell = []
with open(selectedfile, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            selectedwell.append(l.split()[0])

#
# Get all EK values
#
EK1all = np.loadtxt('../qc/herg25oc1-staircaseramp-EK_all.txt')
EK2all = np.loadtxt('../qc/herg25oc1-staircaseramp-EK_all-sweep2.txt')


#
# Fitted parameters
#
obtained_parameters1 = []
obtained_parameters2 = []
EK1 = []
EK2 = []
for cell in selectedwell:
    try:  # can remove try later; fitting was still running
        # sweep 1
        param_file1 = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
                (file_dir, sweep1, file_name, cell, fit_seed)
        ps1 = np.loadtxt(param_file1)
        # sweep 2
        param_file2 = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
                (file_dir, sweep2, file_name, cell, fit_seed)
        ps2 = np.loadtxt(param_file2)

        obtained_parameters1.append(ps1)
        obtained_parameters2.append(ps2)
        EK1.append(EK1all[WELL_ID.index(cell)])
        EK2.append(EK2all[WELL_ID.index(cell)])
    except IOError:
        continue
obtained_parameters1 = np.asarray(obtained_parameters1)
obtained_parameters2 = np.asarray(obtained_parameters2)
ekdiff = np.asarray(EK1) - np.asarray(EK2)
n_param = obtained_parameters1.shape[1]

# Some checks
assert(len(obtained_parameters1) == len(obtained_parameters2))
assert(obtained_parameters1.shape[1] == obtained_parameters2.shape[1])
assert(n_param == axes.shape[0] * axes.shape[1])
assert(len(ekdiff) == len(obtained_parameters1))


#
# Plot
#
for i_p in range(n_param):

    ai, aj = int(i_p / 3), i_p % 3

    pdiff_i = obtained_parameters1[:, i_p] - obtained_parameters2[:, i_p]

    # Title
    axes[ai, aj].text(0.975, 0.025, param_name[i_p], fontsize=20,
                      ha='right', va='bottom',
                      transform=axes[ai, aj].transAxes)

    # Linear regression QQ
    m, c, r, _, _ = stats.linregress(ekdiff, pdiff_i)
    vmin = np.min(ekdiff)
    vmax = np.max(ekdiff)
    vrange = vmax - vmin
    v = np.linspace(vmin - vrange * 0.1,
                    vmax + vrange * 0.1,
                    21)

    # Plot
    axes[ai, aj].scatter(ekdiff, pdiff_i, marker='x')
    axes[ai, aj].plot(v, m * v + c, ls='--', c='#7f7f7f',
            label=r'$R^2=$%.4f' % r ** 2)
    axes[ai, aj].legend(loc=2, fontsize=14)


#
# Final adjustment and save
#
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('%s/sweeps-comparison-ek.png' % savedir, bbox_inch='tight')
# plt.savefig('%s/sweeps-comparison-ek.pdf' % savedir, format='pdf',
#         bbox_inch='tight')
plt.close()

