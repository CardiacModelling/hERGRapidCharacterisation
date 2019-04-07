#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import string

import protocols
import model_ikr as m
from releakcorrect import I_releak, score_leak, protocol_leak_check

from scipy.optimize import fmin
# Set seed
np.random.seed(101)

savedir = './figs/paper'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

cell_ranking_file = './paper-rank-cells.txt'

#
# Protocol info
#
protocol_funcs = {
    'staircaseramp': protocols.leak_staircase,
    'pharma': protocols.pharma,  # during drug application
    'apab': 'protocol-apab.csv',
    'apabv3': 'protocol-apabv3.csv',
    'ap05hz': 'protocol-ap05hz.csv',
    'ap1hz': 'protocol-ap1hz.csv',
    'ap2hz': 'protocol-ap2hz.csv',
    'sactiv': None,
    'sinactiv': None,
}
protocol_dir = '../protocol-time-series'
protocol_list = [
        'staircaseramp',
        'pharma',
        'apab',
        'apabv3',
        'ap1hz',
        'ap2hz',
        ]

data_dir_staircase = '../data'
data_dir = '../data-autoLC'
file_dir = './out'
file_list = [
        'herg25oc1',
        ]
temperatures = np.array([25.0])
temperatures += 273.15  # in K
fit_seed = '542811797'

isNorm = True
norm_method = 1
isSmooth = True
smooth_win = 51  # seems okay


#
# Do a very very tailored version........ :(
#
fig = plt.figure(figsize=(16, 7))
grid = plt.GridSpec(28, 3, hspace=0.0, wspace=0.2)
axes = np.empty([4, int(len(protocol_list) / 2)], dtype=object)
# long list here:
for i in range(int(len(protocol_list) / 2)):
    # First 'row'
    axes[0, i] = fig.add_subplot(grid[0:6, i]) # , sharex=axes[2, i])
    axes[0, i].set_xticklabels([])
    axes[1, i] = fig.add_subplot(grid[6:12, i]) # , sharex=axes[2, i])

    # Second 'row'
    axes[2, i] = fig.add_subplot(grid[16:22, i]) # , sharex=axes[5, i])
    axes[2, i].set_xticklabels([])
    axes[3, i] = fig.add_subplot(grid[22:28, i]) # , sharex=axes[5, i])
axes[0, 0].set_ylabel('Model', fontsize=14)
axes[1, 0].set_ylabel('Data', fontsize=14)
axes[2, 0].set_ylabel('Model', fontsize=14)
axes[3, 0].set_ylabel('Data', fontsize=14)
axes[-1, int(len(protocol_list) / 2 / 2)].set_xlabel('Time [s]', fontsize=18)
axes[1, 0].text(-0.2, -0.25, 'Normalised current', rotation=90, fontsize=18,
                transform=axes[1, 0].transAxes,
                horizontalalignment='center',
                verticalalignment='center')


#
# Model
#
prt2model = {}
for prt in protocol_list:

    protocol_def = protocol_funcs[prt]
    if type(protocol_def) is str:
        protocol_def = '%s/%s' % (protocol_dir, protocol_def)

    prt2model[prt] = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                        protocol_def=protocol_def,
                        temperature=temperatures[0],  # K
                        transform=None,
                        useFilterCap=False)  # ignore capacitive spike


#
# All cells
#

# Get ranking
RANKED_CELLS = []
with open(cell_ranking_file, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            RANKED_CELLS.append(l.split()[0])

import seaborn as sns
# colour_list = sns.cubehelix_palette(len(SORTED_CELLS))
colour_list = sns.color_palette('GnBu_d', n_colors=len(RANKED_CELLS))
colour_list.as_hex()

for i_prt, prt in enumerate(protocol_list):

    # Calculate axis index
    ai, aj = 2 * int(i_prt / 3), i_prt % 3

    # Title
    if prt == 'staircaseramp':
        axes[ai, aj].set_title('Calibration', fontsize=16)
    else:
        axes[ai, aj].set_title('Validation %s' % i_prt, fontsize=16)

    # Time point
    times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_list[0],
        prt), delimiter=',', skiprows=1)

    # Protocol
    model = prt2model[prt]
    voltage = model.voltage(times) * 1000

    # Pre-load a reference trace to do normalisation
    if norm_method == 1 and isNorm:
        ref_file = 'herg25oc1'
        ref_cell = 'D19'
        if prt == 'staircaseramp':
            ref_data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir_staircase,
                ref_file, prt, ref_cell), delimiter=',', skiprows=1)
        else:
            ref_data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, ref_file,
                prt, ref_cell), delimiter=',', skiprows=1)
            # Re-leak correct the leak corrected data...
            g_releak = fmin(score_leak, [0.0], args=(ref_data, voltage, times,
                                protocol_leak_check[prt]), disp=False)
            ref_data = I_releak(g_releak[0], ref_data, voltage)
        assert(ref_data.shape == times.shape)

        # Set axes limit for normalisation
        maximum = np.percentile(ref_data, 99.5)
        minimum = np.percentile(ref_data, 0.5)
        maximum += 0.25 * np.abs(maximum)
        minimum -= 0.5 * np.abs(minimum)
        axes[ai, aj].set_ylim([minimum, maximum])
        axes[ai + 1, aj].set_ylim([minimum, maximum])

    for i_CELL, CELL in enumerate(RANKED_CELLS):

        file_name, cell = CELL[:-3], CELL[-3:]

        # Data
        if prt == 'staircaseramp':
            data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir_staircase,
                file_name, prt, cell), delimiter=',', skiprows=1)
        else:
            data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
                prt, cell), delimiter=',', skiprows=1)
            # Re-leak correct the leak corrected data...
            g_releak = fmin(score_leak, [0.0], args=(data, voltage, times,
                                protocol_leak_check[prt]), disp=False)
            data = I_releak(g_releak[0], data, voltage)
        if isSmooth:
            from scipy.signal import savgol_filter
            data = savgol_filter(data, window_length=smooth_win, polyorder=3)
        assert(data.shape == times.shape)

        # Fitted parameters
        param_file = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
                (file_dir, file_name, file_name, cell, fit_seed)
        obtained_parameters = np.loadtxt(param_file)

        # Simulation
        simulation = model.simulate(obtained_parameters, times)

        # Normalisation
        if norm_method == 0:
            # just pick the max (susceptible to nise)
            norm_d = np.max(data) if isNorm else 1.
            norm_s = np.max(simulation) if isNorm else 1.
        elif norm_method == 1:
            # Kylie's method, use a reference trace 
            # (should give the most similar plots)
            from scipy.optimize import minimize
            res_d = minimize(lambda x: np.sum(np.abs(data / x
                                                     - ref_data)), x0=1.0)
            norm_d = res_d.x[0] if isNorm else 1.
            res_s = minimize(lambda x: np.sum(np.abs(simulation / x
                                                     - ref_data)), x0=norm_d)
            norm_s = res_s.x[0] if isNorm else 1.
        elif norm_method == 2:
            # use 95th percentile (less susceptible to nise)
            norm_d = np.percentile(data, 95) if isNorm else 1.
            norm_s = np.percentile(simulation, 95) if isNorm else 1.
        else:
            raise ValueError('Unknown normalisation method, choose' +
                             ' norm_method from 0-2')

        # Plot
        axes[ai, aj].plot(times, simulation / norm_s,
                          lw=0.5, alpha=0.5,
                          c=colour_list[len(RANKED_CELLS) - i_CELL - 1])
        axes[ai + 1, aj].plot(times, data / norm_d,
                              lw=0.5, alpha=0.5,
                              c=colour_list[len(RANKED_CELLS) - i_CELL - 1])


#
# Final adjustment and save
#
grid.tight_layout(fig, pad=1.0, rect=(0.02, 0, 1, 1))
grid.update(wspace=0.12, hspace=0.0)
plt.savefig('%s/fitting-and-validation-selected-cells.png' % savedir,
            bbox_inch='tight', pad_inches=0)
# plt.savefig('%s/fitting-and-validation-selected-cells.pdf' % savedir,
#             format='pdf', bbox_inch='tight', pad_inches=0)

print('Done')
