#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import string

import protocols
from protocols import est_g_staircase
import model_ikr as m
from releakcorrect import I_releak, score_leak, protocol_leak_check

from scipy.optimize import fmin
import seaborn as sns

# Set seed
np.random.seed(101)

savedir = './figs/paper'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

savedirlr = './figs/paper-low-res'
if not os.path.isdir(savedirlr):
    os.makedirs(savedirlr)

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
    'sactiv': protocols.sactiv,
    'sinactiv': protocols.sinactiv,
}
protocol_dir = '../protocol-time-series'
protocol_list = [
    'staircaseramp',
    'sactiv',
    'sinactiv',
    'pharma',
    'apab',
    'apabv3',
    'ap05hz',
    'ap1hz',
    'ap2hz',
]

# IV protocol special treatment
protocol_iv = [
    'sactiv',
    'sinactiv',
]
protocol_iv_times = {
    'sactiv': protocols.sactiv_times,
    'sinactiv': protocols.sinactiv_times,
}
protocol_iv_convert = {
    'sactiv': protocols.sactiv_convert,
    'sinactiv': protocols.sinactiv_convert,
}
protocol_iv_args = {
    'sactiv': protocols.sactiv_iv_arg,
    'sinactiv': protocols.sinactiv_iv_arg,
}
protocol_iv_v = {
    'sactiv': protocols.sactiv_v,
    'sinactiv': protocols.sinactiv_v,
}

data_dir_staircase = '../data'
data_dir = '../data-autoLC'
file_dir = './out'
file_list = [
        'herg25oc',
        'herg37oc',
        ]
temperatures = np.array([25.0, 37.0])
temperatures += 273.15  # in K
fit_seed = '542811797'

add_inset = {  # prt: (figsize, (xmin, xmax), (ymin, ymax), inc_loc, mark_loc)
    'apab': ((3.5, 1), (0.05, 0.6), (-25 / 1000., 120 / 1000.), 1, (2, 4)),
    'apabv3': ((3.5, 1), (0.05, 0.7), (-25 / 1000., 70 / 1000.), 1, (2, 4)),
    'ap05hz': ((2, 1), (1.9, 2.7), (-25 / 1000., 70 / 1000.), 1, (2, 4)),
    'ap1hz': ((2, 1), (2.9, 3.7), (-25 / 1000., 120 / 1000.), 2, (3, 1)),
}

isNorm = True
norm_method = 1


#
# Do a very very tailored version........ :(
#
fig = plt.figure(figsize=(16, 15))
bigygap = 4
n_ygrid = 18
grid = plt.GridSpec(3 * n_ygrid + 2 * bigygap, 3, hspace=0.0, wspace=0.2)
axes = np.empty([9, int(len(protocol_list) / 3)], dtype=object)
# long list here:
for i in range(int(len(protocol_list)/3)):
    # First 'row'
    if i == 0:
        axes[0, i] = fig.add_subplot(grid[0:4, i])
        axes[0, i].set_xticklabels([])
        axes[1, i] = fig.add_subplot(grid[4:11, i])
        axes[2, i] = fig.add_subplot(grid[11:18, i])
    else:
        axes[0, i] = fig.add_subplot(grid[0:4, i])
        axes[1, i] = fig.add_subplot(grid[7:12, i])
        axes[2, i] = fig.add_subplot(grid[12:17, i])
    axes[1, i].set_xticklabels([])

    # Second 'row'
    n_shift = n_ygrid + bigygap
    axes[3, i] = fig.add_subplot(grid[n_shift+0:n_shift+4, i])
    axes[3, i].set_xticklabels([])
    axes[4, i] = fig.add_subplot(grid[n_shift+4:n_shift+11, i])
    axes[5, i] = fig.add_subplot(grid[n_shift+11:n_shift+18, i])
    axes[4, i].set_xticklabels([])

    # Third 'row'
    n_shift = 2 * (n_ygrid + bigygap)
    axes[6, i] = fig.add_subplot(grid[n_shift+0:n_shift+4, i])
    axes[6, i].set_xticklabels([])
    axes[7, i] = fig.add_subplot(grid[n_shift+4:n_shift+11, i])
    axes[8, i] = fig.add_subplot(grid[n_shift+11:n_shift+18, i])
    axes[7, i].set_xticklabels([])

axes[0, 0].set_ylabel('Voltage\n[mV]', fontsize=14)
axes[1, 0].set_ylabel('Current\n$T=%s^circ$C' % (temperatures[0] - 273.15), fontsize=14)
axes[2, 0].set_ylabel('Current\n$T=%s^circ$C' % (temperatures[1] - 273.15), fontsize=14)
axes[3, 0].set_ylabel('Voltage\n[mV]', fontsize=14)
axes[4, 0].set_ylabel('Current\n$T=%s^circ$C' % (temperatures[0] - 273.15), fontsize=14)
axes[5, 0].set_ylabel('Current\n$T=%s^circ$C' % (temperatures[1] - 273.15), fontsize=14)
axes[6, 0].set_ylabel('Voltage\n[mV]', fontsize=14)
axes[7, 0].set_ylabel('Current\n$T=%s^circ$C' % (temperatures[0] - 273.15), fontsize=14)
axes[8, 0].set_ylabel('Current\n$T=%s^circ$C' % (temperatures[1] - 273.15), fontsize=14)
for i in range(int(len(protocol_list) / 3)):
    axes[-1, i].set_xlabel('Time [s]', fontsize=18)

# Liudmila suggested common y-axis
for i in range(3):
    ai = 3 * i + 1
    for j in range(3):
        aj = j
        if i == 0 and j == 0:  # staircaseramp
            axes[ai, aj].set_ylim((-1, 1.5))
            axes[ai + 1, aj].set_ylim((-1, 1.5))
        else:
            axes[ai, aj].set_ylim((-0.25, 1.5))
            axes[ai + 1, aj].set_ylim((-0.25, 1.5))

# Gary suggested, add inset
axins = np.empty(axes.shape, dtype=object)
for i_prt, prt in enumerate(protocol_list):
    ai, aj = 3 * int(i_prt / 3), i_prt % 3
    if prt in add_inset.keys():
        axins[ai + 1, aj] = inset_axes(axes[ai + 1, aj], *add_inset[prt][0],
                loc=add_inset[prt][3])
        axins[ai + 1, aj].set_facecolor("#f1f1f1")
        axins[ai + 1, aj].set_xlim(add_inset[prt][1])
        axins[ai + 1, aj].set_ylim(add_inset[prt][2])
        axins[ai + 1, aj].set_xticks([])
        axins[ai + 1, aj].set_yticks([])
        mark_inset(axes[ai + 1, aj], axins[ai + 1, aj],
                loc1=add_inset[prt][4][0], loc2=add_inset[prt][4][1],
                fc="#f1f1f1", ec="0.5")

# Add special x,y-label for IV protocols
for ai in [1, 2]:
    for aj in [1, 2]:
        axes[ai, aj].text(0.98, 0.02, 'Voltage [mV]', fontsize=14, ha='right',
                va='bottom', transform=axes[ai, aj].transAxes)
# change y-label to right
for aj in [1, 2]:
    axes[1, aj].yaxis.tick_right()
    axes[2, aj].yaxis.tick_right()
    axes[1, aj].yaxis.set_label_position("right")
    axes[2, aj].yaxis.set_label_position("right")
    axes[0, aj].set_xlabel('Time [s]', fontsize=14)
for ai in [1, 2]:
    axes[ai, -1].set_ylabel('Normalised\ncurrents', fontsize=12)
    axes[ai, 1].set_ylim(-0.05, 1.05)
    axes[ai, 1].set_xlim(-50, 40)
    axes[ai, 2].set_ylim(-2, 1.05)
    axes[ai, 2].set_xlim(-100, 40)

# Set y-ticklabels for protocols
# TODO


#
# Protocol
#
file_name = file_list[0] + '1'
temperature = temperatures[0]

# Model
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

for i_prt, prt in enumerate(protocol_list):

    # Calculate axis index
    ai, aj = 3 * int(i_prt / 3), i_prt % 3

    # Title
    if prt == 'staircaseramp':
        axes[ai, aj].set_title('Calibration', fontsize=16)
    else:
        axes[ai, aj].set_title('Validation %s' % i_prt, fontsize=16)

    # Time point
    times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name,
        prt), delimiter=',', skiprows=1)

    # Protocol
    model = prt2model[prt]
    if prt not in protocol_iv:
        times_sim = np.copy(times)
        voltage = model.voltage(times_sim) * 1000
    else:
        times_sim = protocol_iv_times[prt](times[1] - times[0])
        voltage = model.voltage(times_sim) * 1000
        voltage, t = protocol_iv_convert[prt](voltage, times_sim)
        assert(np.mean(np.abs(t - times)) < 1e-8)
    axes[ai, aj].set_ylim((np.min(voltage) - 10, np.max(voltage) + 15))

    # Plot raw data
    # Add label!
    axes[ai, aj].text(-0.1, 1.1, string.ascii_uppercase[i_prt],
                      transform=axes[ai, aj].transAxes, size=20,
                      weight='bold')
    
    if prt not in protocol_iv:
        # protocol
        axes[ai, aj].plot(times, voltage, c='#696969')
    else:
        # protocol
        for i in range(voltage.shape[1]):
            axes[ai, aj].plot(times, voltage[:, i], c='#696969')


#
# All cells
#

norm_data_all = []
for i_prt, prt in enumerate(protocol_list):

    # Time point
    times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name,
        prt), delimiter=',', skiprows=1)

    # Protocol
    model = prt2model[prt]  # only use for getting voltage!
    if prt not in protocol_iv:
        times_sim = np.copy(times)
        voltage = model.voltage(times_sim) * 1000
    else:
        times_sim = protocol_iv_times[prt](times[1] - times[0])
        voltage = model.voltage(times_sim) * 1000
        voltage, t = protocol_iv_convert[prt](voltage, times_sim)
        assert(np.mean(np.abs(t - times)) < 1e-8)

    for i_T, (file_name, T) in enumerate(zip(file_list, temperatures)):

        # Calculate axis index
        ai, aj = 3 * int(i_prt / 3) + 1 + i_T, i_prt % 3

        # Get ranking
        cell_ranking_file = './manualselection/paper-rank-%s.txt' % file_name

        RANKED_CELLS = []
        with open(cell_ranking_file, 'r') as f:
            for l in f:
                if not l.startswith('#'):
                    RANKED_CELLS.append(l.split()[0])
        # RANKED_CELLS = RANKED_CELLS[:10]  # TODO remove
        # colour_list = sns.cubehelix_palette(len(SORTED_CELLS))
        colour_list = sns.color_palette('Blues', n_colors=len(RANKED_CELLS))
        colour_list = colour_list.as_hex()

        for i_CELL, CELL in enumerate(RANKED_CELLS):

            file_name, cell = CELL[:-3], CELL[-3:]

            # Data
            if prt == 'staircaseramp':
                data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir_staircase,
                    file_name, prt, cell), delimiter=',', skiprows=1)
            elif prt not in protocol_iv:
                data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
                    prt, cell), delimiter=',', skiprows=1)
                # Re-leak correct the leak corrected data...
                g_releak = fmin(score_leak, [0.0], args=(data, voltage, times,
                                    protocol_leak_check[prt]), disp=False)
                data = I_releak(g_releak[0], data, voltage)
            else:
                data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
                        prt, cell), delimiter=',', skiprows=1)
                for i in range(data.shape[1]):
                    g_releak = fmin(score_leak, [0.0], args=(data[:, i],
                                        voltage[:, i], times,
                                        protocol_leak_check[prt]), disp=False)
                    data[:, i] = I_releak(g_releak[0], data[:, i], voltage[:, i])
            assert(len(data) == len(times))

            # Normalisation
            if prt == 'staircaseramp':
                norm_data = est_g_staircase(data, times, p0=[800, 0.025],
                                            debug=False) if isNorm else 1.
                if i_CELL == 0:
                    norm_data_all.append([])
                norm_data_all[-1].append(norm_data)
            else:
                norm_data = norm_data_all[i_T][i_CELL]

            # Plot
            if prt not in protocol_iv:
                axes[ai, aj].plot(times, data / norm_data, 
                                lw=0.2, alpha=0.1,
                                c=colour_list[i_CELL])
            else:
                iv_i = protocols.get_corrected_iv(data, times,
                                                  *protocol_iv_args[prt]())
                iv_v = protocol_iv_v[prt]() * 1000  # mV
                axes[ai, aj].plot(iv_v, iv_i / np.max(iv_i), lw=0.5, alpha=0.5,
                                  c=colour_list[i_CELL])

            # Plot inset
            if i_T == 0:
                if prt in add_inset.keys():  # not in provtocol_iv
                    axins[ai, aj].plot(times, data / norm_data, 
                                    lw=0.2, alpha=0.1,
                                    c=colour_list[i_CELL])


#
# Final adjustment and save
#
grid.tight_layout(fig, pad=0.6)
grid.update(wspace=0.12, hspace=0.0)
plt.savefig('%s/fitting-and-validation-recordings-samey.png' % savedirlr,
            bbox_inch='tight', pad_inches=0, dpi=100)
plt.savefig('%s/fitting-and-validation-recordings-samey.png' % savedir,
            bbox_inch='tight', pad_inches=0, dpi=300)
# plt.savefig('%s/fitting-and-validation-recordings.pdf' % savedir,
#             format='pdf', bbox_inch='tight', pad_inches=0)

print('Done')
