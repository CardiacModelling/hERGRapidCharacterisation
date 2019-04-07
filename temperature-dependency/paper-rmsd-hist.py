#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import string

import protocols
import model_ikr as m
from releakcorrect import I_releak, score_leak, protocol_leak_check

from scipy.optimize import fmin
# Set seed
np.random.seed(101)

try:
    file_id = sys.argv[1]
except IndexError:
    print('Usage: python %s [file_id]' % __file__)
    sys.exit()

savedir = './figs/paper'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

refcell = 'D13'
plot_ref = False


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
    # 'sactiv',
    # 'sinactiv',
    'pharma',
    'apab',
    'apabv3',
    # 'ap05hz',
    'ap1hz',
    'ap2hz',
]
validation_idx = [
    None,
    # 1,
    # 2,
    3,
    4,
    5,
    # 6,
    7,
    8,
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
file_list_tmp = { # TODO
        'herg25oc': 'herg25oc1',
        'herg27oc': 'herg27oc1',
        'herg30oc': 'herg30oc1',
        'herg33oc': 'herg33oc1',
        'herg37oc': 'herg37oc3',
        }
temperatures = {
        'herg25oc': 25.0,
        'herg27oc': 27.0,
        'herg30oc': 30.0,
        'herg33oc': 33.0,
        'herg37oc': 37.0,
        }
fit_seed = '542811797'

file_name = file_list_tmp[file_id]
temperature = temperatures[file_id] + 273.15  # in K

# Load RMSD matrix
rmsd_matrix_file = './figs/rmsd-hist-%s-autoLC-releak/rmsd-matrix.txt' \
                   % file_name
rmsd_cells_file = './figs/rmsd-hist-%s-autoLC-releak/rmsd-matrix-cells.txt' \
                  % file_name

rmsd_matrix = np.loadtxt(rmsd_matrix_file)

with open(rmsd_matrix_file, 'r') as f:
    rmsd_prt = f.readline().strip('\n').strip('#').split()

rmsd_cells = []
with open(rmsd_cells_file, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            rmsd_cells.append(l.strip('\n').split('-')[1])

rankedcolours = ['#1a9850',
                 '#fc8d59',
                 '#d73027',
                 '#7f7f7f']
rankedlabels = [r'$*$',
                u'\u2021',
                r'#',
                u'\u2666']


#
# Do a very very tailored version........ :(
#
fig = plt.figure(figsize=(16, 15))
bigxgap = 12
n_xgrid = 84
bigygap = 5
n_ygrid = 31
grid = plt.GridSpec(2 * n_ygrid + 1 * bigygap, 3 * n_xgrid + 2 * bigxgap,
                    hspace=0.0, wspace=0.0)
axes = np.empty([10, int(len(protocol_list) / 2)], dtype=object)
# long list here:
for i in range(int(len(protocol_list) / 2)):
    i_grid = i * (n_xgrid + bigxgap)
    f_grid = (i + 1) * n_xgrid + i * bigxgap

    # First 'row'
    axes[0, i] = fig.add_subplot(grid[0:3, i_grid:f_grid])
    axes[0, i].set_xticklabels([])
    axes[1, i] = fig.add_subplot(grid[3:9, i_grid:f_grid])
    axes[1, i].set_xticklabels([])
    axes[2, i] = fig.add_subplot(grid[9:15, i_grid:f_grid])
    axes[2, i].set_xticklabels([])
    axes[3, i] = fig.add_subplot(grid[15:21, i_grid:f_grid])
    # Histogram
    axes[4, i] = fig.add_subplot(grid[24:31, i_grid:f_grid])

    # Second 'row'
    n_shift = n_ygrid + bigygap
    axes[5, i] = fig.add_subplot(grid[n_shift+0:n_shift+3, i_grid:f_grid])
    axes[5, i].set_xticklabels([])
    axes[6, i] = fig.add_subplot(grid[n_shift+3:n_shift+9, i_grid:f_grid])
    axes[6, i].set_xticklabels([])
    axes[7, i] = fig.add_subplot(grid[n_shift+9:n_shift+15, i_grid:f_grid])
    axes[7, i].set_xticklabels([])
    axes[8, i] = fig.add_subplot(grid[n_shift+15:n_shift+21, i_grid:f_grid])
    # Histogram
    axes[9, i] = fig.add_subplot(grid[n_shift+24:n_shift+31, i_grid:f_grid])

    # Set x-labels
    axes[3, i].set_xlabel('Time [s]', fontsize=14)
    axes[4, i].set_xlabel('RRMSE', fontsize=14)
    axes[8, i].set_xlabel('Time [s]', fontsize=14)
    axes[9, i].set_xlabel('RRMSE', fontsize=14)

# Set labels
axes[0, 0].set_ylabel('Voltage\n[mV]', fontsize=14)
axes[1, 0].set_ylabel(r'Best ($*$)', fontsize=14, color=rankedcolours[0])
axes[2, 0].set_ylabel(u'Median (\u2021)', fontsize=14, color=rankedcolours[1])
axes[3, 0].set_ylabel(r'90%ile (#)', fontsize=14, color=rankedcolours[2])
axes[4, 0].set_ylabel('Frequency\n[N=%s]' % len(rmsd_cells), fontsize=14)
axes[5, 0].set_ylabel('Voltage\n[mV]', fontsize=14)
axes[6, 0].set_ylabel(r'Best ($*$)', fontsize=14, color=rankedcolours[0])
axes[7, 0].set_ylabel(u'Median (\u2021)', fontsize=14, color=rankedcolours[1])
axes[8, 0].set_ylabel(r'90%ile (#)', fontsize=14, color=rankedcolours[2])
axes[9, 0].set_ylabel('Frequency\n[N=%s]' % len(rmsd_cells), fontsize=14)

axes[2, 0].text(-0.25, 0.5, 'Current [pA]', rotation=90, fontsize=18,
        transform=axes[2, 0].transAxes, ha='center', va='center')
axes[7, 0].text(-0.25, 0.5, 'Current [pA]', rotation=90, fontsize=18,
        transform=axes[7, 0].transAxes, ha='center', va='center')

# Set y-ticklabels for protocols
# TODO


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
                        temperature=temperature,  # K
                        transform=None,
                        useFilterCap=False)  # ignore capacitive spike


#
# Plot
#
for i_prt, prt in enumerate(protocol_list):

    # Calculate axis index
    ai, aj = 5 * int(i_prt / 3), i_prt % 3

    # Title
    if prt == 'staircaseramp':
        axes[ai, aj].set_title('Calibration', fontsize=16)
    else:
        axes[ai, aj].set_title('Validation %s' % validation_idx[i_prt],
                fontsize=16)

    # Add label!
    axes[ai, aj].text(-0.1, 1.4, string.ascii_uppercase[i_prt],
                      transform=axes[ai, aj].transAxes, size=20,
                      weight='bold')

    # Time point
    if prt == 'staircaseramp':
        times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir_staircase,
                file_name, prt), delimiter=',', skiprows=1)
    else:
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

    # Plot protocol
    if prt not in protocol_iv:
        axes[ai, aj].plot(times, voltage, c='#696969')
    else:
        # protocol
        for i in range(voltage.shape[1]):
            axes[ai, aj].plot(times, voltage[:, i], c='#696969')

    # Calculate ranking
    rmsd = rmsd_matrix[:, rmsd_prt.index(prt)]
    best_cell = np.argmin(rmsd)
    median_cell = np.argsort(rmsd)[len(rmsd)//2]
    p90_cell = np.argsort(rmsd)[int(len(rmsd)*0.9)]
    rankedcells = [rmsd_cells[best_cell],
                   rmsd_cells[median_cell],
                   rmsd_cells[p90_cell]]
    rankedvalues = [rmsd[best_cell],
                    rmsd[median_cell],
                    rmsd[p90_cell],]
    if plot_ref:
        rankedvalues.append(rmsd[rmsd_cells.index(refcell)])

    for i_cell, cell in enumerate(rankedcells):
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

        # Fitted parameters
        param_file = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
                (file_dir, file_name, file_name, cell, fit_seed)
        obtained_parameters = np.loadtxt(param_file)

        # Simulation
        simulation = model.simulate(obtained_parameters, times_sim)
        if prt in protocol_iv:
            simulation, t = protocol_iv_convert[prt](simulation, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-8)

        # Plot
        if prt not in protocol_iv:
            # recording
            axes[ai + i_cell + 1, aj].plot(times, data, lw=1, alpha=0.8,
                    c='#1f77b4', label='data')
            # simulation
            if prt == 'staircaseramp':
                axes[ai + i_cell + 1, aj].plot(times, simulation, lw=2,
                        c='#d62728', label='model fit to data')
            else:
                axes[ai + i_cell + 1, aj].plot(times, simulation, lw=2,
                        c='#d62728', label='model prediction')
        else:
            iv_v = protocol_iv_v[prt]() * 1000  # V -> mV
            # recording
            iv_i = protocols.get_corrected_iv(data, times,
                                              *protocol_iv_args[prt]())
            axes[ai + i_cell + 1, aj].plot(iv_v, iv_i / np.max(iv_i), lw=2,
                    alpha=1, c='#1f77b4', label='data')
            # simulation
            iv_i = protocols.get_corrected_iv(simulation, times,
                                              *protocol_iv_args[prt]())
            axes[ai + i_cell + 1, aj].plot(iv_v, iv_i / np.max(iv_i), lw=2,
                    alpha=1, c='#d62728', label='model prediction')
    
    # Plot rmsd histogram
    n, b, _ = axes[ai + 4, aj].hist(rmsd, bins=15, color='#9e9ac8')

    # Add labels
    rankedidx = []
    for i, v in enumerate(rankedvalues):
        idx = np.where(b <= v)[0][-1]
        if idx in rankedidx:
            print('Ref. marker might clash with other markers...')
        axes[ai + 4, aj].text(
                (b[idx] + b[idx + 1]) / 2.,
                n[idx] + 0.12 * np.max(n),
                rankedlabels[i], fontsize=16, color=rankedcolours[i],
                ha='center', va='center')
        if n[idx] == np.max(n):
            axes[ai + 4, aj].set_ylim([0, n[idx] + 6])
        rankedidx.append(idx)

    axes[ai + 4, aj].set_ylim(0, 1.25 * np.max(n))


#
# Final adjustment and save
#
axes[1, 0].legend()
axes[1, 1].legend()
grid.tight_layout(fig, pad=0.6, rect=(0.02, 0.0, 1, 0.99))
grid.update(wspace=0.2, hspace=0.0)
plt.savefig('%s/rmsd-hist-%s.png' % (savedir, file_id), bbox_inch='tight',
        pad_inches=0, dpi=300)

print('Done')
