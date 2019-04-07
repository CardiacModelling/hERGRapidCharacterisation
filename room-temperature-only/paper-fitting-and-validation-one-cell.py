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


#
# Do a very very tailored version........ :(
#
fig = plt.figure(figsize=(16, 7))
grid = plt.GridSpec(28, 3, hspace=0.0, wspace=0.2)
axes = np.empty([4, int(len(protocol_list)/2)], dtype=object)
# long list here:
for i in range(3):
    # First 'row'
    axes[0, i] = fig.add_subplot(grid[0:6, i]) # , sharex=axes[2, i])
    axes[0, i].set_xticklabels([])
    axes[1, i] = fig.add_subplot(grid[6:12, i]) # , sharex=axes[2, i])

    # Second 'row'
    axes[2, i] = fig.add_subplot(grid[16:22, i]) # , sharex=axes[5, i])
    axes[2, i].set_xticklabels([])
    axes[3, i] = fig.add_subplot(grid[22:28, i]) # , sharex=axes[5, i])
axes[0, 0].set_ylabel('Voltage [mV]', fontsize=14)
axes[1, 0].set_ylabel('Current [pA]', fontsize=14)
axes[2, 0].set_ylabel('Voltage [mV]', fontsize=14)
axes[3, 0].set_ylabel('Current [pA]', fontsize=14)
axes[-1, int(len(protocol_list)/2/2)].set_xlabel('Time [s]', fontsize=18)


#
# Protocol and a cell
#
file_name = file_list[0]
temperature = temperatures[0]
cell = 'D19'

# Model
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

# Fitted parameters
param_file = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
        (file_dir, file_name, file_name, cell, fit_seed)
obtained_parameters = np.loadtxt(param_file)

for i_prt, prt in enumerate(protocol_list):

    # Calculate axis index
    ai, aj = 2 * int(i_prt / 3), i_prt % 3

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
    voltage = model.voltage(times) * 1000

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
    assert(data.shape == times.shape)

    # Simulation
    simulation = model.simulate(obtained_parameters, times)

    # Plot raw data
    # Add label!
    axes[ai, aj].text(-0.1, 1.1, string.ascii_uppercase[i_prt],
                      transform=axes[ai, aj].transAxes, size=20,
                      weight='bold')
    # protocol
    axes[ai, aj].plot(times, voltage, c='#696969')
    # recording
    axes[ai + 1, aj].plot(times, data, lw=1, alpha=0.8, c='#1f77b4')
    # simulation
    if prt == 'staircaseramp':
        axes[ai + 1, aj].plot(times, simulation, label='model fit to data',
                                lw=2, c='#d62728')
    else:
        axes[ai + 1, aj].plot(times, simulation, label='model prediction',
                                lw=2, c='#d62728')


#
# Final adjustment and save
#
axes[1, 0].legend()
axes[1, 1].legend()
grid.tight_layout(fig, pad=0.6)
grid.update(wspace=0.12, hspace=0.0)
plt.savefig('%s/fitting-and-validation-%s_%s.png' % (savedir, file_name, \
            cell), bbox_inch='tight', pad_inches=0)
# plt.savefig('%s/fitting-and-validation-%s_%s.pdf' % (savedir, file_name, \
#             cell), format='pdf', bbox_inch='tight', pad_inches=0)

print('Done')
