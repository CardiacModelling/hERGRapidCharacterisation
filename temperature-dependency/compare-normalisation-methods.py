#!/usr/bin/env python3
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import protocols
from protocols import est_g_staircase
import model_ikr as m
from releakcorrect import I_releak, score_leak, protocol_leak_check

from scipy.optimize import fmin

savedir = './figs'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_dir_staircase = '../data'
data_dir = '../data-autoLC'
file_dir = './out'
file_list = [
        'herg25oc1',
        'herg27oc1',
        'herg30oc1',
        'herg33oc1',
        'herg37oc3',
        ]
temperatures = np.array([25.0, 27.0, 30.0, 33.0, 37.0])
temperatures += 273.15  # in K
fit_seed = 542811797

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
    #'sactiv',
    #'sinactiv',
    'pharma',
    'apab',
    'apabv3',
    'ap05hz',
    'ap1hz',
    'ap2hz',
]
prt_ylim = { # no normalisation
    'staircaseramp': (-1500, 2250),
    'sactiv': (-0.025, 1.025),
    'sinactiv': (-3.25, 1.025),
    'pharma': (-250, 2250),
    'apab': (-250, 2250),
    'apabv3': (-250, 2250),
    'ap05hz': (-250, 2250),
    'ap1hz': (-250, 2250),
    'ap2hz':(-250, 2250),
    }
prt_ylim = { # normalise with fitted conductance value
    'staircaseramp': (-0.02, 0.04),
    'sactiv': (-0.025, 1.025),
    'sinactiv': (-3.25, 1.025),
    'pharma': (-0.005, 0.04),
    'apab': (-0.005, 0.04),
    'apabv3': (-0.005, 0.04),
    'ap05hz': (-0.005, 0.04),
    'ap1hz': (-0.005, 0.04),
    'ap2hz': (-0.005, 0.04),
    }
prt_ylim = { # normalise with extrapolated -120 mV spike value
    'staircaseramp': (-1.0, 1.5),
    'sactiv': (-0.025, 1.025),
    'sinactiv': (-3.25, 1.025),
    'pharma': (-0.25, 1.5),
    'apab': (-0.25, 1.5),
    'apabv3': (-0.25, 1.5),
    'ap05hz': (-0.25, 1.5),
    'ap1hz': (-0.25, 1.5),
    'ap2hz': (-0.25, 1.5),
    }

zoomin = {
    'staircaseramp': [(1.8, 2.5), (14.3, 15.0)],
    'pharma': [(0.64, 0.66), (1.14, 1.16)],
    'apab': [(0.0475, 0.0575), (0.32, 0.33)],
    'apabv3': [(0.05, 0.07), (0.55, 0.70)],
    'ap05hz': [(0.04, 0.07), (2.04, 2.07)],
    'ap1hz': [(0.04, 0.07),
              # (1.04, 1.07),
              # (2.04, 2.07), 
              (3.04, 3.07)],
    'ap2hz': [(0.045, 0.06),
              # (0.545, 0.56),
              # (1.045, 1.06), (1.545, 1.56),
              # (2.045, 2.06), (2.545, 2.56),
              (3.045, 3.06)],
    'sactiv': None,
    'sinactiv': None,
    }

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


fig, axes = plt.subplots(len(protocol_list), 1, sharex=True, figsize=(4, 10))

norm_data_2_all = []
for i_prt, prt in enumerate(protocol_list):
    
    print('Plotting', prt)

    # Time point
    times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, 'herg25oc1',
        prt), delimiter=',', skiprows=1)

    norm_1_peak = []
    norm_2_peak = []

    # Protocol
    model = prt2model[prt]
    if prt not in protocol_iv:
        times_sim = np.copy(times)
        voltage = model.voltage(times) * 1000
    else:
        times_sim = protocol_iv_times[prt](times[1] - times[0])
        voltage = model.voltage(times_sim) * 1000
        voltage, t = protocol_iv_convert[prt](voltage, times_sim)
        assert(np.mean(np.abs(t - times)) < 1e-8)

    # Temperatures
    for i_T, T in enumerate(temperatures):

        file_name = file_list[i_T]

        selectedfile = './manualselection/manualselected-%s.txt' % (file_name)
        selectedwell = []
        with open(selectedfile, 'r') as f:
            for l in f:
                if not l.startswith('#'):
                    selectedwell.append(l.split()[0])
        selectedwell = selectedwell[:50]  # TODO remove
        print('Getting', file_name)

        for i_cell, cell in enumerate(selectedwell):
            # Data
            if prt == 'staircaseramp':
                data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir_staircase,
                        file_name, prt, cell), delimiter=',', skiprows=1)
            elif prt not in protocol_iv:
                data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
                        prt, cell), delimiter=',', skiprows=1)
                # Set seed
                np.random.seed(101)
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
                    data[:, i] = I_releak(g_releak[0], data[:, i],
                            voltage[:, i])
            assert(len(data) == len(times))

            # Fitted parameters
            param_file = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
                    (file_dir, file_name, file_name, cell, fit_seed)
            parameters = np.loadtxt(param_file)

            # Normalisation constant for data
            norm_data_1 = parameters[0]

            if prt == 'staircaseramp':
                norm_data_2 = est_g_staircase(data, times, p0=[800, 0.025],
                                            debug=False)
                if i_cell == 0:
                    norm_data_2_all.append([])
                norm_data_2_all[-1].append(norm_data_2)
            else:
                norm_data_2 = norm_data_2_all[i_T][i_cell]

            if i_cell == 0:
                norm_1_peak.append([])
                norm_2_peak.append([])
            norm_1_peak[-1].append(
                    np.percentile(data / norm_data_1, 99.9))
            norm_2_peak[-1].append(
                    np.percentile(data / norm_data_2, 99.9))

    mean_norm_1_peak = np.mean(norm_1_peak, axis=1)
    mean_norm_2_peak = np.mean(norm_2_peak, axis=1)
    err_norm_1_peak = np.std(norm_1_peak, axis=1)
    err_norm_2_peak = np.std(norm_2_peak, axis=1)

    axes[i_prt].errorbar(temperatures,
            mean_norm_1_peak / np.mean(mean_norm_1_peak),
            yerr=err_norm_1_peak / np.mean(mean_norm_1_peak), marker='o',
            label='norm with fitted g')
    axes[i_prt].errorbar(temperatures,
            mean_norm_2_peak / np.mean(mean_norm_2_peak),
            yerr=err_norm_2_peak / np.mean(mean_norm_2_peak), marker='o',
            label='norm with staircase spike')
    axes[i_prt].set_title(prt)

axes[0].legend()
axes[-1].set_xlabel('Temperature [K]')
axes[3].set_ylabel('99.9%ile peak')

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('./figs/compare-normalisation-method', bbox_iches='tight')
plt.close()
