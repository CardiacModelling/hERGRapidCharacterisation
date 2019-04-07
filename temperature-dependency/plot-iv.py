#!/usr/bin/env python3
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


file_list_tmp = { # TODO
        'herg25oc':'herg25oc1',
        'herg27oc':'herg27oc1',
        'herg30oc':'herg30oc1',
        'herg33oc':'herg33oc1',
        'herg37oc':'herg37oc3',
        }

file_name = file_list_tmp[file_id]


# Get selected cells
selectedfile = './manualselection/manualv2selected-%s.txt' % (file_name)
selectedwell = []
with open(selectedfile, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            selectedwell.append(l.split()[0])


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
    'sactiv',
    'sinactiv',
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

data_dir = '../data-autoLC'
file_dir = './out'
temperatures = {
        'herg25oc': 25.0,
        'herg27oc': 27.0,
        'herg30oc': 30.0,
        'herg33oc': 33.0,
        'herg37oc': 37.0,
        }
temperature = temperatures[file_id]
temperature += 273.15  # in K
fit_seed = '542811797'


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


for i_prt, prt in enumerate(protocol_list):

    savedir = './figs/rmsd-hist-%s-autoLC-releak/%s-plots' % (file_name, prt)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    for cell in selectedwell:
        
        # Time points
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

        # Plot
        if prt not in protocol_iv:
            # recording
            plt.plot(times, data, lw=1, alpha=0.8, c='#1f77b4',
                    label='data')
        else:
            iv_v = protocol_iv_v[prt]() * 1000  # mV
            # recording
            iv_i = protocols.get_corrected_iv(data, times,
                    *protocol_iv_args[prt]())
            plt.plot(iv_v, iv_i / np.max(iv_i), lw=2, alpha=1,
                    c='#1f77b4', label='data')

        plt.xlabel('V [mV]')
        plt.ylabel('Normalised Current')

        plt.savefig('%s/%s-%s' % (savedir, prt, cell))
        plt.close()


