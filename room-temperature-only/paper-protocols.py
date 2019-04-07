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

savedir = './figs/paper/protocols'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Protocol info
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

# For loading time point data
data_dir = '../data'
data_dir_2 = '../data-autoLC'
file_name = 'herg25oc1'

# Plot setting
figsizes = {
    'staircaseramp': (16, 3),
    'pharma': (10, 3),
    'apab': (10, 3),
    'apabv3': (10, 3),
    'ap05hz': (10, 3),
    'ap1hz': (10, 3),
    'ap2hz': (10, 3),
    'sactiv': (10, 3),
    'sinactiv': (10, 3),
}

colours = {
    'staircaseramp': '#2ca02c',
    'pharma': '#1f77b4',
    'apab': '#1f77b4',
    'apabv3': '#1f77b4',
    'ap05hz': '#1f77b4',
    'ap1hz': '#1f77b4',
    'ap2hz': '#1f77b4',
    'sactiv': '#1f77b4',
    'sinactiv': '#1f77b4',
}

xyticks = {
    'staircaseramp': (range(0, 15, 2), [-120, -80, -40, 0, 40]),
    'pharma': None,
    'apab': None,
    'apabv3': None,
    'ap05hz': None,
    'ap1hz': None,
    'ap2hz': None,
    'sactiv': None,
    'sinactiv': None,
}

for i_prt, prt in enumerate(protocol_list):

    # Model
    protocol_def = protocol_funcs[prt]
    if type(protocol_def) is str:
        protocol_def = '%s/%s' % (protocol_dir, protocol_def)

    model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                    protocol_def=protocol_def,
                    temperature=25.0 + 273.15,  # K
                    transform=None,
                    useFilterCap=False)  # ignore capacitive spike

    # Time point
    try:
        times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name,
                prt), delimiter=',', skiprows=1)
    except IOError:
        times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir_2, file_name,
                prt), delimiter=',', skiprows=1)

    # Protocol
    if prt not in protocol_iv:
        times_sim = np.copy(times)
        voltage = model.voltage(times_sim) * 1000
    else:
        times_sim = protocol_iv_times[prt](times[1] - times[0])
        voltage = model.voltage(times_sim) * 1000
        voltage, t = protocol_iv_convert[prt](voltage, times_sim)
        assert(np.mean(np.abs(t - times)) < 1e-8)

    # Plot
    fig = plt.figure(figsize=figsizes[prt])
    plt.plot(times, voltage, lw=4, c=colours[prt])
    plt.ylim((np.min(voltage) - 10, np.max(voltage) + 15))
    plt.xlim(times[0], times[-1])
    if xyticks[prt] is None:
        plt.xticks([])
        plt.yticks([])
    else:
        plt.yticks(xyticks[prt][1], fontsize=24)
        plt.xticks(xyticks[prt][0], fontsize=24)
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.tight_layout(pad=0.001, w_pad=0.001, h_pad=0.001)
    plt.savefig('%s/protocol-%s.png' % (savedir, prt), transparent=True)

print('Done')
