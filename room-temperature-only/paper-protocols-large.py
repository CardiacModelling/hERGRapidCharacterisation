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

savedir = './figs/paper'
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


#
# Do a very very tailored version........ :(
#
fig = plt.figure(figsize=(16, 10))
bigxgap = 1
n_xgrid = 21
bigygap = 3
n_ygrid = 8
grid = plt.GridSpec(3 * n_ygrid + 2 * bigygap, 3 * n_xgrid + 2 * bigxgap,
                    hspace=0.0, wspace=0.0)
axes = np.empty([3, 3], dtype=object)

for i in range(3):
    ii_grid = i * (n_ygrid + bigygap)
    if_grid = (i + 1) * n_ygrid + i * bigygap

    for j in range(3):
        ji_grid = j * (n_xgrid + bigxgap)
        jf_grid = (j + 1) * n_xgrid + j * bigxgap

        axes[i, j] = fig.add_subplot(grid[ii_grid:if_grid, ji_grid:jf_grid])
        if j > 0:
            axes[i, j].set_yticklabels([])

for i in range(3):
    axes[i, 0].set_ylabel('Voltage [mV]', fontsize=24)
    axes[-1, i].set_xlabel('Time [s]', fontsize=24)

#
# Plot
#
for i_prt, prt in enumerate(protocol_list):

    print(prt)
    ai, aj = int(i_prt / 3), i_prt % 3

    # Title
    if prt == 'staircaseramp':
        axes[ai, aj].set_title('Calibration', fontsize=20)
    else:
        axes[ai, aj].set_title('Validation %s' % i_prt, fontsize=20)

    # Add label!
    axes[ai, aj].text(0.0, 1.1, string.ascii_uppercase[i_prt],
                      transform=axes[ai, aj].transAxes, size=20,
                      weight='bold')

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
    axes[ai, aj].plot(times, voltage, lw=2, c=colours[prt])
    axes[ai, aj].set_ylim(-145, 45)
    # axes[ai, aj].set_ylim((np.min(voltage) - 10, np.max(voltage) + 15))
    axes[ai, aj].set_xlim(times[0], times[-1])
    axes[ai, aj].yaxis.set_ticks(range(-140, 40 + 40, 40))
    axes[ai, aj].tick_params('both', labelsize=16)
    axes[ai, aj].grid(c='#7f7f7f', ls='--', alpha=0.5)

grid.tight_layout(fig, pad=0.6, rect=(0, 0, 0.999, 0.99))
grid.update(wspace=0.4, hspace=0.0)
plt.savefig('%s/protocols.png' % (savedir), dpi=300)

print('Done')
