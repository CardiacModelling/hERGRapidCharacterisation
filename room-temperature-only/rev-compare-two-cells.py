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

savedir = './figs/paper'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

cells = ['B20', 'C17']

import pints.io
basename = './out-mcmc/mcmc-herg25oc1'
load_name = '%s/' % (basename)  # if any prefix in all files

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param


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
        'herg25oc1',
        ]
temperatures = np.array([25.0])
temperatures += 273.15  # in K
fit_seed = '542811797'

file_name = file_list[0]
temperature = temperatures[0]


#
# Where to zoom in
#
norm_zoom = False
zoom_in_win = { # protocol: [(time_start, time_end), ...] in second
    # 'staircaseramp': [(1.8, 2.5), (11.395, 11.415), (13.895, 13.915),
    #                   (14.375, 14.925)],
    'staircaseramp': [(1.875, 2.125), (11.35, 11.45), (13.85, 13.95),
                      (14.375, 14.625)],
    'pharma': [(0.64, 0.66), (1.14, 1.16)],
    'apab': [(0.0475, 0.0575), (0.32, 0.33)],
    'apabv3': [(0.05, 0.07)],
    'ap05hz': [(0.04, 0.07), (2.04, 2.07)],
    'ap1hz': [(0.04, 0.07), (1.04, 1.07),
              (2.04, 2.07), (3.04, 3.07)],
    'ap2hz': [(0.045, 0.06), (0.545, 0.56),
              (1.045, 1.06), (1.545, 1.56),
              (2.045, 2.06), (2.545, 2.56),
              (3.045, 3.06)],
    'sactiv': None,
    'sinactiv': None,
}


#
# Do a very very tailored version........ :(
#
fig = plt.figure(figsize=(10, 5))
n_maxzoom = 7
bigxgap = 12
n_xgrid = n_maxzoom * 6 * 2
bigygap = 4
n_ygrid = 19
grid = plt.GridSpec(n_ygrid, 2 * n_xgrid + 1 * bigxgap,
                    hspace=0.0, wspace=0.0)
axes = np.empty([3, 2], dtype=object)
# long list here:
for i in range(2):
    i_grid = i * (n_xgrid + bigxgap)
    f_grid = (i + 1) * n_xgrid + i * bigxgap

    # First 'row'
    if i == 0:  # staircase-ramp
        axes[0, i] = fig.add_subplot(grid[0:4, i_grid:f_grid])
        axes[0, i].set_xticklabels([])
        axes[1, i] = fig.add_subplot(grid[4:11, i_grid:f_grid])
        axes[2, i] = np.empty(n_maxzoom, dtype=object)  # grid[13:19, _])
    else:
        axes[0, i] = fig.add_subplot(grid[0:7, i_grid:f_grid])
        axes[1, i] = fig.add_subplot(grid[10:19, i_grid:f_grid])

r_ngrid = {
    2: (13, 19),
    5: (n_ygrid + bigygap + 13, n_ygrid + bigygap + 19),
    8: (2 * (n_ygrid + bigygap) + 13, 2 * (n_ygrid + bigygap) + 19),
}

# Do zoom in
# staircase-ramp specifal case
ai = 2
n_zoom = 6
assert(n_xgrid % n_zoom == 0)
cf = int(n_xgrid / n_zoom)
axes[ai, 0][0] = fig.add_subplot(grid[r_ngrid[ai][0]:r_ngrid[ai][1],
                                 0:2*cf])
axes[ai, 0][1] = fig.add_subplot(grid[r_ngrid[ai][0]:r_ngrid[ai][1],
                                 2*cf:3*cf])
axes[ai, 0][2] = fig.add_subplot(grid[r_ngrid[ai][0]:r_ngrid[ai][1],
                                 3*cf:4*cf])
axes[ai, 0][3] = fig.add_subplot(grid[r_ngrid[ai][0]:r_ngrid[ai][1],
                                 4*cf:6*cf])
for i in range(len(zoom_in_win['staircaseramp'])):
    axes[ai, 0][i].set_xticklabels([])
    axes[ai, 0][i].set_xticks([])
    if i > 0:
        axes[ai, 0][i].set_yticklabels([])
        axes[ai, 0][i].set_yticks([])
# the rest
for i_prt, prt in enumerate(protocol_list):
    ai, aj = 3 * int(i_prt / 3) + 2, i_prt % 3

    if prt == 'staircaseramp' or (prt in protocol_iv):
        continue

    n_zoom = len(zoom_in_win[prt])
    assert(n_xgrid % n_zoom == 0)
    n = int(n_xgrid / n_zoom)
    n_shift = aj * (n_xgrid + bigxgap)
    for i in range(n_zoom):
        axes[ai, aj][i] = fig.add_subplot(
                grid[r_ngrid[ai][0]:r_ngrid[ai][1],
                     n_shift + i * n:n_shift + (i + 1) * n])
        axes[ai, aj][i].set_xticklabels([])
        axes[ai, aj][i].set_xticks([])
        if i > 0:
            axes[ai, aj][i].set_yticklabels([])
            axes[ai, aj][i].set_yticks([])

# Set labels
axes[0, 0].set_ylabel('Voltage\n[mV]', fontsize=14)
axes[1, 0].set_ylabel('Current\n[pA]', fontsize=14)
axes[2, 0][0].set_ylabel('Zoom in', fontsize=14)
axes[2, 0][1].text(1, -0.3,
        'Time [s]', fontsize=14, ha='center', va='center',
        transform=axes[2, 0][1].transAxes)

# Add special x,y-label for IV protocols
axes[1, 1].set_xlabel('Voltage [mV]', fontsize=14)
# change y-label to right
for aj in [1]:
    axes[1, aj].yaxis.tick_right()
    axes[1, aj].yaxis.set_label_position("right")
    axes[0, aj].set_xlabel('Time [s]', fontsize=14)
axes[1, -1].set_ylabel('Normalised\ncurrent', fontsize=14)
axes[1, 1].set_ylim(-0.05, 1.05)

# Set y-ticklabels for protocols
# TODO


#
# Protocol and a cell
#

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

for i_cell, cell in enumerate(cells):
    # Fitted parameters
    param_file = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
            (file_dir, file_name, file_name, cell, fit_seed)
    obtained_parameters = np.loadtxt(param_file)

    for i_prt, prt in enumerate(protocol_list):

        # Calculate axis index
        ai, aj = 3 * int(i_prt / 3), i_prt % 3

        # Title
        if prt == 'staircaseramp':
            axes[ai, aj].set_title('Calibration', fontsize=16, loc='left')
        else:
            axes[ai, aj].set_title('Validation %s' % i_prt, fontsize=16,
                    loc='left')

        # Add label!
        if prt not in protocol_iv:
            axes[ai, aj].text(-0.1, 1.1, string.ascii_uppercase[i_prt],
                              transform=axes[ai, aj].transAxes, size=20,
                              weight='bold')
        else:
            axes[ai, aj].text(-0.1, 1.06, string.ascii_uppercase[i_prt],
                              transform=axes[ai, aj].transAxes, size=20,
                              weight='bold')

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

        # Simulation
        simulation = model.simulate(obtained_parameters, times_sim)
        if prt in protocol_iv:
            simulation, t = protocol_iv_convert[prt](simulation, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-8)

        # Plot
        if prt not in protocol_iv:
            '''
            # Maybe normalise for this comparison?
            max_value = np.max(simulation)
            simulation = simulation / max_value
            data = data / max_value
            '''
        
            # protocol
            axes[ai, aj].plot(times, voltage, c='#696969')
            # recording
            axes[ai + 1, aj].plot(times, data, lw=1, alpha=0.4, c='C%s' \
                                  % (i_cell + 4))
            # simulation
            if prt == 'staircaseramp':
                axes[ai + 1, aj].plot(times, simulation, lw=1, c='C%s' \
                                      % (i_cell + 4), label='cell %s' % cell)
            else:
                axes[ai + 1, aj].plot(times, simulation, lw=1, c='C%s' \
                                      % (i_cell + 4), label='cell %s' % cell)
            axes[ai, aj].set_xlim([0, times[-1]])
            axes[ai + 1, aj].set_xlim([0, times[-1]])
        else:
            # protocol
            for i in range(voltage.shape[1]):
                axes[ai, aj].plot(times, voltage[:, i], c='#696969')
            iv_v = protocol_iv_v[prt]() * 1000  # mV
            # recording
            iv_i = protocols.get_corrected_iv(data, times,
                                              *protocol_iv_args[prt]())
            axes[ai + 1, aj].plot(iv_v, iv_i / np.max(iv_i), lw=2, alpha=1,
                                  c='C%s' % (i_cell + 4), ls='--',
                                  label='cell %s data' % cell)
            # simulation
            iv_i = protocols.get_corrected_iv(simulation, times,
                                              *protocol_iv_args[prt]())
            axes[ai + 1, aj].plot(iv_v, iv_i / np.max(iv_i), lw=1, alpha=0.5,
                                  c='C%s' % (i_cell + 4),
                                  label='cell %s prediction' % cell)
            # Load fitting result
            chain_file = '%s%s-chain.csv' % (load_name, cell)
            exp_chains = pints.io.load_samples(chain_file, 1)[0]
            n_samples = len(exp_chains)
            warm_up = int(n_samples * 3. / 4.)
            thinning = 1
            exp_chains = exp_chains[warm_up::thinning, :]
            for i_p, p in enumerate(exp_chains[::2000]):
                print(i_p)

                p = transform_to_model_param(p)

                simulation = model.simulate(p, times_sim)
                if prt in protocol_iv:
                    simulation, t = protocol_iv_convert[prt](simulation, times_sim)
                    assert(np.mean(np.abs(t - times)) < 1e-8)
            
                iv_i = protocols.get_corrected_iv(simulation, times,
                                                  *protocol_iv_args[prt]())
                axes[ai + 1, aj].plot(iv_v, iv_i / np.max(iv_i), lw=1, alpha=0.2,
                                      c='C%s' % (i_cell + 4))
        
        # Plot zoom in version
        if prt not in protocol_iv:
            amplitude = np.max(simulation) - np.min(simulation)
            for i_z, (t_i, t_f) in enumerate(zoom_in_win[prt]):
                # Find closest time
                idx_i = np.argmin(np.abs(times - t_i))
                idx_f = np.argmin(np.abs(times - t_f))
                # Work out the max and min
                if norm_zoom:
                    y_min = np.min(simulation[idx_i:idx_f])
                    y_max = np.max(simulation[idx_i:idx_f])
                    y_amp = y_max - y_min
                    y_min -=  0.2 * y_amp
                    y_max +=  0.2 * y_amp
                    y_amp = y_max - y_min
                else:
                    y_min = np.min(simulation) - 0.2 * amplitude
                    y_max = np.max(simulation) + 0.2 * amplitude
                # Fix ylim if we think we need it to be bigger for second panel
                axes[ai + 1, aj].set_ylim(
                        (min(y_min, axes[ai + 1, aj].get_ylim()[0])),
                        (max(y_max, axes[ai + 1, aj].get_ylim()[1])))
                # set specific ylim
                if prt == 'ap1hz':
                    axes[ai + 1, aj].set_ylim([-40, 90])
                    y_min_t, y_max_t = -40, 90
                elif prt == 'ap2hz':
                    axes[ai + 1, aj].set_ylim([-40, 180])
                    y_min_t, y_max_t = -40, 180
                else:
                    y_min_t, y_max_t = y_min, y_max
                # Work out third panel plot
                if norm_zoom:
                    zoom_in_segment_data = (data[idx_i:idx_f] - y_min) / y_amp
                    zoom_in_segment_sim = (simulation[idx_i:idx_f] - y_min) \
                                            / y_amp
                else:
                    zoom_in_segment_data = data[idx_i:idx_f]
                    zoom_in_segment_sim = simulation[idx_i:idx_f]
                
                axes[ai + 2, aj][i_z].plot(times[idx_i:idx_f],
                        zoom_in_segment_data, lw=1, alpha=0.5,
                        c='C%s' % (i_cell + 4))
                axes[ai + 2, aj][i_z].plot(times[idx_i:idx_f],
                        zoom_in_segment_sim, lw=1, c='C%s' % (i_cell + 4))
                axes[ai + 2, aj][i_z].set_xlim([times[idx_i], times[idx_f]])
                if norm_zoom:
                    axes[ai + 2, aj][i_z].set_ylim([0, 1])
                else:
                    axes[ai + 2, aj][i_z].set_ylim([y_min, y_max])
                if i_cell == 1:
                    # And plot shading over second panels
                    codes = [Path.MOVETO] + [Path.LINETO] * 3 \
                            + [Path.CLOSEPOLY]
                    vertices = np.array([(times[idx_i], y_min_t),
                                            (times[idx_i], y_max_t),
                                            (times[idx_f], y_max_t),
                                            (times[idx_f], y_min_t),
                                            (0, 0)], float)
                    pathpatch = PathPatch(Path(vertices, codes),
                                            facecolor='#2ca02c',
                                            edgecolor='#2ca02c',
                                            alpha=0.75)
                    plt.sca(axes[ai + 1, aj])
                    pyplot_axes = plt.gca()
                    pyplot_axes.add_patch(pathpatch)
                    # Add trapezium over second and third panels
                    top_v = [(times[idx_i], y_min_t), (times[idx_f], y_min_t)]
                    bottom_v = axes[ai + 2, aj][i_z].transData.transform(
                            [(times[idx_f], y_max), (times[idx_i], y_max)])
                    inv = axes[ai + 1, aj].transData.inverted()
                    vertices = list(top_v) + list(inv.transform(bottom_v)) \
                            + list([(0, 0)])
                    pathpatch = PathPatch(Path(vertices, codes),
                                            facecolor='#2ca02c',
                                            edgecolor='#2ca02c',
                                            clip_on=False,
                                            alpha=0.15)
                    plt.sca(axes[ai + 1, aj])
                    pyplot_axes = plt.gca()
                    pyplot_axes.add_patch(pathpatch)

                    # Set arrow and time duration
                    axes[ai + 2, aj][i_z].arrow(1, -0.05, -1, 0,
                            length_includes_head=True,
                            head_width=0.03, head_length=0.05, clip_on=False,
                            fc='k', ec='k',
                            transform=axes[ai + 2, aj][i_z].transAxes)
                    axes[ai + 2, aj][i_z].arrow(0, -0.05, 1, 0,
                            length_includes_head=True,
                            head_width=0.03, head_length=0.05, clip_on=False,
                            fc='k', ec='k',
                            transform=axes[ai + 2, aj][i_z].transAxes)
                    axes[ai + 2, aj][i_z].text(0.5, -0.15,
                            '%s' % np.around(t_f - t_i, decimals=3),
                            transform=axes[ai + 2, aj][i_z].transAxes,
                            horizontalalignment='center',
                            verticalalignment='center')


#
# Final adjustment and save
#
axes[1, 0].legend()
axes[1, 1].legend()
grid.tight_layout(fig, pad=0.6, rect=(0, 0.01, 1, 1))
grid.update(wspace=20, hspace=0.0)
plt.savefig('%s/rev-compare-%s_%s-v-%s-zoom.png' % (savedir, file_name, \
            cells[0], cells[1]), bbox_inch='tight', pad_inches=0, dpi=300)

print('Done')
