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

savedirlr = './figs/paper-low-res'
if not os.path.isdir(savedirlr):
    os.makedirs(savedirlr)

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
        'herg25oc1',
        ]
temperatures = np.array([25.0])
temperatures += 273.15  # in K
fit_seed = '542811797'

isNorm = True
norm_method = 1
isSmooth = True
smooth_win = 51  # seems okay
smooth_order = 3
smooth_win_small = 3
smooth_order_small = 1


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
fig = plt.figure(figsize=(16, 15))
grid = plt.GridSpec(70, 3, hspace=0.0, wspace=0.2)
n_maxzoom = 7
bigxgap = 12
n_xgrid = n_maxzoom * 6 * 2
bigygap = 4
n_ygrid = 22
grid = plt.GridSpec(3 * n_ygrid + 2 * bigygap, 3 * n_xgrid + 2 * bigxgap,
                    hspace=0.0, wspace=0.0)
axes = np.empty([12, int(len(protocol_list) / 3)], dtype=object)
# long list here:
for i in range(int(len(protocol_list) / 3)):
    i_grid = i * (n_xgrid + bigxgap)
    f_grid = (i + 1) * n_xgrid + i * bigxgap

    # First 'row'
    if i == 0:
        axes[0, i] = fig.add_subplot(grid[0:6, i_grid:f_grid])
        axes[0, i].set_xticklabels([])
        axes[1, i] = fig.add_subplot(grid[6:12, i_grid:f_grid])
        axes[2, i] = np.empty(n_maxzoom, dtype=object)
        axes[3, i] = np.empty(n_maxzoom, dtype=object)
    else:
        axes[0, i] = fig.add_subplot(grid[0:10, i_grid:f_grid])
        axes[0, i].set_xticklabels([])
        axes[1, i] = fig.add_subplot(grid[10:20, i_grid:f_grid])

    # Second 'row'
    n_shift = n_ygrid + bigygap
    axes[4, i] = fig.add_subplot(grid[n_shift+0:n_shift+6, i_grid:f_grid])
    axes[4, i].set_xticklabels([])
    axes[5, i] = fig.add_subplot(grid[n_shift+6:n_shift+12, i_grid:f_grid])
    axes[6, i] = np.empty(n_maxzoom, dtype=object)
    axes[7, i] = np.empty(n_maxzoom, dtype=object)

    # Third 'row'
    n_shift = 2 * (n_ygrid + bigygap)
    axes[8, i] = fig.add_subplot(grid[n_shift+0:n_shift+6, i_grid:f_grid])
    axes[8, i].set_xticklabels([])
    axes[9, i] = fig.add_subplot(grid[n_shift+6:n_shift+12, i_grid:f_grid])
    axes[10, i] = np.empty(n_maxzoom, dtype=object)
    axes[11, i] = np.empty(n_maxzoom, dtype=object)

r_ngrid = {
    2: (14, 18),
    3: (18, 22),
    6: (n_ygrid + bigygap + 14, n_ygrid + bigygap + 18),
    7: (n_ygrid + bigygap + 18, n_ygrid + bigygap + 22),
    10: (2 * (n_ygrid + bigygap) + 14, 2 * (n_ygrid + bigygap) + 18),
    11: (2 * (n_ygrid + bigygap) + 18, 2 * (n_ygrid + bigygap) + 22),
}

# Do zoom in
# staircase-ramp specifal case
ai = 2
n_zoom = 6
assert(n_xgrid % n_zoom == 0)
cf = int(n_xgrid / n_zoom)
for ai in [2, 3]:
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
    for ii in [2, 3]:
        ai, aj = 4 * int(i_prt / 3) + ii, i_prt % 3

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
axes[0, 0].set_ylabel('Model', fontsize=14)
axes[1, 0].set_ylabel('Data', fontsize=14)
axes[2, 0][0].set_ylabel('Zoom\nmodel', fontsize=14)
axes[3, 0][0].set_ylabel('Zoom\ndata', fontsize=14)
axes[4, 0].set_ylabel('Model', fontsize=14)
axes[5, 0].set_ylabel('Data', fontsize=14)
axes[6, 0][0].set_ylabel('Zoom\nmodel', fontsize=14)
axes[7, 0][0].set_ylabel('Zoom\ndata', fontsize=14)
axes[8, 0].set_ylabel('Model', fontsize=14)
axes[9, 0].set_ylabel('Data', fontsize=14)
axes[10, 0][0].set_ylabel('Zoom\nmodel', fontsize=14)
axes[11, 0][0].set_ylabel('Zoom\ndata', fontsize=14)
axes[3, 0][1].text(1, -0.5,
        'Time [s]', fontsize=14, ha='center', va='center',
        transform=axes[3, 0][1].transAxes)
axes[-5, 0][0].text(1, -0.5,
        'Time [s]', fontsize=14, ha='center', va='center',
        transform=axes[-5, 0][0].transAxes)
axes[-5, 1][0].text(1, -0.5,
        'Time [s]', fontsize=14, ha='center', va='center',
        transform=axes[-5, 1][0].transAxes)
axes[-5, 2][0].text(0.5, -0.5,
        'Time [s]', fontsize=14, ha='center', va='center',
        transform=axes[-5, 2][0].transAxes)
axes[-1, 0][0].text(1, -0.55,
        'Time [s]', fontsize=18, ha='center', va='center',
        transform=axes[-1, 0][0].transAxes)
axes[-1, 1][1].text(1, -0.55,
        'Time [s]', fontsize=18, ha='center', va='center',
        transform=axes[-1, 1][1].transAxes)
axes[-1, 2][3].text(0.5, -0.55,
        'Time [s]', fontsize=18, ha='center', va='center',
        transform=axes[-1, 2][3].transAxes)
axes[5, 0].text(-0.25, -0.25, 'Normalised current', rotation=90, fontsize=18,
        transform=axes[5, 0].transAxes,
        horizontalalignment='center',
        verticalalignment='center')

for aj in [1, 2]:
    # Add special x,y-label for IV protocols
    axes[1, aj].set_xlabel('Voltage [mV]', fontsize=14)
for ai in [0, 1]:
    axes[ai, 1].set_ylim(-0.05, 1.05)
    axes[ai, 2].set_ylim(-5, 1.2)


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
colour_list_d = sns.color_palette('Blues', n_colors=len(RANKED_CELLS))
colour_list_d = colour_list_d.as_hex()
colour_list_s = sns.color_palette('Reds', n_colors=len(RANKED_CELLS))
colour_list_s = colour_list_s.as_hex()

for i_prt, prt in enumerate(protocol_list):

    # Calculate axis index
    ai, aj = 4 * int(i_prt / 3), i_prt % 3

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
    times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_list[0],
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

    # Pre-load a reference trace to do normalisation
    if norm_method == 1 and isNorm:
        ref_file = 'herg25oc1'
        ref_cell = 'D19'
        if prt == 'staircaseramp':
            ref_data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir_staircase,
                ref_file, prt, ref_cell), delimiter=',', skiprows=1)
        elif prt not in protocol_iv:
            ref_data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, ref_file,
                    prt, ref_cell), delimiter=',', skiprows=1)
            # Re-leak correct the leak corrected data...
            g_releak = fmin(score_leak, [0.0], args=(ref_data, voltage, times,
                                protocol_leak_check[prt]), disp=False)
            ref_data = I_releak(g_releak[0], ref_data, voltage)
        else:
            ref_data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, ref_file,
                    prt, ref_cell), delimiter=',', skiprows=1)
            for i in range(ref_data.shape[1]):
                g_releak = fmin(score_leak, [0.0], args=(ref_data[:, i],
                                    voltage[:, i], times,
                                    protocol_leak_check[prt]), disp=False)
                ref_data[:, i] = I_releak(g_releak[0], ref_data[:, i],
                                          voltage[:, i])
        assert(len(ref_data) == len(times))

        # Set axes limit for normalisation and draw zoom-in boxes
        if prt not in protocol_iv:
            if prt in ['ap1hz', 'ap2hz']:
                maximum = np.percentile(ref_data, 99.99)
                minimum = np.percentile(ref_data, 0.01)
                maximum += 0.25 * np.abs(maximum)
                minimum -= 0.5 * np.abs(minimum)
            else:
                maximum = np.percentile(ref_data, 99.5)
                minimum = np.percentile(ref_data, 0.5)
                maximum += 0.25 * np.abs(maximum)
                minimum -= 0.5 * np.abs(minimum)
            for i in range(2):
                axes[ai + i, aj].set_ylim([minimum, maximum])
                axes[ai + i, aj].set_xlim([times[0], times[-1]])

                # set specific ylim
                if prt == 'ap1hz':
                    axes[ai + i, aj].set_ylim([-40, 90])
                    minimum_t, maximum_t = -40, 90
                elif prt == 'ap2hz':
                    axes[ai + i, aj].set_ylim([-40, 180])
                    minimum_t, maximum_t = -40, 180
                else:
                    minimum_t, maximum_t = minimum, maximum

            # Zoom in ones
            for i_z, (t_i, t_f) in enumerate(zoom_in_win[prt]):
                for i in range(2, 4):
                    axes[ai + i, aj][i_z].set_ylim([minimum, maximum])

                # Find closest time
                idx_i = np.argmin(np.abs(times - t_i))
                idx_f = np.argmin(np.abs(times - t_f))
                
                # And plot gray boxes over second panels
                codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
                vertices = np.array([(times[idx_i], minimum_t),
                                     (times[idx_i], maximum_t),
                                     (times[idx_f], maximum_t),
                                     (times[idx_f], minimum_t),
                                     (0, 0)], float)
                pathpatch1 = PathPatch(Path(vertices, codes),
                                       facecolor='#2ca02c',
                                       edgecolor='#2ca02c',
                                       alpha=0.75)
                pathpatch2 = PathPatch(Path(vertices, codes),
                                       facecolor='#2ca02c',
                                       edgecolor='#2ca02c',
                                       alpha=0.75)
                plt.sca(axes[ai, aj])
                pyplot_axes1 = plt.gca()
                pyplot_axes1.add_patch(pathpatch1)
                plt.sca(axes[ai + 1, aj])
                pyplot_axes2 = plt.gca()
                pyplot_axes2.add_patch(pathpatch2)

                # Set arrow and time duration
                axes[ai + 3, aj][i_z].arrow(1, -0.075, -1, 0,
                        length_includes_head=True,
                        head_width=0.03, head_length=0.05, clip_on=False,
                        fc='k', ec='k', transform=axes[ai + 3, aj][i_z].transAxes)
                axes[ai + 3, aj][i_z].arrow(0, -0.075, 1, 0,
                        length_includes_head=True,
                        head_width=0.03, head_length=0.05, clip_on=False,
                        fc='k', ec='k', transform=axes[ai + 3, aj][i_z].transAxes)
                axes[ai + 3, aj][i_z].text(0.5, -0.2,
                        '%s' % np.around(t_f - t_i, decimals=3),
                        transform=axes[ai + 3, aj][i_z].transAxes,
                        horizontalalignment='center',
                        verticalalignment='center')

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
        raw_data = np.copy(data)
        if isSmooth:
            from scipy.signal import savgol_filter
            if prt not in protocol_iv:
                data = savgol_filter(data, window_length=smooth_win,
                        polyorder=smooth_order)
                weak_filter_data = savgol_filter(data,
                        window_length=smooth_win_small,
                        polyorder=smooth_order_small)
            elif False:
                for i in range(data.shape[1]):
                    data[:, i] = savgol_filter(data[:, i],
                            window_length=smooth_win,
                            polyorder=3)
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

        # Normalisation
        if norm_method == 0:
            # just pick the max (susceptible to nise)
            norm_d = np.max(data) if isNorm else 1.
            norm_s = np.max(simulation) if isNorm else 1.
        elif norm_method == 1:
            # Kylie's method, use a reference trace 
            # (should give the most similar plots)
            from scipy.optimize import minimize
            res_s = minimize(lambda x: np.sum(np.abs(simulation / x
                                                     - ref_data)), x0=1.0)
            norm_s = res_s.x[0] if isNorm else 1.
            res_d = minimize(lambda x: np.sum(np.abs(data / x
                                                     - ref_data)), x0=norm_s)
            norm_d = res_d.x[0] if isNorm else 1.
            if norm_d > 1e2 or not np.isfinite(norm_d):
                # Maybe smoothing making fitting harder?
                norm_d = norm_s
            if norm_s > 1e2 or not np.isfinite(norm_s):
                # Simulation went wrong?!
                raise RuntimeError('Simulation for %s %s %s seems' % \
                        (file_name, cell, prt) + ' problematic')
        elif norm_method == 2:
            # use 95th percentile (less susceptible to nise)
            norm_d = np.percentile(data, 95) if isNorm else 1.
            norm_s = np.percentile(simulation, 95) if isNorm else 1.
        else:
            raise ValueError('Unknown normalisation method, choose' +
                             ' norm_method from 0-2')

        # Plot
        if prt not in protocol_iv:
            # simulation
            axes[ai, aj].plot(times, simulation / norm_s, lw=0.5, alpha=0.5,
                    c=colour_list_s[i_CELL])
            # recording
            axes[ai + 1, aj].plot(times, data / norm_d, lw=0.5, alpha=0.5,
                    c=colour_list_d[i_CELL])
        else:
            iv_v = protocol_iv_v[prt]() * 1000  # mV
            # simulation
            iv_i_s = protocols.get_corrected_iv(simulation, times,
                                                *protocol_iv_args[prt]())
            axes[ai, aj].plot(iv_v, iv_i_s / np.max(iv_i_s), lw=0.5, alpha=0.5,
                    c=colour_list_s[i_CELL])
            # recording
            iv_i_d = protocols.get_corrected_iv(data, times,
                                                *protocol_iv_args[prt]())
            axes[ai + 1, aj].plot(iv_v, iv_i_d / np.max(iv_i_d), lw=0.5,
                    alpha=0.5, c=colour_list_d[i_CELL])
        
        # Plot zoom in version
        if prt not in protocol_iv:
            for i_z, (t_i, t_f) in enumerate(zoom_in_win[prt]):
                # Find closest time
                idx_i = np.argmin(np.abs(times - t_i))
                idx_f = np.argmin(np.abs(times - t_f))
                zoom_in_segment_data = raw_data[idx_i:idx_f]
                zoom_in_segment_sim = simulation[idx_i:idx_f]

                axes[ai + 2, aj][i_z].plot(times[idx_i:idx_f],
                        zoom_in_segment_sim / norm_s,
                        lw=0.5, alpha=0.5, c=colour_list_s[i_CELL])
                axes[ai + 3, aj][i_z].plot(times[idx_i:idx_f],
                        zoom_in_segment_data / norm_d,
                        lw=0.5, alpha=0.5, c=colour_list_d[i_CELL])
                axes[ai + 2, aj][i_z].set_xlim([times[idx_i], times[idx_f]])
                axes[ai + 3, aj][i_z].set_xlim([times[idx_i], times[idx_f]])
                
                # Add trapezium over second and third panels
                if i_CELL == 0:
                    top_v = [(times[idx_i], minimum_t), (times[idx_f], minimum_t)]
                    bottom_v = axes[ai + 2, aj][i_z].transData.transform(
                            [(times[idx_f], maximum),
                             (times[idx_i], maximum)])
                    inv = axes[ai + 1, aj].transData.inverted()
                    codes = [Path.MOVETO] + [Path.LINETO] * 3 \
                            + [Path.CLOSEPOLY]
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


#
# Final adjustment and save
#
grid.tight_layout(fig, pad=1.0, rect=(0.02, 0.03, 1, 1))
grid.update(wspace=20, hspace=0.0)
plt.savefig('%s/fitting-and-validation-selected-cells-zoom.png' % savedirlr,
            bbox_inch='tight', pad_inches=0, dpi=100)
plt.savefig('%s/fitting-and-validation-selected-cells-zoom.png' % savedir,
            bbox_inch='tight', pad_inches=0, dpi=300)
# This pdf version can get up to 40+MB!
plt.savefig('%s/fitting-and-validation-selected-cells-zoom.pdf' % savedir,
            format='pdf', bbox_inch='tight', pad_inches=0)

print('Done')
