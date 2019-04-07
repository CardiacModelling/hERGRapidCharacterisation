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
import seaborn as sns

import protocols
from protocols import est_g_staircase
import model_ikr as m
from releakcorrect import I_releak, score_leak, protocol_leak_check

from scipy.optimize import fmin

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param


try:
    prt = sys.argv[1]
except IndexError:
    print('Usage: python %s [protocol]' % __file__)
    sys.exit()

# Set seed
np.random.seed(101)

savedir = './figs/paper'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

savedirlr = './figs/paper-low-res'
if not os.path.isdir(savedirlr):
    os.makedirs(savedirlr)

# Colours for fan chart
fan_blue = ['#b5c7d5',
        '#adc1d0',
        '#91abbc',
        '#85a0b1',
        '#6b8fa9',
        '#62869f',
        '#587c96',
        '#477390',
        '#3f6c88',
    ]

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
        'herg27oc',
        'herg30oc',
        'herg33oc',
        'herg37oc',
        ]
temperatures = np.array([25.0, 27.0, 30.0, 33.0, 37.0])
temperatures += 273.15  # in K
fit_seed = '542811797'

# Load pseudo2hbm
mean_chains = []
for i_temperature, (file_name, temperature) in enumerate(zip(file_list,
    temperatures)):

    load_file = './out-mcmc/%s-pseudo2hbm-lognorm-mean.txt' % (file_name)
    mean_chain = np.loadtxt(load_file)  # transformed

    mean_chains.append(mean_chain)
mean_chains = np.asarray(mean_chains)

# Eyring and Q10
from temperature_models import eyringA, eyringB, eyringG, eyringT
from temperature_models import q10A, q10B, q10G, q10T
from temperature_models import eyring_transform_to_model_param

eyring_mean = np.loadtxt('%s/eyring-mean.txt' % file_dir)
q10_mean = np.loadtxt('%s/q10-mean.txt' % file_dir)

#
# Where to zoom in
#
norm_zoom = False
zoom_in_win = { # protocol: [[(time_start, time_end), ...] in second,
                #            [(grid_start, grid_end), ...] total 16]
    # 'staircaseramp': [(1.8, 2.5), (11.395, 11.415), (13.895, 13.915),
    #                   (14.375, 14.925)],
    'staircaseramp': [[(1.875, 2.125), (11.35, 11.45), (13.85, 13.95),
                      (14.375, 14.625)],
                      [(0, 5), (5, 8), (8, 11), (11, 16)]],
    'pharma': [(0.64, 0.66), (1.14, 1.16)],
    'apab': [(0.0475, 0.0575), (0.32, 0.33)],
    'apabv3': [(0.05, 0.07)],
    'ap05hz': [[(0.04, 0.07), (2.04, 2.07)],
               [(0, 8), (8, 16)]],
    'ap1hz': [[(0.04, 0.07), (1.04, 1.07),
              (2.04, 2.07), (3.04, 3.07)],
              [(0, 4), (4, 8), (8, 12), (12, 16)]],
    'ap2hz': [[(0.045, 0.06),# (0.545, 0.56),
              (1.045, 1.06),# (1.545, 1.56),
              (2.045, 2.06),# (2.545, 2.56),
              (3.045, 3.06)],
              [(0, 4), (4, 8), (8, 12), (12, 16)]],
    'sactiv': None,
    'sinactiv': None,
}

isNorm = True
norm_method = 1


#
# Do a very very tailored version........ :(
#
fig = plt.figure(figsize=(18, 18))
n_maxzoom = 7
bigygap = 3
n_ygrid = 4 * 6
n_subpanels = 4
n_xgrid_1 = 24
n_xgrid_2 = 16
n_xgrid = n_xgrid_1 + n_xgrid_2 + 1  # extra 1 for space before zoom-in
bigxgap = 4
grid = plt.GridSpec(3 * n_ygrid + 2 * bigygap + 6,
                    2 * n_xgrid + bigxgap, hspace=0.0, wspace=0.2)
axes = np.empty([3 * n_subpanels + 1, 4], dtype=object)
# long list here:
for j in [0, 1]:
    # 0 row for protocol
    jj = 2 * j
    j_shift = n_xgrid + bigxgap
    axes[0, jj] = fig.add_subplot(grid[0:4, j*j_shift:j*j_shift+n_xgrid_1])
    axes[0, jj].set_xticklabels([])
for i in range(3):
    # i 'big row'
    n_shift = i * (n_ygrid + bigygap) + 6  # 6 is for protocol + gap
    n_panel = n_subpanels * i
    for j in range(2):
        if (i * 2 + j + 1) > len(temperatures):
            continue
        # j 'big column'
        j_shift = n_xgrid + bigxgap
        for k in range(4):
            # k 'sub row'
            ai, aj = n_panel + k + 1, 2 * j
            # first 'sub column'
            axes[ai, aj] = fig.add_subplot(
                    grid[n_shift + 6 * k:n_shift + 6 * (k + 1),
                         j * j_shift:j * j_shift + n_xgrid_1])
            if k != 3:
                axes[ai, aj].set_xticklabels([])

            # second 'sub column'
            axes[ai, aj + 1] = np.empty(n_maxzoom, dtype=object)
            n_zoom = len(zoom_in_win[prt][1])
            for l in range(n_zoom):
                j_grid_start = j * j_shift + n_xgrid_1 + 1\
                               + zoom_in_win[prt][1][l][0]  # extra 1 for gap
                j_grid_end = j * j_shift + n_xgrid_1 + 1 \
                             + zoom_in_win[prt][1][l][1]  # extra 1 for gap
                axes[ai, aj + 1][l] = fig.add_subplot(
                        grid[n_shift + 6 * k:n_shift + 6 * (k + 1),
                        j_grid_start:j_grid_end])
                axes[ai, aj + 1][l].set_xticklabels([])
                axes[ai, aj + 1][l].set_xticks([])
                axes[ai, aj + 1][l].set_yticklabels([])
                axes[ai, aj + 1][l].set_yticks([])

# Labels
axes[0, 0].text(-0.15, 0.5, 'Voltage\n[mV]', fontsize=14,
        rotation=90, ha='center', va='center',
        transform=axes[0, 0].transAxes)
Ts_oC = temperatures - 273.15  # in oC
for i in range(3):
    ai = i * n_subpanels + 1
    for j in range(2):
        if (i * 2 + j + 1) > len(temperatures):
            continue
        T_oC = int(temperatures[i * 2 + j] - 273.15)
        axes[ai, 2 * j].set_ylabel('$%d\pm1^\circ$C' % T_oC, fontsize=14)
    axes[ai + 1, 0].text(-0.125, 0, 'Normalised currents', fontsize=18,
            rotation=90, ha='center', va='center',
            transform=axes[ai + 1, 0].transAxes)
axes[-1, 0].set_xlabel('Time [s]', fontsize=18)
axes[-5, 2 * j].set_xlabel('Time [s]', fontsize=18)

# Liudmila suggested common y-axis
n_zoom = len(zoom_in_win[prt][1])
for i in range(3):
    ai = i * n_subpanels + 1
    for j in range(2):
        aj = 2 * j
        if (i * 2 + j + 1) > len(temperatures):
            continue
        if prt == 'staircaseramp':
            for ii in range(4):
                axes[ai + ii, aj].set_ylim((-1, 1.5))
                for l in range(n_zoom):
                    axes[ai + ii, aj + 1][l].set_ylim((-1, 1.5))
        else:
            for ii in range(4):
                axes[ai + ii, aj].set_ylim((-0.25, 1.5))
                for l in range(n_zoom):
                    axes[ai + ii, aj + 1][l].set_ylim((-0.25, 1.5))


#
# Plot!
#

# Normalisation factors for data
if prt == 'staircaseramp':
    norm_data_all = []
else:
    try:
        norm_data_all = np.loadtxt('./out/norm-factors/data.txt',
                skiprows=1).T
    except IOError:
        raise IOError('Expect running for prt=staircaseramp first')

if norm_method == 3:
    if prt == 'staircaseramp':
        norm_sim_all = []
        norm_eyring_all = []
        norm_q10_all = []
    else:
        try:
            norm_sim_all = np.loadtxt('./out/norm-factors/hbm.txt',
                    skiprows=1).T
            norm_eyring_all = np.loadtxt('./out/norm-factors/eyring.txt',
                    skiprows=1).T
            norm_q10_all = np.loadtxt('./out/norm-factors/q10.txt',
                    skiprows=1).T
        except IOError:
            raise IOError('Expect running for prt=staircaseramp first')

for i_T, (file_name, T) in enumerate(zip(file_list, temperatures)):

    # Model
    protocol_def = protocol_funcs[prt]
    if type(protocol_def) is str:
        protocol_def = '%s/%s' % (protocol_dir, protocol_def)

    model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                    protocol_def=protocol_def,
                    temperature=temperatures[0],  # K
                    transform=None,
                    useFilterCap=False)  # ignore capacitive spike

    # Calculate axis index
    ai, aj = n_subpanels * int(i_T / 2) + 1, 2 * (i_T % 2)

    # Add label!
    axes[ai, aj].text(-0.125, 1.05, string.ascii_uppercase[i_T],
                      transform=axes[ai, aj].transAxes, size=20,
                      weight='bold')

    # Time point
    if prt == 'staircaseramp':
        times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir_staircase,
            file_name + '1', prt), delimiter=',', skiprows=1)
    else:
        times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name + '1',
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

    if i_T < 2:
        if prt not in protocol_iv:
            # protocol
            axes[ai - 1, aj].plot(times, voltage, c='#696969')
        else:
            # protocol
            for i in range(voltage.shape[1]):
                axes[ai - 1, aj].plot(times, voltage[:, i], c='#696969')
        axes[ai - 1, aj].set_ylim((np.min(voltage) - 10, np.max(voltage) + 15))
    
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
                            lw=0.2, alpha=0.1, c=colour_list[i_CELL])
        else:
            iv_i = protocols.get_corrected_iv(data, times,
                                              *protocol_iv_args[prt]())
            iv_v = protocol_iv_v[prt]() * 1000  # mV
            axes[ai, aj].plot(iv_v, iv_i / np.max(iv_i), lw=0.5, alpha=0.5,
                              c=colour_list[i_CELL])

        # Zoom in for data
        if prt not in protocol_iv:
            for i_z, (t_i, t_f) in enumerate(zoom_in_win[prt][0]):
                # Find closest time
                idx_i = np.argmin(np.abs(times - t_i))
                idx_f = np.argmin(np.abs(times - t_f))
                # Segments
                zoom_in_segment_data = data[idx_i:idx_f]
                # Plot
                axes[ai, aj + 1][i_z].plot(times[idx_i:idx_f],
                        zoom_in_segment_data / norm_data,
                        lw=0.2, alpha=0.1, c=colour_list[i_CELL])
                axes[ai, aj + 1][i_z].set_xlim([times[idx_i], times[idx_f]])

    # Plot data as background
    data_fancharts_dir = './out/data-fancharts'
    percentiles = np.loadtxt('%s/percentiles.txt' % data_fancharts_dir)
    fan_chart_data_top = np.loadtxt('%s/%s-%s-top.txt' % \
            (data_fancharts_dir, file_name, prt))
    fan_chart_data_bot = np.loadtxt('%s/%s-%s-bot.txt' % \
            (data_fancharts_dir, file_name, prt))
    if prt not in protocol_iv:
        fan_x = np.loadtxt('%s/%s-%s-times.txt' % \
                (data_fancharts_dir, file_name, prt))
    else:
        fan_x = np.loadtxt('%s/%s-%s-voltage.txt' % \
                (data_fancharts_dir, file_name, prt))

    for ii in range(1, 4):
        for i_p, p in enumerate(percentiles):
            alpha = 0.8
            color = fan_blue[i_p]
            top = fan_chart_data_top[:, i_p]
            bot = fan_chart_data_bot[:, i_p]
            axes[ai + ii, aj].fill_between(fan_x, top, bot, color=color,
                    alpha=alpha, linewidth=0)
            if prt not in protocol_iv:
                axes[ai + ii, aj].fill_between(fan_x, top, bot,
                        alpha=alpha, linewidth=0, color=color)
                axes[ai + ii, aj].fill_between(fan_x, top, bot,
                        alpha=alpha, linewidth=0, color=color)

    # Models
    # HBM mean parameters
    hbm_T_mean = transform_to_model_param(
            np.mean(mean_chains[i_T], axis=0))
    simulation = model.simulate(hbm_T_mean, times_sim)

    # Eyring parameters
    eyring_T_mean = eyringT(eyring_mean, T)
    eyring_model_param = eyring_transform_to_model_param(eyring_T_mean, T)
    eyring_sim = model.simulate(eyring_model_param, times_sim)

    # Q10 parameters
    q10_T_mean = q10T(q10_mean, T)
    q10_model_param = eyring_transform_to_model_param(q10_T_mean, T)
    q10_sim = model.simulate(q10_model_param, times_sim)

    if norm_method == 1:
        # Kylie's method, use a reference trace 
        # (should give the most similar plots)
        top = fan_chart_data_top[:, -1]
        bot = fan_chart_data_bot[:, -1]
        ref_data = (top + bot) / 2.  # TODO
        from scipy.optimize import minimize
        res_s = minimize(lambda x: np.sum(
                np.abs(simulation / x - ref_data)),
                x0=np.abs(np.min(simulation)))
        norm_sim = res_s.x[0] if isNorm else 1.
        res_e = minimize(lambda x: np.sum(
                np.abs(eyring_sim / x - ref_data)), x0=norm_sim)
        norm_eyring = res_e.x[0] if isNorm else 1.
        res_q = minimize(lambda x: np.sum(
                np.abs(q10_sim / x - ref_data)), x0=norm_sim)
        norm_q10 = res_q.x[0] if isNorm else 1.
        if (norm_sim > 5e3 or not np.isfinite(norm_sim)):
            # Simulation went wrong?!
            raise RuntimeError('Simulation for HBM %s %s seems' % \
                    (file_name, prt) + ' problematic')
        if (norm_eyring > 5e3 or not np.isfinite(norm_eyring)):
            # Simulation went wrong?!
            raise RuntimeError('Simulation for Eyring %s %s seems' % \
                    (file_name, prt) + ' problematic')
        if (norm_q10 > 5e3 or not np.isfinite(norm_q10)):
            # Simulation went wrong?!
            raise RuntimeError('Simulation for Q10 %s %s seems' % \
                    (file_name, prt) + ' problematic')
    elif norm_method == 3:
        if prt == 'staircaseramp':
            norm_sim = est_g_staircase(simulation, times_sim,
                                       p0=[800, 0.025], debug=False)
            norm_sim_all.append(norm_sim)
        else:
            norm_sim = norm_sim_all[i_T]
        if prt == 'staircaseramp':
            norm_eyring = est_g_staircase(eyring_sim, times_sim,
                                          p0=[800, 0.025], debug=False)
            norm_eyring_all.append(norm_eyring)
        else:
            norm_eyring = norm_eyring_all[i_T]
        if prt == 'staircaseramp':
            norm_q10 = est_g_staircase(q10_sim, times_sim, p0=[800, 0.025],
                                        debug=False)
            norm_q10_all.append(norm_q10)
        else:
            norm_q10 = norm_q10_all[i_T]
    elif norm_method == 4:
        norm_sim = hbm_T_mean[0]
        norm_eyring = eyring_model_param[0]
        norm_q10 = q10_model_param[0]

    # Mean individual cells fit
    if prt in protocol_iv:
        simulation, t = protocol_iv_convert[prt](simulation, times_sim)
        assert(np.mean(np.abs(t - times)) < 1e-8)
        iv_v = protocol_iv_v[prt]() * 1000  # mV
        iv_i = protocols.get_corrected_iv(simulation, times,
                                          *protocol_iv_args[prt]())
        axes[ai + 1, aj].plot(iv_v, iv_i / np.max(iv_i), lw=1.5, alpha=1,
                              c='C1', zorder=1, label='HBM mean')
    else:
        axes[ai + 1, aj].plot(times_sim, simulation / norm_sim, alpha=1,
                lw=1.5, c='C1', zorder=1, label='HBM mean')

    # Eyring
    if prt in protocol_iv:
        eyring_sim, t = protocol_iv_convert[prt](eyring_sim, times_sim)
        assert(np.mean(np.abs(t - times)) < 1e-8)
        iv_v = protocol_iv_v[prt]() * 1000  # mV
        iv_i = protocols.get_corrected_iv(eyring_sim, times,
                                          *protocol_iv_args[prt]())
        axes[ai + 2, aj].plot(iv_v, iv_i / np.max(iv_i), lw=1.5, alpha=1,
                              c='C2', zorder=2, label='Eyring')
    else:
        axes[ai + 2, aj].plot(times_sim,
                          eyring_sim / norm_eyring,
                          alpha=1, lw=1.5, c='C2', zorder=2,
                          label='Eyring')

    # Q10
    if prt in protocol_iv:
        q10_sim, t = protocol_iv_convert[prt](q10_sim, times_sim)
        assert(np.mean(np.abs(t - times)) < 1e-8)
        iv_v = protocol_iv_v[prt]() * 1000  # mV
        iv_i = protocols.get_corrected_iv(q10_sim, times,
                                          *protocol_iv_args[prt]())
        axes[ai + 3, aj].plot(iv_v, iv_i / np.max(iv_i), lw=1.5, alpha=1,
                              c='C3', zorder=3, label='Q10')
    else:
        axes[ai + 3, aj].plot(times_sim,
                          q10_sim / norm_q10,
                          alpha=1, lw=1.5, c='C3', zorder=3, label='Q10')

    # Zoom in
    if prt not in protocol_iv:
        for i_z, (t_i, t_f) in enumerate(zoom_in_win[prt][0]):
            # Data fan chart
            # Find closest time
            idx_fi = np.argmin(np.abs(fan_x - t_i))
            idx_ff = np.argmin(np.abs(fan_x - t_f))
            # Segment and Plot
            for ii in range(1, 4):
                for i_p, p in enumerate(percentiles):
                    alpha = 0.8
                    color = fan_blue[i_p]
                    top = fan_chart_data_top[:, i_p]
                    bot = fan_chart_data_bot[:, i_p]
                    zoom_in_segment_top = bot[idx_fi:idx_ff]
                    zoom_in_segment_bot = top[idx_fi:idx_ff]
                    axes[ai + ii, aj + 1][i_z].fill_between(
                            fan_x[idx_fi:idx_ff],
                            zoom_in_segment_top, zoom_in_segment_bot,
                            color=color, alpha=alpha, linewidth=0)
                    axes[ai + ii, aj + 1][i_z].set_xlim(
                            [fan_x[idx_fi], fan_x[idx_ff]])

            # Models
            # Find closest time
            idx_i = np.argmin(np.abs(times_sim - t_i))
            idx_f = np.argmin(np.abs(times_sim - t_f))
            # Segments
            zoom_in_segment_sim = simulation[idx_i:idx_f]
            zoom_in_segment_eyring = eyring_sim[idx_i:idx_f]
            zoom_in_segment_q10 = q10_sim[idx_i:idx_f]
            # Plot
            axes[ai + 1, aj + 1][i_z].plot(times_sim[idx_i:idx_f],
                    zoom_in_segment_sim / norm_sim,
                    alpha=1, lw=1.5, c='C1', zorder=1)
            axes[ai + 2, aj + 1][i_z].plot(times_sim[idx_i:idx_f],
                    zoom_in_segment_eyring / norm_eyring,
                    alpha=1, lw=1.5, c='C2', zorder=2)
            axes[ai + 3, aj + 1][i_z].plot(times_sim[idx_i:idx_f],
                    zoom_in_segment_q10 / norm_q10,
                    alpha=1, lw=1.5, c='C3', zorder=3)
            for ii in range(1, 4):
                axes[ai + ii, aj + 1][i_z].set_xlim(
                        [times_sim[idx_i], times_sim[idx_f]])

    # Draw zoom-in boxes
    if prt not in protocol_iv:

        if prt == 'staircaseramp':
            minimum = -1.
            maximum = 1.5
        else:
            minimum = -0.25
            maximum = 1.5

        for i_z, (t_i, t_f) in enumerate(zoom_in_win[prt][0]):
            # Find closest time
            idx_i = np.argmin(np.abs(times - t_i))
            idx_f = np.argmin(np.abs(times - t_f))
            
            # And plot gray boxes over second panels
            codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
            vertices = np.array([(times[idx_i], minimum),
                                 (times[idx_i], maximum),
                                 (times[idx_f], maximum),
                                 (times[idx_f], minimum),
                                 (0, 0)], float)
            for ii in range(4):
                pathpatch = PathPatch(Path(vertices, codes),
                                      facecolor='#2ca02c',
                                      edgecolor='#2ca02c',
                                      alpha=0.25)
                plt.sca(axes[ai + ii, aj])
                pyplot_axes = plt.gca()
                pyplot_axes.add_patch(pathpatch)

            # Set arrow and time duration
            axes[ai + 3, aj + 1][i_z].arrow(1, -0.075, -1, 0,
                    length_includes_head=True,
                    head_width=0.03, head_length=0.05, clip_on=False,
                    fc='k', ec='k',
                    transform=axes[ai + 3, aj + 1][i_z].transAxes)
            axes[ai + 3, aj + 1][i_z].arrow(0, -0.075, 1, 0,
                    length_includes_head=True,
                    head_width=0.03, head_length=0.05, clip_on=False,
                    fc='k', ec='k',
                    transform=axes[ai + 3, aj + 1][i_z].transAxes)
            axes[ai + 3, aj + 1][i_z].text(0.5, -0.2,
                    '%s' % np.around(t_f - t_i, decimals=3),
                    transform=axes[ai + 3, aj + 1][i_z].transAxes,
                    horizontalalignment='center',
                    verticalalignment='center')

# Save norm factors
if prt == 'staircaseramp':

    def boolean_indexing(v, fillval=np.nan):
        lens = np.array([len(item) for item in v])
        mask = lens[:,None] > np.arange(lens.max())
        out = np.full(mask.shape,fillval)
        out[mask] = np.concatenate(v)
        return out

    header = 'Order follows `../../../manualselection/paper-rank-*` columns' \
            + ' are %s' % (' '.join(file_list))
    if not os.path.isdir('./out/norm-factors'):
        os.makedirs('./out/norm-factors')
    np.savetxt('./out/norm-factors/data.txt',
            boolean_indexing(norm_data_all).T, header=header)

    if norm_method == 3:
        np.savetxt('./out/norm-factors/hbm.txt',
                boolean_indexing(norm_sim_all).T, header=header)
        np.savetxt('./out/norm-factors/eyring.txt',
                boolean_indexing(norm_eyring_all).T, header=header)
        np.savetxt('./out/norm-factors/q10.txt',
                boolean_indexing(norm_q10_all).T, header=header)


#
# Final adjustment and save
#
grid.tight_layout(fig, pad=0.6, rect=[0.025, 0, 1, 1])
grid.update(wspace=0.12, hspace=0.0)
plt.savefig('%s/fitting-and-validation-eyring-q10-%s.png' % (savedirlr, prt),
            bbox_inch='tight', pad_inches=0, dpi=100)
plt.savefig('%s/fitting-and-validation-eyring-q10-%s.png' % (savedir, prt),
            bbox_inch='tight', pad_inches=0, dpi=300)

print('Done')
