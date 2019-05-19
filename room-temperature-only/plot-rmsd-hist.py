#!/usr/bin/env python2
# coding: utf-8
#
# Plot RMSD histograms for CMA-ES fittings
#

from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
if not '--show' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import protocols
import model_ikr as m
from releakcorrect import I_releak, score_leak, protocol_leak_check

from scipy.optimize import fmin
# Set seed
np.random.seed(101)

# Predefine stuffs
debug = True
BIG_MATRIX = []


def rmsd(t1, t2):
    # Normalised RMSD value between trace 1 ``t1`` and trace 2 ``t2``
    #
    # Note, usually normalise to data, so 
    # - ``t2`` data (or anything as reference)
    # - ``t1`` simulation (or anything for comparison)
    return np.sqrt(np.mean((t1 - t2) ** 2)) / np.sqrt(np.mean(t2 ** 2))


#
# Protocols
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
        'pharma',
        'apab',
        'apabv3',
        'ap05hz',
        'ap1hz',
        'ap2hz',
        'sactiv',
        'sinactiv',
        ]
prt_names = ['Staircase', 'pharma', 'EAD', 'DAD', 'AP05Hz', 'AP1Hz', 'AP2Hz', 'actIV', 'inactIV']

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
data_dir_staircase = '../data'
file_dir = './out'
file_list = [
        'herg25oc1',
        ]
temperatures = np.array([25.0])
temperatures += 273.15  # in K
fit_seed = '542811797'
withfcap = False

#
# Get new parameters and traces
#
for i_temperature, (file_name, temperature) in enumerate(zip(file_list,
    temperatures)):

    savepath = './figs/rmsd-hist-%s-autoLC-releak' % file_name
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    logfile = savepath + '/rmsd-values.txt'
    with open(logfile, 'w') as f:
        f.write('Start logging...\n')


    print('Reading %s' % file_name)
    with open(logfile, 'a') as f:
        f.write(file_name + '...\n')

    # Get selected cells
    files_dir = os.path.realpath(os.path.join(file_dir, file_name))
    searchwfcap = '-fcap' if withfcap else ''
    selectedfile = './manualv2selected-%s.txt' % (file_name)
    selectedwell = []
    with open(selectedfile, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                selectedwell.append(l.split()[0])

    for prt in protocol_list:

        with open(logfile, 'a') as f:
            f.write('%s...\n' % prt)

        # Model
        protocol_def = protocol_funcs[prt]
        if type(protocol_def) is str:
            protocol_def = '%s/%s' % (protocol_dir, protocol_def)

        model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                        protocol_def=protocol_def,
                        temperature=temperature,  # K
                        transform=None,
                        useFilterCap=False)  # ignore capacitive spike

        # Time points
        times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name,
                prt), delimiter=',', skiprows=1)

        # Voltage protocol
        if prt not in protocol_iv:
            times_sim = np.copy(times)
            voltage = model.voltage(times)
        else:
            times_sim = protocol_iv_times[prt](times[1] - times[0])
            voltage = model.voltage(times_sim) * 1000
            voltage, t = protocol_iv_convert[prt](voltage, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-8)

        # Initialisation
        i = 0
        RMSD = []
        outvalues = []
        RMSD_cells = []
        VALUES = []
        SIMS = []
        for cell in selectedwell:
            # Fitted parameters
            param_file = '%s/%s-staircaseramp-%s-solution%s-%s.txt' % \
                    (files_dir, file_name, cell, searchwfcap, fit_seed)
            obtained_parameters = np.loadtxt(param_file)

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
            if False:
                for _ in range(5):
                    assert(all(simulation == 
                        model.simulate(obtained_parameters, times_sim)))
            if prt in protocol_iv:
                simulation, t = protocol_iv_convert[prt](simulation, times_sim)
                assert(np.mean(np.abs(t - times)) < 1e-8)
                iv_v = protocol_iv_v[prt]() * 1000  # mV
                # simulation
                iv_i_s = protocols.get_corrected_iv(simulation, times,
                                                    *protocol_iv_args[prt]())
                # recording
                iv_i_d = protocols.get_corrected_iv(data, times,
                                                    *protocol_iv_args[prt]())
                # normalise and replace 'simulation', 'data', and 'times'
                simulation = iv_i_s / np.max(iv_i_s)
                data = iv_i_d / np.max(iv_i_d)

            RMSD.append(rmsd(simulation, data))
            RMSD_cells.append((file_name, cell))
            VALUES.append(data)
            SIMS.append(simulation)
            if i == 0 and debug:
                if prt not in protocol_iv:
                    plot_x = np.copy(times)
                    plt.xlabel('Time')
                else:
                    plot_x = np.copy(iv_v)
                    plt.xlabel('Voltage')
                plt.plot(plot_x, data)
                plt.plot(plot_x, simulation)
                plt.ylabel('Current')
                print('Debug rmsd: ' + str(rmsd(simulation, data)))
                plt.savefig('%s/rmsd-hist-%s-debug.png' % (savepath, prt))
                plt.close('all')
            i += 1

        BIG_MATRIX.append(RMSD)

        best_cell = np.argmin(RMSD)
        worst_cell = np.argmax(RMSD)
        median_cell = np.argsort(RMSD)[len(RMSD)//2]
        p75_cell = np.argsort(RMSD)[int(len(RMSD)*0.75)]
        p90_cell = np.argsort(RMSD)[int(len(RMSD)*0.9)]
        to_plot = {
            'best': best_cell,
            'worst': worst_cell,
            'median': median_cell,
            '75percent': p75_cell,
            '90percent': p90_cell,
        }

        #
        # Plot
        #
        # Plot histograms
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        axes.hist(RMSD, 20)
        axes.set_ylabel('Frequency (N=%s)' % len(selectedwell))
        axes.set_xlabel(r'RMSE / RMSD$_0$')
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('%s/rmsd-hist-%s.png' % (savepath, prt))
        plt.close('all')

        # Plot extreme cases
        for n, i in to_plot.iteritems():
            ID, CELL = RMSD_cells[i][0], RMSD_cells[i][1]
            values = VALUES[i]
            sim = SIMS[i]
            if prt not in protocol_iv:
                plot_x = np.copy(times)
                plt.xlabel('Time')
            else:
                plot_x = np.copy(iv_v)
                plt.xlabel('Voltage')
            plt.plot(plot_x, values)
            plt.plot(plot_x, sim)
            plt.ylabel('Current')
            plt.savefig('%s/rmsd-hist-%s-plot-%s.png'% (savepath, prt, n))
            plt.close('all')
            print('%s %s %s rmsd: '%(n, ID, CELL) + str(rmsd(sim, values)))
            with open(logfile, 'a') as f:
                f.write('%s %s %s rmsd: '%(n, ID, CELL)\
                        + str(rmsd(sim, values)) + '\n')

        # Plot all in sorted RMSD order
        rmsd_argsort = np.argsort(RMSD)
        with open(logfile, 'a') as f:
            f.write('---\n')
        savedir = '%s/rmsd-hist-%s-plots' % (savepath, prt)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        for ii, i in enumerate(rmsd_argsort):
            ID, CELL = RMSD_cells[i][0], RMSD_cells[i][1]
            values = VALUES[i]
            sim = SIMS[i]
            if prt not in protocol_iv:
                plot_x = np.copy(times)
                plt.xlabel('Time')
            else:
                plot_x = np.copy(iv_v)
                plt.xlabel('Voltage')
            plt.plot(plot_x, values)
            plt.plot(plot_x, sim)
            plt.ylabel('Current')
            plt.savefig('%s/rank_%s-%s-%s.png'%(savedir, str(ii).zfill(3), ID,\
                    CELL))
            plt.close('all')
            with open(logfile, 'a') as f:
                f.write('rank %s %s %s rmsd: ' % (str(ii).zfill(2), ID, CELL)\
                        + str(rmsd(sim, values)) + '\n')
        with open(logfile, 'a') as f:
            f.write('---\n')

    #
    # Play around with the big matrix
    #
    BIG_MATRIX = np.array(BIG_MATRIX)
    # sorted by 'best fit'
    sorted_as = BIG_MATRIX[0, :].argsort()
    # apply sort
    RMSD_cells = [RMSD_cells[i][0]+'-'+RMSD_cells[i][1] for i in sorted_as]
    BIG_MATRIX = BIG_MATRIX[:, sorted_as]

    # maybe just color by rank; scipy.stats.rankdata()

    fig, ax = plt.subplots(figsize=(10, 100))
    # vmin, vmax here is a bit arbitrary...
    vmin = 0
    vmax = 2
    im = ax.matshow(BIG_MATRIX.T, cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)
    # .T is needed for the ordering i,j below!
    # do some tricks with the colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=np.arange(vmin, vmax))
    # change the current axis back to ax
    plt.sca(ax)
    for i in range(BIG_MATRIX.shape[0]):
        for j in range(BIG_MATRIX.shape[1]):
            c = BIG_MATRIX[i, j]
            ax.text(i, j, '%.2f'%c, va='center', ha='center')
    plt.yticks(np.arange(BIG_MATRIX.shape[1]), RMSD_cells)
    plt.xticks(np.arange(BIG_MATRIX.shape[0]), prt_names)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('%s/rmsd-matrix.png' % savepath, bbox_inch='tight')
    plt.close('all')

    #
    # Save matrix
    #
    np.savetxt('%s/rmsd-matrix.txt' % savepath, BIG_MATRIX.T,
            header=' '.join(protocol_list))
    with open('%s/rmsd-matrix-cells.txt' % savepath, 'w') as f:
        for c in RMSD_cells:
            f.write(c + '\n')

    #
    # Gary's plotmatrix type plot
    #
    fig, axes = plt.subplots(BIG_MATRIX.shape[0], BIG_MATRIX.shape[0],
                             figsize=(12, 12))
    for i in range(BIG_MATRIX.shape[0]):
        for j in range(BIG_MATRIX.shape[0]):
            if i == j:
                # Do nothing
                axes[i,j ].set_xticks([])
                axes[i,j ].set_yticks([])
            elif i < j:
                axes[i, j].set_visible(False)
            elif i > j:
                axes[i, j].scatter(BIG_MATRIX[j], BIG_MATRIX[i])
            if j == 0:
                axes[i, j].set_ylabel(prt_names[i])
            if i == len(prt_names) - 1:
                axes[i, j].set_xlabel(prt_names[j])
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('%s/rmsd-gary-matrix.png' % savepath, bbox_inch='tight')
    plt.close('all')


