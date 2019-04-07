#!/usr/bin/env python2
import sys
sys.path.append('../lib')
import os
import numpy as np
if '--show' not in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import glob

import protocols
import model_ikr as m

from scipy.optimize import fmin
# Set seed
np.random.seed(101)

from releakcorrect import I_releak, score_leak, protocol_leak_check

savepath = './figs'
if not os.path.isdir(savepath):
    os.makedirs(savepath)

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
# Where to zoom in
#
norm_zoom = False
zoom_in_win = { # protocol: [(time_start, time_end), ...] in second
    # 'staircaseramp': [(1.8, 2.5), (11.395, 11.415), (13.895, 13.915),
    #                   (14.375, 14.925)],
    'staircaseramp': [(1.875, 2.125), (11.35, 11.45), (13.85, 13.95),
                      (14.375, 14.625)],
    'pharma': [(0.64, 0.66), (1.14, 1.16)],
    'apab': [(0.035, 0.065), (0.32, 0.33)],
    'apabv3': [(0.05, 0.07)],
    'ap05hz': None,
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
# Get new parameters and traces
#
for i_temperature, (file_name, temperature) in enumerate(zip(file_list,
    temperatures)):

    print('Plotting %s' % file_name)

    savedir = '%s/%s-autoLC-releak-zoom' % (savepath, file_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    # Get selected cells
    files_dir = os.path.realpath(os.path.join(file_dir, file_name))
    searchwfcap = '-fcap' if withfcap else ''
    selectedfile = './manualselected-%s.txt' % (file_name)
    selectedwell = []
    with open(selectedfile, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                selectedwell.append(l.split()[0])

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

    for cell in selectedwell[:]:
        # Fitted parameters
        param_file = '%s/%s-staircaseramp-%s-solution%s-%s.txt' % \
                (files_dir, file_name, cell, searchwfcap, fit_seed)
        obtained_parameters = np.loadtxt(param_file)

        # Create figure
        # Do a very very tailored version........ :(
        fig = plt.figure(figsize=(16, 12))
        grid = plt.GridSpec(40, 3, hspace=0.0, wspace=0.2)
        axes = np.empty([6, int(len(protocol_list)/2)], dtype=object)
        # long list here:
        for i in range(int(len(protocol_list)/2)):
            # First 'row'
            axes[0, i] = fig.add_subplot(grid[0:5, i]) # , sharex=axes[2, i])
            axes[0, i].set_xticklabels([])
            axes[1, i] = fig.add_subplot(grid[5:10, i]) # , sharex=axes[2, i])
            axes[2, i] = fig.add_subplot(grid[13:18, i])
            axes[2, i].set_xticklabels([])  # last one is zoom in

            # Second 'row'
            axes[3, i] = fig.add_subplot(grid[22:27, i]) # , sharex=axes[5, i])
            axes[3, i].set_xticklabels([])
            axes[4, i] = fig.add_subplot(grid[27:32, i]) # , sharex=axes[5, i])
            axes[5, i] = fig.add_subplot(grid[35:40, i])
            axes[5, i].set_xticklabels([])  # last one is zoom in

            if norm_zoom:
                axes[2, i].set_yticklabels([])
                axes[5, i].set_yticklabels([])
        # Set labels
        axes[0, 0].set_ylabel('Voltage [mV]', fontsize=14)
        axes[1, 0].set_ylabel('Current [pA]', fontsize=14)
        axes[2, 0].set_ylabel('Zoom in', fontsize=14)
        axes[3, 0].set_ylabel('Voltage [mV]', fontsize=14)
        axes[4, 0].set_ylabel('Current [pA]', fontsize=14)
        axes[5, 0].set_ylabel('Zoom in', fontsize=14)
        axes[-1, len(protocol_list) // 2 // 2].set_xlabel('Time [s]',
                fontsize=18)

        for i_prt, prt in enumerate(protocol_list):
            # Time points
            times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name,
                prt), delimiter=',', skiprows=1)

            # Simulation
            model = prt2model[prt]
            simulation = model.simulate(obtained_parameters, times)
            if False:
                for _ in range(5):
                    assert(all(simulation == 
                        model.simulate(obtained_parameters, times)))
            voltage = model.voltage(times) * 1000  # V -> mV

            # Data
            if prt == 'staircaseramp':
                data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir_staircase,
                    file_name, prt, cell), delimiter=',', skiprows=1)
                data_new = np.copy(data)
            else:
                data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
                    prt, cell), delimiter=',', skiprows=1)
                # Re-leak correct the leak corrected data...
                g_releak = fmin(score_leak, [0.0], args=(data, voltage, times,
                                    protocol_leak_check[prt]))
                data_new = I_releak(g_releak[0], data, voltage)
                assert(data_new.shape == times.shape)
                # TODO: Save corrected data later...
            assert(data.shape == times.shape)

            # Plot
            ai = (i_prt // (len(protocol_list) // 2)) * 3
            aj = i_prt % (len(protocol_list) // 2)
            amplitude = np.max(simulation) - np.min(simulation)
            if prt == 'staircaseramp':
                axes[ai, aj].set_title('Calibration', fontsize=16)
                # Fix ylim using simulation
                axes[ai + 1, aj].set_ylim([
                    np.min(simulation) - 0.05 * amplitude,
                    np.max(simulation) + 0.05 * amplitude])
            else:
                axes[ai, aj].set_title('Validation %s' % i_prt, fontsize=16)
                # Fix ylim using simulation
                axes[ai + 1, aj].set_ylim([
                    np.min(simulation) - 0.3 * amplitude,
                    np.max(simulation) + 0.3 * amplitude])
            axes[ai, aj].plot(times, voltage)
            axes[ai + 1, aj].plot(times, data, alpha=0.2, label='Data')
            axes[ai + 1, aj].plot(times, data_new, alpha=0.5, label='New data')
            axes[ai + 1, aj].plot(times, simulation, label='Model')
            # Plot zoom in version
            zoom_in_data = []
            zoom_in_data_new = []
            zoom_in_simulation = []
            zoom_in_line_break = []
            for t_i, t_f in zoom_in_win[prt]:
                # Find closest time
                idx_i = np.argmin(np.abs(times - t_i))
                idx_f = np.argmin(np.abs(times - t_f))
                # Work out the max and min
                if norm_zoom:
                    y_min = np.min(simulation[idx_i:idx_f])
                    y_max = np.max(simulation[idx_i:idx_f])
                    y_amp = y_max - y_min
                    y_min -=  0.3 * y_amp
                    y_max +=  0.3 * y_amp
                    y_amp = y_max - y_min
                else:
                    y_min = np.min(simulation) - 0.25 * amplitude
                    y_max = np.max(simulation) + 0.25 * amplitude
                # And plot gray boxes over second panels
                codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
                vertices = np.array([(times[idx_i], y_min),
                                     (times[idx_i], y_max),
                                     (times[idx_f], y_max),
                                     (times[idx_f], y_min),
                                     (0, 0)], float)
                pathpatch = PathPatch(Path(vertices, codes),
                                      facecolor='#fa9fb5',
                                      edgecolor='#fa9fb5',
                                      alpha=0.75)
                plt.sca(axes[ai + 1, aj])
                pyplot_axes = plt.gca()
                pyplot_axes.add_patch(pathpatch)
                # Work out third panel plot
                if norm_zoom:
                    zoom_in_segment_data = (data[idx_i:idx_f] - y_min) / y_amp
                    zoom_in_segment_data_new = (data_new[idx_i:idx_f] - y_min)\
                                               / y_amp
                    zoom_in_segment_sim = (simulation[idx_i:idx_f] - y_min) \
                                          / y_amp
                else:
                    zoom_in_segment_data = data[idx_i:idx_f]
                    zoom_in_segment_data_new = data_new[idx_i:idx_f]
                    zoom_in_segment_sim = simulation[idx_i:idx_f]
                zoom_in_data = np.append(zoom_in_data, zoom_in_segment_data)
                zoom_in_data_new = np.append(zoom_in_data_new,
                                             zoom_in_segment_data_new)
                zoom_in_simulation = np.append(zoom_in_simulation,
                        zoom_in_segment_sim)
                zoom_in_line_break.append(len(zoom_in_segment_sim))
            axes[ai + 2, aj].plot(zoom_in_data, alpha=0.2)
            axes[ai + 2, aj].plot(zoom_in_data_new, alpha=0.5)
            axes[ai + 2, aj].plot(zoom_in_simulation)
            for x in np.cumsum(zoom_in_line_break)[:-1]:
                axes[ai + 2, aj].axvline(x, color='k')
            axes[ai + 2, aj].set_xlim([0, len(zoom_in_simulation)])
            if norm_zoom:
                axes[ai + 2, aj].set_ylim([0, 1])
            else:
                axes[ai + 2, aj].set_ylim([y_min, y_max])
        axes[1, 0].legend()
        grid.tight_layout(fig, pad=0.6)
        grid.update(wspace=0.12, hspace=0.0)
        if '--show' not in sys.argv:
            plt.savefig('%s/%s.png' % (savedir, cell),
                        bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.close('all')
        print('Done ' + file_name + cell)
    del(prt2model)

