#!/usr/bin/env python2
import sys
sys.path.append('../lib')
import os
import numpy as np
if '--show' not in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

import protocols
import model_ikr as m

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

data_dir = '../data'
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

    print('Plotting %s' % file_name)

    savedir = '%s/%s' % (savepath, file_name)
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
        fig = plt.figure(figsize=(16, 7))
        grid = plt.GridSpec(28, 3, hspace=0.0, wspace=0.2)
        axes = np.empty([4, len(protocol_list) // 2], dtype=object)
        for i in range(3):
            # First 'row'
            if i != 2:
                axes[0, i] = fig.add_subplot(grid[0:6, i]) # , sharex=axes[2, i])
                axes[0, i].set_xticklabels([])
                axes[1, i] = fig.add_subplot(grid[6:12, i]) # , sharex=axes[2, i])
            else:
                axes[0, i] = fig.add_subplot(grid[0:4, i]) # , sharex=axes[2, i])
                axes[1, i] = fig.add_subplot(grid[7:12, i]) # , sharex=axes[2, i])
            # Second 'row'
            axes[2, i] = fig.add_subplot(grid[16:22, i]) # , sharex=axes[5, i])
            axes[2, i].set_xticklabels([])
            axes[3, i] = fig.add_subplot(grid[22:28, i]) # , sharex=axes[5, i])
        # Set labels
        axes[0, 0].set_ylabel('Voltage [mV]', fontsize=14)
        axes[1, 0].set_ylabel('Current [pA]', fontsize=14)
        axes[2, 0].set_ylabel('Voltage [mV]', fontsize=14)
        axes[3, 0].set_ylabel('Current [pA]', fontsize=14)
        axes[-1, len(protocol_list) // 2 // 2].set_xlabel('Time [s]',
                fontsize=18)

        for i_prt, prt in enumerate(protocol_list):
            # Data
            data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name, prt,
                cell), delimiter=',', skiprows=1)
            times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name,
                prt), delimiter=',', skiprows=1)
            assert(data.shape == times.shape)

            # Simulation
            model = prt2model[prt]
            simulation = model.simulate(obtained_parameters, times)
            if False:
                for _ in range(5):
                    assert(all(simulation == 
                        model.simulate(obtained_parameters, times)))
            voltage = model.voltage(times)

            # Plot
            ai = (i_prt // (len(protocol_list) // 2)) * 2
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
            axes[ai + 1, aj].plot(times, data, alpha=0.5, label='Data')
            axes[ai + 1, aj].plot(times, simulation, label='Model')
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

