#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

isNorm = True

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
protocol = 'pharma'

data_dir = '../data'
file_dir = './out'
file_list = [
        ['herg25oc1'],
        ['herg27oc1'],
        ['herg30oc1'],
        ['herg33oc1'],
        ['herg37oc3'],
        ]
temperatures = np.array([25.0, 27.0, 30.0, 33.0, 37.0])
temperatures += 273.15  # in K
fit_seed = '542811797'
withfcap = False

#
# Load up all selected cells
#
for i_temperature, (file_names, temperature) in enumerate(zip(file_list,
    temperatures)):

    RANK = {}

    for file_name in file_names:
        # Get selected cells
        files_dir = os.path.realpath(os.path.join(file_dir, file_name))
        searchwfcap = '-fcap' if withfcap else ''
        selectedfile = './manualselection/manualv2selected-%s.txt' % (file_name)
        selectedwell = []
        with open(selectedfile, 'r') as f:
            for l in f:
                if not l.startswith('#'):
                    selectedwell.append(l.split()[0])
        print(file_name + ' selected ' + str(len(selectedwell)) + ' cells')

        # Model
        protocol_def = protocol_funcs[protocol]
        if type(protocol_def) is str:
            protocol_def = '%s/%s' % (protocol_dir, protocol_def)
        model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                        protocol_def=protocol_def,
                        temperature=temperature,  # K
                        transform=None,
                        useFilterCap=False)  # ignore capacitive spike

        for cell in selectedwell:
            # Fitted parameters
            param_file = '%s/%s-staircaseramp-%s-solution%s-%s.txt' % \
                    (files_dir, file_name, cell, searchwfcap, fit_seed)
            obtained_parameters = np.loadtxt(param_file)

            # Data
            data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
                protocol, cell), delimiter=',', skiprows=1)
            times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name,
                protocol), delimiter=',', skiprows=1)
            assert(data.shape == times.shape)

            # Simulation
            simulation = model.simulate(obtained_parameters, times)
            if False:
                for _ in range(5):
                    assert(all(simulation == 
                        model.simulate(obtained_parameters, times)))
            voltage = model.voltage(times)

            norm = np.max(simulation) if isNorm else 1.

            idx = (np.abs(times - 1.1)).argmin()
            idx_r = 100  # dt=0.2ms -> 20 ms
            rank_value = np.mean(simulation[idx - idx_r:idx + idx_r]) / norm
            RANK[file_name + cell] = rank_value

    #
    # Rank and output
    #
    RANKED_CELLS = [key for key, value in sorted(RANK.iteritems(), \
                                                 key=lambda (k,v): (v,k))]

    with open('./manualselection/paper-rank-%s.txt' % file_names[0][:-1],
            'w') as f:
        f.write('# experiment-id + cell-id, ordered using %s\n' % __file__)
        for c in RANKED_CELLS:
            f.write(c + '\n')

'''
# Can use the following for colouring
import seaborn as sns
# color_list = sns.cubehelix_palette(len(SORTED_CELLS))
color_list = sns.color_palette('Blues', n_colors=len(SORTED_CELLS))
color_list.as_hex()
'''

print('Done')
