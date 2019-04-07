#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib/')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pints

import model_ikr as m
import protocols

import string
WELL_ID = [l+str(i).zfill(2)
           for l in string.ascii_uppercase[:16]
           for i in range(1,25)]

if '--regular' in sys.argv:
    regularsweep = True
    savedir = './fakedata-voltageoffset-regularsweep'
else:
    regularsweep = False
    savedir = './fakedata-voltageoffset'

if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_dir = '../data'
qc_dir = '../qc'
file_list = ['herg25oc1']
temperatures = [25.0]
temperatures = np.asarray(temperatures) + 273.15  # K

file_name = file_list[0]
temperature = temperatures[0]

# Protocol info
protocol_funcs = {
    'staircaseramp': 'protocol-staircaseramp.csv',
    'pharma': 'protocol-pharma.csv',
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
    # 'sactiv',
    # 'sinactiv',
    'pharma',
    'apab',
    'apabv3',
    'ap05hz',
    'ap1hz',
    'ap2hz',
]

# Model
prt2model = {}
for prt in protocol_list:

    protocol_def = protocol_funcs[prt]
    if type(protocol_def) is str:
        protocol_def = '%s/%s' % (protocol_dir, protocol_def)

    prt2model[prt] = m.ModelWithVoltageOffset(
                        '../mmt-model-files/kylie-2017-IKr.mmt',
                        protocol_def=protocol_def,
                        temperature=temperature,  # K
                        transform=None,
                        useFilterCap=False)  # ignore capacitive spike

# Estimated EK
if not regularsweep:
    selectedfile = './manualv2selected-%s.txt' % (file_name)
    selectedwell = []
    with open(selectedfile, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                selectedwell.append(l.split()[0])

    est_ek_all = np.loadtxt('%s/%s-staircaseramp-EK_all.txt' \
                            % (qc_dir, file_name))
    est_ek = []
    est_v_err = []
    expected_ek = prt2model['staircaseramp'].EK() * 1000  # V -> mV
    for c in selectedwell:
        est_ek.append(est_ek_all[WELL_ID.index(c)])
        # Assume our observed EK error correlated to voltage error 
        est_v_err.append(expected_ek - est_ek[-1])
elif regularsweep:
    est_v_err = np.arange(-5, 5 + 0.5, 0.5)
else:
    raise RuntimeError()

print('Est. mean (mV): ' + str(np.mean(est_v_err)))
print('Est. std. (mV): ' + str(np.std(est_v_err)))
print('min (mV): ' + str(np.min(est_v_err)))
print('max (mV): ' + str(np.max(est_v_err)))

# Parameters
parameter_file = 'herg25oc1-staircaseramp-mcmcsimplemean-542811797.txt'
parameters = np.loadtxt('%s/%s' % ('./out', parameter_file))

for i_prt, prt in enumerate(protocol_list):
    print('Generating protocol ' + prt, end='')

    model = prt2model[prt]

    # Time point
    times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name,
            prt), delimiter=',', skiprows=1)

    # Save setting
    np.savetxt('%s/%s-%s-times.csv' % (savedir, file_name, prt), times,
            delimiter=',', comments='', header='\"time\"')
    np.savetxt('%s/%s-%s-voltageoffset.csv' % (savedir, file_name, prt),
            est_v_err,delimiter=',', comments='',
            header='\"voltage offset (mV)\"')

    for i_vo, vo in enumerate(est_v_err):
        param_vo = np.append(parameters, vo / 1000.)  # mV -> V

        simulation = model.simulate(param_vo, times)
        for _ in range(5):
            assert(np.sum(np.abs(model.simulate(param_vo, times) \
                                 - simulation)) < 1e-15)

        np.savetxt('%s/%s-%s-sim-%s.csv' % (savedir, file_name, prt, i_vo),
                simulation, delimiter=',', comments='', header='\"current\"')

        voltage = model.voltage(times)
        plt.plot(times, voltage)
    plt.savefig(file_name + prt)
    plt.close()
    print(' ')

print('Done')
