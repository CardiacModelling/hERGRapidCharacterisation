#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib/')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model_ikr as m
import protocols

import string
WELL_ID = [l+str(i).zfill(2)
           for l in string.ascii_uppercase[:16]
           for i in range(1,25)]

savedir = './figs/paper'

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
}
protocol_dir = '../protocol-time-series'
protocol_list = [
    'staircaseramp',
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

expected_ek = prt2model['staircaseramp'].EK() * 1000  # V -> mV


# Estimated EK
selectedfile = './manualv2selected-%s.txt' % (file_name)
selectedwell = []
with open(selectedfile, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            selectedwell.append(l.split()[0])

est_ek_all = np.loadtxt('%s/%s-staircaseramp-EK_all.txt' \
                        % (qc_dir, file_name))
est_cm_all = np.loadtxt('%s/%s-staircaseramp-Cm_before.txt' \
                        % (qc_dir, file_name))
est_rseal_all = np.loadtxt('%s/%s-staircaseramp-Rseal_before.txt' \
                        % (qc_dir, file_name))
est_rseries_all = np.loadtxt('%s/%s-staircaseramp-Rseries_before.txt' \
                        % (qc_dir, file_name))
est_leak_all = np.loadtxt('%s/%s-staircaseramp-leak_before.txt' \
                        % (qc_dir, file_name))

est_ek = []
est_cm = []  # pF
est_rseal = []  # GOhm
est_rseries = []  # MOhm
est_gleak = []
est_eleak = []
for c in selectedwell:
    est_ek.append(est_ek_all[WELL_ID.index(c)])
    est_cm.append(est_cm_all[WELL_ID.index(c)] / 1e-12)
    est_rseal.append(est_rseal_all[WELL_ID.index(c)] / 1e9)
    est_rseries.append(est_rseries_all[WELL_ID.index(c)] / 1e6)
    est_gleak.append(est_leak_all[WELL_ID.index(c)][0])
    est_eleak.append(est_leak_all[WELL_ID.index(c)][1])

d_ek = np.array(est_ek) - expected_ek

# Plot
fig = plt.figure(figsize=(21, 4))

plt.subplot(1, 5, 1)
plt.scatter(est_rseal, d_ek, alpha=0.75)

plt.ylabel(r'$\Delta V^j$ [mV]', fontsize=16)
plt.xlabel(r'$R_{seal}$ [$G\Omega$]', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(1, 5, 2)
plt.scatter(est_cm, d_ek, alpha=0.75)

plt.ylabel(r'$\Delta V^j$ [mV]', fontsize=16)
plt.xlabel(r'$C_{m}$ [$pF$]', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(1, 5, 3)
plt.scatter(est_rseries, d_ek, alpha=0.75)

plt.ylabel(r'$\Delta V^j$ [mV]', fontsize=16)
plt.xlabel(r'$R_{series}$ [$M\Omega$]', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(1, 5, 4)
plt.scatter(est_gleak, d_ek, alpha=0.75)

plt.ylabel(r'$\Delta V^j$ [mV]', fontsize=16)
plt.xlabel(r'$g_{leak}$ [$pS$]', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(1, 5, 5)
plt.scatter(est_eleak, d_ek, alpha=0.75)

plt.ylabel(r'$\Delta V^j$ [mV]', fontsize=16)
plt.xlabel(r'$E_{leak}$ [$mV$]', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Done
plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
plt.subplots_adjust(hspace=0)
plt.savefig('%s/rev-err-scatter.png' % savedir, bbox_inch='tight')
plt.savefig('%s/rev-err-scatter.pdf' % savedir, format='pdf', bbox_inch='tight')
plt.close()

print('Done')
