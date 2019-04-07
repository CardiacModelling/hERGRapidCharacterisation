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
from scipy import stats

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

# Estimated EK
selectedfile = './manualv2selected-%s.txt' % (file_name)
selectedwell = []
with open(selectedfile, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            selectedwell.append(l.split()[0])

est_ek_all = np.loadtxt('%s/%s-staircaseramp-EK_all.txt' \
                        % (qc_dir, file_name))
est_ek = []
for c in selectedwell:
    est_ek.append(est_ek_all[WELL_ID.index(c)])

expected_ek = prt2model['staircaseramp'].EK() * 1000  # V -> mV

print('Expected EK (mV): ' + str(expected_ek))
print('Est. EK median (mV): ' + str(np.median(est_ek)))
print('Est. EK mean (mV): ' + str(np.mean(est_ek)))
print('Est. EK std. (mV): ' + str(np.std(est_ek)))
print('min EK (mV): ' + str(np.min(est_ek)))
print('max EK (mV): ' + str(np.max(est_ek)))

# Plot
fig = plt.figure(figsize=(11, 4))

# linear regression on IV
plt.subplot(1, 2, 1)
plt.axhline(0, ls='-', c='#7f7f7f')
t = np.loadtxt('../data/herg25oc1-staircaseramp-times.csv', delimiter=',',
        skiprows=1)
cell = 'A01'
i = np.loadtxt('../data/herg25oc1-staircaseramp-%s.csv' % cell, delimiter=',',
        skiprows=1)
v = np.loadtxt('../protocol-time-series/protocol-staircaseramp.csv',
        delimiter=',', skiprows=1)[::2, 1]
t_i = 14.41  # s
t_f = t_i + 0.1  # s
win = np.logical_and(t > t_i, t < t_f)
plt.plot(v[win], i[win], label='Data')

p = np.poly1d(np.polyfit(v[win], i[win], 3))
r = []
for i in p.r:
    if np.min(v[win]) < i <= np.max(v[win]) \
                             and (np.isreal(i) or np.abs(i.imag) < 1e-8):
        r.append(i)
if len(r) == 1:
    ek =  r[0].real
elif len(r) > 1:
    ek = np.max(r).real
else:
    ek = np.inf
plt.plot(v[win], p(v[win]), c='#d62728', ls='--', label='Fitted')
plt.axvline(ek, ls='-', c='C2', label=r'Estimated $E_K$')
# plt.axvline(expected_ek, ls='--', c='k', label=r'Expected $E_K$')
plt.xlabel('Voltage [mV]', fontsize=16)
plt.ylabel('Current [pA]', fontsize=16)
plt.legend(fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


# histogram
plt.subplot(1, 2, 2)
plt.hist(est_ek, bins=20, alpha=0.75)

plt.axvline(expected_ek, c='#ff7f0e', ls='--', label=r'Expected $E_K$')

plt.xlabel(r'Estimated $E_K$ [mV]', fontsize=16)
plt.ylabel('Frequency (N=%s)' % len(est_ek), fontsize=16)
plt.legend(fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Force integer only
from matplotlib.ticker import MaxNLocator
fig.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlim([-92.5, -77.5])

# Done
plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
plt.subplots_adjust(hspace=0)
plt.savefig('%s/ek-hist.pdf' % savedir, format='pdf', bbox_inch='tight')
plt.close()

print('Done')
