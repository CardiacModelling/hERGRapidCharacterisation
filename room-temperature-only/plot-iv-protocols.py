#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import string

WELL_ID = [l+str(i).zfill(2) for l in string.ascii_uppercase[:16] for i in range(1,25)]

import protocols
import model_ikr as m
from releakcorrect import I_releak, score_leak, protocol_leak_check

from scipy.optimize import fmin
# Set seed
np.random.seed(101)

savedir = './figs'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Protocol info
protocol_funcs = {
    'sactiv': protocols.sactiv,
    'sinactiv': protocols.sinactiv,
}

# File info
data_dir = '../data-autoLC'
file_dir = './out'
file_name = 'herg25oc1'
cell = 'D19'
temperature = 25.0 + 273.15  # in K
fit_seed = '542811797'

# Model
model_act = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                    protocol_def=protocol_funcs['sactiv'],
                    temperature=temperature,  # K
                    transform=None,
                    useFilterCap=False)  # ignore capacitive spike

model_inact = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                      protocol_def=protocol_funcs['sinactiv'],
                      temperature=temperature,  # K
                      transform=None,
                      useFilterCap=False)  # ignore capacitive spike

# Fitted parameters
param_file = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
        (file_dir, file_name, file_name, cell, fit_seed)
obtained_parameters = np.loadtxt(param_file)


# Time point
times_act = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name,
        'sactiv'), delimiter=',', skiprows=1)
times_inact = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, file_name,
        'sinactiv'), delimiter=',', skiprows=1)
times_sim_act = protocols.sactiv_times(times_act[1] - times_act[0])
times_sim_inact = protocols.sinactiv_times(times_inact[1] - times_inact[0])

# Protocol
voltage_act = model_act.voltage(times_sim_act) * 1000
voltage_inact = model_inact.voltage(times_sim_inact) * 1000
va, ta = protocols.sactiv_convert(voltage_act, times_sim_act)
vi, ti = protocols.sinactiv_convert(voltage_inact, times_sim_inact)
assert(len(ta) == len(times_act))
assert(len(ti) == len(times_inact))
assert(np.mean(np.abs(ta - times_act)) < 1e-8)
assert(np.mean(np.abs(ti - times_inact)) < 1e-8)

# Data
if True:
    data_act = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
            'sactiv', cell), delimiter=',', skiprows=1)
    data_inact = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
            'sinactiv', cell), delimiter=',', skiprows=1)
    # Re-leak correct the leak corrected data...
    for i in range(data_act.shape[1]):
        g_releak = fmin(score_leak, [0.0], args=(data_act[:, i], va[:, i], ta,
                            protocol_leak_check['sactiv']), disp=False)
        data_act[:, i] = I_releak(g_releak[0], data_act[:, i], va[:, i])
    for i in range(data_inact.shape[1]):
        g_releak = fmin(score_leak, [0.0], args=(data_inact[:, i], vi[:, i],
                            ti, protocol_leak_check['sinactiv']), disp=False)
        data_inact[:, i] = I_releak(g_releak[0], data_inact[:, i], vi[:, i])
else:
    data_act = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
            'sactiv', cell), delimiter=',', skiprows=1)
    data_inact = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
            'sinactiv', cell), delimiter=',', skiprows=1)
assert(len(data_act) == len(times_act))
assert(len(data_inact) == len(times_inact))

# Simulation
simulation_act = model_act.simulate(obtained_parameters, times_sim_act)
simulation_inact = model_inact.simulate(obtained_parameters, times_sim_inact)

# Convert simulation output format
sa, ta = protocols.sactiv_convert(simulation_act, times_sim_act)
si, ti = protocols.sinactiv_convert(simulation_inact, times_sim_inact)
assert(len(ta) == len(times_act))
assert(len(ti) == len(times_inact))
assert(np.mean(np.abs(ta - times_act)) < 1e-8)
assert(np.mean(np.abs(ti - times_inact)) < 1e-8)

# Plot
fig = plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 2)
iv_i_data = protocols.get_corrected_iv(data_act, ta,
                                       *protocols.sactiv_iv_arg(), debug=True)
iv_i_sim = protocols.get_corrected_iv(sa, ta, *protocols.sactiv_iv_arg())
plt.plot(protocols.sactiv_v() * 1e3, iv_i_data / np.max(iv_i_data),
         label='data')
plt.plot(protocols.sactiv_v() * 1e3, iv_i_sim / np.max(iv_i_sim),
         label='simulation')
plt.legend()
plt.xlabel('V [mV]')
plt.ylabel('I [normalised]')

plt.subplot(2, 2, 1)
for i in range(va.shape[1]):
    plt.plot(ta, va[:, i], c='C%s' % i)
plt.ylabel('V [mV]')

plt.subplot(2, 2, 3)
for i in range(va.shape[1]):
    plt.plot(ta, data_act[:, i], c='C%s' % i, alpha=0.5)
    plt.plot(ta, sa[:, i], c='C%s' % i)
plt.ylabel('I [pA]')
plt.xlabel('t [s]')
plt.savefig('%s/test-sactiv-%s_%s.png' % (savedir, file_name, cell))
plt.close()


fig = plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 2)
iv_i_data = protocols.get_corrected_iv(data_inact, ti,
                                       *protocols.sinactiv_iv_arg(), debug=True)
iv_i_sim = protocols.get_corrected_iv(si, ti, *protocols.sinactiv_iv_arg())
plt.plot(protocols.sinactiv_v() * 1e3, iv_i_data / np.max(iv_i_data),
         label='data')
plt.plot(protocols.sinactiv_v() * 1e3, iv_i_sim / np.max(iv_i_sim),
         label='simulation')
plt.legend()
plt.ylabel('I [normalised]')

plt.subplot(2, 2, 4)
gv_g_data = iv_i_data / (protocols.sinactiv_v() - model_inact.EK()) / 1e3
gv_g_sim = iv_i_sim / (protocols.sinactiv_v() - model_inact.EK()) / 1e3
plt.plot(protocols.sinactiv_v() * 1e3, gv_g_data,# / np.max(gv_g_data),
         label='data')
plt.plot(protocols.sinactiv_v() * 1e3, gv_g_sim,# / np.max(gv_g_sim),
         label='simulation')
EK = np.loadtxt('../qc/%s-staircaseramp-EK_all.txt' % file_name)
estimated_EK = EK[WELL_ID.index(cell)]
gv_g_data = iv_i_data / (protocols.sinactiv_v() * 1e3 - estimated_EK)
plt.plot(protocols.sinactiv_v() * 1e3, gv_g_data,# / np.max(gv_g_data), c='C0',
         ls='--', label='data (corrected EK)')
plt.legend()
plt.ylabel('g * O [nS]')
plt.xlabel('V [mV]')

plt.subplot(2, 2, 1)
for i in range(vi.shape[1]):
    plt.plot(ti, vi[:, i], c='C%s' % i)
plt.ylabel('V [mV]')

plt.subplot(2, 2, 3)
for i in range(vi.shape[1]):
    plt.plot(ti, data_inact[:, i], c='C%s' % i, alpha=0.5)
    plt.plot(ti, si[:, i], c='C%s' % i)
plt.ylabel('I [pA]')
plt.xlabel('t [s]')
plt.savefig('%s/test-sinactiv-%s_%s.png' % (savedir, file_name, cell))
plt.close()

