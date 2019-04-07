#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import protocols
import model_ikr as m
from releakcorrect import I_releak, score_leak, protocol_leak_check

from scipy.optimize import fmin

savedir = './figs'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_dir_staircase = '../data'
data_dir = '../data-autoLC'
file_dir = './out'
file_list = [
        'herg25oc1',
        'herg27oc1',
        'herg30oc1',
        'herg33oc1',
        'herg37oc3',
        ]
temperatures = np.array([25.0, 27.0, 30.0, 33.0, 37.0])
temperatures += 273.15  # in K
fit_seed = 542811797


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
prt_ylim = [
    (-1500, 2250),
    (-0.025, 1.025),
    (-3.25, 1.025),
    (-250, 2250),
    (-250, 2250),
    (-250, 2250),
    (-250, 2250),
    (-250, 2250),
    (-250, 2250),
]
prt_ylim = [
    (-0.02, 0.04),
    (-0.025, 1.025),
    (-3.25, 1.025),
    (-0.005, 0.04),
    (-0.005, 0.04),
    (-0.005, 0.04),
    (-0.005, 0.04),
    (-0.005, 0.04),
    (-0.005, 0.04),
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

# Eyring and Q10
from temperature_models import eyringA, eyringB, eyringG, eyringT
from temperature_models import q10A, q10B, q10G, q10T
from temperature_models import eyring_transform_to_model_param

eyring_mean = np.loadtxt('%s/eyring-mean.txt' % file_dir)
eyring_std = np.loadtxt('%s/eyring-std.txt' % file_dir)
q10_mean = np.loadtxt('%s/q10-mean.txt' % file_dir)
q10_std = np.loadtxt('%s/q10-std.txt' % file_dir)

# Model
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

# Plot
for i_prt, prt in enumerate(protocol_list):
    
    fig, axes = plt.subplots(2, len(temperatures), figsize=(16, 6))
    print('Plotting', prt)

    # Time point
    times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, 'herg25oc1',
        prt), delimiter=',', skiprows=1)

    # Protocol
    model = prt2model[prt]
    if prt not in protocol_iv:
        times_sim = np.copy(times)[::5]
        voltage = model.voltage(times) * 1000
    else:
        times_sim = protocol_iv_times[prt](times[1] - times[0])
        voltage = model.voltage(times_sim) * 1000
        voltage, t = protocol_iv_convert[prt](voltage, times_sim)
        assert(np.mean(np.abs(t - times)) < 1e-8)
    
    # Temperatures
    for i_T, T in enumerate(temperatures):

        axes[0, i_T].set_title(r'T = %s$^o$C' % (T - 273.15))
        if prt not in protocol_iv:
            axes[0, i_T].plot(times, voltage, c='#7f7f7f')
        else:
            for i in range(voltage.shape[1]):
                axes[0, i_T].plot(times, voltage[:, i], c='#696969')

        file_name = file_list[i_T]

        selectedfile = './manualselection/manualselected-%s.txt' % (file_name)
        selectedwell = []
        with open(selectedfile, 'r') as f:
            for l in f:
                if not l.startswith('#'):
                    selectedwell.append(l.split()[0])
        print('Getting', file_name)
        if i_T < 4:
            selectedwell = selectedwell[:50]

        # Eyring parameters
        np.random.seed(int(T))  # 'different cell at different T'
        eyring_T_mean = eyringT(eyring_mean, T)
        eyring_T_std = eyringT(eyring_std, T)
        eyring_param = np.random.normal(eyring_T_mean, eyring_T_std,
                size=(len(selectedwell), len(eyring_mean)))
        eyring_model_param = eyring_transform_to_model_param(
                eyring_param.T, T).T

        # Q10 parameters
        np.random.seed(int(T))  # 'different cell at different T'
        q10_T_mean = q10T(q10_mean, T)
        q10_T_std = q10T(q10_std, T)
        q10_param = np.random.normal(q10_T_mean, q10_T_std,
                size=(len(selectedwell), len(q10_mean)))
        q10_model_param = eyring_transform_to_model_param(q10_param.T, T).T
        
        for i_cell, cell in enumerate(selectedwell):
            # Data
            if prt == 'staircaseramp':
                data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir_staircase,
                        file_name, prt, cell), delimiter=',', skiprows=1)
            elif prt not in protocol_iv:
                data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
                        prt, cell), delimiter=',', skiprows=1)
                # Set seed
                np.random.seed(101)
                # Re-leak correct the leak corrected data...
                g_releak = fmin(score_leak, [0.0], args=(data, voltage, times,
                                    protocol_leak_check[prt]), disp=False)
                data = I_releak(g_releak[0], data, voltage)
            else:
                data = np.loadtxt('%s/%s-%s-%s.csv' % (data_dir, file_name,
                        prt, cell), delimiter=',', skiprows=1)
                for i in range(data.shape[1]):
                    # Set seed
                    np.random.seed(101)
                    g_releak = fmin(score_leak, [0.0], args=(data[:, i],
                                        voltage[:, i], times,
                                        protocol_leak_check[prt]), disp=False)
                    data[:, i] = I_releak(g_releak[0], data[:, i],
                            voltage[:, i])
            assert(len(data) == len(times))

            # Fitted parameters
            param_file = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
                    (file_dir, file_name, file_name, cell, fit_seed)
            parameters = np.loadtxt(param_file)

            # Plot
            if prt in protocol_iv:
                iv_v = protocol_iv_v[prt]() * 1000  # mV
                iv_i = protocols.get_corrected_iv(data, times,
                                                  *protocol_iv_args[prt]())
                axes[1, i_T].plot(iv_v, iv_i / np.max(iv_i), lw=0.4, alpha=0.5,
                                      c='C0', zorder=0)
            else:
                axes[1, i_T].plot(times, data / parameters[0], alpha=0.5,
                        lw=0.3, c='C0', zorder=0)

            # Individual fit
            simulation = model.simulate(parameters, times_sim)
            if prt in protocol_iv:
                simulation, t = protocol_iv_convert[prt](simulation, times_sim)
                assert(np.mean(np.abs(t - times)) < 1e-8)
                iv_v = protocol_iv_v[prt]() * 1000  # mV
                iv_i = protocols.get_corrected_iv(simulation, times,
                                                  *protocol_iv_args[prt]())
                axes[1, i_T].plot(iv_v, iv_i / np.max(iv_i), lw=0.4, alpha=0.5,
                                      c='C1', zorder=1)
            else:
                axes[1, i_T].plot(times_sim, simulation / parameters[0],
                        alpha=0.5, lw=0.4, c='C1', zorder=1)


            # Eyring
            eyring_sim = model.simulate(eyring_model_param[i_cell], times_sim)
            if prt in protocol_iv:
                eyring_sim, t = protocol_iv_convert[prt](eyring_sim, times_sim)
                assert(np.mean(np.abs(t - times)) < 1e-8)
                iv_v = protocol_iv_v[prt]() * 1000  # mV
                iv_i = protocols.get_corrected_iv(eyring_sim, times,
                                                  *protocol_iv_args[prt]())
                axes[1, i_T].plot(iv_v, iv_i / np.max(iv_i), lw=0.4, alpha=0.5,
                                      c='C2', zorder=2, label='Eyring')
            else:
                axes[1, i_T].plot(times_sim,
                                  eyring_sim / eyring_model_param[i_cell][0],
                                  alpha=0.5, lw=0.4, c='C2', zorder=2,
                                  label='Eyring')

            # Q10
            q10_sim = model.simulate(q10_model_param[i_cell], times_sim)
            if prt in protocol_iv:
                q10_sim, t = protocol_iv_convert[prt](q10_sim, times_sim)
                assert(np.mean(np.abs(t - times)) < 1e-8)
                iv_v = protocol_iv_v[prt]() * 1000  # mV
                iv_i = protocols.get_corrected_iv(q10_sim, times,
                                                  *protocol_iv_args[prt]())
                axes[1, i_T].plot(iv_v, iv_i / np.max(iv_i), lw=0.4, alpha=0.5,
                                      c='C3', zorder=3, label='Q10')
                axes[1, i_T].grid()
            else:
                axes[1, i_T].plot(times_sim,
                                  q10_sim / q10_model_param[i_cell][0],
                                  alpha=0.5, lw=0.4, c='C3', zorder=3,
                                  label='Q10')

        axes[1, i_T].set_ylim(prt_ylim[i_prt])

    # Save fig
    axes[1, 2].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Voltage [mV]')
    axes[1, 0].set_ylabel('Current [pA]')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('%s/quick-plots/%s.png' % (savedir, prt), bbox_iches='tight')
    plt.close('all')
