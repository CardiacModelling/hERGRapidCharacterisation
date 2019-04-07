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

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

savedir = './figs/Vandenberg2006/quick-test-Q10s/'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_dir_staircase = data_dir = '../data'
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

# Load pseudo2hbm
mean_chains = []
for i_temperature, (file_name, temperature) in enumerate(zip(file_list,
    temperatures)):

    load_file = './out-mcmc/%s-pseudo2hbm-lognorm-mean.txt' % (file_name[:-1])
    mean_chain = np.loadtxt(load_file)  # transformed

    mean_chains.append(mean_chain)
mean_chains = np.asarray(mean_chains)

# Eyring and Q10
from temperature_models import eyringA, eyringB, eyringG, eyringT
from temperature_models import q10A, q10B, q10G, q10T
from temperature_models import eyring_transform_to_model_param

eyring_mean = np.loadtxt('%s/eyring-mean.txt' % file_dir)
q10_mean = np.loadtxt('%s/q10-mean.txt' % file_dir)

# Vandenberg et al. 2006 Q10
q10_tref = 273.15 + 25  # K
mean_param_tref = transform_to_model_param(np.mean(mean_chains[0], axis=0))
q10_A = mean_param_tref[[1, 3, 5, 7]]
q10_jamie = np.array([2.1, 1.7, 2.5, 2.6])  # from the paper
q10_jamie_re = np.array([7.400, 3.692, 2.750, 4.436])  # re-est. from models
a_jamie = np.log(q10_jamie) / 10.
c_jamie = np.log(q10_A) - np.log(q10_jamie) * q10_tref / 10.
a_jamie_re = np.log(q10_jamie_re) / 10.
c_jamie_re = np.log(q10_A) - np.log(q10_jamie_re) * q10_tref / 10.
b_jamie = b_jamie_re = mean_param_tref[[2, 4, 6, 8]]
q10_param_jamie = [q10_mean[0]]
q10_param_jamie_re = [q10_mean[0]]
for i in range(4):
    q10_param_jamie.append([a_jamie[i], c_jamie[i]])
    q10_param_jamie.append([b_jamie[i], np.NaN])
    q10_param_jamie_re.append([a_jamie_re[i], c_jamie_re[i]])
    q10_param_jamie_re.append([b_jamie_re[i], np.NaN])
q10_param_jamie = np.asarray(q10_param_jamie)
q10_param_jamie_re = np.asarray(q10_param_jamie_re)

if False:
    # Quick check above method works for the q10_mean
    q10_test = np.array([13.128, 1.198, 3.085, 4.086])
    a_test = np.log(q10_test) / 10.
    c_test = np.log(q10_A) - np.log(q10_test) * q10_tref / 10.
    b_test = mean_param_tref[[2, 4, 6, 8]]
    q10_param_test = [q10_mean[0]]
    for i in range(4):
        q10_param_test.append([a_test[i], c_test[i]])
        q10_param_test.append([b_test[i], np.NaN])
    q10_param_test = np.asarray(q10_param_test)
    print(q10_param_test)
    print(q10_mean)
    sys.exit()

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
    if prt not in protocol_iv:
        times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, 'herg25oc1',
                prt), delimiter=',', skiprows=1)
    else:
        times = np.loadtxt('%s/%s-%s-times.csv' % ('../data-autoLC',
                'herg25oc1', prt), delimiter=',', skiprows=1)

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

        # HBM mean parameters
        hbm_T_mean = transform_to_model_param(
                np.mean(mean_chains[i_T], axis=0))

        # Eyring parameters
        eyring_T_mean = eyringT(eyring_mean, T)
        eyring_model_param = eyring_transform_to_model_param(eyring_T_mean, T)

        # Q10 parameters
        q10_T_mean = q10T(q10_mean, T)
        q10_model_param = eyring_transform_to_model_param(q10_T_mean, T)

        # Q10 parameters from Vandenberg et al. 2006
        q10_jamie_T = q10T(q10_param_jamie, T)
        q10_jamie_model_param = eyring_transform_to_model_param(
                q10_jamie_T, T)
        q10_jamie_re_T = q10T(q10_param_jamie_re, T)
        q10_jamie_re_model_param = eyring_transform_to_model_param(
                q10_jamie_re_T, T)

        if (i_T == 4 or i_T == 0) and ('--showparam' in sys.argv):
            print('temperature', T - 273.15)
            print('HBM: ', hbm_T_mean)
            print('Eyring: ', eyring_model_param)
            print('RMSD Eyring: ',
                    np.sqrt(np.mean(
                            (hbm_T_mean[1:] - eyring_model_param[1:]) ** 2)))
            print('Q10: ', q10_model_param)
            print('RMSD Q10: ',
                    np.sqrt(np.mean(
                            (hbm_T_mean[1:] - q10_model_param[1:]) ** 2)))
        
        # Mean individual cells fit
        simulation = model.simulate(hbm_T_mean, times_sim)
        if prt in protocol_iv:
            simulation, t = protocol_iv_convert[prt](simulation, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-8)
            iv_v = protocol_iv_v[prt]() * 1000  # mV
            iv_i = protocols.get_corrected_iv(simulation, times,
                                              *protocol_iv_args[prt]())
            axes[1, i_T].plot(iv_v, iv_i / np.max(iv_i), lw=1.5, alpha=1,
                                  c='C1', zorder=1, label='HBM mean')
        else:
            axes[1, i_T].plot(times_sim, simulation / hbm_T_mean[0], alpha=1,
                    lw=1.5, c='C1', zorder=1, label='HBM mean')
    

        # Eyring
        eyring_sim = model.simulate(eyring_model_param, times_sim)
        if prt in protocol_iv:
            eyring_sim, t = protocol_iv_convert[prt](eyring_sim, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-8)
            iv_v = protocol_iv_v[prt]() * 1000  # mV
            iv_i = protocols.get_corrected_iv(eyring_sim, times,
                                              *protocol_iv_args[prt]())
            axes[1, i_T].plot(iv_v, iv_i / np.max(iv_i), lw=1.5, alpha=1,
                                  c='C2', zorder=2, label='Eyring')
        else:
            axes[1, i_T].plot(times_sim,
                              eyring_sim / eyring_model_param[0],
                              alpha=1, lw=1.5, c='C2', zorder=2,
                              label='Eyring')

        # Q10
        q10_sim = model.simulate(q10_model_param, times_sim)
        if prt in protocol_iv:
            q10_sim, t = protocol_iv_convert[prt](q10_sim, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-8)
            iv_v = protocol_iv_v[prt]() * 1000  # mV
            iv_i = protocols.get_corrected_iv(q10_sim, times,
                                              *protocol_iv_args[prt]())
            axes[1, i_T].plot(iv_v, iv_i / np.max(iv_i), lw=1.5, alpha=1,
                                  c='C3', zorder=3, label='Q10')
        else:
            axes[1, i_T].plot(times_sim,
                              q10_sim / q10_model_param[0],
                              alpha=1, lw=1.5, c='C3', zorder=3, label='Q10')

        # Q10 from Vandenberg et al. 2006
        q10_sim_1 = model.simulate(q10_jamie_model_param, times_sim)
        q10_sim_2 = model.simulate(q10_jamie_re_model_param, times_sim)
        if prt in protocol_iv:
            q10_sim_1, t = protocol_iv_convert[prt](q10_sim_1, times_sim)
            q10_sim_2, t = protocol_iv_convert[prt](q10_sim_2, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-8)
            iv_v = protocol_iv_v[prt]() * 1000  # mV
            iv_i_1 = protocols.get_corrected_iv(q10_sim_1, times,
                                                *protocol_iv_args[prt]())
            iv_i_2 = protocols.get_corrected_iv(q10_sim_2, times,
                                                *protocol_iv_args[prt]())
            axes[1, i_T].plot(iv_v, iv_i_1 / np.max(iv_i_1), lw=1.5, alpha=1,
                                  c='C4', zorder=4,
                                  label='Vandenberg et al. 2006')
            axes[1, i_T].plot(iv_v, iv_i_2 / np.max(iv_i_2), lw=1.5, alpha=1,
                                  c='C5', zorder=5,
                                  label='Vandenberg et al. 2006 re-est.')
            axes[1, i_T].grid()
        else:
            axes[1, i_T].plot(times_sim,
                              q10_sim_1 / q10_jamie_model_param[0],
                              alpha=1, lw=1.5, c='C4', zorder=4,
                              label='Vandenberg et al. 2006')
            axes[1, i_T].plot(times_sim,
                              q10_sim_2 / q10_jamie_re_model_param[0],
                              alpha=1, lw=1.5, c='C5', zorder=5,
                              label='Vandenberg et al. 2006 re-est.')

        axes[1, i_T].set_ylim(prt_ylim[i_prt])

    # Save fig
    axes[1, 0].legend()
    axes[1, 2].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Voltage [mV]')
    axes[1, 0].set_ylabel('Current [pA]')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('%s/%s.png' % (savedir, prt),
            bbox_iches='tight')
    plt.close('all')
