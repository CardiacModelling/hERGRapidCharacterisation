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
from protocols import est_g_staircase
import model_ikr as m

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

savedir = './figs/Zhou1998/quick-test-Q10s/'
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

# Colours for fan chart
fan_blue = ['#b5c7d5',
        '#adc1d0',
        '#91abbc',
        '#85a0b1',
        '#6b8fa9',
        '#62869f',
        '#587c96',
        '#477390',
        '#3f6c88',
    ]


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
prt_ylim = { # no normalisation
    'staircaseramp': (-1500, 2250),
    'sactiv': (-0.025, 1.025),
    'sinactiv': (-3.25, 1.025),
    'pharma': (-250, 2250),
    'apab': (-250, 2250),
    'apabv3': (-250, 2250),
    'ap05hz': (-250, 2250),
    'ap1hz': (-250, 2250),
    'ap2hz':(-250, 2250),
    }
prt_ylim = { # normalise with fitted conductance value
    'staircaseramp': (-0.02, 0.04),
    'sactiv': (-0.025, 1.025),
    'sinactiv': (-3.25, 1.025),
    'pharma': (-0.005, 0.04),
    'apab': (-0.005, 0.04),
    'apabv3': (-0.005, 0.04),
    'ap05hz': (-0.005, 0.04),
    'ap1hz': (-0.005, 0.04),
    'ap2hz': (-0.005, 0.04),
    }
prt_ylim = { # normalise with extrapolated -120 mV spike value
    'staircaseramp': (-1.0, 1.5),
    'sactiv': (-0.025, 1.025),
    'sinactiv': (-3.25, 1.025),
    'pharma': (-0.25, 1.5),
    'apab': (-0.25, 1.5),
    'apabv3': (-0.25, 1.5),
    'ap05hz': (-0.25, 1.5),
    'ap1hz': (-0.25, 1.5),
    'ap2hz': (-0.25, 1.5),
    }

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

zoomin = {
    'staircaseramp': [(1.8, 2.5), (14.3, 15.0)],
    'pharma': [(0.64, 0.66), (1.14, 1.16)],
    'apab': [(0.0475, 0.0575), (0.32, 0.33)],
    'apabv3': [(0.05, 0.07), (0.55, 0.70)],
    'ap05hz': [(0.04, 0.07), (2.04, 2.07)],
    'ap1hz': [(0.04, 0.07),
              # (1.04, 1.07),
              # (2.04, 2.07), 
              (3.04, 3.07)],
    'ap2hz': [(0.045, 0.06),
              # (0.545, 0.56),
              # (1.045, 1.06), (1.545, 1.56),
              # (2.045, 2.06), (2.545, 2.56),
              (3.045, 3.06)],
    'sactiv': None,
    'sinactiv': None,
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

# Zhou et al. 1998 Q10
q10_tref = 273.15 + 25  # K
mean_param_tref = transform_to_model_param(np.mean(mean_chains[0], axis=0))
q10_A = mean_param_tref[[1, 3, 5, 7]]
q10_zhou = np.array([6.25, 1.36, 3.55, 3.65])  # from the paper
q10_zhou_re = np.array([10.668, 2.016, 3.421, 2.991])  # re-est. from models
a_zhou = np.log(q10_zhou) / 10.
c_zhou = np.log(q10_A) - np.log(q10_zhou) * q10_tref / 10.
a_zhou_re = np.log(q10_zhou_re) / 10.
c_zhou_re = np.log(q10_A) - np.log(q10_zhou_re) * q10_tref / 10.
b_zhou = b_zhou_re = mean_param_tref[[2, 4, 6, 8]]
q10_param_zhou = [q10_mean[0]]
q10_param_zhou_re = [q10_mean[0]]
for i in range(4):
    q10_param_zhou.append([a_zhou[i], c_zhou[i]])
    q10_param_zhou.append([b_zhou[i], np.NaN])
    q10_param_zhou_re.append([a_zhou_re[i], c_zhou_re[i]])
    q10_param_zhou_re.append([b_zhou_re[i], np.NaN])
q10_param_zhou = np.asarray(q10_param_zhou)
q10_param_zhou_re = np.asarray(q10_param_zhou_re)

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
norm_sim_all = []
norm_eyring_all = []
norm_q10_all = []
norm_q10_1_all = []
norm_q10_2_all = []
for i_prt, prt in enumerate(protocol_list):
    
    if prt not in protocol_iv: 
        fig, axes = plt.subplots(4, len(temperatures), figsize=(16, 6))
    else:
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
    for i_T, (file_name, T) in enumerate(zip(file_list, temperatures)):

        axes[0, i_T].set_title(r'T = %s$^o$C' % (T - 273.15))
        if prt not in protocol_iv:
            axes[0, i_T].plot(times, voltage, c='#7f7f7f')
        else:
            for i in range(voltage.shape[1]):
                axes[0, i_T].plot(times, voltage[:, i], c='#696969')

        # Data
        data_fancharts_dir = './out/data-fancharts'
        percentiles = np.loadtxt('%s/percentiles.txt' % data_fancharts_dir)
        fan_chart_data_top = np.loadtxt('%s/%s-%s-top.txt' % \
                (data_fancharts_dir, file_name, prt))
        fan_chart_data_bot = np.loadtxt('%s/%s-%s-bot.txt' % \
                (data_fancharts_dir, file_name, prt))
        if prt not in protocol_iv:
            fan_x = np.loadtxt('%s/%s-%s-times.txt' % \
                    (data_fancharts_dir, file_name, prt))
        else:
            fan_x = np.loadtxt('%s/%s-%s-voltage.txt' % \
                    (data_fancharts_dir, file_name, prt))

        # Zoom in
        if prt not in protocol_iv:
            twin1d = np.logical_and(fan_x > zoomin[prt][0][0],
                                    fan_x < zoomin[prt][0][1])
            twin2d = np.logical_and(fan_x > zoomin[prt][1][0],
                                    fan_x < zoomin[prt][1][1])
            twin1 = np.logical_and(times_sim > zoomin[prt][0][0],
                                   times_sim < zoomin[prt][0][1])
            twin2 = np.logical_and(times_sim > zoomin[prt][1][0],
                                   times_sim < zoomin[prt][1][1])

        # Plot data
        for i_p, p in enumerate(percentiles):
            alpha = 0.8
            color = fan_blue[i_p]
            top = fan_chart_data_top[:, i_p]
            bot = fan_chart_data_bot[:, i_p]
            axes[1, i_T].fill_between(fan_x, top, bot, color=color,
                    alpha=alpha, linewidth=0)
            if prt not in protocol_iv:
                axes[2, i_T].fill_between(fan_x[twin1d],
                        top[twin1d], bot[twin1d],
                        alpha=alpha, linewidth=0, color=color)
                axes[3, i_T].fill_between(fan_x[twin2d],
                        top[twin2d], bot[twin2d],
                        alpha=alpha, linewidth=0, color=color)

        # HBM mean parameters
        hbm_T_mean = transform_to_model_param(
                np.mean(mean_chains[i_T], axis=0))

        # Eyring parameters
        eyring_T_mean = eyringT(eyring_mean, T)
        eyring_model_param = eyring_transform_to_model_param(eyring_T_mean, T)

        # Q10 parameters
        q10_T_mean = q10T(q10_mean, T)
        q10_model_param = eyring_transform_to_model_param(q10_T_mean, T)

        # Q10 parameters from Zhou et al. 1998
        q10_zhou_T = q10T(q10_param_zhou, T)
        q10_zhou_model_param = eyring_transform_to_model_param(
                q10_zhou_T, T)
        q10_zhou_re_T = q10T(q10_param_zhou_re, T)
        q10_zhou_re_model_param = eyring_transform_to_model_param(
                q10_zhou_re_T, T)

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
        # norm_sim = hbm_T_mean[0]
        if prt == 'staircaseramp':
            norm_sim = est_g_staircase(simulation, times_sim,
                                       p0=[800, 0.025], debug=False)
            norm_sim_all.append(norm_sim)
        else:
            norm_sim = norm_sim_all[i_T]
        if prt in protocol_iv:
            simulation, t = protocol_iv_convert[prt](simulation, times_sim)
            assert(np.mean(np.abs(t - times)) < 1e-8)
            iv_v = protocol_iv_v[prt]() * 1000  # mV
            iv_i = protocols.get_corrected_iv(simulation, times,
                                              *protocol_iv_args[prt]())
            axes[1, i_T].plot(iv_v, iv_i / np.max(iv_i), lw=1.5, alpha=1,
                                  c='C1', zorder=1, label='HBM mean')
        else:
            axes[1, i_T].plot(times_sim, simulation / norm_sim, alpha=1,
                    lw=1.5, c='C1', zorder=1, label='HBM mean')
            axes[2, i_T].plot(times_sim[twin1],
                    simulation[twin1] / norm_sim,
                    lw=1.5, c='C1', zorder=1, label='HBM mean')
            axes[3, i_T].plot(times_sim[twin2],
                    simulation[twin2] / norm_sim,
                    lw=1.5, c='C1', zorder=1, label='HBM mean')

        # Eyring
        eyring_sim = model.simulate(eyring_model_param, times_sim)
        # norm_eyring = eyring_model_param[0]
        if prt == 'staircaseramp':
            norm_eyring = est_g_staircase(eyring_sim, times_sim,
                                          p0=[800, 0.025], debug=False)
            norm_eyring_all.append(norm_eyring)
        else:
            norm_eyring = norm_eyring_all[i_T]
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
                              eyring_sim / norm_eyring,
                              alpha=1, lw=1.5, c='C2', zorder=2,
                              label='Eyring')
            axes[2, i_T].plot(times_sim[twin1],
                              eyring_sim[twin1] / norm_eyring,
                              alpha=1, lw=1.5, c='C2', zorder=2,
                              label='Eyring')
            axes[3, i_T].plot(times_sim[twin2],
                              eyring_sim[twin2] / norm_eyring,
                              alpha=1, lw=1.5, c='C2', zorder=2,
                              label='Eyring')

        # Q10
        q10_sim = model.simulate(q10_model_param, times_sim)
        # norm_q10 = q10_model_param[0]
        if prt == 'staircaseramp':
            norm_q10 = est_g_staircase(q10_sim, times_sim, p0=[800, 0.025],
                                        debug=False)
            norm_q10_all.append(norm_q10)
        else:
            norm_q10 = norm_q10_all[i_T]
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
                              q10_sim / norm_q10,
                              alpha=1, lw=1.5, c='C3', zorder=3, label='Q10')
            axes[2, i_T].plot(times_sim[twin1],
                              q10_sim[twin1] / norm_q10,
                              alpha=1, lw=1.5, c='C3', zorder=3, label='Q10')
            axes[3, i_T].plot(times_sim[twin2],
                              q10_sim[twin2] / norm_q10,
                              alpha=1, lw=1.5, c='C3', zorder=3, label='Q10')

        # Q10 from Zhou et al. 1998
        q10_sim_1 = model.simulate(q10_zhou_model_param, times_sim)
        q10_sim_2 = model.simulate(q10_zhou_re_model_param, times_sim)
        # norm_q10_1 = q10_zhou_model_param[0]
        # norm_q10_2 = q10_zhou_re_model_param[0]
        if prt == 'staircaseramp':
            norm_q10_1 = est_g_staircase(q10_sim_1, times_sim, p0=[800, 0.025],
                                        debug=False)
            norm_q10_1_all.append(norm_q10_1)
            norm_q10_2 = est_g_staircase(q10_sim_2, times_sim, p0=[800, 0.025],
                                        debug=False)
            norm_q10_2_all.append(norm_q10_2)
        else:
            norm_q10_1 = norm_q10_1_all[i_T]
            norm_q10_2 = norm_q10_2_all[i_T]
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
                                  label='Zhou et al. 1998')
            axes[1, i_T].plot(iv_v, iv_i_2 / np.max(iv_i_2), lw=1.5, alpha=1,
                                  c='C5', zorder=5,
                                  label='Zhou et al. 1998 re-est.')
            axes[1, i_T].grid()
        else:
            axes[1, i_T].plot(times_sim,
                              q10_sim_1 / norm_q10_1,
                              alpha=1, lw=1.5, c='C4', zorder=4,
                              label='Zhou et al. 1998')
            axes[1, i_T].plot(times_sim,
                              q10_sim_2 / norm_q10_2,
                              alpha=1, lw=1.5, c='C5', zorder=5,
                              label='Zhou et al. 1998 re-est.')
            axes[2, i_T].plot(times_sim[twin1],
                              q10_sim_1[twin1] / norm_q10_1,
                              alpha=1, lw=1.5, c='C4', zorder=4,
                              label='Zhou et al. 1998')
            axes[2, i_T].plot(times_sim[twin1],
                              q10_sim_2[twin1] / norm_q10_2,
                              alpha=1, lw=1.5, c='C5', zorder=5,
                              label='Zhou et al. 1998 re-est.')
            axes[3, i_T].plot(times_sim[twin2],
                              q10_sim_1[twin2] / norm_q10_1,
                              alpha=1, lw=1.5, c='C4', zorder=4,
                              label='Zhou et al. 1998')
            axes[3, i_T].plot(times_sim[twin2],
                              q10_sim_2[twin2] / norm_q10_2,
                              alpha=1, lw=1.5, c='C5', zorder=5,
                              label='Zhou et al. 1998 re-est.')

        axes[1, i_T].set_ylim(prt_ylim[prt])

    # Save fig
    axes[1, 0].legend()
    axes[1, 2].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Voltage [mV]')
    axes[1, 0].set_ylabel('Current [pA]')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('%s/%s.png' % (savedir, prt),
            bbox_iches='tight')
    plt.close('all')
