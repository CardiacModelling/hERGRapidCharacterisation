#!/usr/bin/env python2
# 
# Try to reproduce similar figures in Vandenberg et al. 2006
# In particular its Figure 3 and 5.
#
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import myokit
import pickle

import model_ikr as m
from protocols import Vandenberg2006_isochronal_tail_current as prt_act
from protocols import Vandenberg2006_double_pulse as prt_inact

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

plot_kylie = False

debug = False
common_conductance = 1.0
fit_seed = 542811797
temperatures = ['25.0', '33.0'][::-1]
color_mid = {'25.0':'#6baed6',
         '33.0':'#fd8d3c',
         '37.0':'C2',}


file_list = {
        '25.0': 'herg25oc',
        '33.0': 'herg33oc',
        '37.0': 'herg37oc',
    }

# Fan chart
fan_red = [
    '#ec9999',
    # '#e77c7c',
    # '#e25f60',
    '#dd4243',
    # '#d62728',  #
    '#b92223',
    '#9d1d1d',
    # '#801718',
]

fan_blue = ['#b5c7d5',
        # '#adc1d0',
        # '#91abbc',
        '#85a0b1',
        # '#6b8fa9',
        # '#62869f',
        '#587c96',
        # '#477390',
        '#3f6c88',
    ]

fan_green = [
        '#94e294',
        # '#7ada7a',
        # '#5fd35f',
        '#52cf52',  # 4c4?
        # '#3b3',
        # '#2ca02c',  #
        '#289328',
        '#1d6b1d',
    ]

color_fan = {
        '25.0':fan_blue,
        '33.0':fan_red,
        '37.0':fan_green,
        }

# Load pseudo2hbm
mean_chains = []
cov_chains = []
for temperature in temperatures:
    file_name = file_list[temperature]

    load_file = './out-mcmc/%s-pseudo2hbm-lognorm-mean.txt' % (file_name)
    mean_chain = np.loadtxt(load_file)  # transformed

    load_file = './out-mcmc/%s-pseudo2hbm-lognorm-cov.pkl' % (file_name)
    cov_chain = pickle.load(open(load_file, "rb"))  # transformed

    mean_chains.append(mean_chain)
    cov_chains.append(cov_chain)
mean_chains = np.asarray(mean_chains)
cov_chains = np.asarray(cov_chains)

'''
# Eyring and Q10
from temperature_models import eyringA, eyringB, eyringG, eyringT
from temperature_models import q10A, q10B, q10G, q10T
from temperature_models import eyring_transform_to_model_param

eyring_mean = np.loadtxt('%s/eyring-mean.txt' % file_dir)
q10_mean = np.loadtxt('%s/q10-mean.txt' % file_dir)
'''


#
# Functions to get I-V curves
#
def get_IV(folded_current, n_steps, t_start, t_end):
    # Simple method to find minimum or maximum
    times = folded_current['time']
    time_window = np.where(np.logical_and(times > t_start, times <= t_end))[0]
    time_window_90 = time_window[int(len(time_window) * 0.05):
            int(len(time_window) * 0.25)]
    I = []
    for i in range(n_steps):
        if (folded_current[str(i) + '.current'][time_window_90] <= 0).all():
            peak_I = np.min(folded_current[str(i) + '.current'][time_window])
        else:
            peak_I = np.max(folded_current[str(i) + '.current'][time_window])
        I.append(peak_I)
    return I


def get_corrected_IV(folded_current, n_steps, t_start, t_end, debug=False):
    # use 2-parameters exponential fit to the tail
    import scipy
    def exp_func(t, a, b):
        # do a "proper exponential" decay fit
        # i.e. shift the t to t' where t' has zero at the start of the 
        # voltage step
        return - a * np.exp( -b * (t - x[0]))
    times = folded_current['time']
    time_window = np.where(np.logical_and(times > t_start, times <= t_end))[0]
    I = np.zeros(n_steps)
    i_trim = 200  # assuming DT=1e-4 s -> 20 ms
    i_fit_until = 1000  # assuming DT=1e-4 s -> 100 ms
    if debug:
        fig = plt.figure()
    for i in range(n_steps):
        # trim off the first i_trim (20ms) in case it is still shooting down...
        x = times[time_window[0] + i_trim:time_window[0] + i_fit_until]
        y = folded_current[str(i) + '.current'][time_window[0] + i_trim:
                                                time_window[0] + i_fit_until]
        # if np.mean(y) < 0:
        try:
            popt, pcov = scipy.optimize.curve_fit(exp_func, x, y)
            fitted = exp_func(times[time_window[0]:
                                    time_window[0] + i_fit_until], *popt)
            I[i] = np.max(fitted[0])
        except:
            raise Exception('CANNOT FIT TO voltage step %d' % i)
        # else:
        #     I[i] = np.max(y)
        if debug:
            plt.plot(times[time_window[0] - 500:time_window[-1] + 500],
                     folded_current[str(i) + '.current'][time_window[0] -
                         500:time_window[-1] + 500],
                         c='#d62728' if i != 0 else 'C1',
                         zorder=0 if i != 0 else 10)
            plt.plot(times[time_window[0]:time_window[0] + i_fit_until],
                    fitted, '--', c='#1f77b4', zorder=0 if i != 0 else 10)
            plt.plot(times[time_window][0], I[i], 'kx')
    if debug:
        plt.axvline(x=times[time_window[0] + i_trim])
        plt.axvline(x=times[time_window[0] + i_fit_until])
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Vandenberg2006/fig5a2/'
                    'Vandenberg2006-fig5a2-%sC-%s-%s.png'
                    % (temperature, file_name, cell))
            plt.close()

        # Plot Figure 5B2 for this cell too
        plt.plot(I)
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Vandenberg2006/fig5a2/'
                    'Vandenberg2006-fig5a2-%sC-%s-%s-2.png'
                    % (temperature, file_name, cell))
            plt.close()
    return I


# Simulate

times_act, ttotal_act, tmeasure_act = prt_act(None, return_times=True)
I_activations = {}
I_activations_cov = {}
av_steps = prt_act(None, return_voltage=True)

times_inact, ttotal_inact, tmeasure_inact = prt_inact(None, return_times=True)
I_inactivations = {}
g_inactivations = {}
g_inactivations_cov = {}
iv_steps = prt_inact(None, return_voltage=True)

p_cov = {}

for i_T, temperature in enumerate(temperatures):

    I_activations[temperature] = []
    I_inactivations[temperature] = []
    g_inactivations[temperature] = []

    # Model
    model_act = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                        protocol_def=prt_act,
                        temperature=273.15 + float(temperature),  # K
                        transform=None,
                        useFilterCap=False,  # ignore capacitive spike
                        effEK=False,  # OK to switch this off here
                        concK=[4.8 + 0.3, 120 + 20])
    model_inact = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                          protocol_def=prt_inact,
                          temperature=273.15 + float(temperature),  # K
                          transform=None,
                          useFilterCap=False,  # ignore capacitive spike
                          effEK=False,  # OK to switch this off here
                          concK=[4.8 + 0.3, 120 + 20])

    T = 273.15 + float(temperature)

    # HBM mean parameters
    hbm_T_mean = transform_to_model_param(
            np.mean(mean_chains[i_T], axis=0))

    '''
    # Eyring parameters
    eyring_T_mean = eyringT(eyring_mean, T)
    eyring_param = eyring_transform_to_model_param(eyring_T_mean, T)

    # Q10 parameters
    q10_T_mean = q10T(q10_mean, T)
    q10_param = eyring_transform_to_model_param(q10_T_mean, T)
    '''

    p = hbm_T_mean
    p[0] = common_conductance

    a = myokit.DataLog()
    a['time'] = times_act
    a['current'] = model_act.simulate(p, times_act)
    a['voltage'] = model_act.voltage(times_act)
    a.set_time_key('time')
    a = a.fold(ttotal_act)

    i = myokit.DataLog()
    i['time'] = times_inact
    i['current'] = model_inact.simulate(p, times_inact)
    i['voltage'] = model_inact.voltage(times_inact)
    i.set_time_key('time')
    i = i.fold(ttotal_inact)

    if debug and False:
        # Figure 3A
        for ii in range(len(av_steps)):
            plt.plot(a['time'], a[str(ii)+'.current'])
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Vandenberg2006/' + 
                    'Vandenberg2006-fig3a-%sC.png'%temperature)
            plt.close()

    if debug and False:
        # Figure 5A
        for ii in range(len(iv_steps)):
            plt.plot(i['time'], i[str(ii)+'.current'])
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Vandenberg2006/'
                    + 'Vandenberg2006-fig5a-%sC.png'%temperature)
            plt.close()

    I_activation = get_IV(a, len(av_steps),
                          tmeasure_act, tmeasure_act + 0.5)
    I_activations[temperature].append(I_activation)

    I_inactivation = get_corrected_IV(i, len(iv_steps), 
                                      tmeasure_inact,
                                      tmeasure_inact + 0.5,
                                      debug=False)
    I_inactivations[temperature].append(I_inactivation)

    conductance_inactivation = \
            I_inactivation / (iv_steps - model_inact.EK())
    g_inactivations[temperature].append(conductance_inactivation)

    del(a, i)

    np.random.seed(101)
    i_act_cov_all = []
    g_inact_cov_all = []
    p_cov_all = []
    for s in range(50):

        p = np.random.multivariate_normal(mean_chains[i_T][s, :],
                cov_chains[i_T][s, :, :])
        p = transform_to_model_param(p)
        p[0] = common_conductance

        a = myokit.DataLog()
        a['time'] = times_act
        a['current'] = model_act.simulate(p, times_act)
        a['voltage'] = model_act.voltage(times_act)
        a.set_time_key('time')
        a = a.fold(ttotal_act)

        i = myokit.DataLog()
        i['time'] = times_inact
        i['current'] = model_inact.simulate(p, times_inact)
        i['voltage'] = model_inact.voltage(times_inact)
        i.set_time_key('time')
        i = i.fold(ttotal_inact)

        I_activation = get_IV(a, len(av_steps),
                              tmeasure_act, tmeasure_act + 0.5)
        I_activation = I_activation / np.min(I_activation)
        i_act_cov_all.append(I_activation)

        I_inactivation = get_corrected_IV(i, len(iv_steps), 
                                          tmeasure_inact,
                                          tmeasure_inact + 0.5,
                                          debug=False)
        conductance_inactivation = \
                I_inactivation / (iv_steps - model_inact.EK())
        conductance_inactivation = conductance_inactivation \
                / np.max(conductance_inactivation)
        g_inact_cov_all.append(conductance_inactivation)

        av_steps_tmp = np.around(np.array(av_steps) * 1e3, 1)
        iv_steps_tmp = np.around(np.array(iv_steps) * 1e3, 1)
        v_steps = iv_steps_tmp[:]
        po = []
        for v in v_steps:
            # act
            if v in av_steps_tmp:
                i = np.where(np.abs(av_steps_tmp - v) < 1e-5)[0][0]
                act = I_activation[i]
            elif v > 30.0:  # mV
                act = 1.0  # approx. as fully open
            elif v < -70.0:  # mV
                act = 0.0  # approx. as fully closed
            else:
                print('...')
            # inact
            if v in iv_steps_tmp:
                i = np.where(np.abs(iv_steps_tmp - v) < 1e-5)[0][0]
                inact = conductance_inactivation[i]
            else:
                print('...')
            po.append(act * inact)

        p_cov_all.append(po)

        del(a, i)

    I_activations_cov[temperature] = []
    g_inactivations_cov[temperature] = []
    p_cov[temperature] = []

    percentiles = [90, 60, 30]
    fan_chart_data_top = []
    fan_chart_data_bot = []
    for i_p, p in enumerate(percentiles):
        top = np.nanpercentile(i_act_cov_all, 50 + p / 2., axis=0)
        bot = np.nanpercentile(i_act_cov_all, 50 - p / 2., axis=0)
        I_activations_cov[temperature].append([top, bot])

        top = np.nanpercentile(g_inact_cov_all, 50 + p / 2., axis=0)
        bot = np.nanpercentile(g_inact_cov_all, 50 - p / 2., axis=0)
        g_inactivations_cov[temperature].append([top, bot])

        top = np.nanpercentile(p_cov_all, 50 + p / 2., axis=0)
        bot = np.nanpercentile(p_cov_all, 50 - p / 2., axis=0)
        p_cov[temperature].append([top, bot])


# Kylie
if plot_kylie:
    # Model
    model_act = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                        protocol_def=prt_act,
                        temperature=273.15 + 22,  # K
                        transform=None,
                        useFilterCap=False,  # ignore capacitive spike
                        effEK=False,  # OK to switch this off here
                        concK=[4.8 + 0.3, 120 + 20])
    model_inact = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                          protocol_def=prt_inact,
                          temperature=273.15 + 22,  # K
                          transform=None,
                          useFilterCap=False,  # ignore capacitive spike
                          effEK=False,  # OK to switch this off here
                          concK=[4.8 + 0.3, 120 + 20])
    path_to_solutions = '../room-temperature-only/kylie-room-temperature'
    last_solution = path_to_solutions + '/last-solution_log-mean.txt'
    obtained_parameters = np.loadtxt(last_solution)
    obtained_parameters[0] = common_conductance

    a = myokit.DataLog()
    a['time'] = times_act
    a['current'] = model_act.simulate(obtained_parameters, times_act)
    a['voltage'] = model_act.voltage(times_act)
    a.set_time_key('time')
    a = a.fold(ttotal_act)

    i = myokit.DataLog()
    i['time'] = times_inact
    i['current'] = model_inact.simulate(obtained_parameters, times_inact)
    i['voltage'] = model_inact.voltage(times_inact)
    i.set_time_key('time')
    i = i.fold(ttotal_inact)

    kI_activation = get_IV(a, len(av_steps),
                          tmeasure_act, tmeasure_act + 0.5)

    kI_inactivation = get_corrected_IV(i, len(iv_steps), 
                                      tmeasure_inact,
                                      tmeasure_inact + 0.5,
                                      debug=False)

    kg_inactivation = \
            kI_inactivation / (iv_steps - model_inact.EK())
    del(a, i)

# convert units V -> mV
av_steps = np.around(np.array(av_steps) * 1e3, 1)
iv_steps = np.around(np.array(iv_steps) * 1e3, 1)


# Figure 3C
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9, 4))

for temperature in temperatures:
    for i, I_activation in enumerate(I_activations[temperature]):
        axes[1].plot(av_steps, I_activation/np.min(I_activation),
                 c=color_mid[temperature], lw=1.5,
                 label='__nolegend__' if i else temperature+' $^o$C')
# fan charts
for temperature in temperatures:
    for i_p, p in enumerate(percentiles):
        alpha = 0.8
        color = color_fan[temperature][i_p]
        top, bot = I_activations_cov[temperature][i_p]
        axes[1].fill_between(av_steps, top, bot, color=color,
                alpha=alpha, linewidth=0)
# Kylie
if plot_kylie:
    axes[1].plot(av_steps, kI_activation/np.min(kI_activation),
             c='r', label='Beattie et al. 2018 22 $^o$C')
# Vandenberg
u1 = np.loadtxt('vandenberg-et-al-2006/vandenberg-et-al-2006-fig3c-22oc',
        delimiter=',', skiprows=1)
u2 = np.loadtxt('vandenberg-et-al-2006/vandenberg-et-al-2006-fig3c-32oc',
        delimiter=',', skiprows=1)
axes[0].plot(u1[:, 0], u1[:, 1], marker='s', label=r'Vandenberg et al. 2006 22 $^o$C')
axes[0].plot(u2[:, 0], u2[:, 1], marker='s', label=r'Vandenberg et al. 2006 32 $^o$C')

for i in range(2):
    axes[i].legend()
    axes[i].set_xlabel('Voltage [mV]', fontsize=14)
    axes[i].set_xlim([-124, 55])
axes[0].set_ylabel('Normalised tail peak current', fontsize=14)

plt.tight_layout(pad=0.05, w_pad=0.15, h_pad=0.15)
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('figs/paper/re-Vandenberg2006-fig3c.png')
    plt.savefig('figs/paper/re-Vandenberg2006-fig3c.pdf', format='pdf',
            bbox_inches='tight')
    plt.close()


# Figure 5D
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9, 4))

for temperature in temperatures:
    for i, g_inactivation in enumerate(g_inactivations[temperature]):
        axes[1].plot(iv_steps, g_inactivation/np.max(g_inactivation),
                 c=color_mid[temperature], lw=1.5,
                 label='__nolegend__' if i else temperature+' $^o$C')
# fan charts
for temperature in temperatures:
    for i_p, p in enumerate(percentiles):
        alpha = 0.8
        color = color_fan[temperature][i_p]
        top, bot = g_inactivations_cov[temperature][i_p]
        axes[1].fill_between(iv_steps, top, bot, color=color,
                alpha=alpha, linewidth=0)
# Kylie
if plot_kylie:
    axes[1].plot(iv_steps, kg_inactivation/np.max(kg_inactivation),
             c='r', label='Beattie et al. 2018 22 $^o$C')
# Vandenberg
w1 = np.loadtxt('vandenberg-et-al-2006/vandenberg-et-al-2006-fig5d-22oc',
        delimiter=',', skiprows=1)
w2 = np.loadtxt('vandenberg-et-al-2006/vandenberg-et-al-2006-fig5d-32oc',
        delimiter=',', skiprows=1)
axes[0].plot(w1[4:, 0], w1[4:, 1] / np.max(w1[4:, 1]), marker='s',
        label=r'Vandenberg et al. 2006 22 $^o$C')
axes[0].plot(w2[4:, 0], w2[4:, 1] / np.max(w2[4:, 1]), marker='s',
        label=r'Vandenberg et al. 2006 32 $^o$C')

for i in range(2):
    axes[i].legend()
    axes[i].set_xlabel('Voltage [mV]', fontsize=14)
    axes[i].set_xlim([-124, 55])
axes[0].set_ylabel('Normalised conductance', fontsize=14)

plt.tight_layout(pad=0.05, w_pad=0.15, h_pad=0.15)
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('figs/paper/re-Vandenberg2006-fig5d.png')
    plt.savefig('figs/paper/re-Vandenberg2006-fig5d.pdf', format='pdf',
            bbox_inches='tight')
    plt.close()


# Figure 6
for temperature in temperatures:
    v_steps = iv_steps[:]
    for ii, (I_activation, g_inactivation) in \
                enumerate(zip(I_activations[temperature],
                              g_inactivations[temperature])):
        po = []
        for v in v_steps:
            # act
            if v in av_steps:
                i = np.where(np.abs(av_steps - v) < 1e-5)[0][0]
                act = I_activation[i] / np.min(I_activation)
            elif v > 30.0:  # mV
                act = 1.0  # approx. as fully open
            elif v < -70.0:  # mV
                act = 0.0  # approx. as fully closed
            else:
                print('...')
            # inact
            if v in iv_steps:
                i = np.where(np.abs(iv_steps - v) < 1e-5)[0][0]
                inact = g_inactivation[i] / np.max(g_inactivation)
            else:
                print('...')
            po.append(act * inact)
        plt.plot(v_steps, po, c=color_mid[temperature],
                 label='_nolegend_' if ii else temperature+' $^o$C')

# fan charts
for temperature in temperatures:
    v_steps = iv_steps[:]
    for i_p, p in enumerate(percentiles):
        alpha = 0.8
        color = color_fan[temperature][i_p]
        top, bot = p_cov[temperature][i_p]
        plt.fill_between(v_steps, top, bot, color=color,
                alpha=alpha, linewidth=0)

# Kylie
if plot_kylie:
    v_steps = iv_steps[:]
    po = []
    for v in v_steps:
        # act
        if v in av_steps:
            i = np.where(np.abs(av_steps - v) < 1e-5)[0][0]
            act = kI_activation[i] / np.min(kI_activation)
        elif v > 30.0:  # mV
            act = 1.0  # approx. as fully open
        elif v < -70.0:  # mV
            act = 0.0  # approx. as fully closed
        else:
            print('...')
        # inact
        if v in iv_steps:
            i = np.where(np.abs(iv_steps - v) < 1e-5)[0][0]
            inact = kg_inactivation[i] / np.max(kg_inactivation)
        else:
            print('...')
        po.append(act * inact)
    plt.plot(v_steps, po, c='r', label='Beattie et al. 2018 22 $^o$C')

# Vandenberg
v1 = np.loadtxt('vandenberg-et-al-2006/vandenberg-et-al-2006-fig6-22oc',
        delimiter=',', skiprows=1)
v2 = np.loadtxt('vandenberg-et-al-2006/vandenberg-et-al-2006-fig6-32oc',
        delimiter=',', skiprows=1)
plt.scatter(v1[:, 0], v1[:, 1], label=r'Vandenberg et al. 2006 22 $^o$C')
plt.scatter(v2[:, 0], v2[:, 1], label=r'Vandenberg et al. 2006 32 $^o$C')

plt.legend()
plt.xlabel('Voltage [mV]', fontsize=14)
plt.ylabel('Open probability', fontsize=14)

if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('figs/paper/re-Vandenberg2006-fig6.png')
    plt.savefig('figs/paper/re-Vandenberg2006-fig6.pdf', format='pdf',
            bbox_inches='tight')
    plt.close()


# Figure 6 v2
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9, 4))

for temperature in temperatures:
    v_steps = iv_steps[:]
    for ii, (I_activation, g_inactivation) in \
                enumerate(zip(I_activations[temperature],
                              g_inactivations[temperature])):
        po = []
        for v in v_steps:
            # act
            if v in av_steps:
                i = np.where(np.abs(av_steps - v) < 1e-5)[0][0]
                act = I_activation[i] / np.min(I_activation)
            elif v > 30.0:  # mV
                act = 1.0  # approx. as fully open
            elif v < -70.0:  # mV
                act = 0.0  # approx. as fully closed
            else:
                print('...')
            # inact
            if v in iv_steps:
                i = np.where(np.abs(iv_steps - v) < 1e-5)[0][0]
                inact = g_inactivation[i] / np.max(g_inactivation)
            else:
                print('...')
            po.append(act * inact)
        axes[1].plot(v_steps, po, c=color_mid[temperature], lw=1.5,
                 label='_nolegend_' if ii else temperature+' $^o$C')

# fan charts
for temperature in temperatures:
    v_steps = iv_steps[:]
    for i_p, p in enumerate(percentiles):
        alpha = 0.8
        color = color_fan[temperature][i_p]
        top, bot = p_cov[temperature][i_p]
        axes[1].fill_between(v_steps, top, bot, color=color,
                alpha=alpha, linewidth=0)

# Kylie
if plot_kylie:
    v_steps = iv_steps[:]
    po = []
    for v in v_steps:
        # act
        if v in av_steps:
            i = np.where(np.abs(av_steps - v) < 1e-5)[0][0]
            act = kI_activation[i] / np.min(kI_activation)
        elif v > 30.0:  # mV
            act = 1.0  # approx. as fully open
        elif v < -70.0:  # mV
            act = 0.0  # approx. as fully closed
        else:
            print('...')
        # inact
        if v in iv_steps:
            i = np.where(np.abs(iv_steps - v) < 1e-5)[0][0]
            inact = kg_inactivation[i] / np.max(kg_inactivation)
        else:
            print('...')
        po.append(act * inact)
    axes[1].plot(v_steps, po, c='r', label='Beattie et al. 2018 22 $^o$C')

# Vandenberg
v1 = np.loadtxt('vandenberg-et-al-2006/vandenberg-et-al-2006-fig6-22oc',
        delimiter=',', skiprows=1)
v2 = np.loadtxt('vandenberg-et-al-2006/vandenberg-et-al-2006-fig6-32oc',
        delimiter=',', skiprows=1)
axes[0].plot(v1[:, 0], v1[:, 1], marker='s', label=r'Vandenberg et al. 2006 22 $^o$C')
axes[0].plot(v2[:, 0], v2[:, 1], marker='s', label=r'Vandenberg et al. 2006 32 $^o$C')

for i in range(2):
    axes[i].legend()
    axes[i].set_xlabel('Voltage [mV]', fontsize=14)
    axes[i].set_xlim([-124, 55])
axes[0].set_ylabel('Open probability', fontsize=14)

plt.tight_layout(pad=0.05, w_pad=0.15, h_pad=0.15)

if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('figs/paper/re-Vandenberg2006-fig6-v2.png')
    plt.savefig('figs/paper/re-Vandenberg2006-fig6-v2.pdf', format='pdf',
            bbox_inches='tight')
    plt.close()

## eof
