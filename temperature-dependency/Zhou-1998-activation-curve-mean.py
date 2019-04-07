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
from protocols import Zhou1998_isochronal_tail_current as prt_act

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

plot_kylie = False

debug = False
common_conductance = 1.0
fit_seed = 542811797
temperatures = ['25.0', '37.0'][::-1]
color = {'25.0':'#6baed6',
         '33.0':'C2',
         '37.0':'#fd8d3c',}


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
        '33.0':fan_green,
        '37.0':fan_red,
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

for i_T, temperature in enumerate(temperatures):

    I_activations[temperature] = []

    # Model
    model_act = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                        protocol_def=prt_act,
                        temperature=273.15 + float(temperature),  # K
                        transform=None,
                        useFilterCap=False,  # ignore capacitive spike
                        effEK=False,  # OK to switch this off here
                        concK=[4., 130.])

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

    if debug and False:
        # Figure 8A
        for ii in range(len(av_steps)):
            plt.plot(a['time'], a[str(ii)+'.current'])
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Vandenberg2006/' + 
                    'Vandenberg2006-fig8a-%sC.png'%temperature)
            plt.close()

    I_activation = get_IV(a, len(av_steps),
                          tmeasure_act, tmeasure_act + 0.5)
    I_activations[temperature].append(I_activation)

    del(a)

    np.random.seed(101)
    i_act_cov_all = []
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


        I_activation = get_IV(a, len(av_steps),
                              tmeasure_act, tmeasure_act + 0.5)
        I_activation = I_activation / np.max(I_activation)
        i_act_cov_all.append(I_activation)

        del(a)

    I_activations_cov[temperature] = []

    percentiles = [90, 60, 30]
    fan_chart_data_top = []
    fan_chart_data_bot = []
    for i_p, p in enumerate(percentiles):
        top = np.nanpercentile(i_act_cov_all, 50 + p / 2., axis=0)
        bot = np.nanpercentile(i_act_cov_all, 50 - p / 2., axis=0)
        I_activations_cov[temperature].append([top, bot])

# Kylie
if plot_kylie:
    temperature = 22 + 273.15
    # Model
    model_act = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                        protocol_def=prt_act,
                        temperature=temperature,  # K
                        transform=None,
                        useFilterCap=False,  # ignore capacitive spike
                        effEK=False,  # OK to switch this off here
                        concK=[4., 130.])
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

    kI_activation = get_IV(a, len(av_steps),
                          tmeasure_act, tmeasure_act + 0.5)

    del(a)


# convert units V -> mV
av_steps = np.around(np.array(av_steps) * 1e3, 1)

# Figure 8B
for temperature in temperatures:
    for i, I_activation in enumerate(I_activations[temperature]):
        if i == 0:
            plt.plot(av_steps, I_activation/np.max(I_activation),
                     c=color[temperature],
                     label=temperature+' $^o$C')
        else:
            plt.plot(av_steps, I_activation/np.max(I_activation),
                     c=color[temperature])
# fan charts
for temperature in temperatures:
    for i_p, p in enumerate(percentiles):
        alpha = 0.8
        color = color_fan[temperature][i_p]
        top, bot = I_activations_cov[temperature][i_p]
        plt.fill_between(av_steps, top, bot, color=color,
                alpha=alpha, linewidth=0)
# Kylie
if plot_kylie:
    plt.plot(av_steps, kI_activation/np.max(kI_activation),
             c='r',
             label='Beattie et al. 2018 22 $^o$C')

# Zhou
v1 = np.loadtxt('zhou-et-al-1998/zhou-et-al-1998-fig8b-23oc.csv',
        delimiter=',', skiprows=1)
v2 = np.loadtxt('zhou-et-al-1998/zhou-et-al-1998-fig8b-35oc.csv',
        delimiter=',', skiprows=1)
plt.scatter(v1[:, 0], v1[:, 1], label=r'Zhou et al. 1998 23 $^o$C')
plt.scatter(v2[:, 0], v2[:, 1], label=r'Zhou et al. 1998 35 $^o$C')

plt.xlabel('Voltage [mV]')
plt.ylabel('Normalised tail peak current')
plt.legend()
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('figs/paper/re-Zhou2006-fig8b.png')
    plt.close()

## eof
