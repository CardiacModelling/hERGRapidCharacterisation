#!/usr/bin/env python2
# 
# Try to reproduce Q10 values in Vandenberg et al. 2006
# See Fig. 8-10.
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

import model_ikr as m
from protocols import Vandenberg2006_envelope_of_tails as prt1
from protocols import Vandenberg2006_triple_pulse as prt2

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param


debug = False
common_conductance = 1.0
fit_seed = 542811797
temperatures = ['25.0', '37.0'][::-1]
color = {'25.0':'#6baed6',
         '37.0':'#fd8d3c'}


FILE_LIST = {
        '25.0': ['herg25oc'],
        '37.0': ['herg37oc'],
    }


#
# Functions to fit time constants
#


def fit_tau_single_exp_simple(current, times, debug=False):
    # use 2-parameters exponential fit to the tail
    from scipy.optimize import curve_fit
    def exp_func(t, a, b):
        # do a "proper exponential" decay fit
        # i.e. shift the t to t' where t' has zero at the start of the 
        # voltage step
        return a * (1.0 - np.exp(-t / b))
    x = times
    y = current
    try:
        popt, pcov = curve_fit(exp_func, x, y, p0=[y[-1], 300e-3])
        fitted = exp_func(x, *popt)
        tau = 1e3 * popt[1]  # [ms]
    except:
        raise Exception('Maybe not here!')
    if debug:
        fig = plt.figure()
        plt.plot(x, y, marker='s', c='#d62728')
        plt.plot(x, fitted, '--', c='#1f77b4')
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Vandenberg2006/Vandenberg2006-fig8c-%sC.png'
                    % temperature)
            plt.close()
    return tau


def fit_tau_single_exp(current, times,
                       t_start, t_end,
                       t_trim, t_fit_until,
                       debug=False):
    # use 2-parameters exponential fit to the tail
    from scipy.optimize import curve_fit
    def exp_func(t, a, b):
        # do a "proper exponential" decay fit
        # i.e. shift the t to t' where t' has zero at the start of the 
        # voltage step
        return a * np.exp( -b * (t - x[0]))
    time_window = np.where(np.logical_and(times > t_start, times <= t_end))[0]
    i_trim = np.argmin(np.abs(times - (t_start + t_trim))) - time_window[0]
    i_fit_until = np.argmin(np.abs(times - (t_start + t_fit_until))) \
                  - time_window[0]
    # trim off the first i_trim (100ms) in case it is still shooting up...
    x = times[time_window[0] + i_trim:time_window[0] + i_fit_until]
    y = current[time_window[0] + i_trim:
                time_window[0] + i_fit_until]
    try:
        popt, pcov = curve_fit(exp_func, x, y)
        fitted = exp_func(times[time_window[0]:
                                time_window[0] + i_fit_until], *popt)
        tau = 1e3 / popt[1]  # [ms]
    except:
        raise Exception('Maybe not here!')
    if debug:
        fig = plt.figure()
        plt.plot(times[time_window[0] - 500:time_window[-1] + 500],
                 current[time_window[0] - 500:time_window[-1] + 500],
                 c='#d62728')
        plt.plot(times[time_window[0]:time_window[0] + i_fit_until], fitted,
                 '--', c='#1f77b4')
        plt.plot(times[time_window][0], fitted[0], 'kx')
        plt.axvline(x=times[time_window[0] + i_trim])
        plt.axvline(x=times[time_window[0] + i_fit_until])
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Vandenberg2006/Vandenberg2006-fig9a-%sC.png'
                    % temperature)
            plt.close()
    return tau


def fit_tau_double_exp(current, times,
                       t_start, t_end,
                       t_trim, t_fit_until,
                       debug=False):
    # use 4-parameters 'double exponential' fit to the current
    from scipy.optimize import curve_fit
    def exp_func(t, a, b, c, d):
        # Shift the t to t' where t' has zero at the start of the 
        # voltage step
        return a * np.exp( -(t - x[0]) / b) - c * np.exp( -(t - x[0]) / d)
    time_window = np.where(np.logical_and(times > t_start, times <= t_end))[0]
    i_trim = np.argmin(np.abs(times - (t_start + t_trim))) - time_window[0]
    i_fit_until = np.argmin(np.abs(times - (t_start + t_fit_until))) \
                  - time_window[0]
    # trim off the first i_trim (100ms) in case it is still shooting up...
    x = times[time_window[0] + i_trim:time_window[0] + i_fit_until]
    y = current[time_window[0] + i_trim:
                time_window[0] + i_fit_until]
    try:
        popt, pcov = curve_fit(exp_func, x, y, p0=[0.1, 1.1e-3, 0.1, 12.4e-3],
                               bounds=(1e-6, [1, 20e-3, 1, 200e-3]))
        fitted = exp_func(times[time_window[0]:
                                time_window[0] + i_fit_until], *popt)
        tau_1 = 1e3 * popt[1]  # [ms]
        tau_2 = 1e3 * popt[3]  # [ms]
    except:
        raise Exception('Maybe not here!')
    if debug:
        fig = plt.figure()
        plt.plot(times[time_window[0] - 500:time_window[-1] + 500],
                 current[time_window[0] - 500:time_window[-1] + 500],
                 c='#d62728')
        plt.plot(times[time_window[0]:time_window[0] + i_fit_until], fitted,
                 '--', c='#1f77b4')
        plt.axvline(x=times[time_window[0] + i_trim])
        plt.axvline(x=times[time_window[0] + i_fit_until])
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Vandenberg2006/Vandenberg2006-fig9c-%sC.png'
                    % temperature)
            plt.close()
    return tau_1, tau_2


#
# Simulate
#

times_prt1, ttotal_prt1, tmeasure_prt1 = prt1(None, return_times=True)
tau_act = {}

times_prt2, ttotal_prt2, tmeasure_prt2 = prt2(None, return_times=True)
tau_deact = {}
tau_inact = {}
tau_rec = {}

for temperature in temperatures:

    tau_act[temperature] = []
    tau_deact[temperature] = []
    tau_inact[temperature] = []
    tau_rec[temperature] = []

    # Model
    model_prt1 = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                         protocol_def=prt1,
                         temperature=273.15 + float(temperature),  # K
                         transform=None,
                         useFilterCap=False,  # ignore capacitive spike
                         effEK=False)  # OK to switch this off here
    model_prt2 = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                         protocol_def=prt2,
                         temperature=273.15 + float(temperature),  # K
                         transform=None,
                         useFilterCap=False,  # ignore capacitive spike
                         effEK=False)  # OK to switch this off here

    # ts = np.array([0.17, 0.33, 0.61, 1.1, 2.0, 3.5])
    ts = np.array([0.05, 0.1, 0.17, 0.33, 0.61, 1.1, 2.0, 3.5])
    m_t = []
    for t in ts:
        def prt1_t(m, c=False):
            return prt1(m, c, thold=t, vhold=40e-3)
        m_t.append(m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                           protocol_def=prt1_t,
                           temperature=273.15 + float(temperature),  # K
                           transform=None,
                           useFilterCap=False,
                           effEK=False)  # OK to switch this off here
                           )

    file_list = FILE_LIST[temperature]
    for file_name in file_list:

        load_file = './out-mcmc/%s-pseudo2hbm-lognorm-mean.txt' % (file_name)
        mean_chain = np.loadtxt(load_file)  # transformed
        hbm_T_mean = transform_to_model_param(
                np.mean(mean_chain, axis=0))
        selectedwell = transform_to_model_param(mean_chain[:70, :].T).T
        selectedwell = [hbm_T_mean] + list(selectedwell)

        for i_cell, p in enumerate(selectedwell):
            p[0] = common_conductance
            p = np.append(p, 0)  # No leak current

            i_prt2 = model_prt2.simulate(p, times_prt2)

            i_prt1 = []
            for i_t, t in enumerate(ts):
                times, _, tm = prt1(None, thold=t, return_times=True)
                c = m_t[i_t].simulate(p, times)
                i_max_range = np.where(np.logical_and(times > tm,
                            times < tm + 50e-3))[0]
                i_prt1.append(np.max(c[i_max_range]))

                # Figure 8
                if i_cell == 0 and debug:
                    plt.plot(times, c)
                    plt.scatter(tm, #+ times[np.argmax(c[i_max_range])],
                                i_prt1[-1], color='r', marker='s')

            if i_cell == 0 and debug:
                plt.xlabel('Time [ms]')
                plt.ylabel(r'Current [$g_{Kr}=%s$]' % common_conductance)
                plt.savefig('figs/Vandenberg2006/' + 
                        'Vandenberg2006-fig8ab-%sC.png' % temperature)
                plt.close()

                # Figure 8C
                plt.plot(ts * 1e3, i_prt1, marker='s')
                # plt.xscale('log')
                plt.xlabel('Time [ms]')
                plt.ylabel(r'Current [$g_{Kr}=%s$]' % common_conductance)
                plt.savefig('figs/Vandenberg2006/' +
                        'Vandenberg2006-fig8cii-%sC.png' % temperature)
                plt.close()


            tau1 = fit_tau_single_exp_simple(i_prt1, ts, debug=False)
            tau_act[temperature].append(tau1)

            tau2 = fit_tau_single_exp(i_prt2, times_prt2,
                                      tmeasure_prt2[0], tmeasure_prt2[0] + 0.2,
                                      t_trim=0, t_fit_until=10e-3,
                                      debug=False)
            tau_deact[temperature].append(tau2)

            tau3, tau4 = fit_tau_double_exp(i_prt2, times_prt2,
                                            tmeasure_prt2[1],
                                            tmeasure_prt2[1] + 0.5,
                                            t_trim=0, t_fit_until=100e-3,
                                            debug=False)
            tau_inact[temperature].append(tau3)
            tau_rec[temperature].append(tau4)


#
# Q10s
#
import itertools
T1 = 37.0
T2 = 25.0

# activation
a1 = tau_act[str(T1)][0]
a2 = tau_act[str(T2)][0]
q10_act_hbm = (a1 / a2) ** (10.0 / (T2 - T1))
tau_act[str(T1)] = tau_act[str(T1)][1:]
tau_act[str(T2)] = tau_act[str(T2)][1:]
a1 = np.mean(tau_act[str(T1)])
a2 = np.mean(tau_act[str(T2)])
q10_act = (a1 / a2) ** (10.0 / (T2 - T1))
q10_act_combinations = []
for a1i, a2i in itertools.product(tau_act[str(T1)], tau_act[str(T2)]):
    q10_act_combinations.append((a1i / a2i) ** (10.0 / (T2 - T1)))
q10_act_mean = np.mean(q10_act_combinations)
q10_act_std = np.std(q10_act_combinations)

# deactivation
a1 = tau_deact[str(T1)][0]
a2 = tau_deact[str(T2)][0]
q10_deact_hbm = (a1 / a2) ** (10.0 / (T2 - T1))
tau_deact[str(T1)] = tau_deact[str(T1)][1:]
tau_deact[str(T2)] = tau_deact[str(T2)][1:]
a1 = np.mean(tau_deact[str(T1)])
a2 = np.mean(tau_deact[str(T2)])
q10_deact = (a1 / a2) ** (10.0 / (T2 - T1))
q10_deact_combinations = []
for a1i, a2i in itertools.product(tau_deact[str(T1)], tau_deact[str(T2)]):
    q10_deact_combinations.append((a1i / a2i) ** (10.0 / (T2 - T1)))
q10_deact_mean = np.mean(q10_deact_combinations)
q10_deact_std = np.std(q10_deact_combinations)

# inactivation
a1 = tau_inact[str(T1)][0]
a2 = tau_inact[str(T2)][0]
q10_inact_hbm = (a1 / a2) ** (10.0 / (T2 - T1))
tau_inact[str(T1)] = tau_inact[str(T1)][1:]
tau_inact[str(T2)] = tau_inact[str(T2)][1:]
a1 = np.mean(tau_inact[str(T1)])
a2 = np.mean(tau_inact[str(T2)])
q10_inact = (a1 / a2) ** (10.0 / (T2 - T1))
q10_inact_combinations = []
for a1i, a2i in itertools.product(tau_inact[str(T1)], tau_inact[str(T2)]):
    q10_inact_combinations.append((a1i / a2i) ** (10.0 / (T2 - T1)))
q10_inact_mean = np.mean(q10_inact_combinations)
q10_inact_std = np.std(q10_inact_combinations)

# recovery
a1 = tau_rec[str(T1)][0]
a2 = tau_rec[str(T2)][0]
q10_rec_hbm = (a1 / a2) ** (10.0 / (T2 - T1))
tau_rec[str(T1)] = tau_rec[str(T1)][1:]
tau_rec[str(T2)] = tau_rec[str(T2)][1:]
a1 = np.mean(tau_rec[str(T1)])
a2 = np.mean(tau_rec[str(T2)])
q10_rec = (a1 / a2) ** (10.0 / (T2 - T1))
q10_rec_combinations = []
for a1i, a2i in itertools.product(tau_rec[str(T1)], tau_rec[str(T2)]):
    q10_rec_combinations.append((a1i / a2i) ** (10.0 / (T2 - T1)))
q10_rec_mean = np.mean(q10_rec_combinations)
q10_rec_std = np.std(q10_rec_combinations)

# round up
q10_act = round(q10_act, 3)
q10_act_mean = round(q10_act_mean, 3)
q10_act_std = round(q10_act_std, 3)
q10_deact = round(q10_deact, 3)
q10_deact_mean = round(q10_deact_mean, 3)
q10_deact_std = round(q10_deact_std, 3)
q10_inact = round(q10_inact, 3)
q10_inact_mean = round(q10_inact_mean, 3)
q10_inact_std = round(q10_inact_std, 3)
q10_rec = round(q10_rec, 3)
q10_rec_mean = round(q10_rec_mean, 3)
q10_rec_std = round(q10_rec_std, 3)

print('----' * 20)
print('Q10_activation:', q10_act_mean, '+/-', q10_act_std,
        '(mean tau estimate: %s)' % q10_act,
        '(hbm estimate: %s)' % q10_act_hbm)
print('Q10_deactivation:', q10_deact_mean, '+/-', q10_deact_std,
        '(mean tau estimate: %s)' % q10_deact,
        '(hbm estimate: %s)' % q10_deact_hbm)
print('Q10_inactivation:', q10_inact_mean, '+/-', q10_inact_std,
        '(mean tau estimate: %s)' % q10_inact,
        '(hbm estimate: %s)' % q10_inact_hbm)
print('Q10_recovery:', q10_rec_mean, '+/-', q10_rec_std,
        '(mean tau estimate: %s)' % q10_rec,
        '(hbm estimate: %s)' % q10_rec_hbm)

## eof
