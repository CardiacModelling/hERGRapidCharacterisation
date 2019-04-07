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


debug = False
debug2 = True
common_conductance = 1.0
fit_seed = 542811797
temperatures = ['25.0', '37.0'][::-1]
color = {'25.0':'#6baed6',
         '37.0':'#fd8d3c'}


FILE_LIST = {
        '25.0': ['herg25oc1'],
        '37.0': ['herg37oc3'],
    }

VandenbergQ10 = [(2.1, 0.1 * np.sqrt(9)),
                 (1.7, 0.1 * np.sqrt(9)),
                 (2.5, 0.2 * np.sqrt(7)),
                 (2.6, 0.1 * np.sqrt(7))
                 ]


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

for i_T, temperature in enumerate(temperatures):

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
    ts = np.array([0.02, 0.05, 0.1, 0.17, 0.33, 0.61, 1.1, 2.0, 3.5])
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

        selectedfile = './manualselection/manualselected-%s.txt' % (file_name)
        selectedwell = []
        with open(selectedfile, 'r') as f:
            for l in f:
                if not l.startswith('#'):
                    selectedwell.append(l.split()[0])
        print('Getting ', file_name)
        selectedwell = selectedwell[:90]

        for i_cell, cell in enumerate(selectedwell):
            pfile = './out/%s/%s-staircaseramp-%s-solution-%s.txt' \
                    % (file_name, file_name, cell, fit_seed)
            p = np.loadtxt(pfile)
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

            if debug2:
                # Figure 8C
                plt.plot(ts * 1e3, i_prt1, marker='s', c=color[temperature],
                         label='__nolegend__' if i_T else '%s' % temperature)
                # plt.xscale('log')
                plt.xlabel('Time [ms]')
                plt.ylabel(r'Current [$g_{Kr}=%s$]' % common_conductance)

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

if debug2:
    plt.savefig('figs/Vandenberg2006/Vandenberg2006-fig8cii.png')
    plt.close()


#
# Q10s
#
import itertools
T1 = 37.0
T2 = 25.0
dT = 10.0 / (T2 - T1)
dT1 = dT2 = 1.0

# activation
# method 1
a1 = np.mean(tau_act[str(T1)])
a2 = np.mean(tau_act[str(T2)])
da1 = np.std(tau_act[str(T1)])
da2 = np.std(tau_act[str(T2)])
q10 = (a1 / a2) ** (dT)
T1_mean_q10_act = (a1 / np.asarray(tau_act[str(T2)])) ** (dT)
T2_mean_q10_act = (np.asarray(tau_act[str(T1)]) / a2) ** (dT)
mean_q10_act = q10
std_q10_act = np.sqrt((da1 * dT / a1 * q10) ** 2 + (da2 * dT / a2 * q10) ** 2
        + (dT1 * q10 * np.log(a1 / a2) * dT ** 2 / 10.0) ** 2
        + (dT2 * q10 * np.log(a1 / a2) * dT ** 2 / 10.0) ** 2)
# method 2
q10_act_combinations = []
for a1i, a2i in itertools.product(tau_act[str(T1)], tau_act[str(T2)]):
    q10_act_combinations.append((a1i / a2i) ** (dT))
q10_act_mean = np.mean(q10_act_combinations)
q10_act_std = np.std(q10_act_combinations)

# deactivation
# method 1
a1 = np.mean(tau_deact[str(T1)])
a2 = np.mean(tau_deact[str(T2)])
da1 = np.std(tau_deact[str(T1)])
da2 = np.std(tau_deact[str(T2)])
q10 = (a1 / a2) ** (dT)
T1_mean_q10_deact = (a1 / np.asarray(tau_deact[str(T2)])) ** (dT)
T2_mean_q10_deact = (np.asarray(tau_deact[str(T1)]) / a2) ** (dT)
mean_q10_deact = q10
std_q10_deact = np.sqrt((da1 * dT / a1 * q10) ** 2 + (da2 * dT / a2 * q10) ** 2
        + (dT1 * q10 * np.log(a1 / a2) * dT ** 2 / 10.0) ** 2
        + (dT2 * q10 * np.log(a1 / a2) * dT ** 2 / 10.0) ** 2)
# method 2
q10_deact_combinations = []
for a1i, a2i in itertools.product(tau_deact[str(T1)], tau_deact[str(T2)]):
    q10_deact_combinations.append((a1i / a2i) ** (dT))
q10_deact_mean = np.mean(q10_deact_combinations)
q10_deact_std = np.std(q10_deact_combinations)

# inactivation
# method 1
a1 = np.mean(tau_inact[str(T1)])
a2 = np.mean(tau_inact[str(T2)])
da1 = np.std(tau_inact[str(T1)])
da2 = np.std(tau_inact[str(T2)])
q10 = (a1 / a2) ** (dT)
T1_mean_q10_inact = (a1 / np.asarray(tau_inact[str(T2)])) ** (dT)
T2_mean_q10_inact = (np.asarray(tau_inact[str(T1)]) / a2) ** (dT)
mean_q10_inact = q10
std_q10_inact = np.sqrt((da1 * dT / a1 * q10) ** 2 + (da2 * dT / a2 * q10) ** 2
        + (dT1 * q10 * np.log(a1 / a2) * dT ** 2 / 10.0) ** 2
        + (dT2 * q10 * np.log(a1 / a2) * dT ** 2 / 10.0) ** 2)
# method 2
q10_inact_combinations = []
for a1i, a2i in itertools.product(tau_inact[str(T1)], tau_inact[str(T2)]):
    q10_inact_combinations.append((a1i / a2i) ** (dT))
q10_inact_mean = np.mean(q10_inact_combinations)
q10_inact_std = np.std(q10_inact_combinations)

# recovery
# method 1
a1 = np.mean(tau_rec[str(T1)])
a2 = np.mean(tau_rec[str(T2)])
da1 = np.std(tau_rec[str(T1)])
da2 = np.std(tau_rec[str(T2)])
q10 = (a1 / a2) ** (dT)
T1_mean_q10_rec = (a1 / np.asarray(tau_rec[str(T2)])) ** (dT)
T2_mean_q10_rec = (np.asarray(tau_rec[str(T1)]) / a2) ** (dT)
mean_q10_rec = q10
std_q10_rec = np.sqrt((da1 * dT / a1 * q10) ** 2 + (da2 * dT / a2 * q10) ** 2
        + (dT1 * q10 * np.log(a1 / a2) * dT ** 2 / 10.0) ** 2
        + (dT2 * q10 * np.log(a1 / a2) * dT ** 2 / 10.0) ** 2)
# method 2
q10_rec_combinations = []
for a1i, a2i in itertools.product(tau_rec[str(T1)], tau_rec[str(T2)]):
    q10_rec_combinations.append((a1i / a2i) ** (dT))
q10_rec_mean = np.mean(q10_rec_combinations)
q10_rec_std = np.std(q10_rec_combinations)

# round up
mean_q10_act = round(mean_q10_act, 3)
std_q10_act = round(std_q10_act, 3)
q10_act_mean = round(q10_act_mean, 3)
q10_act_std = round(q10_act_std, 3)
mean_q10_deact = round(mean_q10_deact, 3)
std_q10_deact = round(std_q10_deact, 3)
q10_deact_mean = round(q10_deact_mean, 3)
q10_deact_std = round(q10_deact_std, 3)
mean_q10_inact = round(mean_q10_inact, 3)
std_q10_inact = round(std_q10_inact, 3)
q10_inact_mean = round(q10_inact_mean, 3)
q10_inact_std = round(q10_inact_std, 3)
mean_q10_rec = round(mean_q10_rec, 3)
std_q10_rec = round(std_q10_rec, 3)
q10_rec_mean = round(q10_rec_mean, 3)
q10_rec_std = round(q10_rec_std, 3)

# N pairs of cells are within Vandenberg et al. 2006 estimation (1st std)
# method 1
T1_mean_n_act = np.sum(np.logical_and(
    VandenbergQ10[0][0] - VandenbergQ10[0][1] < T1_mean_q10_act,
    VandenbergQ10[0][0] + VandenbergQ10[0][1] > T1_mean_q10_act))
T2_mean_n_act = np.sum(np.logical_and(
    VandenbergQ10[0][0] - VandenbergQ10[0][1] < T2_mean_q10_act,
    VandenbergQ10[0][0] + VandenbergQ10[0][1] > T2_mean_q10_act))
T1_mean_n_deact = np.sum(np.logical_and(
    VandenbergQ10[0][0] - VandenbergQ10[0][1] < T1_mean_q10_deact,
    VandenbergQ10[0][0] + VandenbergQ10[0][1] > T1_mean_q10_deact))
T2_mean_n_deact = np.sum(np.logical_and(
    VandenbergQ10[0][0] - VandenbergQ10[0][1] < T2_mean_q10_deact,
    VandenbergQ10[0][0] + VandenbergQ10[0][1] > T2_mean_q10_deact))
T1_mean_n_inact = np.sum(np.logical_and(
    VandenbergQ10[0][0] - VandenbergQ10[0][1] < T1_mean_q10_inact,
    VandenbergQ10[0][0] + VandenbergQ10[0][1] > T1_mean_q10_inact))
T2_mean_n_inact = np.sum(np.logical_and(
    VandenbergQ10[0][0] - VandenbergQ10[0][1] < T2_mean_q10_inact,
    VandenbergQ10[0][0] + VandenbergQ10[0][1] > T2_mean_q10_inact))
T1_mean_n_rec = np.sum(np.logical_and(
    VandenbergQ10[0][0] - VandenbergQ10[0][1] < T1_mean_q10_rec,
    VandenbergQ10[0][0] + VandenbergQ10[0][1] > T1_mean_q10_rec))
T2_mean_n_rec = np.sum(np.logical_and(
    VandenbergQ10[0][0] - VandenbergQ10[0][1] < T2_mean_q10_rec,
    VandenbergQ10[0][0] + VandenbergQ10[0][1] > T2_mean_q10_rec))
# method 2
n_act = np.sum(np.logical_and(
    VandenbergQ10[0][0] - VandenbergQ10[0][1] < q10_act_combinations,
    VandenbergQ10[0][0] + VandenbergQ10[0][1] > q10_act_combinations))
n_deact = np.sum(np.logical_and(
    VandenbergQ10[1][0] - VandenbergQ10[1][1] < q10_deact_combinations,
    VandenbergQ10[1][0] + VandenbergQ10[1][1] > q10_deact_combinations))
n_inact = np.sum(np.logical_and(
    VandenbergQ10[2][0] - VandenbergQ10[2][1] < q10_inact_combinations,
    VandenbergQ10[2][0] + VandenbergQ10[2][1] > q10_inact_combinations))
n_rec = np.sum(np.logical_and(
    VandenbergQ10[3][0] - VandenbergQ10[3][1] < q10_rec_combinations,
    VandenbergQ10[3][0] + VandenbergQ10[3][1] > q10_rec_combinations))

print('----' * 20)
print('Method 1: Compute mean and std of tau at each temperature.')
print('Q10_activation:', mean_q10_act, '+/-', std_q10_act)
print('Q10_deactivation:', mean_q10_deact, '+/-', std_q10_deact)
print('Q10_inactivation:', mean_q10_inact, '+/-', std_q10_inact)
print('Q10_recovery:', mean_q10_rec, '+/-', std_q10_rec)
print(T1_mean_n_act, T2_mean_n_act)
print(T1_mean_n_deact, T2_mean_n_deact)
print(T1_mean_n_inact, T2_mean_n_inact)
print(T1_mean_n_rec, T2_mean_n_rec)
print('----' * 20)
print('Method 2: Compute all cells combinations between two temperatures.')
print('Q10_activation:', q10_act_mean, '+/-', q10_act_std)
print('Q10_deactivation:', q10_deact_mean, '+/-', q10_deact_std)
print('Q10_inactivation:', q10_inact_mean, '+/-', q10_inact_std)
print('Q10_recovery:', q10_rec_mean, '+/-', q10_rec_std)
print('With %s pairs within Vandenberg et al. 2006 estimation' \
        % min(n_act, n_deact, n_inact, n_rec))
print('----' * 20)
print('Vandenberg et al. 2006')
print('Q10_activation:', VandenbergQ10[0][0], '+/-', VandenbergQ10[0][1])
print('Q10_deactivation:', VandenbergQ10[1][0], '+/-', VandenbergQ10[1][1])
print('Q10_inactivation:', VandenbergQ10[2][0], '+/-', VandenbergQ10[2][1])
print('Q10_recovery:', VandenbergQ10[3][0], '+/-', VandenbergQ10[3][1])

## eof
