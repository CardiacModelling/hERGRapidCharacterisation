#!/usr/bin/env python2
# 
# Try to reproduce Q10 values in Zhou et al. 1998
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
from protocols import Zhou1998_activation_deactivation as prt1
from protocols import Zhou1998_inactivation as prt2
from protocols import Zhou1998_recovery as prt3


debug = True
common_conductance = 1.0
fit_seed = 542811797
temperatures = ['25.0', '37.0'][::-1]
color = {'25.0':'#6baed6',
         '37.0':'#fd8d3c'}


FILE_LIST = {
        '25.0': ['herg25oc1'],
        '37.0': ['herg37oc3'],
    }

#
# Zhou et al. 1998 Q10 values
#
ZhouQ10 = []

T1 = 23.0
T2 = 35.0
dT1 = dT2 = 1.0

def compute_q10(ks, dks, Ts, dTs):
    # Parameters:
    # ks, Ts each should have two elements for rate constant and temperature
    # dks, dTs are the standard deviations of ks, Ts respectively
    #
    # Return:
    # mean Q10 and standard deviation of Q10
    a1, a2 = ks
    da1, da2 = dks
    T1, T2 = Ts
    dT1, dT2 = dTs
    dT = 10.0 / (T2 - T1)

    q10 = (a1 / a2) ** (dT)
    mean_q10 = q10
    std_q10 = np.sqrt((da1 * dT / a1 * q10) ** 2
            + (da2 * dT / a2 * q10) ** 2
            + (dT1 * q10 * np.log(a1 / a2) * dT ** 2 / 10.0) ** 2
            + (dT2 * q10 * np.log(a1 / a2) * dT ** 2 / 10.0) ** 2)
    return (mean_q10, std_q10)

ZhouQ10.append(compute_q10((947., 105.),
                           (87. * np.sqrt(6), 15. * np.sqrt(6)),
                           (T1, T2), (dT1, dT2)))
ZhouQ10.append(compute_q10((216., 149.),
                           (19. * np.sqrt(3), 27. * np.sqrt(3)),
                           (T1, T2), (dT1, dT2)))
ZhouQ10.append(compute_q10((14.2, 3.1),
                           (1.3 * np.sqrt(3), 0.3 * np.sqrt(3)),
                           (T1, T2), (dT1, dT2)))
ZhouQ10.append(compute_q10((8.5, 1.8),
                           (0.6 * np.sqrt(3), 0.1 * np.sqrt(3)),
                           (T1, T2), (dT1, dT2)))

if debug:
    savedir = './figs/Zhou1998/'
    for p in ['-act', '-deact', '-inact', '-recovery']:
        if not os.path.isdir(savedir + 'Zhou1998-fig%s' % (p)):
            os.makedirs(savedir + 'Zhou1998-fig%s' % (p))


#
# Functions to fit time constants
#

def fit_tau_single_exp(current, times,
                       t_start, t_end,
                       t_trim, t_fit_until,
                       func, p0=None,
                       debug=False, debugout=''):
    # use 2-parameters exponential fit to the tail
    from scipy.optimize import curve_fit
    if func == 0:
        def exp_func(t, a, b):
            # do a "proper exponential" decay fit
            # i.e. shift the t to t' where t' has zero at the start of the 
            # voltage step
            return a * (1.0 - np.exp(-t / b))
    elif func == 1:
        def exp_func(t, a, b):
            # do a "proper exponential" decay fit
            # i.e. shift the t to t' where t' has zero at the start of the 
            # voltage step
            return a * np.exp(-(t - x[0]) / b)
    time_window = np.where(np.logical_and(times > t_start, times <= t_end))[0]
    i_trim = np.argmin(np.abs(times - (t_start + t_trim))) - time_window[0]
    i_fit_until = np.argmin(np.abs(times - (t_start + t_fit_until))) \
                  - time_window[0]
    # trim off the first i_trim (100ms) in case it is still shooting up...
    x = times[time_window[0] + i_trim:time_window[0] + i_fit_until]
    if func == 0:
        x = np.copy(x - times[time_window[0]])
    y = current[time_window[0] + i_trim:
                time_window[0] + i_fit_until]
    try:
        popt, pcov = curve_fit(exp_func, x, y, p0=p0)
        tau = 1e3 * popt[1]  # [ms]
    except:
        raise Exception('Maybe not here!')
    if debug:
        fig = plt.figure()
        plt.plot(times[time_window[0] - 500:time_window[-1] + 500],
                 current[time_window[0] - 500:time_window[-1] + 500],
                 c='#d62728')
        plot_times = times[time_window[0]:time_window[0] + i_fit_until]
        if func == 0:
            fitted_times = plot_times - times[time_window[0]]
        elif func == 1:
            fitted_times = plot_times
        fitted = exp_func(fitted_times, *popt)
        plt.plot(plot_times, fitted, '--', c='#1f77b4')
        plt.plot(times[time_window][0], fitted[0], 'kx')
        plt.axvline(x=times[time_window[0] + i_trim])
        plt.axvline(x=times[time_window[0] + i_fit_until])
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Zhou1998/Zhou1998-fig%s/%sC-%s.png'
                    % (debugout[0], temperature, debugout[1]))
            plt.close()
    return tau


def fit_tau_double_exp(current, times,
                       t_start, t_end,
                       t_trim, t_fit_until,
                       debug=False, debugout=''):
    # use 4-parameters 'double exponential' fit to the current
    from scipy.optimize import curve_fit
    def exp_func(t, a, b, c, d):
        # Shift the t to t' where t' has zero at the start of the 
        # voltage step
        return a * np.exp( -(t - x[0]) / b) + c * np.exp( -(t - x[0]) / d)
    time_window = np.where(np.logical_and(times > t_start, times <= t_end))[0]
    i_trim = np.argmin(np.abs(times - (t_start + t_trim))) - time_window[0]
    i_fit_until = np.argmin(np.abs(times - (t_start + t_fit_until))) \
                  - time_window[0]
    # trim off the first i_trim (100ms) in case it is still shooting up...
    x = times[time_window[0] + i_trim:time_window[0] + i_fit_until]
    y = current[time_window[0] + i_trim:
                time_window[0] + i_fit_until]
    try:
        popt, pcov = curve_fit(exp_func, x, y, p0=[0.1, 2, 0.1, 3.8],
                               bounds=([-1, 1e-6, -1, 1e-6], [1, 5, 1, 10]))
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
            plt.savefig('figs/Zhou1998/Zhou1998-fig%s/%sC-%s.png'
                    % (debugout[0], temperature, debugout[1]))
            plt.close()
    return tau_1, tau_2


#
# Simulate
#

times_prt1, ttotal_prt1, tmeasure_prt1 = prt1(None, return_times=True)
tau_act = {}
tau_deact = {}

times_prt2, ttotal_prt2, tmeasure_prt2 = prt2(None, return_times=True)
tau_inact = {}

times_prt3, ttotal_prt3, tmeasure_prt3 = prt3(None, return_times=True)
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
    model_prt3 = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                         protocol_def=prt3,
                         temperature=273.15 + float(temperature),  # K
                         transform=None,
                         useFilterCap=False,  # ignore capacitive spike
                         effEK=False)  # OK to switch this off here

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

            i_prt1 = model_prt1.simulate(p, times_prt1)
            i_prt2 = model_prt2.simulate(p, times_prt2)
            i_prt3 = model_prt3.simulate(p, times_prt3)

            tau1 = fit_tau_single_exp(i_prt1, times_prt1,
                                      tmeasure_prt1[0], tmeasure_prt1[0] + 5,
                                      t_trim=0.5, t_fit_until=3,
                                      func=0,
                                      debug=debug, debugout=['-act', cell])
            tau_act[temperature].append(tau1)

            tau2 = fit_tau_single_exp(i_prt1, times_prt1,
                                      tmeasure_prt1[1], tmeasure_prt1[1] + 5,
                                      # t_trim=200e-3, t_fit_until=3.5,
                                      t_trim=200e-3, t_fit_until=1.5,
                                      # t_trim=2, t_fit_until=4,
                                      func=1,
                                      debug=debug, debugout=['-deact', cell])
            tau_deact[temperature].append(tau2)

            tau3 = fit_tau_single_exp(i_prt2, times_prt2,
                                      tmeasure_prt2, tmeasure_prt2 + 20e-3,
                                      t_trim=0.1e-3, t_fit_until=4e-3,
                                      func=1, p0=[0.025, 5e-3],
                                      debug=debug, debugout=['-inact', cell])
            tau_inact[temperature].append(tau3)

            tau4 = fit_tau_single_exp(i_prt3, times_prt3,
                                      tmeasure_prt3, tmeasure_prt3 + 20e-3,
                                      t_trim=0, t_fit_until=12e-3,
                                      func=0,
                                      debug=debug,
                                      debugout=['-recovery', cell])
            tau_rec[temperature].append(tau4)


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

# N pairs of cells are within Zhou et al. 1998 estimation (1st std)
# method 1
T1_mean_n_act = np.sum(np.logical_and(
    ZhouQ10[0][0] - ZhouQ10[0][1] < T1_mean_q10_act,
    ZhouQ10[0][0] + ZhouQ10[0][1] > T1_mean_q10_act))
T2_mean_n_act = np.sum(np.logical_and(
    ZhouQ10[0][0] - ZhouQ10[0][1] < T2_mean_q10_act,
    ZhouQ10[0][0] + ZhouQ10[0][1] > T2_mean_q10_act))
T1_mean_n_deact = np.sum(np.logical_and(
    ZhouQ10[0][0] - ZhouQ10[0][1] < T1_mean_q10_deact,
    ZhouQ10[0][0] + ZhouQ10[0][1] > T1_mean_q10_deact))
T2_mean_n_deact = np.sum(np.logical_and(
    ZhouQ10[0][0] - ZhouQ10[0][1] < T2_mean_q10_deact,
    ZhouQ10[0][0] + ZhouQ10[0][1] > T2_mean_q10_deact))
T1_mean_n_inact = np.sum(np.logical_and(
    ZhouQ10[0][0] - ZhouQ10[0][1] < T1_mean_q10_inact,
    ZhouQ10[0][0] + ZhouQ10[0][1] > T1_mean_q10_inact))
T2_mean_n_inact = np.sum(np.logical_and(
    ZhouQ10[0][0] - ZhouQ10[0][1] < T2_mean_q10_inact,
    ZhouQ10[0][0] + ZhouQ10[0][1] > T2_mean_q10_inact))
T1_mean_n_rec = np.sum(np.logical_and(
    ZhouQ10[0][0] - ZhouQ10[0][1] < T1_mean_q10_rec,
    ZhouQ10[0][0] + ZhouQ10[0][1] > T1_mean_q10_rec))
T2_mean_n_rec = np.sum(np.logical_and(
    ZhouQ10[0][0] - ZhouQ10[0][1] < T2_mean_q10_rec,
    ZhouQ10[0][0] + ZhouQ10[0][1] > T2_mean_q10_rec))
# method 2
n_act = np.sum(np.logical_and(
    ZhouQ10[0][0] - ZhouQ10[0][1] < q10_act_combinations,
    ZhouQ10[0][0] + ZhouQ10[0][1] > q10_act_combinations))
n_deact = np.sum(np.logical_and(
    ZhouQ10[1][0] - ZhouQ10[1][1] < q10_deact_combinations,
    ZhouQ10[1][0] + ZhouQ10[1][1] > q10_deact_combinations))
n_inact = np.sum(np.logical_and(
    ZhouQ10[2][0] - ZhouQ10[2][1] < q10_inact_combinations,
    ZhouQ10[2][0] + ZhouQ10[2][1] > q10_inact_combinations))
n_rec = np.sum(np.logical_and(
    ZhouQ10[3][0] - ZhouQ10[3][1] < q10_rec_combinations,
    ZhouQ10[3][0] + ZhouQ10[3][1] > q10_rec_combinations))

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
print('With %s pairs within Zhou et al. 1998 estimation' \
        % min(n_act, n_deact, n_inact, n_rec))
print('----' * 20)
print('Zhou et al. 1998')
print('Q10_activation:', ZhouQ10[0][0], '+/-', ZhouQ10[0][1])
print('Q10_deactivation:', ZhouQ10[1][0], '+/-', ZhouQ10[1][1])
print('Q10_inactivation:', ZhouQ10[2][0], '+/-', ZhouQ10[2][1])
print('Q10_recovery:', ZhouQ10[3][0], '+/-', ZhouQ10[3][1])

## eof
