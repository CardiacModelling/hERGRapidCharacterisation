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

import model_ikr as m
from protocols import Vandenberg2006_conductance as prt

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param


debug = False
common_conductance = 1.0
fit_seed = 542811797
temperatures = ['25.0', '27.0', '30.0', '33.0'][::-1]
color = {'25.0':'C1',
         '27.0':'C2',
         '30.0':'C3',
         '33.0':'C4',
         '37.0':'C5',}


file_list = {
        '25.0': 'herg25oc',
        '27.0': 'herg27oc',
        '30.0': 'herg30oc',
        '33.0': 'herg33oc',
        '37.0': 'herg37oc',
    }

# Load pseudo2hbm
mean_chains = []
for temperature in temperatures:
    file_name = file_list[temperature]

    load_file = './out-mcmc/%s-pseudo2hbm-lognorm-mean.txt' % (file_name)
    mean_chain = np.loadtxt(load_file)  # transformed

    mean_chains.append(mean_chain)
mean_chains = np.asarray(mean_chains)

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


def get_tau_correction(folded_current, n_steps, t_start, t_end, debug=False):
    # use 2-parameters exponential fit to the tail
    import scipy
    def exp_func(t, a, b):
        # do a "proper exponential" decay fit
        # i.e. shift the t to t' where t' has zero at the start of the 
        # voltage step
        return - a * np.exp( -b * (t - x[0]))
    times = folded_current['time']
    time_window = np.where(np.logical_and(times > t_start, times <= t_end))[0]
    tau = np.zeros(n_steps)
    i_trim = 50  # assuming DT=1e-4 s -> 10 ms
    i_fit_until = 150  # assuming DT=1e-4 s -> 30 ms
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
            tau[i] = 1. / popt[1] * 1e3  # ms
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
    if debug:
        plt.axvline(x=times[time_window[0] + i_trim])
        plt.axvline(x=times[time_window[0] + i_fit_until])
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Vandenberg2006/fig2a/'
                    'Vandenberg2006-fig2a2-%sC-%s.png'
                    % (temperature, file_name))
            plt.close()

        # Plot Figure 5B2 for this cell too
        plt.plot(tau)
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Vandenberg2006/fig2a/'
                    'Vandenberg2006-fig2a2-%sC-%s-debug.png'
                    % (temperature, file_name))
            plt.close()
    return tau


# Simulate

times, ttotal, tmeasure = prt(None, return_times=True)
I = {}
v_steps = prt(None, return_voltage=True)

for i_T, temperature in enumerate(temperatures):

    I[temperature] = []

    # Model
    model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                    protocol_def=prt,
                    temperature=273.15 + float(temperature),  # K
                    transform=None,
                    useFilterCap=False,  # ignore capacitive spike
                    effEK=False,  # OK to switch this off here
                    concK=[4.8 + 0.3, 120 + 20])

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
    p[0] = common_conductance  # TODO: Try not set common conductance?

    i = myokit.DataLog()
    i['time'] = times
    i['current'] = model.simulate(p, times)
    i['voltage'] = model.voltage(times)
    i.set_time_key('time')
    i = i.fold(ttotal)

    if debug and False:
        # Figure 2A
        for ii in range(len(iv_steps)):
            plt.plot(i['time'], i[str(ii)+'.current'])
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Vandenberg2006/'
                    + 'Vandenberg2006-fig2a-%sC.png'%temperature)
            plt.close()

    ii = get_IV(i, len(v_steps), 
                          tmeasure[0],
                          tmeasure[0] + 0.04)
    tau = get_tau_correction(i, len(v_steps), 
                          tmeasure[1],
                          tmeasure[1] + 0.2,
                          debug=debug)
    I[temperature].append(ii * np.exp(10 / tau))  # TODO negative?!

    del(i)

# convert units V -> mV
v_steps = np.around(np.array(v_steps) * 1e3, 1)

# Figure 2B
plt.figure(figsize=(8, 8))
for temperature in temperatures:
    for i, ii in enumerate(I[temperature]):
        plt.scatter(v_steps, ii,
                    c=color[temperature],
                    label='__nolegend__' if i else temperature+' $^o$C')

plt.legend()
plt.axvline(0, color='#7f7f7f')
plt.axhline(0, color='#7f7f7f')
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('figs/Vandenberg2006/Vandenberg2006-fig2b.png')
    plt.close()

## eof
