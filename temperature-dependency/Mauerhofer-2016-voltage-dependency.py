#!/usr/bin/env python2
# 
# Try to reproduce similar figures in Mauerhofer et al. 2016
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
from protocols import Mauerhofer2016_voltage_activation as prt_act
from protocols import Mauerhofer2016_voltage_ssinactivation as prt_inact


debug = False
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
            # give it a bound for fitting: 
            # 1. "decay => all positive"  or  maybe some not 'decay'?
            #    => a bit negative...
            # 2. all current < 500 A/F...
            # 3. delay tend to be slow! (in unit of second though!)
            popt, pcov = scipy.optimize.curve_fit(exp_func, x, y)
            #         , bounds=(-10., [500., 10.]))
            fitted = exp_func(times[time_window[0]:
                                    time_window[0] + i_fit_until], *popt)
            I[i] = np.max(fitted[0])
        except:
            # give up, just print out a warning and use old method
            print('WARNING: CANNOT FIT TO voltage step %d'%(i))
            raise Exception('Maybe not here!')
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
            plt.savefig('figs/Mauerhofer2016/fit-debug/'
                    'Mauerhofer2016-%sC-%s-%s.png'
                    % (temperature, file_name, cell))
            plt.close()

        # Plot Figure 5B2 for this cell too
        plt.plot(I)
        if '--show' in sys.argv:
            plt.show()
        else:
            plt.savefig('figs/Mauerhofer2016/fit-debug/'
                    'Mauerhofer2016-%sC-%s-%s-2.png'
                    % (temperature, file_name, cell))
            plt.close()
    return I


# Simulate

times_act, ttotal_act, tmeasure_act = prt_act(None, return_times=True)
I_activations = {}
av_steps = prt_act(None, return_voltage=True)

times_inact, ttotal_inact, tmeasure_inact = prt_inact(None, return_times=True)
I_inactivations = {}
g_inactivations = {}
iv_steps = prt_inact(None, return_voltage=True)

for temperature in temperatures:

    I_activations[temperature] = []
    I_inactivations[temperature] = []
    g_inactivations[temperature] = []

    # Model
    model_act = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                        protocol_def=prt_act,
                        temperature=273.15 + float(temperature),  # K
                        transform=None,
                        useFilterCap=False,  # ignore capacitive spike
                        effEK=False)  # OK to switch this off here
    model_inact = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                          protocol_def=prt_inact,
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
        selectedwell = selectedwell[:10]

        for i_cell, cell in enumerate(selectedwell):
            print(file_name, cell)
            pfile = './out/%s/%s-staircaseramp-%s-solution-%s.txt' \
                    % (file_name, file_name, cell, fit_seed)
            p = np.loadtxt(pfile)
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

            if True and i_cell == 0:
                # Figure 1A
                for ii in range(len(av_steps)):
                    plt.plot(a['time'], a[str(ii)+'.current'])
                if '--show' in sys.argv:
                    plt.show()
                else:
                    plt.savefig('figs/Mauerhofer2016/' + 
                            'Mauerhofer2016-fig1a-%sC.png'%temperature)
                    plt.close()

            if debug and False:
                # Figure 2A
                for ii in range(len(iv_steps)):
                    plt.plot(i['time'], i[str(ii)+'.current'])
                if '--show' in sys.argv:
                    plt.show()
                else:
                    plt.savefig('figs/Mauerhofer2016/'
                            + 'Mauerhofer2016-fig2a-%sC.png'%temperature)
                    plt.close()

            I_activation = get_IV(a, len(av_steps),
                                  tmeasure_act, tmeasure_act + 0.5)
            I_activations[temperature].append(I_activation)

            I_inactivation = get_corrected_IV(i, len(iv_steps), 
                                              tmeasure_inact,
                                              tmeasure_inact + 0.5,
                                              debug=debug)
            I_inactivations[temperature].append(I_inactivation)

            conductance_inactivation = \
                    I_inactivation / (iv_steps - model_inact.EK())
            g_inactivations[temperature].append(conductance_inactivation)

            del(a, i)

# convert units V -> mV
av_steps = np.around(np.array(av_steps) * 1e3, 1)
iv_steps = np.around(np.array(iv_steps) * 1e3, 1)


# Figure 1C
for temperature in temperatures:
    for i, I_activation in enumerate(I_activations[temperature]):
        if i == 0:
            plt.plot(av_steps, I_activation/np.min(I_activation),
                     c=color[temperature],
                     label=temperature+' $^o$C')
        else:
            plt.plot(av_steps, I_activation/np.min(I_activation),
                     c=color[temperature])
plt.legend()
plt.xlabel('Voltage [mV]')
plt.ylabel('Normalised tail peak current')
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('figs/Mauerhofer2016/Mauerhofer2016-fig1c.png')
    plt.close()



# Figure 2B
for temperature in temperatures:
    for i, I_inactivation in enumerate(I_inactivations[temperature]):
        if i == 0:
            plt.plot(iv_steps, I_inactivation,
                     c=color[temperature],
                     label=temperature+' $^o$C')
        else:
            plt.plot(iv_steps, I_inactivation,
                     c=color[temperature])
plt.legend()
plt.xlabel('Voltage [mV]')
plt.ylabel('Peak current [$g=%s$]' % common_conductance)
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('figs/Mauerhofer2016/Mauerhofer2016-fig2b.png')
    plt.close()

# Figure 2C (extra)
for temperature in temperatures:
    for i, g_inactivation in enumerate(g_inactivations[temperature]):
        if i == 0:
            plt.plot(iv_steps, g_inactivation,
                     c=color[temperature],
                     label=temperature+' $^o$C')
        else:
            plt.plot(iv_steps, g_inactivation,
                     c=color[temperature])
plt.legend()
plt.xlabel('Voltage [mV]')
plt.ylabel('Conductance [$g=%s$]' % common_conductance)
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('figs/Mauerhofer2016/Mauerhofer2016-fig2c-extra.png')
    plt.close()

# Figure 2C
for temperature in temperatures:
    for i, g_inactivation in enumerate(g_inactivations[temperature]):
        if i == 0:
            plt.plot(iv_steps, g_inactivation/np.max(g_inactivation),
                     c=color[temperature],
                     label=temperature+' $^o$C')
        else:
            plt.plot(iv_steps, g_inactivation/np.max(g_inactivation),
                     c=color[temperature])
plt.legend()
plt.xlabel('Voltage [mV]')
plt.ylabel('Normalised conductance')
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('figs/Mauerhofer2016/Mauerhofer2016-fig2c.png')
    plt.close()


# Extra
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
        plt.plot(v_steps, po, c=color[temperature],
                 label='_nolegend_' if ii else temperature+' $^o$C')
plt.legend()
plt.xlabel('Voltage [mV]')
plt.ylabel('Open probability')
if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('figs/Mauerhofer2016/Mauerhofer2016-extra-open-prob.png')
    plt.close()

## eof
