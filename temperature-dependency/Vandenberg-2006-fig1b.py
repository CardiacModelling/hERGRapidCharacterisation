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

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

prt = 'vandenberg-et-al-2006/vandenberg-et-al-2006-fig1b-corrected.csv'
prt_timeseries = np.loadtxt(prt, delimiter=',', skiprows=1)
times = prt_timeseries[:, 0]
voltage = prt_timeseries[:, 1]
''' # -02hz
time_interest = (45, 45.2888)  # in second
time_interest_idx = np.logical_and(times > time_interest[0],
        times < time_interest[1])
#'''
''' # -2hz
time_interest = (9.5, 9.7888)  # in second
time_interest_idx = np.logical_and(times > time_interest[0],
        times < time_interest[1])
#'''
#''' # '1 beat'
time_interest_idx = [True] * len(times)
#'''

debug = False
common_conductance = 1.0
temperatures = ['25.0', '37.0'][::-1]


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

color = {'25.0':fan_blue[-1],
         '33.0':fan_green[-1],
         '37.0':fan_red[-1],}

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


# Simulate and plot
fig, axes = plt.subplots(2, 2, figsize=(12, 6),
        gridspec_kw={'height_ratios':[1, 2.5]})

axes[0, 1].plot(times[time_interest_idx], voltage[time_interest_idx],
        color='#7f7f7f')
axes[0, 1].set_ylabel('Voltage [mV]', fontsize=14)
axes[0, 1].set_xticks([])

for i_T, temperature in enumerate(temperatures):

    # Model
    model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                    protocol_def=prt,
                    temperature=273.15 + float(temperature),  # K
                    transform=None,
                    useFilterCap=False,  # ignore capacitive spike
                    effEK=False,  # OK to switch this off here
                    concK=[4.8 + 0.3, 120 + 20])

    # HBM mean parameters
    hbm_T_mean = transform_to_model_param(np.mean(mean_chains[i_T], axis=0))

    p = hbm_T_mean
    p[0] = common_conductance

    c = model.simulate(p, times)

    axes[1, 1].plot(times[time_interest_idx], c[time_interest_idx],
            c=color[temperature],
            label=str(int(float(temperature))) + r'$^\circ$C')

    np.random.seed(101)
    i_cov_all = []
    for s in range(120):

        p = np.random.multivariate_normal(mean_chains[i_T][s, :],
                cov_chains[i_T][s, :, :])
        p = transform_to_model_param(p)
        p[0] = common_conductance

        c = model.simulate(p, times)
        i_cov_all.append(c)

    percentiles = [90, 60, 30]
    for i_p, p in enumerate(percentiles):
        top = np.nanpercentile(i_cov_all, 50 + p / 2., axis=0)
        bot = np.nanpercentile(i_cov_all, 50 - p / 2., axis=0)

        alpha = 0.8
        c = color_fan[temperature][i_p]
        axes[1, 1].fill_between(times[time_interest_idx],
                top[time_interest_idx],
                bot[time_interest_idx],
                color=c, alpha=alpha, linewidth=0)


axes[1, 1].set_ylabel('Current [$g=%s$]' % common_conductance, fontsize=14)
axes[1, 1].set_xlabel('Times [s]', fontsize=14)

axes[1, 1].legend()
axes[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

# New stuffs
data_dir = '../data-autoLC'
protocol_dir = '../protocol-time-series'
data_fancharts_dir = './out/data-fancharts'
prt = 'ap05hz'
file_list_tmp = {
        '25.0': 'herg25oc1',
        '33.0': 'herg33oc1',
        '37.0': 'herg37oc3',
    }

protocol_def = 'protocol-%s.csv' % prt
protocol_def = '%s/%s' % (protocol_dir, protocol_def)
model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + float(temperatures[0]),  # K
                transform=None,
                useFilterCap=False)  # ignore capacitive spike
times = np.loadtxt('%s/%s-%s-times.csv' % (data_dir, 'herg25oc1',
    prt), delimiter=',', skiprows=1)
idx = np.logical_and(times > 0, times < 0.75)
axes[0, 0].plot(times[idx], model.voltage(times)[idx] * 1000, color='#7f7f7f')
axes[0, 0].set_ylabel('Voltage [mV]', fontsize=14)
axes[0, 0].set_xticks([])

for i_T, temperature in enumerate(temperatures):
    file_name = file_list_tmp[temperature]
    # Plot data as background
    percentiles = np.loadtxt('%s/percentiles.txt' % data_fancharts_dir)
    fan_chart_data_top = np.loadtxt('%s/%s-%s-top.txt' % \
            (data_fancharts_dir, file_name, prt))
    fan_chart_data_bot = np.loadtxt('%s/%s-%s-bot.txt' % \
            (data_fancharts_dir, file_name, prt))
    fan_x = np.loadtxt('%s/%s-%s-times.txt' % \
            (data_fancharts_dir, file_name, prt))

    for i_p, p in enumerate(percentiles):
        alpha = 0.8
        color = color_fan[temperature][i_p]
        top = fan_chart_data_top[:, i_p]
        bot = fan_chart_data_bot[:, i_p]
        axes[1, 0].fill_between(fan_x[idx], top[idx], bot[idx], color=color,
                alpha=alpha, linewidth=0)
axes[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[1, 0].set_ylabel('Normalised current', fontsize=12)
axes[1, 0].set_xlabel('Times [s]', fontsize=14)

# Done
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(hspace=0.12)

if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('figs/paper/re-Vandenberg2006-fig1b.png', dpi=200,
        bbox_iches='tight')
    # plt.savefig('figs/paper/re-Vandenberg2006-fig1b.pdf', format='pdf',
    #     bbox_iches='tight')
    plt.close()

## eof
