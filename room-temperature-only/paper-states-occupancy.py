#!/usr/bin/env python2
import sys
sys.path.append('../lib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import protocols
import model_ikr as m

# Setup
temperature = 23.0 + 273.15  # K
sine_prt = protocols.sine_wave
sine_model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                    protocol_def=sine_prt,
                    temperature=temperature,  # K
                    transform=None,
                    useFilterCap=False)
t_sine = np.arange(0, 8, 0.5e-3)
stair_prt = protocols.leak_staircase
stair_model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                    protocol_def=stair_prt,
                    temperature=temperature,  # K
                    transform=None,
                    useFilterCap=False)
t_stair = np.arange(0, 15.5, 0.5e-3)
p = np.loadtxt('kylie-room-temperature/last-solution_C5.txt')
# Change conductance unit nS->pS (new parameter use V, but here mV)
p[0] = p[0] * 1e3
labels = ['O', 'I', 'C', 'IC']
states = ['ikr.open', 'ikr.active']
colors = ['#b3cde3', '#fbb4ae', '#ccebc5', '#decbe4']
suffix = ''


def compute_states(hh_open, hh_active):
    # return O, I, C, IC
    O = hh_open * hh_active
    I = hh_open * (1. - hh_active)
    C = hh_active * (1. - hh_open)
    IC = (1. - hh_open) * (1. - hh_active)
    return O, I, C, IC


# Simulate
v_sine = sine_model.voltage(t_sine) * 1000
d_sine = sine_model.simulate(p, t_sine, extra_log=states)
i_sine = d_sine['ikr.IKr']
o_sine, a_sine = d_sine[states[0]], d_sine[states[1]]
states_sine = compute_states(o_sine, a_sine)
v_stair = stair_model.voltage(t_stair) * 1000
d_stair = stair_model.simulate(p, t_stair, extra_log=states)
i_stair = d_stair['ikr.IKr']
o_stair, a_stair = d_stair[states[0]], d_stair[states[1]]
states_stair = compute_states(o_stair, a_stair)

# Plot sine wave
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 4))
axes[0].plot(t_sine, v_sine)
axes[1].plot(t_sine, i_sine)
axes[2].stackplot(t_sine,
                  states_sine[0], states_sine[1],
                  states_sine[2], states_sine[3],
                  labels=labels,
                  colors=colors)
axes[2].legend()
axes[0].set_ylabel('Voltage\n[mV]', fontsize=16, rotation=0, labelpad=40)
axes[1].set_ylabel('Current\n[pA]', fontsize=16, rotation=0, labelpad=40)
axes[2].set_ylabel('State\noccupancy', fontsize=16, rotation=0, labelpad=40)
axes[2].set_xlabel('Time [s]', fontsize=16)
axes[2].set_xlim([0, 8])
axes[2].set_ylim([0, 0.995])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(hspace=0)
plt.savefig('figs/paper/states-occupancy-sine%s.png' % suffix,
            bbox_inch='tight', dpi=300)
plt.savefig('figs/paper/states-occupancy-sine%s.pdf' % suffix, format='pdf',
            bbox_inch='tight')
plt.close('all')

# Plot staircase
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 4))
axes[0].plot(t_stair, v_stair)
axes[1].plot(t_stair, i_stair)
axes[2].stackplot(t_stair,
                  states_stair[0], states_stair[1],
                  states_stair[2], states_stair[3],
                  labels=labels,
                  colors=colors)
axes[2].legend()
axes[0].set_ylabel('Voltage\n[mV]', fontsize=16, rotation=0, labelpad=40)
axes[1].set_ylabel('Current\n[pA]', fontsize=16, rotation=0, labelpad=40)
axes[2].set_ylabel('State\noccupancy', fontsize=16, rotation=0, labelpad=40)
axes[2].set_xlabel('Time [s]', fontsize=16)
axes[2].set_xlim([0, 15.5])
axes[2].set_ylim([0, 0.995])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(hspace=0)
plt.savefig('figs/paper/states-occupancy-stair%s.png' % suffix,
            bbox_inch='tight', dpi=300)
plt.savefig('figs/paper/states-occupancy-stair%s.pdf' % suffix, format='pdf',
            bbox_inch='tight')
plt.close('all')
