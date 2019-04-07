#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param


savedir = './figs/paper'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

file_dir = './out'
file_list = [
        'herg25oc',
        'herg27oc',
        'herg30oc',
        'herg33oc',
        'herg37oc',
        ]
temperatures = np.array([25.0, 27.0, 30.0, 33.0, 37.0])
temperatures += 273.15  # in K
fit_seed = 542811797

# Load pseudo2hbm
mean_chains = []
for i_temperature, (file_name, temperature) in enumerate(zip(file_list,
    temperatures)):

    load_file = './out-mcmc/%s-pseudo2hbm-lognorm-mean.txt' % (file_name)
    mean_chain = np.loadtxt(load_file)  # transformed

    mean_chains.append(mean_chain)
mean_chains = np.asarray(mean_chains)

# Eyring and Q10
from temperature_models import eyringA, eyringB, eyringG, eyringT
from temperature_models import q10A, q10B, q10G, q10T
from temperature_models import eyring_transform_to_model_param

eyring_mean = np.loadtxt('%s/eyring-mean.txt' % file_dir)
q10_mean = np.loadtxt('%s/q10-mean.txt' % file_dir)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))
axes[0, 0].set_ylabel(r'$a_\infty$', fontsize=14)
axes[0, 1].set_ylabel(r'$\tau_a$ [s]', fontsize=14)
axes[1, 0].set_ylabel(r'$r_\infty$', fontsize=14)
axes[1, 1].set_ylabel(r'$\tau_r$ [ms]', fontsize=14)
for i in range(2):
    axes[0, i].tick_params(labelbottom=False)
    axes[1, i].set_xlabel('Voltage [mV]', fontsize=14)
v = np.linspace(-100, 60, 200)  # mV

import seaborn as sns
# colour_list = sns.cubehelix_palette(len(SORTED_CELLS))
colour_list = sns.color_palette('coolwarm', n_colors=len(temperatures))
colour_list = colour_list.as_hex()

for i_T, T in enumerate(temperatures):

    # HBM mean parameters
    hbm_T_mean = transform_to_model_param(
            np.mean(mean_chains[i_T], axis=0))

    # Eyring parameters
    eyring_T_mean = eyringT(eyring_mean, T)
    eyring_param = eyring_transform_to_model_param(eyring_T_mean, T)

    # Q10 parameters
    q10_T_mean = q10T(q10_mean, T)
    q10_param = eyring_transform_to_model_param(q10_T_mean, T)

    # k1 
    k1_hbm = hbm_T_mean[1] * np.exp(hbm_T_mean[2] * v * 1e-3)
    k1_eyring = eyring_param[1] * np.exp(eyring_param[2] * v * 1e-3)
    k1_q10 = q10_param[1] * np.exp(q10_param[2] * v * 1e-3)
    # k2
    k2_hbm = hbm_T_mean[3] * np.exp(-1 * hbm_T_mean[4] * v * 1e-3)
    k2_eyring = eyring_param[3] * np.exp(-1 * eyring_param[4] * v * 1e-3)
    k2_q10 = q10_param[3] * np.exp(-1 * q10_param[4] * v * 1e-3)
    # k3
    k3_hbm = hbm_T_mean[5] * np.exp(hbm_T_mean[6] * v * 1e-3)
    k3_eyring = eyring_param[5] * np.exp(eyring_param[6] * v * 1e-3)
    k3_q10 = q10_param[5] * np.exp(q10_param[6] * v * 1e-3)
    # k4
    k4_hbm = hbm_T_mean[7] * np.exp(-1 * hbm_T_mean[8] * v * 1e-3)
    k4_eyring = eyring_param[7] * np.exp(-1 * eyring_param[8] * v * 1e-3)
    k4_q10 = q10_param[7] * np.exp(-1 * q10_param[8] * v * 1e-3)

    # HBM
    a_inf_hbm = k1_hbm / (k1_hbm + k2_hbm)
    a_tau_hbm = 1e3 / (k1_hbm + k2_hbm)
    r_inf_hbm = k4_hbm / (k3_hbm + k4_hbm)
    r_tau_hbm = 1e3 / (k3_hbm + k4_hbm)

    # Eyring
    a_inf_eyring = k1_eyring / (k1_eyring + k2_eyring)
    a_tau_eyring = 1e3 / (k1_eyring + k2_eyring)
    r_inf_eyring = k4_eyring / (k3_eyring + k4_eyring)
    r_tau_eyring = 1e3 / (k3_eyring + k4_eyring)

    # Q10
    a_inf_q10 = k1_q10 / (k1_q10 + k2_q10)
    a_tau_q10 = 1e3 / (k1_q10 + k2_q10)
    r_inf_q10 = k4_q10 / (k3_q10 + k4_q10)
    r_tau_q10 = 1e3 / (k3_q10 + k4_q10)

    # Plot
    axes[0, 0].plot(v, a_inf_hbm, ls='-', c=colour_list[i_T],
            label='_nolegend_' if i_T else 'HBM')
    axes[0, 0].plot(v, a_inf_eyring, ls='--', c=colour_list[i_T],
            label='_nolegend_' if i_T else 'Eyring')
    axes[0, 0].plot(v, a_inf_q10, ls=':', c=colour_list[i_T],
            label='_nolegend_' if i_T else r'$Q_{10}$')

    axes[0, 1].plot(v, a_tau_hbm * 1e-3, ls='-', c=colour_list[i_T])
    axes[0, 1].plot(v, a_tau_eyring * 1e-3, ls='--', c=colour_list[i_T])
    axes[0, 1].plot(v, a_tau_q10 * 1e-3, ls=':', c=colour_list[i_T])

    axes[1, 0].plot(v, r_inf_hbm, ls='-', c=colour_list[i_T])
    axes[1, 0].plot(v, r_inf_eyring, ls='--', c=colour_list[i_T])
    axes[1, 0].plot(v, r_inf_q10, ls=':', c=colour_list[i_T])

    axes[1, 1].plot(v, r_tau_hbm, ls='-', c=colour_list[i_T])
    axes[1, 1].plot(v, r_tau_eyring, ls='--', c=colour_list[i_T])
    axes[1, 1].plot(v, r_tau_q10, ls=':', c=colour_list[i_T])

plt.tight_layout(pad=0.1, w_pad=0.125, h_pad=0.25)

# Colorbar
fig.subplots_adjust(top=0.9)
cbar_ax = fig.add_axes([0.1, 0.95, 0.8, 0.0325])
cmap = ListedColormap(colour_list)
cbar = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
        orientation='horizontal')
cbar.ax.get_xaxis().set_ticks([])
for j, lab in enumerate(temperatures - 273.15):
    cbar.ax.text((2 * j + 1) / 10.0, .5, int(lab), ha='center', va='center')
cbar.set_label(r'Temperature [$^\circ$C]', fontsize=14)

# Save fig
axes[0, 0].legend()
axes[0, 0].set_ylim([-0.05, 1.05])
axes[1, 0].set_ylim([-0.05, 1.05])
plt.savefig('%s/eyring-q10-ss-tau.pdf' % (savedir), format='pdf',
        bbox_iches='tight')
plt.savefig('%s/eyring-q10-ss-tau.png' % (savedir), bbox_iches='tight')
plt.close('all')
