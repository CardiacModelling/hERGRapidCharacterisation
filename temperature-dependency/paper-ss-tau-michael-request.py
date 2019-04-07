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

# Plot
fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
axes[0, 0].set_ylabel(r'Steady state probability', fontsize=12)
axes[1, 0].set_ylabel(r'Time constant [ms]', fontsize=14)
for i in range(2):
    axes[-1, i].set_xlabel('Voltage [mV]', fontsize=14)
v = np.linspace(-100, 60, 200)  # mV

import seaborn as sns
# colour_list = sns.cubehelix_palette(len(SORTED_CELLS))
colour_list = sns.color_palette('coolwarm', n_colors=len(temperatures))
colour_list = colour_list.as_hex()

for i_T, T in enumerate(temperatures):

    # HBM mean parameters
    hbm_T_mean = transform_to_model_param(
            np.mean(mean_chains[i_T], axis=0))

    # k1 
    k1_hbm = hbm_T_mean[1] * np.exp(hbm_T_mean[2] * v * 1e-3)
    # k2
    k2_hbm = hbm_T_mean[3] * np.exp(-1 * hbm_T_mean[4] * v * 1e-3)
    # k3
    k3_hbm = hbm_T_mean[5] * np.exp(hbm_T_mean[6] * v * 1e-3)
    # k4
    k4_hbm = hbm_T_mean[7] * np.exp(-1 * hbm_T_mean[8] * v * 1e-3)

    # HBM
    a_inf_hbm = k1_hbm / (k1_hbm + k2_hbm)
    a_tau_hbm = 1e3 / (k1_hbm + k2_hbm)
    r_inf_hbm = k4_hbm / (k3_hbm + k4_hbm)
    r_tau_hbm = 1e3 / (k3_hbm + k4_hbm)

    # Plot
    for i in range(2):
        axes[0, i].plot(v, a_inf_hbm, ls='-', lw=1.5, c=colour_list[i_T],
                label='_nolegend_' if i_T else r'$a_\infty$', alpha=1-0.7*i)
        axes[0, i].plot(v, r_inf_hbm, ls='--', lw=1.5, c=colour_list[i_T],
                label='_nolegend_' if i_T else r'$r_\infty$', alpha=1-0.7*i)

    axes[1, 0].plot(v, a_tau_hbm, ls='-', lw=1.5, c=colour_list[i_T],
            label='_nolegend_' if i_T else r'$\tau_a$')

    axes[1, 1].plot(v, r_tau_hbm, ls='--', lw=1.5, c=colour_list[i_T],
            label='_nolegend_' if i_T else r'$\tau_r$')

    axes[0, 1].plot(v, a_inf_hbm * r_inf_hbm, ls=':', lw=2,
            c=colour_list[i_T],
            label='_nolegend_' if i_T else r'$a_\infty \times r_\infty$')

    # Some values
    print('Temperature: %s K' % T)
    print('a_\\infty V_{1/2} = %s mV' % v[np.argmin(np.abs(a_inf_hbm - 0.5))])
    print('\\tau_a V_{max} = %s mV' % v[np.argmax(a_tau_hbm)])
    print('max(\\tau_a) = %s ms' % np.max(a_tau_hbm))
    print('r_\\infty V_{1/2} = %s mV' % v[np.argmin(np.abs(r_inf_hbm - 0.5))])
    print('\\tau_r V_{max} = %s mV' % v[np.argmax(r_tau_hbm)])
    print('max(\\tau_r) = %s ms' % np.max(r_tau_hbm))

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
axes[0, 0].set_ylim([-0.05, 1.05])
axes[0, 1].set_ylim([-0.01, 0.26])
axes[0, 1].set_xlim([-65, 45])
for i in range(2):
    for j in range(2):
        axes[i, j].legend()
plt.savefig('%s/ss-tau-michael-request.pdf' % (savedir), format='pdf',
        bbox_iches='tight')
plt.savefig('%s/ss-tau-michael-request.png' % (savedir), bbox_iches='tight')
plt.close('all')
