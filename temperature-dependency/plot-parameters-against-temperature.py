#!/usr/bin/env python2
from __future__ import print_function
import os
import sys
import re
import glob
import numpy as np
if '--show' not in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

#''' some selected cells
savedir = './figs'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

file_dir = './out'
file_list = [
        'herg25oc1',
        'herg27oc1',
        'herg30oc1',
        'herg33oc1',
        'herg37oc3',
        ]
temperatures = np.array([25.0, 27.0, 30.0, 33.0, 37.0])
temperatures += 273.15  # in K
fit_seed = '542811797'
withfcap = False
#'''

#
# Define some parameters and labels
#
labels = [r'$g$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
          r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$']
param_A = [1, 3, 5, 7]
param_B = [2, 4, 6, 8]

#
# Option to plot
#
axes_opt = ['normal_axes', 'Eyring_axes']
plot_axes = axes_opt[1]
plot_stair_param = True
plot_g = True

#
# Create figure
#
if plot_g:
    _, axes = plt.subplots(3, 4, sharex=True, figsize=(15, 8))
else:
    _, axes = plt.subplots(2, 4, sharex=True, figsize=(15, 5))

#
# Add labels
#
for i, p_i in enumerate(param_A):
    # i for axes index
    # p_i for parameter index (used by labels)
    if plot_axes == 'normal_axes':
        axes[0, i].set_xlabel('T [K]')
        axes[0, i].set_ylabel('A [s]')
    elif plot_axes == 'Eyring_axes':
        axes[0, i].set_xlabel('1/T [1/K]')
        axes[0, i].set_ylabel('ln(A/T)')
    axes[0, i].set_title(labels[p_i])
for i, p_i in enumerate(param_B):
    # i for axes index
    # p_i for parameter index (used by labels)
    if plot_axes == 'normal_axes':
        axes[1, i].set_xlabel('T [K]')
        axes[1, i].set_ylabel('B [V]')
    elif plot_axes == 'Eyring_axes':
        axes[1, i].set_xlabel('1/T [1/K]')
        axes[1, i].set_ylabel('B [1/V]')
    axes[1, i].set_title(labels[p_i])
# for conductance
if plot_g:
    if plot_axes == 'normal_axes':
        axes[2, 0].set_xlabel('T [K]')
        axes[2, 0].set_ylabel('g')
    elif plot_axes == 'Eyring_axes':
        axes[2, 0].set_xlabel('1/T [1/K]')
        axes[2, 0].set_ylabel('g')
    for i in range(1, len(param_A)):
        axes[2, i].axis('off')


# Loop through files
mean_param = np.inf * np.ones((9, len(temperatures)))
for i_temperature, (file_name, temperature) in enumerate(zip(file_list,
    temperatures)):
    ######################################################################
    ## Get fitting results
    ######################################################################
    files_dir = os.path.realpath(os.path.join(file_dir, file_name))
    searchwfcap = '-fcap' if withfcap else ''
    selectedfile = './manualselection/manualselected-%s.txt' % (file_name)
    selectedwell = []
    with open(selectedfile, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                selectedwell.append(l.split()[0])
    print('Getting ', file_name)

    all_cell_param = []
    for cell in selectedwell:
        param_file = '%s/%s-staircaseramp-%s-solution%s-%s.txt' % (files_dir,
                file_name, cell, searchwfcap, fit_seed)
        # if temperature == 37.0 + 273.15:
        #     param_file = '%s/%s-staircaseramp-%s-solution%s-effEK-%s.txt' \
        #         % (files_dir, file_name, cell, searchwfcap, fit_seed)

        try:
            obtained_parameters = np.loadtxt(param_file)
        except FileNotFoundError:
            continue
        all_cell_param.append(obtained_parameters)

        #
        # Plot the stuffs
        #
        # param A
        for i, p_i in enumerate(param_A):
            # i for axes index
            # obtained_parameters use p_i to get parameter i
            if plot_axes == 'normal_axes':
                # plot A agains T
                y = obtained_parameters[p_i]
                x = temperature
            elif plot_axes == 'Eyring_axes':
                # plot ln(A/2.0836E10*T) against 1/T
                y = np.log( obtained_parameters[p_i] / temperature )
                x = 1. / temperature
            axes[0, i].plot(x, y, c='k', ls='', marker='.')

        # param B
        for i, p_i in enumerate(param_B):
            # i for axes index
            # obtained_parameters use p_i to get parameter i
            if plot_axes == 'normal_axes':
                # plot A agains T
                y = obtained_parameters[p_i]
                x = temperature
            elif plot_axes == 'Eyring_axes':
                # plot B against 1/T (should give 1 dof only!)
                y = obtained_parameters[p_i]
                x = 1. / temperature
            axes[1, i].plot(x, y, c='k', ls='', marker='.')

        # conductance
        if plot_g:
            y = obtained_parameters[0]
            if plot_axes == 'normal_axes':
                x = temperature
            elif plot_axes == 'Eyring_axes':
                x = 1. / temperature
            axes[2, 0].plot(x, y, c='k', ls='', marker='.')

    # Calculate mean
    # mean_param[:, i_temperature] = np.mean(all_cell_param, axis=0)
    all_cell_param = np.array(all_cell_param)
    '''
    for i in range(9):
        upper = np.percentile(all_cell_param[:, i], 90)
        lower = np.percentile(all_cell_param[:, i], 10)
        tmp = all_cell_param[:, i][np.logical_and(lower < all_cell_param[:, i],
                    all_cell_param[:, i] < upper)]
        mean_param[i, i_temperature] = np.mean(tmp)
    '''
    for i in range(9):
        mean_param[i, i_temperature] = np.mean(all_cell_param[:, i])
    #'''

#
# Plot mean parameters
#
# param A
for i, p_i in enumerate(param_A):
    # i for axes index
    # obtained_parameters use p_i to get parameter i
    if plot_axes == 'normal_axes':
        # plot A agains T
        y = mean_param[p_i, :]
        x = temperatures
    elif plot_axes == 'Eyring_axes':
        # plot ln(A/2.0836E10*T) against 1/T
        y = np.log( mean_param[p_i, :] / temperatures )
        x = 1. / temperatures
    axes[0, i].plot(x, y, c='r', ls='--', marker='x',
                    # label='80 percentile mean')
                    label='mean')
axes[0, 0].legend()

# param B
for i, p_i in enumerate(param_B):
    # i for axes index
    # obtained_parameters use p_i to get parameter i
    if plot_axes == 'normal_axes':
        # plot A agains T
        y = mean_param[p_i, :]
        x = temperatures
    elif plot_axes == 'Eyring_axes':
        # plot B against 1/T (should give 1 dof only!)
        y = mean_param[p_i, :]
        x = 1. / temperatures
    axes[1, i].plot(x, y, c='r', ls='--', marker='x')

if plot_g:
    y = mean_param[0, :]
    if plot_axes == 'normal_axes':
        x = temperatures
    elif plot_axes == 'Eyring_axes':
        x = 1. / temperatures
    axes[2, 0].plot(x, y, c='r', ls='--', marker='x')
    # axes[2, 0].set_ylim([np.min(y) * 0.25, np.max(y) * 4])

#
# Kylie's parameters
#
path_to_solutions = '../room-temperature-only/kylie-room-temperature'
cells_kylie = ['C' + str(i) for i in range(1, 9)]
all_param = []
temperature = 22.5 + 273.15
for cell in cells_kylie:
    last_solution = glob.glob(path_to_solutions+'/*%s*'%cell)[0]
    obtained_parameters = np.loadtxt(last_solution)
    # Change conductance unit nS->pS (new parameter use V, but here mV)
    obtained_parameters[0] = obtained_parameters[0] * 1e3
    all_param.append(obtained_parameters)
    # Plot
    # param A
    for i, p_i in enumerate(param_A):
        # i for axes index
        # obtained_parameters use p_i to get parameter i
        if plot_axes == 'normal_axes':
            # plot A agains T
            y = obtained_parameters[p_i]
            x = temperature
        elif plot_axes == 'Eyring_axes':
            # plot ln(A/2.0836E10*T) against 1/T
            y = np.log( obtained_parameters[p_i] / temperature )
            x = 1. / temperature
        axes[0, i].plot(x, y, c='#ff7f0e', ls='', marker='s')

    # param B
    for i, p_i in enumerate(param_B):
        # i for axes index
        # obtained_parameters use p_i to get parameter i
        if plot_axes == 'normal_axes':
            # plot A agains T
            y = obtained_parameters[p_i]
            x = temperature
        elif plot_axes == 'Eyring_axes':
            # plot B against 1/T (should give 1 dof only!)
            y = obtained_parameters[p_i]
            x = 1. / temperature
        axes[1, i].plot(x, y, c='#ff7f0e', ls='', marker='s')

    # conductance
    if plot_g:
        y = obtained_parameters[0]
        if plot_axes == 'normal_axes':
            x = temperature
        elif plot_axes == 'Eyring_axes':
            x = 1. / temperature
        axes[2, 0].plot(x, y, c='#ff7f0e', ls='', marker='s')
kylie_mean_param = np.mean(all_param, axis=0)


plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

if '--show' in sys.argv:
    plt.show()
else:
    withfcap = 'fcap-' if withfcap else ''
    withEffEK = 'effEK-'
    plt.savefig('%s/temperature-parameters-%s%s%s.png'%(savedir, withEffEK,\
                withfcap, plot_axes), bbox_iches='tight')

