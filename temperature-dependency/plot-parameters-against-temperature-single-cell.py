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
        'herg25oc1-staircaseramp-A01-fcap-solution-1027799654.txt',
        'herg27oc1-staircaseramp-A01-fcap-solution-1027799654.txt',
        'herg30oc1-staircaseramp-A01-fcap-solution-1027799654.txt',
        'herg33oc1-staircaseramp-A03-fcap-solution-1027799654.txt',
        'herg37oc3-staircaseramp-B06-fcap-solution-1027799654.txt',
        ]
temperatures = [25.0, 27.0, 30.0, 33.0, 37.0]
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
for file_name, temperature in zip(file_list, temperatures):
        ######################################################################
        ## Get data
        ######################################################################
        data_file = os.path.realpath(os.path.join(file_dir, 
                                                  file_name))

        #
        # Load earlier result
        #
        #'''
        obtained_parameters = np.loadtxt(data_file)

        print('temperature = ' + str(temperature))

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


plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

if '--show' in sys.argv:
    plt.show()
else:
    plt.savefig('%s/test-fcap-temperature-parameters-%s.png'%(savedir, plot_axes),
                bbox_iches='tight')

