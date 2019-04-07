#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from scipy import stats
from scipy.optimize import curve_fit

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

#
# Option to plot
#
axes_opt = ['normal_axes', 'Eyring_axes']
if '--normal' in sys.argv:
    plot_axes = axes_opt[0]
    from scipy.stats import norm
    from temperature_models import eyring_transform_ppf, eyring_transform_mean
    from temperature_models import eyring_transform_to_model_param
else:
    plot_axes = axes_opt[1]
print('Plotting in %s' % plot_axes)
plot_kylie_param = False
plot_g = True


savedir = './figs/paper'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

savedir_data = './out'
if not os.path.isdir(savedir_data):
    os.makedirs(savedir_data)

file_dir = './out-mcmc'
file_list = [
        'herg25oc',
        'herg27oc',
        'herg30oc',
        'herg33oc',
        'herg37oc',
        ]
temperatures = np.array([25.0, 27.0, 30.0, 33.0, 37.0])
temperatures += 273.15  # in K
assert(len(file_list) == len(temperatures))

# Control fitting seed --> OR DONT
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Fit seed: ', fit_seed)
np.random.seed(fit_seed)


#
# Define some parameters and labels
#
labels = [r'$g$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
          r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$']
param_A = [1, 3, 5, 7]
param_B = [2, 4, 6, 8]


#
# Create figure
#
if plot_g:
    _, axes = plt.subplots(3, 4, figsize=(15, 7))
else:
    _, axes = plt.subplots(2, 4, figsize=(15, 4.5))


#
# Add labels
#
if plot_axes == 'normal_axes':
    axes[0, 0].set_ylabel(r'A [s$^{-1}$]', fontsize=16)
elif plot_axes == 'Eyring_axes':
    axes[0, 0].set_ylabel(r'$\ln$(A/T)', fontsize=16)

for i, p_i in enumerate(param_A):
    axes[0, i].tick_params(labelbottom=False)
    axes[0, i].set_title(labels[p_i], loc='left', fontsize=16)

if plot_axes == 'normal_axes':
    axes[1, 0].set_ylabel(r'B [V$^{-1}$]', fontsize=16)
elif plot_axes == 'Eyring_axes':
    axes[1, 0].set_ylabel(r'B [V$^{-1}$]', fontsize=16)

for i, p_i in enumerate(param_B):
    if not plot_g:
        if plot_axes == 'normal_axes':
            axes[1, i].set_xlabel('Temperature [K]', fontsize=16)
        elif plot_axes == 'Eyring_axes':
            axes[1, i].set_xlabel(r'T$^{-1}$ [K$^{-1}$]', fontsize=16)
            axes[1, i].ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    else:
        if i > 0:
            if plot_axes == 'normal_axes':
                axes[1, i].set_xlabel('Temperature [K]', fontsize=16)
            elif plot_axes == 'Eyring_axes':
                axes[1, i].set_xlabel(r'T$^{-1}$ [K$^{-1}$]', fontsize=16)
                axes[1, i].ticklabel_format(axis='x', style='sci',
                        scilimits=(0,0))
        else:
            axes[1, i].tick_params(labelbottom=False)
    axes[1, i].set_title(labels[p_i], loc='left', fontsize=16)

# for conductance
if plot_g:
    if plot_axes == 'normal_axes':
        axes[2, 0].set_xlabel('Temperature [K]', fontsize=16)
        axes[2, 0].set_ylabel('g', fontsize=16)
    elif plot_axes == 'Eyring_axes':
        axes[2, 0].set_xlabel(r'T$^{-1}$ [K$^{-1}$]', fontsize=16)
        axes[2, 0].set_ylabel('g', fontsize=16)
        axes[2, 0].ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    for i in range(1, len(param_A)):
        axes[2, i].axis('off')
    axes[2, 0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))


#
# Load pseudo2hbm
#
mean_chains = []
for i_temperature, (file_name, temperature) in enumerate(zip(file_list,
    temperatures)):

    print('Getting', file_name)
    load_file = '%s/%s-pseudo2hbm-lognorm-mean.txt' % (file_dir, file_name)
    mean_chain = np.loadtxt(load_file)  # transformed

    # detransform parameters
    mean_chains.append(transform_to_model_param(mean_chain.T).T)
mean_chains = np.asarray(mean_chains)


#
# Fit and plot
#


# param A
for i, p_i in enumerate(param_A):
    # i for axes index
    # obtained_parameters use p_i to get parameter i

    # Fit
    p = mean_chains[:, :, p_i]  # [T, samples, params]
    fit_y = np.log(p.T / temperatures).T
    fit_x = 1. / temperatures

    # Plot
    if plot_axes == 'normal_axes':
        # plot A agains T
        y = list(p)
        x = list(temperatures)
    elif plot_axes == 'Eyring_axes':
        # plot ln(A/2.0836E10*T) against 1/T
        y = list(fit_y)
        x = list(fit_x)

    # Data (violin)
    violinwidth = (x[1] - x[0]) / 3
    violins = axes[0, i].violinplot(y, x, showmeans=True, showmedians=False,
            showextrema=False, widths=violinwidth)

    for v in violins['bodies']:
        v.set_facecolor('C1')
        v.set_edgecolor('black')
        v.set_alpha(0.7)


# param B
for i, p_i in enumerate(param_B):
    # i for axes index
    # obtained_parameters use p_i to get parameter i

    # Fit
    p = mean_chains[:, :, p_i]  # [T, samples, params]
    fit_y = p
    fit_x = 1. / temperatures

    # Plot
    if plot_axes == 'normal_axes':
        # plot B agains T
        y = list(p)
        x = list(temperatures)
    elif plot_axes == 'Eyring_axes':
        # plot B against 1/T (should give 1 dof only!)
        y = list(fit_y)
        x = list(fit_x)

    # Data (violin)
    violinwidth = (x[1] - x[0]) / 3
    violins = axes[1, i].violinplot(y, x, showmeans=True, showmedians=False,
            showextrema=False, widths=violinwidth)

    for v in violins['bodies']:
        v.set_facecolor('C1')
        v.set_edgecolor('black')
        v.set_alpha(0.7)


if plot_g:

    # Fit
    p = mean_chains[:, :, 0]  # [T, samples, params]
    fit_y = p
    fit_x = 1. / temperatures

    # Plot
    y = list(p)
    if plot_axes == 'normal_axes':
        x = list(temperatures)
    elif plot_axes == 'Eyring_axes':
        x = list(fit_x)

    # Data (violin)
    violinwidth = (x[1] - x[0]) / 3
    violins = axes[2, 0].violinplot(y, x, showmeans=True, showmedians=False,
            showextrema=False, widths=violinwidth)

    for v in violins['bodies']:
        v.set_facecolor('C1')
        v.set_edgecolor('black')
        v.set_alpha(0.7)


#
# Kylie's parameters
#
if plot_kylie_param:
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


# Done
# axes[0, 0].legend()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(hspace=0.125)

plt.savefig('%s/parameters-%s.png' % (savedir, plot_axes), dpi=200,
        bbox_iches='tight')
plt.savefig('%s/parameters-%s.pdf' % (savedir, plot_axes), format='pdf',
        bbox_iches='tight')
plt.close('all')

