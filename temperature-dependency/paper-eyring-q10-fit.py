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

# Functions defined in Eyring plot; x=1/T
from temperature_models import eyringA, eyringB, eyringG
from temperature_models import q10A, q10B, q10G


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

# Colours for fan chart
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
        '#adc1d0',
        '#91abbc',
        '#85a0b1',
        '#6b8fa9',
        '#62869f',
        '#587c96',
        '#477390',
        '#3f6c88',
    ]


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

eyring_param_mean = np.zeros((9, 2))
eyring_param_std = np.zeros((9, 2))
q10_param_mean = np.zeros((9, 2))
q10_param_std = np.zeros((9, 2))

# param A
for i, p_i in enumerate(param_A):
    # i for axes index
    # obtained_parameters use p_i to get parameter i

    # Fit
    p = mean_chains[:, :, p_i]  # [T, samples, params]
    fit_y = np.log(p.T / temperatures).T
    fit_x = 1. / temperatures
    fit_y_mean = np.mean(fit_y, axis=1)
    fit_y_std = np.std(fit_y, axis=1)

    # m_mean, c_mean, r_mean, _, _ = stats.linregress(fit_x, fit_y_mean)
    # m_std, c_std, r_std, _, _ = stats.linregress(fit_x, fit_y_std)
    eyring_mean, _ = curve_fit(eyringA, fit_x, fit_y_mean)
    eyring_std, _ = curve_fit(eyringA, fit_x, fit_y_std)
    q10_mean, _ = curve_fit(q10A, fit_x, fit_y_mean)
    q10_std, _ = curve_fit(q10A, fit_x, fit_y_std)

    # Save fit
    eyring_param_mean[p_i, :] = np.copy(eyring_mean)
    eyring_param_std[p_i, :] = np.copy(eyring_std)
    q10_param_mean[p_i, :] = np.copy(q10_mean)
    q10_param_std[p_i, :] = np.copy(q10_std)

    # Plot
    if plot_axes == 'normal_axes':
        # plot A agains T
        y = list(p)
        x = list(temperatures)
    elif plot_axes == 'Eyring_axes':
        # plot ln(A/2.0836E10*T) against 1/T
        y = list(fit_y)
        x = list(fit_x)

    # Fitted (fan chart)
    for ii_std, i_std in enumerate(range(1, 4)[::-1]):

        if plot_axes == 'normal_axes':
            percentile_top = norm.cdf(i_std)
            percentile_bot = norm.cdf(-1 * i_std)
            eyring_top = []
            eyring_bot = []
            q10_top = []
            q10_bot = []
            for i_T, T in enumerate(x):
                parameyring = [(eyringA(1. / T, *eyring_mean) + np.log(T)),
                               eyringA(1. / T, *eyring_std)]
                eyring_top.append(
                        eyring_transform_ppf(parameyring, p_i, percentile_top))
                eyring_bot.append(
                        eyring_transform_ppf(parameyring, p_i, percentile_bot))
                paramq10 = [(q10A(1. / T, *q10_mean) + np.log(T)),
                            q10A(1. / T, *q10_std)]
                q10_top.append(
                        eyring_transform_ppf(paramq10, p_i, percentile_top))
                q10_bot.append(
                        eyring_transform_ppf(paramq10, p_i, percentile_bot))

        elif plot_axes == 'Eyring_axes':
            eyring_top = eyringA(fit_x, *eyring_mean) + \
                    i_std * eyringA(fit_x, *eyring_std)
            eyring_bot = eyringA(fit_x, *eyring_mean) - \
                    i_std * eyringA(fit_x, *eyring_std)
            q10_top = q10A(fit_x, *q10_mean) + i_std * q10A(fit_x, *q10_std)
            q10_bot = q10A(fit_x, *q10_mean) - i_std * q10A(fit_x, *q10_std)

        alpha = 0.7
        color = fan_green[ii_std]
        axes[0, i].fill_between(x, eyring_top, eyring_bot, color=color,
                alpha=alpha, linewidth=0, zorder=-1,
                label='__nolegend__' if 2 - ii_std else 'Eyring')

        color = fan_red[ii_std]
        axes[0, i].fill_between(x, q10_top, q10_bot, color=color,
                alpha=alpha, linewidth=0, zorder=-2,
                label='__nolegend__' if 2 - ii_std else r'$Q_{10}$')

    # Fitted (mean)
    if plot_axes == 'normal_axes':
        meaneyring = []
        meanq10 = []
        for i_T, T in enumerate(x):
            parameyring = [(eyringA(1. / T, *eyring_mean) + np.log(T)),
                           eyringA(1. / T, *eyring_std)]
            meaneyring.append(eyring_transform_mean(parameyring, p_i))
            paramq10 = [(q10A(1. / T, *q10_mean) + np.log(T)),
                        q10A(1. / T, *q10_std)]
            meanq10.append(eyring_transform_mean(paramq10, p_i))
    elif plot_axes == 'Eyring_axes':
        meaneyring = eyringA(fit_x, *eyring_mean)
        meanq10 = q10A(fit_x, *q10_mean)
    axes[0, i].plot(x, meaneyring, c=fan_green[-1], lw=2, ls='--')
    axes[0, i].plot(x, meanq10, c=fan_red[-1], lw=2, ls=':')

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
    fit_y_mean = np.mean(fit_y, axis=1)
    fit_y_std = np.std(fit_y, axis=1)

    # m_mean, c_mean, r_mean, _, _ = stats.linregress(fit_x, fit_y_mean)
    # m_std, c_std, r_std, _, _ = stats.linregress(fit_x, fit_y_std)
    eyring_mean, _ = curve_fit(eyringB, fit_x, fit_y_mean)
    eyring_std, _ = curve_fit(eyringB, fit_x, fit_y_std)
    # q10_mean, _ = curve_fit(q10B, fit_x, fit_y_mean)
    # q10_std, _ = curve_fit(q10B, fit_x, fit_y_std)
    q10_mean = [fit_y_mean[0]]  # only est. from room temperature
    q10_std = [fit_y_std[0]]

    # Save fit
    eyring_param_mean[p_i, :] = np.copy(eyring_mean)
    eyring_param_std[p_i, :] = np.copy(eyring_std)
    q10_param_mean[p_i, :] = np.append(q10_mean, np.NaN)
    q10_param_std[p_i, :] = np.append(q10_std, np.NaN)

    # Plot
    if plot_axes == 'normal_axes':
        # plot B agains T
        y = list(p)
        x = list(temperatures)
    elif plot_axes == 'Eyring_axes':
        # plot B against 1/T (should give 1 dof only!)
        y = list(fit_y)
        x = list(fit_x)

    # Fitted (fan chart)
    for ii_std, i_std in enumerate(range(1, 4)[::-1]):

        if plot_axes == 'normal_axes':
            percentile_top = norm.cdf(i_std)
            percentile_bot = norm.cdf(-1 * i_std)
            eyring_top = []
            eyring_bot = []
            q10_top = []
            q10_bot = []
            for i_T, T in enumerate(x):
                parameyring = [eyringB(1. / T, *eyring_mean),
                               eyringB(1. / T, *eyring_std)]
                eyring_top.append(
                        eyring_transform_ppf(parameyring, p_i, percentile_top))
                eyring_bot.append(
                        eyring_transform_ppf(parameyring, p_i, percentile_bot))
                paramq10 = [q10B(1. / T, *q10_mean),
                            q10B(1. / T, *q10_std)]
                q10_top.append(
                        eyring_transform_ppf(paramq10, p_i, percentile_top))
                q10_bot.append(
                        eyring_transform_ppf(paramq10, p_i, percentile_bot))

        elif plot_axes == 'Eyring_axes':
            eyring_top = eyringB(fit_x, *eyring_mean) + \
                    i_std * eyringB(fit_x, *eyring_std)
            eyring_bot = eyringB(fit_x, *eyring_mean) - \
                    i_std * eyringB(fit_x, *eyring_std)
            q10_top = q10B(fit_x, *q10_mean) + i_std * q10B(fit_x, *q10_std)
            q10_bot = q10B(fit_x, *q10_mean) - i_std * q10B(fit_x, *q10_std)

        alpha = 0.7
        color = fan_green[ii_std]
        axes[1, i].fill_between(x, eyring_top, eyring_bot, color=color,
                alpha=alpha, linewidth=0, zorder=-1,
                label='__nolegend__' if 2 - ii_std else 'Eyring')

        color = fan_red[ii_std]
        axes[1, i].fill_between(x, q10_top, q10_bot, color=color,
                alpha=alpha, linewidth=0, zorder=-2,
                label='__nolegend__' if 2 - ii_std else r'$Q_{10}$')

    # Fitted (mean)
    if plot_axes == 'normal_axes':
        meaneyring = []
        meanq10 = []
        for i_T, T in enumerate(x):
            parameyring = [eyringB(1. / T, *eyring_mean),
                           eyringB(1. / T, *eyring_std)]
            meaneyring.append(eyring_transform_mean(parameyring, p_i))
            paramq10 = [q10B(1. / T, *q10_mean),
                        q10B(1. / T, *q10_std)]
            meanq10.append(eyring_transform_mean(paramq10, p_i))
    elif plot_axes == 'Eyring_axes':
        meaneyring = eyringB(fit_x, *eyring_mean)
        meanq10 = q10B(fit_x, *q10_mean)
    axes[1, i].plot(x, meaneyring, c=fan_green[-1], lw=2, ls='--')
    axes[1, i].plot(x, meanq10, c=fan_red[-1], lw=2, ls=':')

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
    fit_y_mean = np.mean(fit_y, axis=1)
    fit_y_std = np.std(fit_y, axis=1)

    eyring_mean, _ = curve_fit(eyringG, fit_x, fit_y_mean)
    eyring_std, _ = curve_fit(eyringG, fit_x, fit_y_std)
    q10_mean, _ = curve_fit(q10G, fit_x, fit_y_mean)
    q10_std, _ = curve_fit(q10G, fit_x, fit_y_std)

    # Save fit
    eyring_param_mean[0, :] = np.append(eyring_mean, np.NaN)
    eyring_param_std[0, :] = np.append(eyring_std, np.NaN)
    q10_param_mean[0, :] = np.append(q10_mean, np.NaN)
    q10_param_std[0, :] = np.append(q10_std, np.NaN)

    # Plot
    y = list(p)
    if plot_axes == 'normal_axes':
        x = list(temperatures)
    elif plot_axes == 'Eyring_axes':
        x = list(fit_x)

    # Fitted (fan chart)
    for ii_std, i_std in enumerate(range(1, 4)[::-1]):

        if plot_axes == 'normal_axes':
            percentile_top = norm.cdf(i_std)
            percentile_bot = norm.cdf(-1 * i_std)
            eyring_top = []
            eyring_bot = []
            q10_top = []
            q10_bot = []
            for i_T, T in enumerate(x):
                parameyring = [eyringG(1. / T, *eyring_mean),
                               eyringG(1. / T, *eyring_std)]
                eyring_top.append(
                        eyring_transform_ppf(parameyring, p_i, percentile_top))
                eyring_bot.append(
                        eyring_transform_ppf(parameyring, p_i, percentile_bot))
                paramq10 = [q10G(1. / T, *q10_mean),
                            q10G(1. / T, *q10_std)]
                q10_top.append(
                        eyring_transform_ppf(paramq10, p_i, percentile_top))
                q10_bot.append(
                        eyring_transform_ppf(paramq10, p_i, percentile_bot))

        elif plot_axes == 'Eyring_axes':
            eyring_top = eyringG(fit_x, *eyring_mean) + \
                    i_std * eyringG(fit_x, *eyring_std)
            eyring_bot = eyringG(fit_x, *eyring_mean) - \
                    i_std * eyringG(fit_x, *eyring_std)
            q10_top = q10G(fit_x, *q10_mean) + i_std * q10G(fit_x, *q10_std)
            q10_bot = q10G(fit_x, *q10_mean) - i_std * q10G(fit_x, *q10_std)

        alpha = 0.7
        color = fan_green[ii_std]
        axes[2, 0].fill_between(x, eyring_top, eyring_bot, color=color,
                alpha=alpha, linewidth=0, zorder=-1,
                label='__nolegend__' if 2 - ii_std else 'Eyring')

        color = fan_red[ii_std]
        axes[2, 0].fill_between(x, q10_top, q10_bot, color=color,
                alpha=alpha, linewidth=0, zorder=-2,
                label='__nolegend__' if 2 - ii_std else r'$Q_{10}$')

    # Fitted (mean)
    if plot_axes == 'normal_axes':
        meaneyring = []
        meanq10 = []
        for i_T, T in enumerate(x):
            parameyring = [eyringG(1. / T, *eyring_mean),
                           eyringG(1. / T, *eyring_std)]
            meaneyring.append(eyring_transform_mean(parameyring, p_i))
            paramq10 = [q10G(1. / T, *q10_mean),
                        q10G(1. / T, *q10_std)]
            meanq10.append(eyring_transform_mean(paramq10, p_i))
    elif plot_axes == 'Eyring_axes':
        meaneyring = eyringG(fit_x, *eyring_mean)
        meanq10 = q10G(fit_x, *q10_mean)
    axes[2, 0].plot(x, meaneyring, c=fan_green[-1], lw=2, ls='--')
    axes[2, 0].plot(x, meanq10, c=fan_red[-1], lw=2, ls=':')

    # Data (violin)
    violinwidth = (x[1] - x[0]) / 3
    violins = axes[2, 0].violinplot(y, x, showmeans=True, showmedians=False,
            showextrema=False, widths=violinwidth)

    for v in violins['bodies']:
        v.set_facecolor('C1')
        v.set_edgecolor('black')
        v.set_alpha(0.7)

# Save fitted parameters
np.savetxt('%s/eyring-mean.txt' % savedir_data, eyring_param_mean)
np.savetxt('%s/eyring-std.txt' % savedir_data, eyring_param_std)
np.savetxt('%s/q10-mean.txt' % savedir_data, q10_param_mean)
np.savetxt('%s/q10-std.txt' % savedir_data, q10_param_std)


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


#
# What if Q10 ~ 2-3?!
#
# param A
for i, p_i in enumerate(param_A):
    # i for axes index
    # obtained_parameters use p_i to get parameter i

    # Parameters
    p = mean_chains[:, :, p_i]  # [T, samples, params]
    fit_y = p
    fit_x = 1. / temperatures
    fit_y_mean = np.mean(fit_y, axis=1)

    q10_tref = temperatures[0]
    q10_A = fit_y_mean[0]
    q10_top = 3
    q10_bot = 2
    a_top = np.log(q10_top) / 10.
    a_bot = np.log(q10_bot) / 10.
    c_top = np.log(q10_A) - np.log(q10_top) * q10_tref / 10.
    c_bot = np.log(q10_A) - np.log(q10_bot) * q10_tref / 10.

    # Plot
    if plot_axes == 'normal_axes':
        x = list(temperatures)
        top = q10A(fit_x, a_top, c_top)
        bot = q10A(fit_x, a_bot, c_bot)
        tmp = np.ones(9)
        q10_top = []
        q10_bot = []
        for i_T, T in enumerate(x):
            tmp[p_i] = top[i_T]
            q10_top.append(eyring_transform_to_model_param(tmp, T)[p_i])
            tmp[p_i] = bot[i_T]
            q10_bot.append(eyring_transform_to_model_param(tmp, T)[p_i])
        del(tmp)
    elif plot_axes == 'Eyring_axes':
        x = list(fit_x)
        q10_top = q10A(fit_x, a_top, c_top)
        q10_bot = q10A(fit_x, a_bot, c_bot)
    axes[0, i].fill_between(x, q10_top, q10_bot, color='#7f7f7f',
            alpha=0.25, linewidth=0, zorder=-3, label=r'$Q_{10}\in[2,3]$')
    axes[0, i].plot(x, q10_top, c='#7f7f7f', ls='--', alpha=0.5, zorder=-3)
    axes[0, i].plot(x, q10_bot, c='#7f7f7f', ls='--', alpha=0.5, zorder=-3)


#
# Make a table to print out all params!
#
if plot_g:
    eyring_rowlabel = [r'$k_1$', r'$k_2$', r'$k_3$', r'$k_4$']
    eyring_collabel = [r'$\Delta S$ (JK$^{-1}$mol$^{-1}$)',
                       r'$\Delta H$ (Jmol$^{-1}$)',
                       r'$z_e$',
                       r'$D$ (V$^{-1}$)']
    # Eyring slope A = - b = - \Delta H / k_B
    #     ---> \Delta H = - (Eyring slope A) * k_B
    # Eyring interception A = \ln(a) = \ln((k_B / h) \exp(\Delta S / k_B))
    #     ---> \Delta S = k_B * ( (Eyring interception A) - \ln(k_B / h) )
    # Eyring slope B = c = (z_e \cdot e) / k_B
    #     ---> z_e = (Eyring slope B) * k_B / e
    # Eyring interception B = d = D
    #     ---> D = (Eyring interception B)
    q10_rowlabel = [r'$k_1$', r'$k_2$', r'$k_3$', r'$k_4$']
    q10_collabel = [r'$Q_{10}$', r'$\alpha$ (s$^{-1}$)',
                    r'$\beta$ (V$^{-1}$)'] #, r'$T_{ref}$ (K)']
    # Q10 a = \ln(Q10) / 10oC
    #     ---> Q10 = \exp( 10oC * (Q10 a) )
    # Q10 c = \ln(A) - \ln(Q10) * T_{ref} / 10oC
    #     ---> A = \exp( (Q10 c) + (Q10 a) * T_{ref} )
    # Q10 b = B
    #     ---> B = (Q10 b)

    k_B = 1.38064852E-23  # J/K
    R = 8.3144598  # J/K/mol
    h = 6.62607015E-34  # Js
    e = 1.6021766208E-19  # C
    eyring_rows = []
    q10_rows = []
    for pa, pb in zip(param_A, param_B):
        ey_m_A, ey_c_A = eyring_param_mean[pa]
        DH = -1 * ey_m_A * R  # * k_B
        DS = (ey_c_A - np.log(k_B / h)) * R  # * k_B
        ey_m_B, ey_c_B = eyring_param_mean[pb]
        ze = ey_m_B * k_B / e
        D = ey_c_B
        eyring_rows.append([float('%.6g' % DS),
                            float('%.6g' % DH),
                            float('%.6g' % ze),
                            float('%.6g' % D)])

        T_ref = 273.15 + 25
        q10_a, q10_c = q10_param_mean[pa]
        Q10 = np.exp(10.0 * q10_a)
        A = np.exp(q10_c + q10_a * T_ref)
        q10_b, _ = q10_param_mean[pb]
        B = q10_b
        q10_rows.append([float('%.6g' % Q10),
                         float('%.6g' % A),
                         float('%.6g' % B),])
                         # float('%.6g' % T_ref)])

    # The table
    eyring_table = axes[2, 0].table(cellText=eyring_rows,
                                    rowLabels=eyring_rowlabel,
                                    colLabels=eyring_collabel,
                                    clip_on=False,
                                    bbox=(1.2, -0.15, 1.7, 0.85),
                                    transform=axes[2, 0].transAxes)
    eyring_table.auto_set_font_size(False)
    eyring_table.set_fontsize(12)

    q10_table = axes[2, 0].table(cellText=q10_rows,
                                 rowLabels=q10_rowlabel,
                                 colLabels=q10_collabel,
                                 clip_on=False,
                                 bbox=(3.2, -0.15, 1.2, 0.85),
                                 transform=axes[2, 0].transAxes)
    q10_table.auto_set_font_size(False)
    q10_table.set_fontsize(12)


# Done
axes[0, 0].legend()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.subplots_adjust(hspace=0.175)

plt.savefig('%s/eyring-q10-fit-%s.png' % (savedir, plot_axes), dpi=200,
        bbox_iches='tight')
plt.savefig('%s/eyring-q10-fit-%s.pdf' % (savedir, plot_axes), format='pdf',
        bbox_iches='tight')
plt.close('all')

