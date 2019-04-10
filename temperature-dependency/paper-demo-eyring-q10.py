#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.optimize import curve_fit

# Functions defined in Eyring plot; x=1/T
from temperature_models import eyringA, eyringB, simpleeyringB
from temperature_models import q10A, q10B

plot_axes = 'Eyring_axes'

savedir = './figs/paper'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

T_fit = np.linspace(22.0, 37.0, 20) + 273.15
T_plot = np.linspace(-10.0, 50.0, 100) + 273.15

# Parameters in Eyring plot axes, 'typical values' (p7, p8 mean)
pA_eyring_gen = [-1e4, 4e1]
pB_eyring_gen = [3e4, -7e1]

yA_fit_eyring_gen = eyringA(1. / T_fit, *pA_eyring_gen)
yB_fit_eyring_gen = eyringB(1. / T_fit, *pB_eyring_gen)
yA_eyring_gen = eyringA(1. / T_plot, *pA_eyring_gen)
yB_eyring_gen = eyringB(1. / T_plot, *pB_eyring_gen)

# Fit to pA_eyring_gen, pB_eyring_gen
pA_eyring = pA_eyring_gen  # by definition the same
pB_eyring = curve_fit(simpleeyringB, 1. / T_fit, yB_fit_eyring_gen)[0]
pA_q10 = curve_fit(q10A, 1. / T_fit, yA_fit_eyring_gen)[0]
pB_q10 = [yB_fit_eyring_gen[0]]  # Fix to the lowest temperature value

# Plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(4, 6))

axes[0].plot(1. / T_plot, yA_eyring_gen, ls='-', label='Generalised Eyring')
axes[0].plot(1. / T_plot, yA_eyring_gen, ls='--', label='Typical Eyring')
axes[0].plot(1. / T_plot, q10A(1. / T_plot, *pA_q10), ls=':',
        label=r'Q$_{10}$')

axes[0].set_ylabel(r'$\ln$(A / T)', fontsize=12)
# axes[0].set_xlabel(r'T$^{-1}$ [K$^{-1}$]', fontsize=12)
axes[0].legend()

axes[1].plot(1. / T_plot, yB_eyring_gen, ls='-', label='Generalised Eyring')
axes[1].plot(1. / T_plot, simpleeyringB(1. / T_plot, *pB_eyring), ls='--',
        label='Typical Eyring')
axes[1].plot(1. / T_plot, q10B(1. / T_plot, *pB_q10), ls=':',
        label=r'Q$_{10}$')

axes[1].set_ylabel(r'|B| [V$^{-1}$]', fontsize=12)
axes[1].set_xlabel(r'T$^{-1}$ [K$^{-1}$]', fontsize=12)

# Draw box
codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
for i in range(2):
    '''
    if i == 0:
        minimum = np.min(yA_fit_eyring_gen)
        maximum = np.max(yA_fit_eyring_gen)
    elif i == 1:
        minimum = np.min(yB_fit_eyring_gen)
        maximum = np.max(yB_fit_eyring_gen)
    amp = maximum - minimum
    minimum -= 0.1 * amp
    maximum += 0.1 * amp
    '''
    minimum, maximum = -100, 100
    #'''

    vertices = np.array([(1. / T_fit[-1], minimum),
                         (1. / T_fit[-1], maximum),
                         (1. / T_fit[0], maximum),
                         (1. / T_fit[0], minimum),
                         (0, 0)], float)
    pathpatch = PathPatch(Path(vertices, codes),
                          facecolor='#2ca02c',
                          edgecolor='#2ca02c',
                          alpha=0.125)
    plt.sca(axes[i])
    pyplot_axes = plt.gca()
    pyplot_axes.add_patch(pathpatch)

# Annotate
n = int(0.1 * len(T_plot))
axes[1].annotate('Any straight line',
        xy=(1. / T_plot[n], yB_eyring_gen[n]),
        xycoords='data',
        xytext=(0.00322, 43),
        textcoords='data',
        # fontsize=12,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
axes[1].annotate('any straight line\npassing through origin',
        xy=(1. / T_plot[4 * n], simpleeyringB(1. / T_plot[4 * n], *pB_eyring)),
        xycoords='data',
        xytext=(0.00344, 23),
        textcoords='data',
        # fontsize=12,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
axes[1].annotate('Horizontal line',
        xy=(1. / T_plot[-n], pB_q10[0]),
        xycoords='data',
        xytext=(0.0031, 38),
        textcoords='data',
        # fontsize=12,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

# Done
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
plt.subplots_adjust(hspace=0.05)

plt.savefig('%s/demo-eyring-q10-%s.png' % (savedir, plot_axes), dpi=200,
        bbox_iches='tight')
plt.savefig('%s/demo-eyring-q10-%s.pdf' % (savedir, plot_axes), format='pdf',
        bbox_iches='tight')
plt.close('all')
