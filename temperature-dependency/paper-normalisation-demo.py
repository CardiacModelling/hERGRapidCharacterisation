#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib/')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import model_ikr as m
import protocols

savedir = './figs/paper'

if not os.path.isdir(savedir):
    os.makedirs(savedir)

cell = 'B11'
filename = 'herg37oc3'
temperatures = [37.0]
temperatures = np.asarray(temperatures) + 273.15  # K
temperature = temperatures[0]

# Protocol info
protocol_funcs = {
    'staircaseramp': 'protocol-staircaseramp.csv',
}
protocol_dir = '../protocol-time-series'
protocol_list = [
    'staircaseramp',
]

# Model
prt2model = {}
for prt in protocol_list:

    protocol_def = protocol_funcs[prt]
    if type(protocol_def) is str:
        protocol_def = '%s/%s' % (protocol_dir, protocol_def)

    prt2model[prt] = m.ModelWithVoltageOffset(
                        '../mmt-model-files/kylie-2017-IKr.mmt',
                        protocol_def=protocol_def,
                        temperature=temperature,  # K
                        transform=None,
                        useFilterCap=False)  # ignore capacitive spike

# Load data
t = np.loadtxt('../data/%s-staircaseramp-times.csv' % filename, delimiter=',',
        skiprows=1)
i = np.loadtxt('../data/%s-staircaseramp-%s.csv' % (filename, cell),
        delimiter=',', skiprows=1)
v = np.loadtxt('../protocol-time-series/protocol-staircaseramp.csv',
        delimiter=',', skiprows=1)[::2, 1]

# Plot
fig = plt.figure(figsize=(9, 5))
grid = plt.GridSpec(16, 13, hspace=0.0, wspace=0.0)
axes = []
axes.append(fig.add_subplot(grid[0:2, 0:13]))
axes.append(fig.add_subplot(grid[2:6, 0:13]))
axes.append(fig.add_subplot(grid[8:16, 0:8]))
# axes.append(fig.add_subplot(grid[8:16, 8:13]))

# 1. Voltage
axes[0].plot(t, v, c='#7f7f7f')
axes[0].set_ylabel('Voltage\n[mV]', fontsize=13)

# 3. Current
axes[1].plot(t, i, c='C0', alpha=0.75, label=r'Data')
# axes[1].legend(fontsize=12)
axes[1].set_ylabel('Current\n[pA]', fontsize=13)
axes[1].set_xlabel('Time [s]', fontsize=13)

# 4. Normalisation

# Same function in protocol.est_g_staircase, but output more things for plot
def est_g_staircase(current, times, p0, t_start=1.9, t_end=1.95,
                    t_trim=0.0125, t_fit_until=0.03):
    # use 2-parameters exponential fit to the tail
    from scipy.optimize import curve_fit
    def exp_func(t, a, b):
        # do a "proper exponential" decay fit
        # i.e. shift the t to t' where t' has zero at the start of the 
        # voltage step
        return -a * np.exp(-(t - x[0]) / b)
    time_window = np.where(np.logical_and(times > t_start, times <= t_end))[0]
    i_trim = np.argmin(np.abs(times - (t_start + t_trim))) - time_window[0]
    i_fit_until = np.argmin(np.abs(times - (t_start + t_fit_until))) \
                  - time_window[0]
    # trim off the first i_trim (100ms) in case it is still shooting up...
    x = times[time_window[0] + i_trim:time_window[0] + i_fit_until]
    y = current[time_window[0] + i_trim:
                time_window[0] + i_fit_until]
    try:
        popt, pcov = curve_fit(exp_func, x, y, p0=p0)
        fitted = exp_func(times[time_window[0]:
                                time_window[0] + i_fit_until], *popt)
        g_est = np.max(np.abs(fitted))
    except:
        raise Exception('Maybe not here!')
    return g_est, [exp_func, popt]

t_i = 1.85  # s
t_f = 2.0  # s
t_step = 1.9  # s
win = np.logical_and(t > t_i, t < t_f)
win_step = np.logical_and(t > t_step, t < t_f)
norm_value, plot_func = est_g_staircase(i, t, p0=[800, 0.025])
axes[2].plot(t[win], i[win], c='C0', alpha=0.75, label='Data')
i_fitted = plot_func[0](t[win_step], *plot_func[1])
axes[2].plot(t[win_step], i_fitted, c='C1', ls='--',
        label='Fit')
axes[2].plot(t_step, -1 * norm_value, 'kx', label='Est. value')
axes[2].axvline(t_step, color='#2ca02c', ls='--')  # where step at

axes[2].legend(fontsize=11)
axes[2].set_xlabel('Time [s]', fontsize=13)
axes[2].set_ylabel('Current [pA]', fontsize=13)
axes[2].set_xlim(t[win][0], t[win][-1])

# Plot shadings
codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
inv = axes[1].transAxes.inverted()
top = inv.transform(axes[0].transAxes.transform((0, 1)))[1]
bottom = 0
left = inv.transform(axes[1].transData.transform((t[win][0], 0)))[0]
right = inv.transform(axes[1].transData.transform((t[win][-1], 0)))[0]
vertices = np.array([(left, bottom),
                        (left, top),
                        (right, top),
                        (right, bottom),
                        (0, 0)], float)
pathpatch = PathPatch(Path(vertices, codes),
                        facecolor='#7f7f7f',
                        edgecolor='#7f7f7f',
                        alpha=0.3,
                        clip_on=False,
                        transform=axes[1].transAxes)
top2 = bottom
bottom2 = inv.transform(axes[2].transAxes.transform((0, 1)))[1]
left2 = inv.transform(axes[2].transAxes.transform((0, 0)))[0]
right2 = inv.transform(axes[2].transAxes.transform((1, 0)))[0]
vertices2 = np.array([(left2, bottom2),
                        (left, top2),
                        (right, top2),
                        (right2, bottom2),
                        (0, 0)], float)
pathpatch2 = PathPatch(Path(vertices2, codes),
                        facecolor='#7f7f7f',
                        edgecolor='#7f7f7f',
                        alpha=0.2,
                        clip_on=False,
                        transform=axes[1].transAxes)
plt.sca(axes[-1])
pyplot_axes = plt.gca()
pyplot_axes.add_patch(pathpatch)
pyplot_axes.add_patch(pathpatch2)

# Done
grid.tight_layout(fig, pad=0.6, rect=(0, 0, 1, 1))
grid.update(wspace=0.0, hspace=0.0)
plt.savefig('%s/normalisation-demo.png' % savedir, bbox_inch='tight')
plt.savefig('%s/normalisation-demo.pdf' % savedir, format='pdf', bbox_inch='tight')
plt.close()

print('Done')
