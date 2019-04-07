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

import string
WELL_ID = [l+str(i).zfill(2)
           for l in string.ascii_uppercase[:16]
           for i in range(1,25)]

savedir = './figs/paper'

if not os.path.isdir(savedir):
    os.makedirs(savedir)

temperatures = [25.0]
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

# Estimated EK
expected_ek = prt2model['staircaseramp'].EK() * 1000  # V -> mV

print('Expected EK (mV): ' + str(expected_ek))

# Load data
t = np.loadtxt('../data-raw/herg25oc1-staircaseramp-times.csv', delimiter=',',
        skiprows=1)
cell = 'A01'
i = np.loadtxt('../data/herg25oc1-staircaseramp-%s.csv' % cell,
        delimiter=',', skiprows=1)
i_b = np.loadtxt('../data-raw/herg25oc1-staircaseramp-%s-before.csv' % cell,
        delimiter=',', skiprows=1)
i_a = np.loadtxt('../data-raw/herg25oc1-staircaseramp-%s-after.csv' % cell,
        delimiter=',', skiprows=1)
v = np.loadtxt('../protocol-time-series/protocol-staircaseramp.csv',
        delimiter=',', skiprows=1)[::2, 1]

# Plot
fig = plt.figure(figsize=(9, 6.5))
grid = plt.GridSpec(20, 13, hspace=0.0, wspace=0.0)
axes = []
axes.append(fig.add_subplot(grid[0:2, 0:13]))
axes.append(fig.add_subplot(grid[2:6, 0:13]))
axes.append(fig.add_subplot(grid[6:10, 0:13]))
axes.append(fig.add_subplot(grid[12:20, 0:6]))
axes.append(fig.add_subplot(grid[12:20, 7:13]))

# 1. Voltage
axes[0].plot(t, v, c='#7f7f7f')
axes[0].set_ylabel('Voltage\n[mV]', fontsize=13)

# 2. Raw current
axes[1].plot(t, i_b, c='C0', alpha=0.75, label='Control')
axes[1].plot(t, i_a, c='C1', alpha=0.75, label='E-4031')
axes[1].legend(fontsize=12)
axes[1].set_ylabel('Current\n[pA]', fontsize=13)

# 3. Current final
axes[2].plot(t, i, c='C2', alpha=0.75, label=r'$I_{Kr}$')
axes[2].legend(fontsize=12)
axes[2].set_ylabel('Current\n[pA]', fontsize=13)
axes[2].set_xlabel('Time [s]', fontsize=13)

# 4. Linear regression leak
t_i = 0.3  # s
t_f = t_i + 0.4  # s
win = np.logical_and(t > t_i, t < t_f)
win_fit = np.logical_and(t > t_i + 0.05, t < t_f - 0.05)
axes[3].plot(v[win], i_b[win], c='C0', alpha=0.75, label='Control')
axes[3].plot(v[win], i_a[win], c='C1', alpha=0.75, label='E-4031')

m, c, r, _, _ = stats.linregress(v[win_fit], i_b[win_fit])
axes[3].plot(v[win], v[win] * m + c, ls='--', lw=1.5, c='C4')
m, c, r, _, _ = stats.linregress(v[win_fit], i_a[win_fit])
axes[3].plot(v[win], v[win] * m + c, ls='--', lw=1.5, c='C4')
axes[3].legend(fontsize=12)
axes[3].set_xlabel('Voltage [mV]', fontsize=13)
axes[3].set_ylabel('Current [pA]', fontsize=13)

# Plot shadings
codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
inv = axes[2].transAxes.inverted()
top = inv.transform(axes[0].transAxes.transform((0, 1)))[1]
bottom = 0
left = inv.transform(axes[2].transData.transform((t[win][0], 0)))[0]
right = inv.transform(axes[2].transData.transform((t[win][-1], 0)))[0]
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
                        transform=axes[2].transAxes)
top2 = bottom
bottom2 = inv.transform(axes[3].transAxes.transform((0, 1)))[1]
left2 = inv.transform(axes[3].transAxes.transform((0, 0)))[0]
right2 = inv.transform(axes[3].transAxes.transform((1, 0)))[0]
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
                        transform=axes[2].transAxes)
plt.sca(axes[-1])
pyplot_axes = plt.gca()
pyplot_axes.add_patch(pathpatch)
pyplot_axes.add_patch(pathpatch2)

# 5. Linear regression EK
axes[4].axhline(0, ls='-', c='#7f7f7f')
t_i = 14.41  # s
t_f = t_i + 0.1  # s
win = np.logical_and(t > t_i, t < t_f)
axes[4].plot(v[win], i[win], c='C2', alpha=0.75, label=r'$I_{Kr}$')

p = np.poly1d(np.polyfit(v[win], i[win], 3))
r = []
for i in p.r:
    if np.min(v[win]) < i <= np.max(v[win]) \
                             and (np.isreal(i) or np.abs(i.imag) < 1e-8):
        r.append(i)
if len(r) == 1:
    ek =  r[0].real
elif len(r) > 1:
    ek = np.max(r).real
else:
    ek = np.inf
axes[4].plot(v[win], p(v[win]), ls='--', lw=1.5, c='C4')
axes[4].axvline(ek, ls='-', c='#d62728', label=r'Est. $E_K$')
axes[4].legend(fontsize=12)
axes[4].set_xlabel('Voltage [mV]', fontsize=13)

# Plot shadings
codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
inv = axes[2].transAxes.inverted()
top = inv.transform(axes[0].transAxes.transform((0, 1)))[1]
bottom = 0
left = inv.transform(axes[2].transData.transform((t[win][0], 0)))[0]
right = inv.transform(axes[2].transData.transform((t[win][-1], 0)))[0]
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
                        transform=axes[2].transAxes)
top2 = bottom
bottom2 = inv.transform(axes[4].transAxes.transform((0, 1)))[1]
left2 = inv.transform(axes[4].transAxes.transform((0, 0)))[0]
right2 = inv.transform(axes[4].transAxes.transform((1, 0)))[0]
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
                        transform=axes[2].transAxes)
plt.sca(axes[-1])
pyplot_axes = plt.gca()
pyplot_axes.add_patch(pathpatch)
pyplot_axes.add_patch(pathpatch2)

# Done
grid.tight_layout(fig, pad=0.6, rect=(0, 0, 1, 1))
grid.update(wspace=0.0, hspace=0.0)
plt.savefig('%s/ramps.pdf' % savedir, format='pdf', bbox_inch='tight')
plt.close()

print('Done')
