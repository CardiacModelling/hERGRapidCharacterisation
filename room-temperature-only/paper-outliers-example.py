#/usr/bin/env python
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

savedir = './figs/paper'

if not os.path.isdir(savedir):
    os.makedirs(savedir)

c11 = 'A01'
c12 = 'G13'
c13 = 'H18'
c21 = 'A04'
c22 = 'A15'
c23 = 'E05'
c31 = 'C18'
c32 = 'C22'
c33 = 'H08'

t = np.loadtxt('../data/herg25oc1-staircaseramp-times.csv', delimiter=',',
        skiprows=1)

i11 = np.loadtxt('../data/herg25oc1-staircaseramp-%s.csv' % c11, delimiter=',',
        skiprows=1)
i12 = np.loadtxt('../data/herg25oc1-staircaseramp-%s.csv' % c12, delimiter=',',
        skiprows=1)
i13 = np.loadtxt('../data/herg25oc1-staircaseramp-%s.csv' % c13, delimiter=',',
        skiprows=1)
i21 = np.loadtxt('../data/herg25oc1-staircaseramp-%s.csv' % c21, delimiter=',',
        skiprows=1)
i22 = np.loadtxt('../data/herg25oc1-staircaseramp-%s.csv' % c22, delimiter=',',
        skiprows=1)
i23 = np.loadtxt('../data/herg25oc1-staircaseramp-%s.csv' % c23, delimiter=',',
        skiprows=1)
i31 = np.loadtxt('../data/herg25oc1-staircaseramp-%s.csv' % c31, delimiter=',',
        skiprows=1)
i32 = np.loadtxt('../data/herg25oc1-staircaseramp-%s.csv' % c32, delimiter=',',
        skiprows=1)
i33 = np.loadtxt('../data/herg25oc1-staircaseramp-%s.csv' % c33, delimiter=',',
        skiprows=1)

v = np.loadtxt('../protocol-time-series/protocol-staircaseramp.csv',
        delimiter=',', skiprows=1)[::2, 1]

fig = plt.figure(figsize=(8, 8))
grid = plt.GridSpec(19, 1, hspace=0.0, wspace=0.0)
axes = []
axes.append(fig.add_subplot(grid[0:1, 0:1]))
axes.append(fig.add_subplot(grid[1:3, 0:1]))
axes.append(fig.add_subplot(grid[3:5, 0:1]))
axes.append(fig.add_subplot(grid[5:7, 0:1]))
axes.append(fig.add_subplot(grid[7:9, 0:1]))
axes.append(fig.add_subplot(grid[9:11, 0:1]))
axes.append(fig.add_subplot(grid[11:13, 0:1]))
axes.append(fig.add_subplot(grid[13:15, 0:1]))
axes.append(fig.add_subplot(grid[15:17, 0:1]))
axes.append(fig.add_subplot(grid[17:19, 0:1]))

axes[0].plot(t, v, c='#7f7f7f')
axes[0].set_ylabel('Voltage\n[mV]', fontsize=12)

axes[1].plot(t, i11, alpha=0.75, c='C2')
axes[2].plot(t, i12, alpha=0.75, c='C2')
axes[3].plot(t, i13, alpha=0.75, c='C2')

axes[4].plot(t, i21, alpha=0.75, c='C1')
axes[5].plot(t, i22, alpha=0.75, c='C1')
axes[6].plot(t, i23, alpha=0.75, c='C1')
axes[5].set_ylabel('Current\n[pA]', fontsize=12)

axes[7].plot(t, i31, alpha=0.75, c='C3')
axes[8].plot(t, i32, alpha=0.75, c='C3')
axes[9].plot(t, i33, alpha=0.75, c='C3')
axes[9].set_xlabel('Time [s]', fontsize=12)

axes[0].set_ylim((-140, 60))
for i in range(1, 10):
    axes[i].set_ylim((-750, 800))

grid.tight_layout(fig, pad=0.6, rect=(0, 0, 1, 1))
grid.update(wspace=0.0, hspace=0.0)
plt.savefig('%s/outliers.png' % savedir, bbox_inch='tight', dpi=300)
plt.savefig('%s/outliers.pdf' % savedir, format='pdf', bbox_inch='tight')
plt.close()

print('Done')
