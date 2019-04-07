#!/usr/bin/env python2

# Plot conductance histogram and conductance/capacitance histogram

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import string

WELL_ID = [l+str(i).zfill(2) for l in string.ascii_uppercase[:16] for i in range(1,25)]

file_dir = './out'
qc_dir = '../qc'
file_name = 'herg25oc1'
fit_seed = '542811797'

selectedwell = []
with open('manualv2selected-%s.txt' % file_name, 'r') as f:
    for l in f:
        if not l.startswith('#'):
            selectedwell.append(l.split()[0])

Cm_b = np.loadtxt('%s/%s-staircaseramp-Cm_before.txt' % (qc_dir, file_name))
Cm_a = np.loadtxt('%s/%s-staircaseramp-Cm_after.txt' % (qc_dir, file_name))

g = []
gCm_b = []
gCm_a = []
Cm_sa = []
Cm_sb = []
for c in selectedwell:
    # Cell index
    idx = WELL_ID.index(c)

    # Fitted parameters
    param_file = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
            (file_dir, file_name, file_name, c, fit_seed)
    p = np.loadtxt(param_file)  # p[0] = conductance
    g.append(p[0])  # pS
    gCm_b.append(g[-1] / (Cm_b[idx] * 1e12))  # pS/pF
    gCm_a.append(g[-1] / (Cm_a[idx] * 1e12))  # pS/pF
    Cm_sb.append(Cm_b[idx] * 1e12)
    Cm_sa.append(Cm_a[idx] * 1e12)

# Plot
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.hist(g, bins=40, label='g [pS]')
plt.ylabel('Frequency (N=%s)' % len(g))
plt.legend()
plt.subplot(2, 1, 2)
plt.hist(gCm_b, bins=40, label='g/Cm (before E4031) [pS/pF]')
plt.hist(gCm_a, bins=40, label='g/Cm (after E4031) [pS/pF]')
plt.ylabel('Frequency (N=%s)' % len(gCm_b))
plt.xlabel('Value')
plt.legend()
plt.savefig('figs/conductance-hists.png')
plt.close()

fig = plt.figure()
plt.scatter(g, gCm_b, marker='o', label='Cm from before E4031')
plt.scatter(g, gCm_a, marker='x', label='Cm from after E4031')
plt.xlabel('g [pS]')
plt.ylabel('g/Cm [pS/pF]')
plt.legend()
plt.savefig('figs/conductance-scatter.png')
plt.close()

fig = plt.figure()
plt.scatter(g, Cm_sb, marker='o', label='Cm from before E4031')
plt.scatter(g, Cm_sa, marker='x', label='Cm from after E4031')
plt.xlabel('g [pS]')
plt.ylabel('Cm [pF]')
plt.legend()
plt.savefig('figs/conductance-capacitance-scatter.png')
plt.close()
