#!/usr/bin/env python2
import sys
import os
import numpy as np
if '--show' not in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

import seaborn as sns

savedir = './figs'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

file_dir = './out'
file_list = [
        'herg25oc1',
        ]
temperatures = np.array([25.0])
temperatures += 273.15  # in K
fit_seed = '542811797'
withfcap = False

#
# Define some parameters and labels
#
labels = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
          r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$']

#
# Get new parameters
#
all_param = []
all_solutions = []
cells_label = []
all_labels = []
for i_temperature, (file_name, temperature) in enumerate(zip(file_list,
    temperatures)):
    files_dir = os.path.realpath(os.path.join(file_dir, file_name))
    searchwfcap = '-fcap' if withfcap else ''
    selectedfile = './manualv2selected-%s.txt' % (file_name)
    selectedwell = []
    with open(selectedfile, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                selectedwell.append(l.split()[0])

    for cell in selectedwell[:]:
        param_file = '%s/%s-staircaseramp-%s-solution%s-%s.txt' % (files_dir,
                file_name, cell, searchwfcap, fit_seed)

        obtained_parameters = np.loadtxt(param_file)
        all_param.append(obtained_parameters)
        all_solutions.extend(obtained_parameters)
        cells_label.extend([r'Automated patch'] * len(labels))
        all_labels.extend(labels)
mean_param = np.mean(all_param, axis=0)

#
# Kylie's parameters
#
path_to_solutions = './kylie-room-temperature'
cells_kylie = ['C' + str(i) for i in range(1, 9)]
all_param = []
for cell in cells_kylie:
    last_solution = glob.glob(path_to_solutions+'/*%s*'%cell)[0]
    obtained_parameters = np.loadtxt(last_solution)
    # Change conductance unit nS->pS (new parameter use V, but here mV)
    obtained_parameters[0] = obtained_parameters[0] * 1e3
    all_param.append(obtained_parameters)
    all_solutions.extend(obtained_parameters)
    cells_label.extend([r'Beattie et al. 2018 at 22$^o$C'] * len(labels))
    all_labels.extend(labels)
kylie_mean_param = np.mean(all_param, axis=0)

#
# seaborn plot
#
plt.figure(figsize=(12, 4))
sns.swarmplot(x=all_labels, y=np.log10(all_solutions), hue=cells_label,
              size=2.25, zorder=1)
plt.plot(np.log10(mean_param), c='b', ls='', marker='s',
         label=r'Automated patch mean', zorder=2)
plt.plot(np.log10(kylie_mean_param), c='r', ls='', marker='s',
         label=r'Beattie et al. 2018 mean', zorder=2)

legend = plt.legend(fontsize=13)
# Change the marker size manually in legend
legend.legendHandles[2]._sizes = [10]
legend.legendHandles[3]._sizes = [10]

plt.ylabel(r'$\log_{10}($Value$)$', fontsize=16)
# plt.xlabel('Parameter', fontsize=32)
plt.xticks(fontsize=16)
plt.yticks(fontsize=14)
plt.savefig('./figs/paper/swarmplot.png', bbox_inches='tight', dpi=300)
plt.savefig('./figs/paper/swarmplot.pdf', format='pdf', bbox_inches='tight')
