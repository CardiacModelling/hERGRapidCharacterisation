#!/usr/bin/env python2
#
# coding: utf-8
#
# Plot covariance/correlation matrix with synthetic voltage error parameters
# on top
#

from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
if not '--show' in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import glob

import plot_hbm_func as plot_func

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

# About files
file_list = ['herg25oc1']
temperatures = [25.0]
fit_seed = '542811797'

file_dir = './out/herg25oc1-fakedata-voltageoffset'
n_fakedata = 124
file_name = file_list[0]
temperature = temperatures[0] + 273.15  # K

saveas = 'figs/voltageoffset/'
n_non_model_param = 1
variable_names = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
                  r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']

if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))

#
# Load
#

# Voltage error fits
vo_parameters = []
vo_transformed_parameters = []
for i in range(n_fakedata):
    param_file = '%s/%s-staircaseramp-sim-%s-solution-%s.txt' % \
            (file_dir, file_name, i, fit_seed)
    vo_parameters.append(np.loadtxt(param_file))
    vo_transformed_parameters.append(
            transform_from_model_param(vo_parameters[-1])
            )
vo_parameters = np.asarray(vo_parameters)
vo_transformed_parameters = np.asarray(vo_transformed_parameters)

# Experimental data
param_exp = []
path_to_exp = './out/herg25oc1-mcmcmean'
files_exp = glob.glob(path_to_exp + '/*.txt')
for file_exp in files_exp:
    p = np.loadtxt(file_exp)
    param_exp.append(p)
param_exp = np.array(param_exp)
exp_transform_parameters = transform_from_model_param(param_exp.T).T
nexp, n_parameters = exp_transform_parameters.shape

# Mean/Covariance
simple_chain_final = np.loadtxt(
        './out-mcmc/herg25oc1-pseudohbm-lognorm-mean.txt')
with open('./out-mcmc/herg25oc1-pseudohbm-lognorm-cov.pkl', 'rb') as f:
    simple_cov_final = pickle.load(f)

# Covariance matrice
simple_cor_final = np.zeros(simple_cov_final.shape)
for i, s in enumerate(simple_cov_final):
    D = np.sqrt(np.diag(s))
    c = s / D / D[:, None]
    simple_cor_final[i, :, :] = c[:, :]


#
# Plot
#

# individual exp: trace
n_percentiles = 99.9

# this is good ref for real data
rho = covariance = np.zeros((9,9))
ref_hyper = None
ref_variable_corr = None

print('Plotting cov...')
fig, axes = plot_func.plot_correlation_and_variable_covariance(
        simple_chain_final[::200],
        simple_cov_final[::200],
        simple_cor_final, corr=True,
        # ref_parameters=[simple_transform_mean, simple_transform_cov],
        )
axes = plot_func.change_labels_correlation_and_variable_covariance(
        axes,
        variable_names
        )

# Add voltage error parameters and individual experiment parameters
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        if i > j:
            axes[i, j].scatter(exp_transform_parameters[:, j],
                    exp_transform_parameters[:, i], marker='o', alpha=0.5,
                    c='#7f7f7f')
            axes[i, j].scatter(vo_transformed_parameters[:, j],
                    vo_transformed_parameters[:, i], marker='o',
                    c='#d62728')
        elif i == j:
            axes[i, j].hist(exp_transform_parameters[:, j], bins=20,
                    density=True, color='#7f7f7f', alpha=0.5, zorder=-10)

# Done
if '--show' in sys.argv:
    plt.show()
else:
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0,
                     rect=(0.001, 0.001, 0.97, 0.965))
    plt.savefig('%scov-plot.png' % saveas, bbox_iches='tight', pad_inches=0)
    # plt.savefig('%scov-plot.pdf'%saveas, format='pdf', bbox_inches='tight')


# Add boxes for Michael
for i in range(0, n_parameters):
    plot_func.addbox(axes, (i, 0), color='#d9d9d9', alpha=0.75)
for j in range(1, n_parameters):
    plot_func.addbox(axes, (0, j), color='#d9d9d9', alpha=0.75)
for i in range(1, 5):
    for j in range(1, 5):
        plot_func.addbox(axes, (i, j), color='#fdb462', alpha=0.35)
for i in range(5, n_parameters):
    for j in range(5, n_parameters):
        plot_func.addbox(axes, (i, j), color='#ccebc5', alpha=0.75)

if '--show' in sys.argv:
    plt.show()
else:
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0,
                     rect=(0.001, 0.001, 0.97, 0.965))
    plt.savefig('%scov-plot-cbox2.png' % saveas, bbox_iches='tight',
                pad_inches=0, dpi=100)

plt.close('all')
