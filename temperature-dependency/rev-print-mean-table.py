#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib')
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

variable_names = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
                  r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']
variable_unit = [r'$[pS]$', r'$[s^{-1}]$', r'$[V^{-1}]$', r'$[s^{-1}]$',
        r'$[V^{-1}]$', r'$[s^{-1}]$', r'$[V^{-1}]$', r'$[s^{-1}]$',
        r'$[V^{-1}]$', r'$[pA]$']

file_list = ['herg25oc', 'herg27oc', 'herg30oc', 'herg33oc', 'herg37oc']
temperatures = [25, 27, 30, 33, 37]

# Make TeX table
tex_table = ''

tex_table += '     '
for (v, u) in zip(variable_names[:-1], variable_unit[:-1]):
    tex_table += ' & ' + v + ' ' + u
tex_table += ' \\\\\n'
tex_table += '\\midrule\n'

tex_table_2 = ''

tex_table_2 += '     '
for (v, u) in zip(variable_names[:-1], variable_unit[:-1]):
    tex_table_2 += ' & ' + v + ' ' + u
tex_table_2 += ' \\\\\n'
tex_table_2 += '\\midrule\n'


for T, file_name in zip(temperatures, file_list):
    # Load result
    file_prefix = './out-mcmc/%s-pseudo2hbm-lognorm' % file_name
    simple_chain_final = np.loadtxt('%s-mean.txt' % file_prefix)
    # with open('%s-cov.pkl' % file_prefix, 'rb') as f:
    #     simple_cov_final = pickle.load(f)

    # Drop warm up and thinning
    n_samples = len(simple_chain_final)
    thinning = 1
    chain_final = simple_chain_final[(n_samples // 3):n_samples:thinning, :]
    # cov_chain_final = simple_cov_final[
    #         (n_samples // 3):n_samples:thinning, :, :]
    # assert(len(chain_final) == len(cov_chain_final))

    # Hyperparameters standard deviation
    # chain_stddev = np.sqrt(cov_chain_final.diagonal(0, 1, 2))
    # assert(len(chain_final) == len(chain_stddev))

    # Take mean
    mean_mean = np.mean(chain_final, axis=0)
    std_mean = np.std(chain_final, axis=0)
    # mean_std = np.mean(chain_stddev, axis=0)

    # Calculate first std
    std_mean_plus = mean_mean + 2*std_mean
    std_mean_minus = mean_mean - 2*std_mean
    # mean_std_plus = mean_mean + 2*mean_std
    # mean_std_minus = mean_mean - 2*mean_std

    # Detransform
    mean_mean = transform_to_model_param(mean_mean)
    std_mean_plus = transform_to_model_param(std_mean_plus)
    std_mean_minus = transform_to_model_param(std_mean_minus)
    # mean_std_plus = transform_to_model_param(mean_std_plus)
    # mean_std_minus = transform_to_model_param(mean_std_minus)

    # Table 1
    tex_table += '\\midrule\n'
    tex_table += '$T=%d^\circ$C ' % T
    for v in mean_mean:
        tex_table += ' & ' \
                + np.format_float_scientific(v, precision=2, exp_digits=1)
                # + '{:.2E}'.format(v)
    tex_table += ' \\\\\n'

    # Table 2
    tex_table_2 += '\\midrule\n'
    tex_table_2 += r'\multirow{2}{*}{$T=%d^\circ$C} ' % T
    for v in std_mean_minus:
        tex_table_2 += ' & ' \
                + np.format_float_scientific(v, precision=2, exp_digits=1)
                # + '{:.2E}'.format(v)
    tex_table_2 += ' \\\\\n'
    tex_table_2 += ' '
    for v in std_mean_plus:
        tex_table_2 += ' & ' \
                + np.format_float_scientific(v, precision=2, exp_digits=1)
                # + '{:.2E}'.format(v)
    tex_table_2 += ' \\\\\n'


print('Mean of the mean of the hyperparameters')
print(tex_table)

print('Standard deviation of the mean of the hyperparameters')
print(tex_table_2)

