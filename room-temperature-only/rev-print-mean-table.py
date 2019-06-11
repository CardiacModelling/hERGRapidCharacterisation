#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib')
import numpy as np
import pickle
import glob

# Set parameter transformation
import parametertransform
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param

variable_names = [r'$g_{Kr}$', r'$p_1$', r'$p_2$', r'$p_3$', r'$p_4$',
                  r'$p_5$', r'$p_6$', r'$p_7$', r'$p_8$', 'noise']

# Load result
file_prefix = './out-mcmc/herg25oc1-pseudohbm-lognorm'
simple_chain_final = np.loadtxt('%s-mean.txt' % file_prefix)
with open('%s-cov.pkl' % file_prefix, 'rb') as f:
    simple_cov_final = pickle.load(f)

# Drop warm up and thinning
n_samples = len(simple_chain_final)
thinning = 1
chain_final = simple_chain_final[(n_samples // 3):n_samples:thinning, :]
cov_chain_final = simple_cov_final[(n_samples // 3):n_samples:thinning, :, :]
assert(len(chain_final) == len(cov_chain_final))

# Detransform
detransform_cov_chain_final = np.zeros(cov_chain_final.shape)
for i, (m, s) in enumerate(zip(chain_final, cov_chain_final)):
    tm = transform_to_model_param(m)
    detransform_cov_chain_final[i, :, :] = np.outer(tm, tm) * s

# Hyperparameters standard deviation
chain_stddev = np.sqrt(detransform_cov_chain_final.diagonal(0, 1, 2))
assert(len(chain_final) == len(chain_stddev))

# Correlation matrice
simple_cor_final = np.zeros(detransform_cov_chain_final.shape)
for i, s in enumerate(detransform_cov_chain_final):
    D = np.sqrt(np.diag(s))
    c = s / D / D[:, None]
    simple_cor_final[i, :, :] = c[:, :]

# Take mean
mean_mean = np.mean(chain_final, axis=0)
mean_std = np.mean(chain_stddev, axis=0)
mean_cor = np.mean(simple_cor_final, axis=0)

# Detransform (mean-log)
mean_mean = transform_to_model_param(mean_mean)


# Make TeX table
tex_table = ''

tex_table += '     '
for v in variable_names[:-1]:
    tex_table += ' & ' + v
tex_table += ' \\\\\n'

tex_table += 'mean '
for v in mean_mean:
    tex_table += ' & ' \
            + np.format_float_scientific(v, precision=2, exp_digits=1)
tex_table += ' \\\\\n'

tex_table += r'$\sigma$ '
for v in mean_std:
    tex_table += ' & ' \
            + np.format_float_scientific(v, precision=2, exp_digits=1)
tex_table += ' \\\\\n'

print('Mean of the mean & standard deviation of the hyperparameters')
print(tex_table)


# Make TeX table for correlation matrix
tex_table_2 = ''

tex_table_2 += '     '
for v in variable_names[:-1]:
    tex_table_2 += ' & ' + v
tex_table_2 += ' \\\\\n'

for i in range(mean_cor.shape[0]):
    tex_table_2 += variable_names[i]
    for j in range(mean_cor.shape[1]):
        tex_table_2 += ' & ' + '%.3f' % mean_cor[i, j]
    tex_table_2 += ' \\\\\n'

print('Mean of the correlation matrix of the hyperparameters')
print(tex_table_2)

