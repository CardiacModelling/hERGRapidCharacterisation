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
variable_unit = [r'$[pS]$', r'$[s^{-1}]$', r'$[V^{-1}]$', r'$[s^{-1}]$', r'$[V^{-1}]$',
                 r'$[s^{-1}]$', r'$[V^{-1}]$', r'$[s^{-1}]$', r'$[V^{-1}]$', r'$[pA]$']

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
for (v, u) in zip(variable_names[:-1], variable_unit[:-1]):
    tex_table += ' & ' + v + ' ' + u
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


# Calculate steady state activation and inactivation mean values
# f(V) = 1 / (1 + e^{-(V - V_0) / k})
# k: slope factor
# V_0: half point

a_k = 1. / (mean_mean[4] + mean_mean[2])
a_V0 = np.log(mean_mean[3] / mean_mean[1]) * a_k

r_k = -1. / (mean_mean[6] + mean_mean[8])
r_V0 = np.log(mean_mean[5] / mean_mean[7]) * r_k

ss_variable = [r'activation $V_{1/2}$', r'activation $k$',
        r'inactivation $V_{1/2}$', r'inactivation $k$']
mean_ss = np.array([a_V0, a_k, r_V0, r_k]) * 1000
Sanguinetti1995 = np.array([-15, 7.9, -49, -28])  # mV

tex_table_3 = ''

tex_table_3 += '     '
for v in ss_variable:
    tex_table_3 += ' & ' + v
tex_table_3 += ' \\\\\n'

tex_table_3 += 'Our mean (n=124)'
for v in mean_ss:
    tex_table_3 += ' & ' + '%.1f' % v
tex_table_3 += ' \\\\\n'

tex_table_3 += '\\citet{sanguinetti1995mechanistic} (n=10)'
for v in Sanguinetti1995:
    tex_table_3 += ' & ' + '%.1f' % v
tex_table_3 += ' \\\\\n'

print(tex_table_3)

def logistic(V, v0, k):
    return 1. / (1. + np.exp(-(V - v0) / k))

plt.figure(figsize=(8, 7))
V = np.linspace(-120, 60, 1000)
plt.plot(V, logistic(V, mean_ss[0], mean_ss[1]), c='C0', ls='-',
        label='mean model activation')
plt.plot(V, logistic(V, mean_ss[2], mean_ss[3]), c='C1', ls='-',
        label='mean model inactivation')
plt.plot(V, logistic(V, Sanguinetti1995[0], Sanguinetti1995[1]), c='C0',
        ls='--', label='Sanguinetti et al. activation')
plt.plot(V, logistic(V, Sanguinetti1995[2], Sanguinetti1995[3]), c='C1',
        ls='--', label='Sanguinetti et al. inactivation')
plt.legend()
plt.ylabel('Steady state probability', fontsize=14)
plt.xlabel('Voltage [mV]', fontsize=14)
plt.savefig('./figs/paper/rev-steady-state-curves', bbox_inches='tight')

