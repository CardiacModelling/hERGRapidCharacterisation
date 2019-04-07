#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.append('../lib/')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pints

import pints.io
import pints.plot

import model_ikr as m
import parametertransform
from priors import BeattieLogPrior as LogPrior
from protocols import leak_staircase as protocol_def

"""
Run MCMC for single experiment-synthetic data study
"""

try:
    cell = sys.argv[1]
    int(cell)
except:
    print('Usage: python %s [int:cell_id]' % os.path.basename(__file__))
    sys.exit()

cov_seed = 101

savedir = 'out-mcmc/syn-%s' % (cov_seed)
if not os.path.isdir(savedir):
    os.makedirs(savedir)

files_dir = './out'

temperature = 25.0
useFilterCap = False

# Set parameter transformation
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param


#
# Start setting up problems
#

# Control fitting seed --> OR DONT
np.random.seed(int(cell))
fit_seed = np.random.randint(0, 2**30)
print('Using seed: ', fit_seed)
np.random.seed(fit_seed)

# Load true parameters
parameters = np.loadtxt('out/syn-%s-true/solution-%s.txt' % (cov_seed, cell))
parameters = transform_from_model_param(parameters)

# Check
path2mean = '../room-temperature-only/kylie-room-temperature/' \
            + 'last-solution_C5.txt'
mean = np.loadtxt(path2mean)
# Change conductance unit nS->pS (new parameter use V, but here mV)
mean[0] = mean[0] * 1e3
mean = transform_from_model_param(mean)
covariance = np.loadtxt('./out/cov-%s.txt' % cov_seed)
tmp = np.random.multivariate_normal(mean, covariance)
assert(np.sum(np.abs(tmp - parameters)) < 1e-10)
del(tmp, path2mean, mean, covariance)

# Load model
model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + temperature,  # K
                transform=transform_to_model_param,
                useFilterCap=useFilterCap)  # ignore capacitive spike

# Load time points
times = np.loadtxt('../data/herg25oc1-staircaseramp-times.csv',
                   delimiter=',', skiprows=1) # headers
times = np.arange(times[0], times[-1], 0.5e-3)  # dt=0.5ms

# Generate syn. data
fakedatanoise = 11.0  # roughly what the recordings are, 10-12 pA
data = model.simulate(parameters, times)
data += np.random.normal(0, fakedatanoise, size=data.shape)
if useFilterCap:
    # Apply capacitance filter to data
    data = data * model.cap_filter(times)

noise_sigma = np.std(data[:200])
print('Estimated noise level: ', noise_sigma)


#
# MCMC
#

# Create Pints stuffs
problem = pints.SingleOutputProblem(model, times, data)
loglikelihood = pints.UnknownNoiseLogLikelihood(problem)
param_bound_prior = LogPrior(transform_to_model_param,
                             transform_from_model_param)
noise_bound_prior = pints.UniformLogPrior([1e-2 * noise_sigma],
                                          [1e2 * noise_sigma])
logprior = pints.ComposedLogPrior(param_bound_prior,
                                  noise_bound_prior)
logposterior = pints.LogPosterior(loglikelihood, logprior)

# Load fitting result
x0 = np.loadtxt('out/syn-%s/solution-%s.txt' % (cov_seed, cell))
transform_x0 = transform_from_model_param(x0)
print('MCMC starting point: ', x0)
transform_x0 = np.append(transform_x0, noise_sigma)  # add noise parameter
x0_list = [transform_x0,
           np.random.normal(transform_x0, 0.25, len(transform_x0)),
           np.random.normal(transform_x0, 0.25, len(transform_x0))
          ]


#
# Run a simple adaptive mcmc routine
#

mcmc = pints.MCMCSampling(logposterior, 3, x0_list,
                          method=pints.PopulationMCMC)
n_iter = 200000
mcmc.set_max_iterations(n_iter)
# mcmc.set_acceptance_rate(0.25)
mcmc.set_initial_phase_iterations(int(0.05 * n_iter))
chains = mcmc.run()


#
# Save results
#

pints.io.save_samples('%s/c%s-chain.csv' % (savedir, cell), *chains)


#
# Simple plotting of results
#

# burn in and thinning
chains_final = chains[:, int(0.5 * n_iter)::5, :]

# transform param
chains_param = []
for c in chains_final:
    c_tmp = np.copy(c)
    c_tmp[:, :-1] = transform_to_model_param(c[:, :-1])
    chains_param.append(c_tmp)

x0 = np.append(x0, noise_sigma)  # add noise parameter

pints.plot.pairwise(chains_param[0], kde=False, ref_parameters=x0)
plt.savefig('%s/c%s-fig1.png' % (savedir, cell))
plt.close('all')

pints.plot.trace(chains_param, ref_parameters=x0)
plt.savefig('%s/c%s-fig2.png' % (savedir, cell))
plt.close('all')

pints.plot.trace(chains_final, ref_parameters=transform_x0)
plt.savefig('%s/c%s-fig2-transformed.png' % (savedir, cell))
plt.close('all')

pints.plot.series(chains_final[0], problem)  # use search space parameters
plt.savefig('%s/c%s-fig3.png' % (savedir, cell))
plt.close('all')

# Check convergence using rhat criterion
print('R-hat:')
print(pints.rhat_all_params(chains_param))

#eof
