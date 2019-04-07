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


file_name = 'herg25oc1'
temperature = 25.0
try:
    cell = sys.argv[1]
except:
    print('Usage: python %s [cell_id]' % os.path.basename(__file__))
    sys.exit()

saveas = 'out-mcmc/mcmc-%s/%s' % (file_name, cell)
if not os.path.isdir(os.path.dirname(saveas)):
    os.makedirs(os.path.dirname(saveas))

data_dir = '../data'
files_dir = './out'
fit_seed = '542811797'
useFilterCap = False

# Control fitting seed --> OR DONT
# control_seed = np.random.randint(0, 2**30)
control_seed = int(fit_seed)
print('Using seed: ', control_seed)
np.random.seed(control_seed)

# Set parameter transformation
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param


#
# Start setting up problems
#

# Load data file names
data_file_name = file_name + '-staircaseramp-' + cell + '.csv'
time_file_name = file_name + '-staircaseramp-times.csv'

# Load data
data = np.loadtxt(data_dir + '/' + data_file_name,
                  delimiter=',', skiprows=1) # headers
times = np.loadtxt(data_dir + '/' + time_file_name,
                   delimiter=',', skiprows=1) # headers
noise_sigma = np.std(data[:500])
print('Estimated noise level: ', noise_sigma)

# Load model
model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                protocol_def=protocol_def,
                temperature=273.15 + temperature,  # K
                transform=transform_to_model_param,
                useFilterCap=useFilterCap)  # ignore capacitive spike
if useFilterCap:
    # Apply capacitance filter to data
    data = data * model.cap_filter(times)

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
param_file = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
        (files_dir, file_name, file_name, cell, fit_seed)
x0 = np.loadtxt(param_file)
transform_x0 = transform_from_model_param(x0)
print('MCMC starting point: ', x0)
transform_x0 = np.append(transform_x0, noise_sigma)  # add noise parameter
x0_list = [transform_x0,
           np.random.normal(transform_x0, 0.5, len(transform_x0)),
           np.random.normal(transform_x0, 0.5, len(transform_x0))
          ]


#
# Run a simple adaptive mcmc routine
#

mcmc = pints.MCMCSampling(logposterior, 3, x0_list,
                          method=pints.PopulationMCMC)
n_iter = 400000
mcmc.set_max_iterations(n_iter)
# mcmc.set_acceptance_rate(0.25)
mcmc.set_initial_phase_iterations(int(0.05 * n_iter))
chains = mcmc.run()


#
# Save results
#

pints.io.save_samples('%s-chain.csv'%saveas, *chains)


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
plt.savefig('%s-fig1.png'%saveas)
plt.close('all')

pints.plot.trace(chains_param, ref_parameters=x0)
plt.savefig('%s-fig2.png'%saveas)
plt.close('all')

pints.plot.trace(chains_final, ref_parameters=transform_x0)
plt.savefig('%s-fig2-transformed.png'%saveas)
plt.close('all')

pints.plot.series(chains_final[0], problem)  # use search space parameters
plt.savefig('%s-fig3.png'%saveas)
plt.close('all')

# Check convergence using rhat criterion
print('R-hat:')
print(pints.rhat_all_params(chains_param))

#eof
