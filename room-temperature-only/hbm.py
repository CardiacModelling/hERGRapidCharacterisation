#!/usr/bin/env python2

# coding: utf-8
# Metropolis Within Gibbs (MWB) with Bayesian Hierarchical Model (BHM)

from __future__ import print_function
import sys
sys.path.append('../lib')
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import pickle

import pints
import pints.io

# Use multiprocessing for the individual exp. eva.
import parallel_evaluation as pe
import multiprocessing

try:
    # Python 3
    import queue
except ImportError:
    import Queue as queue

import model_ikr as m
import parametertransform
from priors import BeattieLogPrior as LogPrior
from priors import MultiPriori
from protocols import leak_staircase as protocol_def

savedir = './out-mcmc/test-HBM'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

data_dir = '../data'  # staircase ramp
files_dir = './out'
qc_dir = '.'
file_list = ['herg25oc1']
temperatures = [25.0]
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
# Setup samplers
#

log_posteriors = []
samplers = []
exp_parameters = []
exp_data = []
for i_file, (file_name, temperature) in enumerate(zip(file_list,
                                                      temperatures)):
    # Load QC
    selectedfile = '%s/manualv2selected-%s.txt' % (qc_dir, file_name)
    selectedwell = []
    with open(selectedfile, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                selectedwell.append(l.split()[0])

    # Load time points
    time_file_name = file_name + '-staircaseramp-times.csv'
    times = np.loadtxt(data_dir + '/' + time_file_name,
                       delimiter=',', skiprows=1) # headers

    # Load model
    model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                    protocol_def=protocol_def,
                    temperature=273.15 + temperature,  # K
                    transform=transform_to_model_param,
                    useFilterCap=useFilterCap)  # ignore capacitive spike

    for i_cell, cell in enumerate(selectedwell[:20]):
        # Load data file names
        data_file_name = file_name + '-staircaseramp-' + cell + '.csv'

        # Load data
        data = np.loadtxt(data_dir + '/' + data_file_name,
                          delimiter=',', skiprows=1) # headers
        noise_sigma = np.std(data[:500])
        print('Estimated noise level: ', noise_sigma)
        exp_data.append(data)

        if useFilterCap:
            # Apply capacitance filter to data
            data = data * model.cap_filter(times)

        # Load fitting result
        param_file = '%s/%s/%s-staircaseramp-%s-solution-%s.txt' % \
                (files_dir, file_name, file_name, cell, fit_seed)
        x0 = np.loadtxt(param_file)
        transform_x0 = transform_from_model_param(x0)
        exp_parameters.append(x0)

        # Create Pints stuffs
        problem = pints.SingleOutputProblem(model, times, data)
        log_likelihood = pints.UnknownNoiseLogLikelihood(problem)

        # Create a prior
        # sigma_x0 = 0.025 * np.abs(transform_x0) * np.eye(len(transform_x0))
        sigma_x0 = 5.0 * np.eye(len(transform_x0))  # big prior, update later
        sigma_noise_sigma = 0.25 * noise_sigma
        param_prior = pints.MultivariateNormalLogPrior(
            transform_x0,  # here transform_x0 is in search space already
            sigma_x0 ** 2  # require covariance matrix
        )
        noise_prior = pints.NormalLogPrior(noise_sigma, sigma_noise_sigma)
        normallogprior = pints.ComposedLogPrior(param_prior,
                                                noise_prior)
        # Boundaries from Beattie et al. 2018
        beattielogprior = LogPrior(transform_to_model_param,
                                   transform_from_model_param)
        noise_bound_prior = pints.UniformLogPrior([1e-2 * noise_sigma],
                                                  [1e2 * noise_sigma])
        beattielogprior = pints.ComposedLogPrior(beattielogprior,
                                                 noise_bound_prior)
        log_prior = MultiPriori([normallogprior, beattielogprior])

        log_posterior = pints.LogPosterior(log_likelihood, log_prior)
        log_posteriors.append(log_posterior)

        print('Score at default parameters: ',
              log_posterior(np.append(transform_x0, noise_sigma)))
        for _ in range(10):
            assert(log_posterior(np.append(transform_x0, noise_sigma)) ==\
                    log_posterior(np.append(transform_x0, noise_sigma)))

        # Create a MCMC sampler
        sampler = pints.PopulationMCMC(list(transform_x0) + [noise_sigma])
        # sampler = pints.AdaptiveCovarianceMCMC(list(transform_x0)
        #                                        + [noise_sigma])
        sampler.set_initial_phase(True)  # switch it off later
        samplers.append(sampler)

n_parameters = len(exp_parameters[0])
nexp = len(samplers)
# quick sanity checks
assert(len(log_posteriors) == nexp)
print('Running hierarchical Bayesian model on ' + str(nexp) + ' cells\n')


#
# Hyperparameters prior setting
#
# k_0: 0 = not imposing mean, no prior;
#      nexp = prior as strong as our experiments
# nu_0: 1 = no prior
#       nexp = scaling this gives prior of $\Sigma$ to be certain at Gamma_0
# mu_0: mean of the prior
# Gamma_0: covariance of the prior
#

k_0 = 0
nu_0 = 1
mu_0 = transform_from_model_param(np.array(exp_parameters))
mu_0 = np.mean(mu_0, axis=0)  # mean of the transformed parameters
assert(len(mu_0) == n_parameters)
estimated_cov = (np.std(
                    transform_from_model_param(
                        np.array(exp_parameters)
                    ), axis=0)
                )**2
Gamma_0 = 1.0e-3 * np.diag(estimated_cov)
Gamma_0[0, 0] = estimated_cov[0] * nu_0  # Let g_Kr varies
assert(len(Gamma_0) == len(mu_0))


#
# Define and start multiprocessing
#

# Determine number of workers
nworkers = max(1, multiprocessing.cpu_count() - 6)
nworkers = min(len(log_posteriors), nworkers)
nworkers = 10
print('\nNumber of processors (in use): ' + str(nworkers) + '\n')

# assume only log_posterior(x) takes the longest time
log_posteriors_in_parallel = pe.ParallelEvaluator(log_posteriors, 
                                                  nworkers=nworkers)
del(log_posteriors)


#
# Do some burn in (but probably don't need this?)
#

n_burn_in = 10000
n_init_phase = 5000
assert(n_init_phase < n_burn_in)
for sample in range(n_burn_in):
    # Track progress
    if sample % 100 == 0 and sample != 0:
        print('.', end='') #, flush=True)
        sys.stdout.flush()
    if sample % 10000 == 0 and sample != 0:
        print(' ')
    # generate samples of hierarchical 1e9 * params
    proposed_points = []
    # ask
    for sampler in samplers:
        proposed_points.append(sampler.ask())
    # assume only log_posterior(x) takes the longest time
    ff = log_posteriors_in_parallel.evaluate(proposed_points)
    # tell
    for sampler, ll in zip(samplers, ff):
        if np.isnan(ll) or np.isinf(ll):
            ll = -np.inf
        sampler.tell(ll)
        # End initial phase
        if sample == n_init_phase:
            sampler.set_initial_phase(False)


#
# Run a hierarchical gibbs-mcmc routine
# 

n_samples = 400000
n_log_chains = 10  # log it during run 10 times...
thinning = 10  # Do thinning already
chain = np.zeros((n_samples / thinning, n_parameters))
cov_chain = np.zeros((n_samples / thinning, n_parameters, n_parameters))
exp_chains = [np.zeros((n_samples / thinning, n_parameters + 1)) # +1 for noise
              for i in range(nexp)]
n_non_model_params = 1
for sample in range(n_samples):
    # Track progress
    if sample % 100 == 0 and sample != 0:
        print('.', end='') #, flush=True)
        sys.stdout.flush()
    if sample % 10000 == 0 and sample != 0:
        print(' ')

    # generate samples from individual experiments
    xs = np.zeros((n_parameters + 1, nexp))  # +1 for noise
    proposed_points = []
    # ask
    for sampler in samplers:
        proposed_points.append(sampler.ask())
    # assume only log_posterior(x) takes the longest time
    ff = log_posteriors_in_parallel.evaluate(proposed_points)
    # tell
    for i, (sampler, ll) in enumerate(zip(samplers, ff)):
        if np.isnan(ll) or np.isinf(ll):
            ll = -np.inf
        xs[:, i] = sampler.tell(ll)
        # thinning, store every (thinning)th sample
        if ((sample % thinning) == (thinning - 1)) and (sample != 0):
            # store sample to individual exp chain
            exp_chains[i][sample/thinning - 1, :] = xs[:, i]

    # No transformation is needed; all are in log-scale.
    # sample mean and covariance from a normal inverse wishart
    # exclude V_shift and noise from top-level parameters
    xhat = np.mean(xs[:-1*n_non_model_params], axis=1)
    C = np.zeros((len(xhat), len(xhat)))
    for x in xs[:-1*n_non_model_params].T:
        C += np.outer(x - xhat, x - xhat)

    k = k_0 + nexp
    nu = nu_0 + nexp
    mu = (k_0 * mu_0 + nexp * xhat) / k
    tmp = xhat - mu_0
    Gamma = Gamma_0 + C + (k_0 * nexp) / k * np.outer(tmp, tmp)

    covariance_sample = scipy.stats.invwishart.rvs(df=nu, scale=Gamma)
    means_sample = scipy.stats.multivariate_normal.rvs(
        mean=mu, cov=covariance_sample / k)

    # thinning, store every (thinning)th sample
    if ((sample % thinning) == (thinning - 1)) and (sample != 0):
        # store sample to chain
        chain[sample/thinning - 1, :] = means_sample
        # store cov matrix to chain
        cov_chain[sample/thinning - 1, :, :] = covariance_sample[:, :]
    
    # replace individual sampler's priors with hierarchical params
    for i, (log_posterior, sampler) in enumerate(
            zip(log_posteriors_in_parallel._function, samplers)):
        # Several hardcoded layers here...
        # log_posterior (from pints.LogPosterior) . _log_prior (get the prior
        # which is wrapper class MultiPriori) . _priors[0] (the actual
        # pints.ComposedLogPrior -- the first element) . _priors[0] (and only
        # interested in the model-parameter prior -- the first element)
        log_posterior._log_prior._priors[0]._priors[0]._mean = means_sample
        log_posterior._log_prior._priors[0]._priors[0]._cov = covariance_sample

    # log chains as it runs (as backup)
    if sample % (n_samples // n_log_chains) == 0 and sample != 0:
        pints.io.save_samples('%s/chain.csv' % savedir, chain)
        pints.io.save_samples('%s/exp_chain.csv' % savedir, *exp_chains)
        with open('%s/cov_chain.pkl' % savedir, 'wb') as output:
            pickle.dump(cov_chain, output)


#
# Save chain
#

pints.io.save_samples('%s/chain.csv' % savedir, chain)
pints.io.save_samples('%s/exp_chain.csv' % savedir, *exp_chains)

with open('%s/cov_chain.pkl' % savedir, 'wb') as output:
    pickle.dump(cov_chain, output)

