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

import model_ikr as m
import parametertransform
from priors import BeattieLogPrior as LogPrior
from priors import prior_parameters
from protocols import leak_staircase as protocol_def

savedir = './out'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

file_dir = '../data'
file_list = [
        'herg25oc1-staircaseramp-A01.csv',
        'herg27oc1-staircaseramp-A01.csv',
        'herg30oc1-staircaseramp-A01.csv',
        'herg33oc1-staircaseramp-A03.csv',
        'herg37oc3-staircaseramp-B06.csv',
        ]
temperatures = [25.0, 27.0, 30.0, 33.0, 37.0]
time_file = 'herg25oc1-staircaseramp-times.csv'
useFilterCap = True

# Control fitting seed --> OR DONT
fit_seed = np.random.randint(0, 2**30)
print('Using seed: ', fit_seed)
np.random.seed(fit_seed)

# Set parameter transformation
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param


for i_file, file_name in enumerate(file_list):
    # Save name
    saveas = file_name[:-4]
    if useFilterCap:
        saveas += '-fcap'

    # Load synthetic data
    data = np.loadtxt(file_dir + '/' + file_name,
                      delimiter=',', skiprows=1) # headers
    times = np.loadtxt(file_dir + '/' + time_file,
                       delimiter=',', skiprows=1) # headers
    noise_sigma = np.std(data[:500])
    print('Estimated noise level: ', noise_sigma)

    # Try prior param
    priorparams = np.asarray(prior_parameters['23.0'])
    transform_priorparams = transform_from_model_param(priorparams)

    # Load model
    model = m.Model('../mmt-model-files/kylie-2017-IKr.mmt',
                    protocol_def=protocol_def,
                    temperature=273.15 + temperatures[i_file],  # K
                    transform=transform_to_model_param,
                    useFilterCap=useFilterCap)  # remove/ignore capacitive spike
    if useFilterCap:
        # Apply capacitance filter to data
        data = data * model.cap_filter(times)

    # Create Pints stuffs
    problem = pints.SingleOutputProblem(model, times, data)
    loglikelihood = pints.KnownNoiseLogLikelihood(problem, noise_sigma)
    logprior = LogPrior(transform_to_model_param, transform_from_model_param)
    logposterior = pints.LogPosterior(loglikelihood, logprior)

    print('Score at default parameters: ', logposterior(transform_priorparams))
    for _ in range(10):
        assert(logposterior(transform_priorparams) ==\
                logposterior(transform_priorparams))

    try:
        N = int(sys.argv[1])
    except IndexError:
        N = 3

    params, logposteriors = [], []

    for i in range(N):

        if i==0:
            x0 = transform_priorparams
        else:
            # Randomly pick a starting point
            x0 = logprior.sample()
        print('Starting point: ', x0)

        # Create optimiser
        print('Starting logposterior: ', logposterior(x0))
        opt = pints.Optimisation(logposterior, x0.T, method=pints.CMAES)
        opt.set_max_iterations(None)
        opt.set_parallel(True)

        # Run optimisation
        try:
            with np.errstate(all='ignore'): # Tell numpy not to issue warnings
                p, s = opt.run()
                p = transform_to_model_param(p)
                params.append(p)
                logposteriors.append(s)
                print('Found solution:          Old parameters:' )
                for k, x in enumerate(p):
                    print(pints.strfloat(x) + '    ' + \
                            pints.strfloat(priorparams[k]))
        except ValueError:
            import traceback
            traceback.print_exc()

    # Order from best to worst
    order = np.argsort(logposteriors)[::-1]  # (use [::-1] for LL)
    logposteriors = np.asarray(logposteriors)[order]
    params = np.asarray(params)[order]

    # Show results
    bestn = min(3, N)
    print('Best %d logposteriors:' % bestn)
    for i in xrange(bestn):
        print(logposteriors[i])
    print('Mean & std of logposterior:')
    print(np.mean(logposteriors))
    print(np.std(logposteriors))
    print('Worst logposterior:')
    print(logposteriors[-1])

    # Extract best 3
    obtained_logposterior0 = logposteriors[0]
    obtained_parameters0 = params[0]

    # Show results
    print('Found solution:          Old parameters:' )
    # Store output
    with open('%s/%s-solution-%s.txt' % (savedir, saveas, fit_seed), 'w') as f:
        for k, x in enumerate(obtained_parameters0):
            print(pints.strfloat(x) + '    ' + pints.strfloat(priorparams[k]))
            f.write(pints.strfloat(x) + '\n')

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    sol0 = problem.evaluate(transform_from_model_param(obtained_parameters0))
    vol = model.voltage(times) * 1e3
    axes[0].plot(times, vol, c='#7f7f7f')
    axes[0].set_ylabel('Voltage [mV]')
    axes[1].plot(times, data, alpha=0.5, label='data')
    axes[1].plot(times, sol0, label='found solution')
    axes[1].legend()
    axes[1].set_ylabel('Current [pA]')
    axes[1].set_xlabel('Time [s]')
    plt.subplots_adjust(hspace=0)
    plt.savefig('%s/%s-solution-%s.png' % (savedir, saveas, fit_seed),
                bbox_inches='tight')
    plt.close()
