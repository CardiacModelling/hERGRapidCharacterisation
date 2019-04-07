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

fakedatanoise = 11.0  # roughly what the recordings are, 10-12 pA
n_fakedata = 124
data_dir = './fakedata-voltageoffset'
file_list = ['herg25oc1']
temperatures = [25.0]
useFilterCap = False

# Control fitting seed --> OR DONT
# fit_seed = np.random.randint(0, 2**30)
fit_seed = 542811797
print('Using seed: ', fit_seed)
np.random.seed(fit_seed)

# Set parameter transformation
transform_to_model_param = parametertransform.log_transform_to_model_param
transform_from_model_param = parametertransform.log_transform_from_model_param


for i_file, (file_name, temperature) in enumerate(zip(file_list, temperatures)):

    # Split each file_name as a separate output dir
    savename = '%s-fakedata-voltageoffset' % file_name
    if not os.path.isdir('%s/%s' % (savedir, savename)):
        os.makedirs('%s/%s' % (savedir, savename))

    for i_cell in range(n_fakedata):
        # Load data file names
        data_file_name = file_name + '-staircaseramp-sim-' + str(i_cell) + \
                         '.csv'
        time_file_name = file_name + '-staircaseramp-times.csv'

        # Save name
        saveas = data_file_name[:-4]
        if useFilterCap:
            saveas += '-fcap'

        # Load data
        data = np.loadtxt(data_dir + '/' + data_file_name,
                          delimiter=',', skiprows=1) # headers
        times = np.loadtxt(data_dir + '/' + time_file_name,
                           delimiter=',', skiprows=1) # headers
        # Add noise
        data += np.random.normal(0.0, fakedatanoise, size=data.shape)
        noise_sigma = np.std(data[:500])
        print('Estimated noise level: ', noise_sigma)

        # Try prior param
        priorparams = np.asarray(prior_parameters['23.0'])
        transform_priorparams = transform_from_model_param(priorparams)

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
        loglikelihood = pints.KnownNoiseLogLikelihood(problem, noise_sigma)
        logprior = LogPrior(transform_to_model_param,
                            transform_from_model_param)
        logposterior = pints.LogPosterior(loglikelihood, logprior)

        print('Score at default parameters: ',
              logposterior(transform_priorparams))
        for _ in range(10):
            assert(logposterior(transform_priorparams) ==\
                    logposterior(transform_priorparams))

        try:
            N = int(sys.argv[1])
        except IndexError:
            N = 3

        params, logposteriors = [], []

        for i in range(N):

            if i == 0:
                x0 = transform_priorparams
            else:
                # Randomly pick a starting point
                x0 = logprior.sample()
            print('Starting point: ', x0)

            # Create optimiser
            print('Starting logposterior: ', logposterior(x0))
            opt = pints.Optimisation(logposterior, x0.T, method=pints.CMAES)
            opt.set_max_iterations(None)
            opt.set_parallel(20)

            # Run optimisation
            try:
                with np.errstate(all='ignore'):
                    # Tell numpy not to issue warnings
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
        obtained_parameters = params[0]

        # Show results
        print('Found solution:          Old parameters:' )
        # Store output
        with open('%s/%s/%s-solution-%s.txt' % (savedir, savename, saveas,\
                    fit_seed), 'w') as f:
            for k, x in enumerate(obtained_parameters):
                print(pints.strfloat(x) + '    ' + \
                      pints.strfloat(priorparams[k]))
                f.write(pints.strfloat(x) + '\n')

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        sol = problem.evaluate(transform_from_model_param(obtained_parameters))
        vol = model.voltage(times) * 1e3
        axes[0].plot(times, vol, c='#7f7f7f')
        axes[0].set_ylabel('Voltage [mV]')
        axes[1].plot(times, data, alpha=0.5, label='data')
        axes[1].plot(times, sol, label='found solution')
        axes[1].legend()
        axes[1].set_ylabel('Current [pA]')
        axes[1].set_xlabel('Time [s]')
        plt.subplots_adjust(hspace=0)
        plt.savefig('%s/%s/%s-solution-%s.png' % (savedir, savename, saveas,\
                    fit_seed), bbox_inches='tight')
        plt.close()
