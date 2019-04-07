#!/usr/bin/env python
import sys
sys.path.append('../lib/')
import numpy as np
import pints

param_names = ['ikr.g',
               'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4',
               'ikr.p5', 'ikr.p6', 'ikr.p7', 'ikr.p8']

prior_parameters = {
        '23.0': [  # 18020906E11
         4.45447163543925762e+03 * 44,  # conductance in ~pA/V
         2.96501897638277834e-01,
         6.59164494089250610e+01,
         4.62717920987949505e-02,
         4.97958442110845709e+01,
         1.07192891002687489e+02,
         7.96790797459219746e+00,
         6.54624351618329303e+00,
         3.14264169312376893e+01,],
        '36.0': [  # 18020702G20
         1.04815824649796104e+03 * 44,  # conductance in ~pA/V
         4.23953853300947081e+00,
         6.47993717018607924e+01,
         6.38998229062508472e-02,
         5.20314034706424806e+01,
         3.34265316080230207e+02,
         3.03767298311978706e+01,
         4.17768924297139890e+01,
         2.74163520256361117e+01,]
    }

defaultparams = np.asarray(prior_parameters['36.0'])
bound = 100  # 1 + 1e-1
lower = defaultparams * bound ** -1
upper = defaultparams * bound


#
# Set up Kylie's prior
#
class BeattieLogPrior(pints.LogPrior):
    """
    Unnormalised prior with constraint on the rate constants.

    # Adapted from 
    https://github.com/pints-team/ikr/blob/master/beattie-2017/beattie.py

    # Added parameter transformation everywhere
    """
    def __init__(self, transform, inv_transform):
        super(BeattieLogPrior, self).__init__()

        # Give it a big bound...
        self.lower_conductance = 1e2
        self.upper_conductance = 5e5

        # change unit...
        self.lower_alpha = 1e-7 * 1e3              # Kylie: 1e-7
        self.upper_alpha = 1e3 * 1e3               # Kylie: 1e3
        self.lower_beta  = 1e-7 * 1e3              # Kylie: 1e-7
        self.upper_beta  = 0.4 * 1e3               # Kylie: 0.4

        self.lower = np.array([
            self.lower_conductance,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
        ])
        self.upper = np.array([
            self.upper_conductance,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
        ])

        self.minf = -float('inf')

        self.rmin = 1.67e-5 * 1e3
        self.rmax = 1000 * 1e3

        self.vmin = -120 * 1e-3
        self.vmax =  60 * 1e-3

        self.transform = transform
        self.inv_transform = inv_transform

    def n_parameters(self):
        return 8 + 1

    def __call__(self, parameters):

        debug = False
        parameters = self.transform(parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return self.minf

        # Check rate constant boundaries
        g, p1, p2, p3, p4, p5, p6, p7, p8 = parameters[:]

        # Check forward rates
        r = p1 * np.exp(p2 * self.vmax)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r1')
            return self.minf
        r = p5 * np.exp(p6 * self.vmax)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r2')
            return self.minf

        # Check backward rates
        r = p3 * np.exp(-p4 * self.vmin)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r3')
            return self.minf
        r = p7 * np.exp(-p8 * self.vmin)
        if np.any(r < self.rmin) or np.any(r > self.rmax):
            if debug: print('r4')
            return self.minf

        return 0

    def _sample_partial(self, v):
        for i in xrange(100):
            a = np.exp(np.random.uniform(
                np.log(self.lower_alpha), np.log(self.upper_alpha)))
            b = np.random.uniform(self.lower_beta, self.upper_beta)
            r = a * np.exp(b * v)
            if r >= self.rmin and r <= self.rmax:
                return a, b
        raise ValueError('Too many iterations')

    def sample(self):
        p = np.zeros(9)

        # Sample forward rates
        p[1:3] = self._sample_partial(self.vmax)
        p[5:7] = self._sample_partial(self.vmax)

        # Sample backward rates
        p[3:5] = self._sample_partial(-self.vmin)
        p[7:9] = self._sample_partial(-self.vmin)

        # Sample conductance
        p[0] = np.random.uniform(
            self.lower_conductance, self.upper_conductance)

        p = self.inv_transform(p)

        # Return
        return p


#
# Multiple priori
#
class MultiPriori(pints.LogPrior):
    """
    Combine multiple priors
    """
    def __init__(self, priors):
        self._priors = priors
        self._n_parameters = self._priors[0].n_parameters()
        for p in self._priors:
            assert(self._n_parameters == p.n_parameters())

    def n_parameters(self):
        return self._n_parameters

    def __call__(self, x):
        t = 0
        for p in self._priors:
            t += p(x)
        return t


