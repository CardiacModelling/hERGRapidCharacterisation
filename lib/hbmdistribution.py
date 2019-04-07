#!/usr/bin/env python2
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import fmin
from pints import fmin as pmin


class PosteriorPredictiveLogNormal(object):
    """
        Calculate the probability from a HBM-like hyperparameters, assuming
        the base distribution is log-Normal
    """
    def __init__(self, mean, cov):
        """
            mean: mean hyperparameter chain, (samples, n_parametres)
            cov: covariance matrix hyperparameter chain, 
                 (samples, n_parameters, n_parameters)
        """
        if len(mean) != len(cov):
            raise ValueError('Number of mean and cov samples must be the'
                             ' same')
        self._mean = np.asarray(mean)
        self._cov = np.asarray(cov)
        self._n_samples, self._n_params = self._mean.shape

    def evaluate_marginal1d(self, n_param, value):
        """
            Evaluate 1D marginal of the `n_param`th parameter at value
            `value`.
        """

        def base1d(x, mu, sigma):
            output = 1/(sigma * np.sqrt(2 * np.pi) * x) * \
                     np.exp( - (np.log(x) - mu)**2 / (2 * sigma**2) )
            return output

        marginal1d = 0
        for t in range(self._n_samples):  # number of samples
            marginal1d += base1d(value,
                                 self._mean[t, n_param],
                                 np.sqrt(self._cov[t, n_param, n_param]))
        return marginal1d / self._n_samples

    def evaluate_marginal1d_cdf(self, n_param, value):
        """
            Evaluate CDF of 1D marginal of the `n_param`th parameter at value
            `value`.
        """

        marginal1dcdf = 0 if isinstance(value, float) else np.zeros(len(value))
        for t in range(self._n_samples):  # number of samples
            marginal1dcdf += lognorm.cdf(
                    (value / np.exp(self._mean[t, n_param])),
                    np.sqrt(self._cov[t, n_param, n_param]))
            # same as lognorm.cdf(value, std, scale=np.exp(mean))
        return marginal1dcdf / self._n_samples

    def evaluate_marginal1d_ppf(self, n_param, p, guess, bounds=(0.5, 2),
                                n_eva=2500):
        """
            Return `theoretical quantile', also known as `percent point
            function', that is the value from CDF of 1D mariginal of the
            `n_param`th parameter that matches `p`, with a guess value `guess`
            
            i.e. inverse of the
                 PosteriorPredictiveLogNormal().evaluate_marginal1d_cdf
        """
        # Cannot simply use lognorm.ppf, as it only gives quantile function of
        # one sample in the chain only
        
        ''' # This is better but too slow!
        def f(x):
            e = np.abs(self.evaluate_marginal1d_cdf(n_param, x[0]) - p) \
                + x[1] ** 2
            return e
        
        guess = 0.1 if guess is None else guess

        # return fmin(f, guess, disp=False)[0]
        return pmin(f, [guess, 0.1],
                    boundaries=([guess * 0.1, 0], [guess * 10, 10]),
                    verbose=False)[0]
        #'''

        guesses = np.linspace(guess * bounds[0], guess * bounds[1], n_eva)
        a = np.argmin((self.evaluate_marginal1d_cdf(n_param, guesses) \
                       - p) ** 2)
        if a == 0 or a == n_eva - 1:
            print('WARNING: `guess` or `bounds` are too far from true value')

        return guesses[a]

    def evaluate_marginal2d(self, n_params, values):
        """
            Evaluate 2D marginal of the `n_params`th parameters at values
            `values` where both inputs are array type.
        """
        raise NotImplementedError

    def ci2d(self, n_params, n_sigma):
        """
            Compute 2D credible region of the `n_params`th parameters at the
            `n_sigma`th sigma level.
        """
        raise NotImplementedError

    def marginal1d(self, n_param, values):
        """
            Evaluate 1D marignal of the `n_param`th parameter for all
            `values`.
        """
        m1d = np.empty(values.shape, dtype=float)
        for i, v in enumerate(values):
            m1d[i] = self.evaluate_marginal1d(n_param, v)
        return m1d

    def sample_approx(self, i_sample):
        """
            Return a sample from a multivariate normal distribution defined
            by the `i_sample` of the `mean`, `cov` samples
        """
        return np.random.multivariate_normal(self.mean[i_sample, :],
                self.cov[i_sample, :, :])

