#!/usr/bin/env python
""" Ion channel temperature dependency models

This module contains functions and models for temperature dependence of
ion channels. Including generalised Eyring relation and Q10
formulation.

"""
import numpy as np

# Functions defined in Eyring plot
# x = 1 / T
# param A: ln(A / T)
# param B: B
# Assume conductance not function of temperature

def eyringA(x, m, c):
    # Eyring param A in Eyring plot
    return m * x + c

def eyringB(x, m, c):
    # Eyring param B in Eyring plot
    return m * x + c

def simpleeyringB(x, m):
    # Simple Eyring param B in Eyring plot
    return m * x

def eyringG(x, g):
    # Eyring conductance in Eyring plot
    return g * np.ones(x.shape)

def eyringT(param, T):
    # Given temperature (T) and parameters of the functions in Eyring
    # plot (param), return Eyring plots values:
    # [conductance, ln(A/T), B, ...]

    assert(len(param) == 9)
    out = np.zeros(len(param))
    x = 1. / T

    out[0] = eyringG(x, param[0, 0])
    for i in [1, 3, 5, 7]:
        out[i] = eyringA(x, *param[i])
    for i in [2, 4, 6, 8]:
        out[i] = eyringB(x, *param[i])

    return out

def q10A(x, m, c):
    # Q10 param A in Eyring plot
    return m / x + np.log(x) + c

def q10B(x, b):
    # Q10 param B in Eyring plot
    return b * np.ones(x.shape)

def q10G(x, g):
    # Q10 conductance in Eyring plot
    return g * np.ones(x.shape)

def q10T(param, T):
    # Given temperature (T) and parameters of the functions in Eyring
    # plot (param), return Eyring plots values:
    # [conductance, ln(A/T), B, ...]

    assert(len(param) == 9)
    out = np.zeros(len(param))
    x = 1. / T

    out[0] = q10G(x, param[0, 0])
    for i in [1, 3, 5, 7]:
        out[i] = q10A(x, *param[i])
    for i in [2, 4, 6, 8]:
        out[i] = q10B(x, param[i, 0])

    return out


# Functions to transform Eyring plot parameters to model parameters

def eyring_transform_to_model_param(param, temperature):
    # param = [conductance, A, B, A, B, A, B, A, B]
    # temperature in K
    out = np.copy(param)

    for i in [1, 3, 5, 7]:
        out[i] = np.exp(out[i]) * temperature

    return out

def eyring_transform_from_model_param(param, temperature):
    # param = [conductance, A, B, A, B, A, B, A, B]
    # temperature in K
    out = np.copy(param)

    for i in [1, 3, 5, 7]:
        out[i] = np.log(out[i] / temperature)

    return out


def eyring_transform_mean(param, n_param):
    """
        Return mean of 1D mariginal of the `n_param`th parameter, where
        `param` specifies the untransformed (normal) distribution
        (mean, std).
    """
    mean, std = param
    if n_param in [1, 3, 5, 7]:
        from scipy.stats import lognorm
        # param A
        return lognorm.mean(std, loc=0, scale=np.exp(mean))
    else:
        from scipy.stats import norm
        # conductance or param B
        return norm.mean(loc=mean, scale=std)


def eyring_transform_cdf(param, n_param, value):
    """
        Evaluate CDF of 1D marginal of the `n_param`th parameter at value
        `value`, given `param` to specify the untransformed (normal)
        distribution (mean, std).
    """
    mean, std = param
    if n_param in [1, 3, 5, 7]:
        from scipy.stats import lognorm
        # param A
        # before is the same as lognorm.cdf((value / np.exp(mean)), std)
        return lognorm.cdf(value, std, loc=0, scale=np.exp(mean))
    else:
        from scipy.stats import norm
        # conductance or param B
        return norm.cdf(value, loc=mean, scale=std)


def eyring_transform_ppf(param, n_param, p):
    """
        Return `theoretical quantile', also known as `percent point
        function', that is the value from CDF of 1D mariginal of the
        `n_param`th parameter that matches `p`, where `param` specifies
        the untransformed (normal) distribution (mean, std).
    """
    mean, std = param
    if n_param in [1, 3, 5, 7]:
        from scipy.stats import lognorm
        # param A
        return lognorm.ppf(p, std, loc=0, scale=np.exp(mean))
    else:
        from scipy.stats import norm
        # conductance or param B
        return norm.ppf(p, loc=mean, scale=std)

