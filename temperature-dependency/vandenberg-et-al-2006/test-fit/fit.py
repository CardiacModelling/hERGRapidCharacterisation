import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data1 = np.loadtxt('Vandenberg2006Fig8C22oC.txt', delimiter=',', skiprows=1)
data2 = np.loadtxt('Vandenberg2006Fig8C37oC.txt', delimiter=',', skiprows=1)

plt.plot(data1[:, 0], data1[:, 1])
plt.plot(data2[:, 0], data2[:, 1])
plt.savefig('test1.png')
plt.close()

def fit_tau_single_exp_simple(current, times, debug=False):
    # use 2-parameters exponential fit to the tail
    from scipy.optimize import curve_fit
    def exp_func(t, a, b):
        # do a "proper exponential" decay fit
        # i.e. shift the t to t' where t' has zero at the start of the 
        # voltage step
        return a * (1.0 - np.exp(-t / b))
    x = times
    y = current
    try:
        popt, pcov = curve_fit(exp_func, x, y, p0=[y[-1], 300e-3])
        fitted = exp_func(x, *popt)
        tau = 1e3 * popt[1]  # [ms]
    except:
        raise Exception('Maybe not here!')
    if debug:
        plt.plot(x, y, marker='s', c='#d62728')
        plt.plot(x, fitted, '--', c='#1f77b4')
    return tau

tau1 = fit_tau_single_exp_simple(data1[:, 1], data1[:, 0], True)
tau2 = fit_tau_single_exp_simple(data2[:, 1], data2[:, 0], True)
plt.savefig('test2.png')
plt.close()
print(tau1, tau2)
print((tau1 / tau2) ** (10.0 / (37.0 - 22.0)))
