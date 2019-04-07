#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import spline

splinefitskip = 2  # not fitting upstroke using spline
dt = 0.1e-3  # in second

trace = np.loadtxt('vandenberg-et-al-2006-fig1b.csv', delimiter=',',
        skiprows=1)

times = trace[:, 0]
voltage = trace[:, 1]

# Correct time
times = (times - times[0]) * 1e-3  # ms -> s
times_new = np.arange(times[0], times[-1], dt)

# Correct voltage
voltage_new = np.zeros(times_new.shape)

is_holding = times_new <= times[splinefitskip - 1]
voltage_new[is_holding] = np.mean(voltage[:splinefitskip])

is_spline = times_new >= times[splinefitskip]
voltage_new[is_spline] = spline(times[splinefitskip:], voltage[splinefitskip:],
        times_new[is_spline])

is_upstroke = np.logical_and(times_new > times[splinefitskip - 1],
                             times_new < times[splinefitskip])
if any(is_upstroke):
    c = np.polyfit([times[splinefitskip - 1], times[splinefitskip]],
                   [voltage[splinefitskip - 1], voltage[splinefitskip]], 1)
    voltage_new[is_upstroke] = np.poly1d(c)(times_new[is_upstroke])

# Save and plot
out = np.array([times_new, voltage_new]).T
np.savetxt('vandenberg-et-al-2006-fig1b-corrected.csv', out, delimiter=',',
        comments='', header='\"time\",\"voltage\"')

plt.plot(out[:, 0], out[:, 1])
plt.scatter(times, voltage)
plt.xlabel('Time [s]')
plt.ylabel('Voltage [mV]')
plt.savefig('vandenberg-et-al-2006-fig1b-corrected')
plt.close()

