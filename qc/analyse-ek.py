#!/usr/bin/env python
import numpy as np

import string
WELL_ID = [l+str(i).zfill(2)
           for l in string.ascii_uppercase[:16]
           for i in range(1,25)]

# Fix seed
np.random.seed(101)

data_dir = '../data-sweep2'
name = 'herg25oc1-staircaseramp'
saveas = './%s-EK_all-sweep2.txt' % name

def fit_EK_poly(staircase_protocol, current, deg=3, V_full=[-70, -110],
        ramp_start=14.41, ramp_end=14.51, dt=2e-4, savefig=None, beforeE4031=None):
    # Fitting EK during last ramp in staircaseramp prt
    #
    # staircase_protocol: full staircase ramp protocol
    # current: corresponding current for the staircase ramp protocol
    # deg: n degree of polynomial for fitting
    # V_full: Full voltage range during the ramp (in the direction of time)
    # ramp_start: starting time of the ramp that matches the input protocol
    # ramp_end: ending time of the ramp that matches the input protocol
    # dt: duration of each time step to work out the index in the input protocol
    # savefig: for debug use, ['save_name', temperature]
    #
    # Note:
    # V_win=[-80, -96] works quite nicely for temperature at ~25oC
    # V_win=[-70, -80] works quite nicely for temperature at ~37oC
    rampi, rampf = int(ramp_start / dt), int(ramp_end / dt)
    assert((rampf - rampi) > deg + 1)
    vmin, vmax = np.min(V_full), np.max(V_full)
    x = staircase_protocol[rampi:rampf]
    y = current[rampi:rampf]
    p = np.poly1d(np.polyfit(x, y, deg))
    if savefig is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, label='Data')
        plt.plot(x, p(x), label='Fitted')
        if beforeE4031 is not None:
            plt.plot(x, beforeE4031[rampi:rampf], label='Before E4031')
        temperature = savefig[1]  # K
        const_R = 8.314472  # J/mol/K
        const_F = 9.64853415e4  # C/mol
        const_Ko = 4.0  # mM (my hERG experiments)
        const_Ki = 110.0  # mM (my hERG experiments)
        RTF = const_R * temperature / const_F  # J/C == V
        EK = RTF * np.log(const_Ko / const_Ki) * 1000  # mV
        plt.axvline(EK, ls='--', c='#7f7f7f', label=r'Expected $E_K$')
        plt.axhline(0, c='#7f7f7f')
        plt.xlabel('Voltage [mV]')
        plt.ylabel('Current [pA]')
        plt.legend(loc=4)
        plt.savefig(savefig[0])
        plt.close()
    # check within range V_full
    r = []
    for i in p.r:
        if vmin < i <= vmax and (np.isreal(i) or np.abs(i.imag) < 1e-8):
            r.append(i)
    print('Found EK: ', r)
    if len(r) == 1:
        return r[0].real
    elif len(r) > 1:
        return np.max(r).real
    else:
        return np.inf


# Load protocol
staircase_protocol_file = '../protocol-time-series/protocol-staircaseramp.csv'
staircase_protocol = np.loadtxt(staircase_protocol_file, delimiter=',',
                                skiprows=1)  # dt=0.1ms

EK = []
for cell in WELL_ID:
    try:
        # Load data
        data_file_name = name + '-' + cell + '.csv'
        data = np.loadtxt(data_dir + '/' + data_file_name,
                          delimiter=',', skiprows=1) # headers
    except IOError:
        EK.append(np.inf)
        continue

    EK.append(fit_EK_poly(staircase_protocol[::2, 1], data, deg=3))

np.savetxt(saveas, EK)
