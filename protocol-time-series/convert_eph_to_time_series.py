#!/usr/bin/env python2
from __future__ import print_function
import sys
import numpy as np

protocolList = [
    'ProtocolChonhERGKinetics_v3.eph',
    'hERGScreenBT.eph',
    'hERGActivationIVForChon.eph',
    'Kv_long_4S_holding.eph',
    'AP_clamp_Chon_1Hz.eph',
    'AP_clamp_Chon_2Hz.eph',
    'AP_clamp_Chon_abnormal_2Hz.eph',
    'AP_clamp_Chon_abnormal_2Hz_v3.eph',
    'ProtocolChonStaircaseRamp.eph',
    'AP_clamp_Chon_05Hz4s.eph',
    'AP_clamp_Chon_1Hz4s.eph',
    'AP_clamp_Chon_2Hz4s.eph',
    'ProtocolChonShortIV_hERGActivation.eph',
    'ProtocolChonShortIV_hERGSSInactivation.eph']
try:
    n_prt = int(sys.argv[1])
except IndexError:
    n_prt = -8
fileName = protocolList[n_prt]
filePath = 'eph_protocol_files/' + fileName
splitSweeps = False  # True if IV-like protocol
dt = 0.1  # ms
startWith = ['Set', 'Ramp']

# Set debug mode
debug = '--debug' in sys.argv
if debug:
    import matplotlib.pyplot as plt

# Save as myokit csv format
usemyokit = False

# DataControl384Output to Short names
D2S = {
    'ProtocolChonhERGKinetics_v3':'whole',  # shorten one for fitting
    'hERGScreenBT':'pharma',  # during drug application
    'hERGActivationIVForChon': 'activ',
    'Kv_long_4S_holding':'4s',
    'AP_clamp_Chon_1Hz':'ap1s',
    'AP_clamp_Chon_2Hz':'ap2s',
    'AP_clamp_Chon_abnormal_2Hz':'apab',
    'AP_clamp_Chon_abnormal_2Hz_v3':'apabv3',
    'ProtocolChonStaircaseRamp': 'staircaseramp',
    'AP_clamp_Chon_05Hz4s': 'ap05hz',
    'AP_clamp_Chon_1Hz4s': 'ap1hz',
    'AP_clamp_Chon_2Hz4s': 'ap2hz',
    'ProtocolChonShortIV_hERGActivation': 'sactiv',
    'ProtocolChonShortIV_hERGSSInactivation': 'sinactiv',
}

# get time length
total_time = 0
with open(filePath, 'r') as f:
    for line in f:
        if line.startswith('# of Sweeps') and splitSweeps:
            nSweeps = int(line.strip().split()[-1])  # for IV only
        line = line.strip()
        if line.startswith('Set') or line.startswith('Ramp'):
            line = line.split()
            total_time += float(line[3])
times = np.arange(0, total_time, dt)  # ms

# init protocol
if splitSweeps:  # for IV only
    protocolNSweeps = np.zeros((len(times), nSweeps))  # mV
protocol = np.zeros(times.shape)  # mV
lastindex = 0
lastvoltage = -80  # assume it starts with membrane voltage -80 mV

# create protocol
print('Type  Vmemb T/F  Voltage [mV]  Duration [ms]  Vincrease [mV]  RecordData T/F  Group  Rep')
with open(filePath, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('Set') or line.startswith('Ramp'):
            print(line)
            line = line.split()
            method, isNegative80mV, voltage, duration, Vinc = line[:5]
            voltage = float(voltage)
            duration = float(duration)
            Vinc = float(Vinc)  # for IV only
            # make sure the auto setting to membrane voltage is done
            voltage = voltage if isNegative80mV == 'F' else -80
            untiltime = times[lastindex] + duration
            untilindex = np.argmin(np.abs(times - untiltime))

            if method == 'Set':
                protocol[lastindex:untilindex] = voltage
            elif method == 'Ramp':
                coefficients = np.polyfit([lastindex, untilindex], 
                                          [lastvoltage, voltage],
                                          1)
                polynomial = np.poly1d(coefficients)
                protocol[lastindex:untilindex] = polynomial(range(lastindex,
                                                            untilindex))
            else:
                raise ValueError('No such method...')

            # For IV only
            if splitSweeps:
                for i in range(nSweeps):
                    protocolNSweeps[lastindex:untilindex, i] = \
                            protocol[lastindex:untilindex] + i * Vinc

            lastvoltage = protocol[untilindex-1]  # store last voltage for ramp
            lastindex = untilindex

# Fill up the last index
protocol[lastindex] = lastvoltage
if splitSweeps:
    for i in range(nSweeps):
        protocolNSweeps[lastindex, i] = lastvoltage + i * Vinc

if debug:
    plt.plot(times, protocol) 
    plt.xlabel('time [ms]')
    plt.ylabel('voltage [mV]')
    plt.show()
elif usemyokit:
    sys.path.append('../../lib')
    import myokit
    d = myokit.DataLog()
    d['time'] = times*1e-3  # s
    d['voltage'] = protocol*1e-3  # V
    d.set_time_key('time')
    d.save_csv('protocol-' + D2S[fileName[:-4]] + '.csv')
else:
    nSweeps = nSweeps if splitSweeps else 1  # only output 1 sweep if not IV
    out = np.zeros((len(times), nSweeps + 1))
    out[:, 0] = times * 1e-3  # s
    if splitSweeps:
        out[:, 1:] = protocolNSweeps[:, :]  # mV
    else:
        out[:, 1] = protocol  # mV
    header = ",\"voltage\"" * nSweeps
    # np.savetxt('protocol-' + D2S[fileName[:-4]] + '.txt', out)
    np.savetxt('protocol-' + D2S[fileName[:-4]] + '.csv', out,
               delimiter=',', comments='', header="\"time\"" + header)

