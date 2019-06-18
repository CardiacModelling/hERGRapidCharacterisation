import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fink = np.loadtxt('outputs_Current_gnuplot_data.csv', delimiter=',',
        skiprows=1, usecols=(0, 1))

def emptystr2num(s):
    if s == '':
        return np.float('Nan')
    else:
        return np.float(s.strip() or 'Nan')
    
cnv = {2: emptystr2num, 3: emptystr2num}
lei = np.loadtxt('outputs_Current_gnuplot_data.csv', delimiter=',',
        skiprows=1, usecols=(2, 3), converters=cnv)

voltage = np.loadtxt('outputs_Membrane_potential_gnuplot_data.csv',
        delimiter=',', skiprows=1, usecols=(0, 1))

fig, axes = plt.subplots(2, 1, figsize=(8, 4))
axes[0].plot(voltage[:, 0], voltage[:, 1], c='#7f7f7f')
axes[1].plot(fink[:, 0], fink[:, 1], label='Fink et al. 2008 model')
axes[1].plot(lei[:, 0], lei[:, 1], label=r'Our model at $37^\circ$C')

axes[0].set_xticks([])
axes[0].set_ylabel('Voltage [mV]')
axes[1].legend()
axes[1].set_ylabel('Current [nA]\n[$g_{Kr}=0.5\mu S$]')
axes[1].set_xlabel('Time [ms]')

plt.tight_layout(pad=0.2, w_pad=0.0, h_pad=0.0)

plt.savefig('Fink-Lei-comparison', bbox_inch='tight', dpi=200)
plt.close()
