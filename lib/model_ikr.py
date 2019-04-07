from __future__ import print_function
import os
import sys
sys.path.append('../lib')
import numpy as np

import pints
import myokit
import myokit.pacing as pacing

###############################################################################
## Defining Model
###############################################################################

vhold = -80e-3
# Default time
DT = 2.0e-04  # maybe not rely on this...

#
# Create ForwardModel
#
class Model(pints.ForwardModel):
    parameters = [
        'ikr.g', 'ikr.p1', 'ikr.p2', 'ikr.p3', 'ikr.p4', 'ikr.p5', 'ikr.p6', 
        'ikr.p7', 'ikr.p8',  
        ]
    
    def __init__(self, model_file, protocol_def, temperature,
                 transform=None, useFilterCap=False, effEK=True,
                 concK=[4.0, 110.0]):
        # model_file: mmt model file for myokit
        # protocol_def: func take model as arg and return tuple
        #               (modified model, step)
        #               or
        #               str to a file name contains protocol time series.
        # temperature: temperature of the experiment to be set (in K).
        # transform: function to transform parameters when call simulate()
        # useFilterCap: apply capacitive filtering if True
        # effEK: correct EK value for 37.0oC experiment (TODO temporary only)
        # concK: set concentration of K, [Ko, Ki]; default [4.0, 110.0] mM
        
        # Load model
        model = myokit.load_model(model_file)
        # Set temperature
        model.get('phys.T').set_rhs(temperature)
        if temperature == 37.0 + 273.15 and effEK:
            print('Using effective EK for 37oC data')
            model.get('potassium.Ko').set_rhs(
                    110 * np.exp( -92.630662854828572 / (8.314472 * (273.15 + 37) / 9.64853415e4 * 1000))
                    )
        # Set concentration
        model.get('potassium.Ko').set_rhs(float(concK[0]))
        model.get('potassium.Ki').set_rhs(float(concK[1]))

        # Compute model EK
        const_R = 8.314472  # J/mol/K
        const_F = 9.64853415e4  # C/mol
        const_Ko = float(concK[0])  # mM (my hERG experiments)
        const_Ki = float(concK[1])  # mM
        RTF = const_R * temperature / const_F  # J/C == V
        self._EK = RTF * np.log(const_Ko / const_Ki)

        # 1. Create pre-pacing protocol
        protocol = pacing.constant(vhold)
        # Create pre-pacing simulation
        self.simulation1 = myokit.Simulation(model, protocol)
        
        # 2. Create specified protocol
        self.useFilterCap = useFilterCap
        if type(protocol_def) is str:
            d = myokit.DataLog.load_csv(protocol_def).npview()
            self.simulation2 = myokit.Simulation(model)
            self.simulation2.set_fixed_form_protocol(
                d['time'],  # s
                d['voltage'] * 1e-3  # mV -> V
            )
            if self.useFilterCap:
                raise ValueError('Cannot use capacitance filtering with the'
                                 + ' given format of protocol_def')
        else:
            if self.useFilterCap:
                model, steps, fcap = protocol_def(model, self.useFilterCap)
                self.fcap = fcap
            else:
                model, steps = protocol_def(model)
            protocol = myokit.Protocol()
            for f, t in steps:
                protocol.add_step(f, t)
            # Create simulation for protocol
            self.simulation2 = myokit.Simulation(model, protocol)
        
        self.simulation2.set_tolerance(1e-12, 1e-14)
        self.simulation2.set_max_step_size(1e-5)

        self.transform = transform
        self.init_state = self.simulation1.state()

    def n_parameters(self):
        # n_parameters() method for Pints
        return len(self.parameters)

    def cap_filter(self, times):
        if self.useFilterCap:
            return self.fcap(times)
        else:
            return None

    def simulate(self, parameters, times, extra_log=[]):
        # simulate() method for Pints

        # Update model parameters
        if self.transform is not None:
            parameters = self.transform(parameters)

        for i, name in enumerate(self.parameters):
            self.simulation1.set_constant(name, parameters[i])
            self.simulation2.set_constant(name, parameters[i])

        # Reset to ensure each simulate has same init condition
        self.simulation1.reset()
        self.simulation2.reset()
        self.simulation1.set_state(self.init_state)
        self.simulation2.set_state(self.init_state)

        # Run!
        try:
            self.simulation1.pre(100)
            self.simulation2.set_state(self.simulation1.state())
            d = self.simulation2.run(np.max(times)+0.02, 
                log_times = times, 
                log = [
                       'ikr.IKr',
                      ] + extra_log,
                #log_interval = 0.025
                ).npview()
        except myokit.SimulationError:
            return float('inf')

        # Apply capacitance filter and return
        if self.useFilterCap:
            d['ikr.IKr'] = d['ikr.IKr'] * self.fcap(times)

        if len(extra_log) > 0:
            return d
        return d['ikr.IKr']

    def voltage(self, times):
        # Return voltage protocol

        # Run
        self.simulation1.reset()
        self.simulation2.reset()
        self.simulation1.set_state(self.init_state)
        self.simulation2.set_state(self.init_state)
        try:
            self.simulation1.pre(100)
            self.simulation2.set_state(self.simulation1.state())
            d = self.simulation2.run(np.max(times)+0.02, 
                log_times = times, 
                log = ['membrane.V'],
                #log_interval = 0.025
                ).npview()
        except myokit.SimulationError:
            return float('inf')
        # Return
        return d['membrane.V'] 

    def EK(self):
        return self._EK
    
    def parameter(self):
        # return the name of the parameters
        return self.parameters

    def name(self):
        # name
        return 'hERG model'


class ModelWithVoltageOffset(Model):
    """
        Simple voltage offset error model
    """
    def __init__(self, model_file, protocol_def, temperature,
                 transform=None, useFilterCap=False):
        super(ModelWithVoltageOffset, self).__init__(model_file, protocol_def,
                temperature, transform, useFilterCap)

        # voltage offset
        self._vo = 0

        # protocol def must be time series
        if type(protocol_def) is not str:
            raise ValueError('Only support time series type protocol')
        d = myokit.DataLog.load_csv(protocol_def).npview()
        self._prt_t = d['time']  # s
        self._prt_v = d['voltage'] * 1e-3  # mV -> V
        
    def set_voltage_offset(self, vo):
        self._vo = vo  # V
        self.simulation2.set_fixed_form_protocol(
            self._prt_t,  # s
            self._prt_v + self._vo   # V
        )

    def voltage_offset(self):
        return self._vo

    def simulate(self, parameters, times, extra_log=[]):
        # simulate() method for Pints

        # Voltage offset
        self.set_voltage_offset(parameters[-1])
        parameters = parameters[:-1]

        # Update model parameters
        if self.transform is not None:
            parameters = self.transform(parameters)

        for i, name in enumerate(self.parameters):
            self.simulation1.set_constant(name, parameters[i])
            self.simulation2.set_constant(name, parameters[i])

        # Reset to ensure each simulate has same init condition
        self.simulation1.reset()
        self.simulation2.reset()
        self.simulation1.set_state(self.init_state)
        self.simulation2.set_state(self.init_state)

        # Run!
        try:
            self.simulation1.pre(100)
            self.simulation2.set_state(self.simulation1.state())
            d = self.simulation2.run(np.max(times)+0.02, 
                log_times = times, 
                log = [
                       'ikr.IKr',
                      ] + extra_log,
                #log_interval = 0.025
                ).npview()
        except myokit.SimulationError:
            return float('inf')

        # Apply capacitance filter and return
        if self.useFilterCap:
            d['ikr.IKr'] = d['ikr.IKr'] * self.fcap(times)

        if len(extra_log) > 0:
            return d
        return d['ikr.IKr']

