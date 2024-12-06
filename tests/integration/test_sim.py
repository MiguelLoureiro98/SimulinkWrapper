"""
This file contains tests targetting the simulations.py module.
"""

import numpy as np
import pandas as pd
from simulinkwrapper.simulations import Sim
from simulinkwrapper.noise import gen_noise_signal

class controller_lookahead(object):

    def __init__(self):

        return;

    def __call__(self, inputs: np.ndarray, refs: np.ndarray) -> np.ndarray:

        return np.array([refs.sum()]);

def test_var_step(): # needs a new model

    pass

def test_fixed_step():

    pass

def test_SISO():

    pass

def test_MIMO(): # needs a new model

    pass

def test_observer():

    pass

def test_varying_params():

    pass

def test_lookahead(): # control action will be 5

    pass

if __name__ == "__main__":

    from simulinkwrapper.simulations import Sim

    model_path = "C:/Users/ml4/Desktop/Projects/Repos/_simulink/WT_pitch_actuator_no_sat";
    model_name = "WT_pitch_actuator_no_sat";
    vars = ["beta"];
    ctrl = ["u"];
    ref = np.ones(shape=(1, 801));

    sim = Sim(model_name, model_path, vars, ctrl, reference_signals=ref, stop_time=0.8, solver_type="fixed_step");

    print("Connected successfully...");

    results = sim.run();
    sim.plot("beta", 0);

    sim.disconnect();

    print("Disconnected successfully...");