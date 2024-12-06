"""
This file contains tests targetting the simulations.py module.
"""

import pytest
import numpy as np
import pandas as pd

if __name__ == "__main__":

    from simulinkwrapper.simulations import Sim

    model_path = "C:/Users/ml4/Desktop/Projects/Repos/_simulink/WT_pitch_actuator_no_sat";
    model_name = "WT_pitch_actuator_no_sat";
    vars = ["beta"];
    ctrl = ["u"];
    ref = np.ones(shape=(1, 10001));

    sim = Sim(model_name, model_path, vars, ctrl, reference_signals=ref, stop_time=10.0);

    print("Connected successfully...");

    results = sim.run();
    sim.plot("beta");

    sim.disconnect();

    print("Disconnected successfully...");