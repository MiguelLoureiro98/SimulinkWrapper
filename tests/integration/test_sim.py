"""
This file contains tests targetting the simulations.py module.
"""

import pytest
import numpy as np
import pandas as pd

if __name__ == "__main__":

    from simulinkwrapper.simulations import Sim

    model = "../_simulink/WT_pitch_actuator_no_sat";
    vars = ["beta"];
    ctrl = ["u"];
    ref = np.ones(shape=(1, 10001));

    sim = Sim(model, vars, ctrl, reference_signals=ref);

    print("Connected successfully...");

    results = sim.run();
    sim.plot("beta", 0);

    sim.disconnect();

    print("Disconnected successfully...");