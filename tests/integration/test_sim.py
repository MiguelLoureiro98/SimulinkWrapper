"""
This file contains tests targetting the simulations.py module.
"""

import pytest

if __name__ == "__main__":

    from simulinkwrapper.simulations import Sim

    model = "../_simulink/WT_pitch_actuator_no_sat";
    vars = ["beta"];
    ctrl = ["u"];

    sim = Sim(model, vars, ctrl);

    print("Connected successfully...");

    sim.disconnect();

    print("Disconnected successfully...");