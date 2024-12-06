import pytest

from simulinkwrapper.simulations import Sim

@pytest.fixture(scope="session")
def instantiate_sim():

    model_path = "C:/Users/ml4/Desktop/Projects/Repos/_simulink/WT_pitch_actuator_no_sat";
    model_name = "WT_pitch_actuator_no_sat";
    vars = ["beta"];

    sim = Sim(model_name, model_path, vars, stop_time=0.1);

    yield sim;

    print("Disconnected successfully.");

    sim.disconnect();