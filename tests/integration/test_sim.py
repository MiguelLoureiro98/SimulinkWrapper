"""
This file contains tests targetting the simulations.py module.
"""

import pytest
import numpy as np
import pandas as pd
from simulinkwrapper.simulations import Sim
from simulinkwrapper.noise import gen_noise_signal

class controller_lookahead(object):

    def __init__(self, sampling_time=0.001):

        self.Ts = sampling_time;

        return;

    def __call__(self, inputs: np.ndarray, refs: np.ndarray) -> np.ndarray:

        return np.array([refs[0, :].sum()]); # make it multi-input??? multiple references???

class dummy_estimator():

    def __init__(self):

        return;

    def __call__(self, measured_vars: np.ndarray):

        return measured_vars;

def test_var_step(): # needs a new model
    """
    Test run method with a variable-step solver.
    """

    data = pd.read_csv("C:/Users/ml4/Desktop/Projects/Repos/SimulinkWrapper/tests/data/WT_pitch_actuator_no_sat_VarStep.csv");

    model_path = "C:/Users/ml4/Desktop/Projects/Repos/_simulink/WT_pitch_actuator_no_sat_VarStep";
    model_name = "WT_pitch_actuator_no_sat_VarStep";
    vars = ["beta"];
    ctrl = ["u"];
    ref = np.ones(shape=(1, 10001));

    sim = Sim(model_name, model_path, vars, ctrl, reference_signals=ref, stop_time=10.0);

    sim_data = sim.run();
    sim.disconnect();

    assert (sim_data["t"] - data["t"]).max() == pytest.approx(0.0, abs=1e-7);
    assert (sim_data["beta"] - data["beta"]).max() == pytest.approx(0.0, abs=1e-7);

    return;

def test_fixed_step():
    """
    Test run method with a fixed-step solver.
    """

    data = pd.read_csv("C:/Users/ml4/Desktop/Projects/Repos/SimulinkWrapper/tests/data/WT_pitch_actuator_no_sat_Step_0.8s.csv");

    model_path = "C:/Users/ml4/Desktop/Projects/Repos/_simulink/WT_pitch_actuator_no_sat";
    model_name = "WT_pitch_actuator_no_sat";
    vars = ["beta"];
    ctrl = ["u"];
    ref = np.ones(shape=(1, 801));

    sim = Sim(model_name, model_path, vars, ctrl, reference_signals=ref, stop_time=0.8, solver_type="fixed_step");

    sim_data = sim.run();
    sim.disconnect();

    assert (sim_data["t"] - data["t"]).max() == pytest.approx(0.0, abs=1e-5);
    assert (sim_data["beta"] - data["beta"]).max() == pytest.approx(0.0, abs=1e-5);

    return;

def test_MIMO(): # needs a new model
    """
    Test run method with multiple inputs and outputs.
    """

    data = pd.read_csv("C:/Users/ml4/Desktop/Projects/Repos/SimulinkWrapper/tests/data/WT_pitch_actuator_no_sat_MIMO.csv");

    model_path = "C:/Users/ml4/Desktop/Projects/Repos/_simulink/WT_pitch_actuator_no_sat_MIMO";
    model_name = "WT_pitch_actuator_no_sat_MIMO";
    vars = ["beta", "beta_dot"];
    ctrl = ["u1", "u2"];
    ref = np.ones(shape=(2, 801));

    sim = Sim(model_name, model_path, vars, ctrl, reference_signals=ref, stop_time=0.8, solver_type="fixed_step");

    sim_data = sim.run();
    sim.disconnect();

    assert (sim_data["t"] - data["t"]).max() == pytest.approx(0.0, abs=1e-5);
    assert (sim_data["beta"] - data["beta"]).max() == pytest.approx(0.0, abs=1e-5);
    assert (sim_data["beta_dot"] - data["beta_dot"]).max() == pytest.approx(0.0, abs=1e-5);

    return;

def test_observer():
    """
    Test run method with a state observer/estimator.
    """

    data = pd.read_csv("C:/Users/ml4/Desktop/Projects/Repos/SimulinkWrapper/tests/data/WT_pitch_actuator_no_sat_Step_0.8s.csv");

    model_path = "C:/Users/ml4/Desktop/Projects/Repos/_simulink/WT_pitch_actuator_no_sat";
    model_name = "WT_pitch_actuator_no_sat";
    vars = ["beta"];
    ctrl = ["u"];
    ref = np.ones(shape=(1, 801));

    sim = Sim(model_name, model_path, vars, ctrl, state_estimator=dummy_estimator(), reference_signals=ref, stop_time=0.8, solver_type="fixed_step");

    sim_data = sim.run();
    sim.disconnect();

    assert (sim_data["t"] - data["t"]).max() == pytest.approx(0.0, abs=1e-7);
    assert (sim_data["beta"] - data["beta"]).max() == pytest.approx(0.0, abs=1e-7);

    return;

def test_varying_params():
    """
    Test run method with varying system parameters.
    """

    data = pd.read_csv("C:/Users/ml4/Desktop/Projects/Repos/SimulinkWrapper/tests/data/WT_pitch_actuator_no_sat_varying_params.csv");

    model_path = "C:/Users/ml4/Desktop/Projects/Repos/_simulink/WT_pitch_actuator_no_sat_varying_params";
    model_name = "WT_pitch_actuator_no_sat_varying_params";
    vars = ["beta"];
    ctrl = ["u"];
    ref = np.ones(shape=(1, 801));
    freq = np.hstack((np.full(shape=201, fill_value=16.5), np.full(shape=600, fill_value=11.1)));
    eps = np.hstack((np.full(shape=201, fill_value=0.1), np.full(shape=600, fill_value=0.6)));

    sim = Sim(model_name, model_path, vars, ctrl, varying_parameters={"natural_frequency": freq, "damping_ratio": eps}, reference_signals=ref, stop_time=0.8);

    sim_data = sim.run();
    sim.disconnect();

    assert (sim_data["t"] - data["t"]).max() == pytest.approx(0.0, abs=1e-2);
    assert (sim_data["beta"] - data["beta"]).max() == pytest.approx(0.0, abs=1e-2);

    return;

def test_lookahead(): # control action will be 5
    """
    Test run method when the reference is known a priori.
    """

    data = pd.read_csv("C:/Users/ml4/Desktop/Projects/Repos/SimulinkWrapper/tests/data/WT_pitch_actuator_no_sat_lookahead.csv");

    model_path = "C:/Users/ml4/Desktop/Projects/Repos/_simulink/WT_pitch_actuator_no_sat_VarStep";
    model_name = "WT_pitch_actuator_no_sat_VarStep";
    vars = ["beta"];
    ctrl = ["u"];
    ref = np.ones(shape=(2, 10001));

    sim = Sim(model_name, model_path, vars, ctrl, controller=controller_lookahead(), reference_signals=ref, stop_time=10.0);

    sim_data = sim.run();
    sim.disconnect();

    assert (sim_data["t"] - data["t"]).max() == pytest.approx(0.0, abs=1e-7);
    assert (sim_data["beta"] - data["beta"]).max() == pytest.approx(0.0, abs=1e-7);

    return;

def test_noise():
    """
    Test run method when noise is present.
    """

    data = pd.read_csv("C:/Users/ml4/Desktop/Projects/Repos/SimulinkWrapper/tests/data/WT_pitch_actuator_no_sat_Step_0.8s.csv");

    model_path = "C:/Users/ml4/Desktop/Projects/Repos/_simulink/WT_pitch_actuator_no_sat";
    model_name = "WT_pitch_actuator_no_sat";
    vars = ["beta"];
    ctrl = ["u"];
    ref = np.ones(shape=(1, 801));
    noise = np.ravel(gen_noise_signal(4, 801));

    sim = Sim(model_name, model_path, vars, ctrl, reference_signals=ref, stop_time=0.8, solver_type="fixed_step", noise_signals={"beta": noise});

    sim_data = sim.run();
    sim.disconnect();
    data["beta"] += noise;
    sim_data["beta"] += noise;

    assert (sim_data["beta"] - data["beta"]).max() == pytest.approx(0.0, abs=1e-5);

    return;

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