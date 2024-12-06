"""
This file contains unit tests targetting the Sim class.
"""

import os
import numpy as np
import pandas as pd

class controller_testing(object):

    def __init__(self, sampling_time: int | float=1.0) -> None:
        
        self.Ts = sampling_time;

        return;

    def __call__(self, inputs: np.ndarray) -> np.ndarray:

        return np.array([0]);

class observer_testing(object):

    def __init__(self) -> None:

        return;

    def __call__(self, measurements: np.ndarray) -> np.ndarray:

        return measurements * 2;

def test_setters(instantiate_sim: callable):
    """
    Test controller and state estimator setter methods.
    """

    sim = instantiate_sim;
    controller = controller_testing();
    state_estimator = observer_testing();
    sim.set_controller(controller);
    sim.set_state_estimator(state_estimator);

    assert isinstance(sim._controller, controller_testing);
    assert isinstance(sim._state_estimator, observer_testing);

    return;

def test_save(instantiate_sim: callable):
    """
    Test save method.
    """

    sim = instantiate_sim;

    sim_data = sim.run();
    file_name = "sim_data";
    formats = ["csv", "pickle", "parquet", "feather"];

    for format in formats:

        sim.save(file_name, format);
    
    pd.testing.assert_frame_equal(pd.read_csv(f"{file_name}.csv"), sim_data);
    pd.testing.assert_frame_equal(pd.read_pickle(f"{file_name}.pickle"), sim_data);
    pd.testing.assert_frame_equal(pd.read_parquet(f"{file_name}.parquet"), sim_data);
    pd.testing.assert_frame_equal(pd.read_feather(f"{file_name}.feather"), sim_data);

    for format in formats:

        os.remove(f"{file_name}.{format}");

    return;