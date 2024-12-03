#   Copyright 2024 Miguel Loureiro

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
This module contains classes to help users simulate dynamical systems modelled in Simulink.

Classes
-------
Sim
    Run a simulation in Simulink.

RLSim
    Run a simulation in Simulink as if it were an environment for Reinforcement Learning agents.
"""

import numpy as np
import pandas as pd
import matlab.engine

class Sim(object):

    def __init__(self, 
                 model_path: str,
                 measured_variables: list[str],
                 controlled_variables: list[str],
                 controller: any=None,
                 state_estimator: any=None,
                 varying_parameters: dict[str, np.ndarray] | None=None,
                 reference_signals: np.ndarray | None=None,
                 reference_lookahead: int=1,
                 noise_signals: dict[str, np.ndarray] | None=None,
                 stop_time: int | float=10.0,
                 time_step: int | float=0.001) -> None:
        
        self._model = model_path;
        self._measurements = measured_variables;
        self._control_vars = controlled_variables;
        self._controller = controller;
        self._state_estimator = state_estimator;
        self._varying_params = varying_parameters;
        self._refs = reference_signals;
        self._lookahead = reference_lookahead;
        self._noise = noise_signals;
        self._settings = {"StopTime": stop_time, "FixedStep": time_step};
        self._eng = matlab.engine.start_matlab();
        self._sim_data = {};

        return;

    def run(self) -> pd.DataFrame:
        """
        Run a simulation.

        This method can be used to run a complete simulation of a Simulink model.
        """

        self._reset();
        self._config_sim();

        pass

    def plot(self) -> None:

        pass

    def disconnect(self) -> None:
        """
        Exits the Matlab Engine.

        This method must be called whenever the user does not intend to run more simulations using a class instance.
        """

        self._eng.quit();
    
        return;

    def _config_sim(self) -> None:
        """
        Configure simulation settings.
        """

        self._eng.eval(f"model = {self._model}';", nargout=0);
        self._eng.eval("load_system(model);", nargout=0);

        for (parameter, value) in self._settings.items():

            self._eng.eval("set_param('{}', '{}', '{}');".format(self._model, parameter, value), nargout=0);

        self._eng.eval("set_param('{}', 'SimulationCommand', 'start', 'SimulationCommand', 'pause');".format(self._model), nargout=0);

        return;

    def _reset(self) -> None:
        """
        Reset simulation data.
        """

        self._sim_data = {};

        return;

class RLSim(object):

    def __init__(self) -> None:

        return;