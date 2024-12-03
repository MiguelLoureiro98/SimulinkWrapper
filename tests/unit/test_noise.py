"""
This file contains unit tests targetting the noise.py module.
"""

import pytest
from simulinkwrapper.noise import gen_noise_signal

@pytest.fixture
def gen_signals():

    return (gen_noise_signal(4, 10000, 1), gen_noise_signal(9, 10000, 1), gen_noise_signal(4, 1000000, 1));

def test_length(gen_signals: callable):

    """
    Check whether the signals have the specified length.
    """

    signal1, signal2, signal3 = gen_signals;

    assert signal1.shape[0] == 1;
    assert signal1.shape[1] == 10000;
    assert signal2.shape[0] == 1;
    assert signal2.shape[1] == 10000;
    assert signal3.shape[0] == 1;
    assert signal3.shape[1] == 1000000;

    return;

def test_mean(gen_signals: callable):

    """
    Check whether the signals have zero mean.
    """

    signal1, signal2, signal3 = gen_signals;

    assert signal1.mean() == pytest.approx(0.0, abs=1.0e-1);
    assert signal2.mean() == pytest.approx(0.0, abs=1.0e-1);
    assert signal3.mean() == pytest.approx(0.0, abs=1.0e-3);

    return;

def test_power(gen_signals: callable):

    """
    Check whether the signals have the specified power.
    """

    signal1, signal2, signal3 = gen_signals;

    assert signal1.std() == pytest.approx(2.0, abs=1.0e-2);
    assert signal2.std() == pytest.approx(3.0, abs=1.0e-2);
    assert signal3.std() == pytest.approx(2.0, abs=1.0e-2);

    return;