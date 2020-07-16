import pytest

from d3rlpy.algos.bear import BEAR
from tests import performance_test
from .algo_test import algo_tester, algo_update_tester
from .algo_test import algo_pendulum_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, 'pixel', 'min_max', 'standard'])
def test_bear(observation_shape, action_size, q_func_type, scaler):
    bear = BEAR(q_func_type=q_func_type, scaler=scaler)
    algo_tester(bear, observation_shape)
    algo_update_tester(bear, observation_shape, action_size)


@pytest.mark.skip(reason='BEAR is computationally expensive.')
def test_bear_performance():
    bear = BEAR(n_epochs=5)
    algo_pendulum_tester(bear, n_trials=3)
