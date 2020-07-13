import pytest

from d3rlpy.algos.cql import CQL
from tests import performance_test
from .algo_test import algo_tester, algo_update_tester, algo_pendulum_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
def test_cql(observation_shape, action_size, q_func_type):
    cql = CQL(q_func_type=q_func_type)
    algo_tester(cql)
    algo_update_tester(cql, observation_shape, action_size)


@pytest.mark.skip(reason='CQL is computationally expensive.')
def test_cql_performance():
    cql = CQL(n_epochs=5)
    algo_pendulum_tester(cql, n_trials=3)
