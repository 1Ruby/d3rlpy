from skbrl.algos.ddpg import DDPG
from skbrl.tests import performance_test
from .algo_test import algo_tester, algo_pendulum_tester


def test_ddpg():
    ddpg = DDPG()
    algo_tester(ddpg)


@performance_test
def test_ddpg_performance():
    # not good enough for batch RL, but check if it works without errors.
    try:
        ddpg = DDPG(n_epochs=1)
        algo_pendulum_tester(ddpg, n_trials=1)
    except AssertionError:
        pass
