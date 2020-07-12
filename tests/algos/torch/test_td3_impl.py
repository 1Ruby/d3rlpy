import pytest

from d3rlpy.algos.torch.td3_impl import TD3Impl
from tests.algos.algo_test import torch_impl_tester, DummyScaler


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('actor_learning_rate', [1e-3])
@pytest.mark.parametrize('critic_learning_rate', [1e-3])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('tau', [0.05])
@pytest.mark.parametrize('reguralizing_rate', [0.0])
@pytest.mark.parametrize('n_critics', [2])
@pytest.mark.parametrize('target_smoothing_sigma', [0.2])
@pytest.mark.parametrize('target_smoothing_clip', [0.5])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('q_func_type', ['mean', 'qr', 'iqn', 'fqf'])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
def test_td3_impl(observation_shape, action_size, actor_learning_rate,
                  critic_learning_rate, gamma, tau, reguralizing_rate,
                  n_critics, target_smoothing_sigma, target_smoothing_clip,
                  eps, use_batch_norm, q_func_type, scaler):
    impl = TD3Impl(observation_shape,
                   action_size,
                   actor_learning_rate,
                   critic_learning_rate,
                   gamma,
                   tau,
                   reguralizing_rate,
                   n_critics,
                   target_smoothing_sigma,
                   target_smoothing_clip,
                   eps,
                   use_batch_norm,
                   q_func_type=q_func_type,
                   use_gpu=False,
                   scaler=scaler)
    torch_impl_tester(impl,
                      discrete=False,
                      deterministic_best_action=q_func_type != 'iqn')
