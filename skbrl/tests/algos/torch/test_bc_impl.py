import pytest

from skbrl.algos.torch.bc_impl import BCImpl
from skbrl.tests.algos.algo_test import torch_impl_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [1e-3])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('use_batch_norm', [True, False])
def test_bc_impl(observation_shape, action_size, learning_rate, eps,
                 use_batch_norm):
    impl = BCImpl(observation_shape,
                  action_size,
                  learning_rate,
                  eps,
                  use_batch_norm,
                  use_gpu=False)
    torch_impl_tester(impl, discrete=False, imitator=True)
