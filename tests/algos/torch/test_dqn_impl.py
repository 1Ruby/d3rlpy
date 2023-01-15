import pytest

from d3rlpy.algos.torch.dqn_impl import DoubleDQNImpl, DQNImpl
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory, QRQFunctionFactory
from tests.algos.algo_test import (
    DummyObservationScaler,
    DummyRewardScaler,
    torch_impl_tester,
)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [2.5e-4])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("observation_scaler", [None, DummyObservationScaler()])
@pytest.mark.parametrize("reward_scaler", [None, DummyRewardScaler()])
def test_dqn_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    n_critics,
    observation_scaler,
    reward_scaler,
):
    impl = DQNImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        learning_rate=learning_rate,
        optim_factory=optim_factory,
        encoder_factory=encoder_factory,
        q_func_factory=q_func_factory,
        gamma=gamma,
        n_critics=n_critics,
        device="cpu:0",
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    torch_impl_tester(impl, discrete=True)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [2.5e-4])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("observation_scaler", [None, DummyObservationScaler()])
@pytest.mark.parametrize("reward_scaler", [None, DummyRewardScaler()])
def test_double_dqn_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    n_critics,
    observation_scaler,
    reward_scaler,
):
    impl = DoubleDQNImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        learning_rate=learning_rate,
        optim_factory=optim_factory,
        encoder_factory=encoder_factory,
        q_func_factory=q_func_factory,
        gamma=gamma,
        n_critics=n_critics,
        device="cpu:0",
        observation_scaler=observation_scaler,
        reward_scaler=reward_scaler,
    )
    torch_impl_tester(impl, discrete=True)
