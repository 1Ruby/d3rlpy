import pytest

from d3rlpy.algos.torch.crr_impl import CRRImpl
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory, QRQFunctionFactory
from tests.algos.algo_impl_test import impl_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("actor_learning_rate", [1e-3])
@pytest.mark.parametrize("critic_learning_rate", [1e-3])
@pytest.mark.parametrize("actor_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("critic_optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("beta", [1.0])
@pytest.mark.parametrize("n_action_samples", [10])
@pytest.mark.parametrize("advantage_type", ["mean"])
@pytest.mark.parametrize("weight_type", ["exp"])
@pytest.mark.parametrize("max_weight", [20.0])
@pytest.mark.parametrize("n_critics", [1])
@pytest.mark.parametrize("tau", [5e-3])
def test_crr_impl(
    observation_shape,
    action_size,
    actor_learning_rate,
    critic_learning_rate,
    actor_optim_factory,
    critic_optim_factory,
    encoder_factory,
    q_func_factory,
    gamma,
    beta,
    n_action_samples,
    advantage_type,
    weight_type,
    max_weight,
    n_critics,
    tau,
):
    impl = CRRImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        actor_optim_factory=actor_optim_factory,
        critic_optim_factory=critic_optim_factory,
        actor_encoder_factory=encoder_factory,
        critic_encoder_factory=encoder_factory,
        q_func_factory=q_func_factory,
        gamma=gamma,
        beta=beta,
        n_action_samples=n_action_samples,
        advantage_type=advantage_type,
        weight_type=weight_type,
        max_weight=max_weight,
        n_critics=n_critics,
        tau=tau,
        device="cpu:0",
    )
    impl_tester(impl, discrete=False)
