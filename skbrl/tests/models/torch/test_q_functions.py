import numpy as np
import pytest
import torch

from skbrl.models.torch.q_functions import create_discrete_q_function
from skbrl.models.torch.q_functions import create_continuous_q_function
from skbrl.models.torch.q_functions import reduce_ensemble
from skbrl.models.torch.q_functions import reduce_quantile_ensemble
from skbrl.models.torch.q_functions import QRQFunction
from skbrl.models.torch.q_functions import ContinuousQRQFunction
from skbrl.models.torch.q_functions import DiscreteQFunction
from skbrl.models.torch.q_functions import EnsembleDiscreteQFunction
from skbrl.models.torch.q_functions import ContinuousQFunction
from skbrl.models.torch.q_functions import EnsembleContinuousQFunction
from skbrl.tests.models.torch.model_test import check_parameter_updates
from skbrl.tests.models.torch.model_test import DummyHead


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('n_ensembles', [1, 2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('use_batch_norm', [False, True])
@pytest.mark.parametrize('use_quantile_regression', [False, True])
def test_create_discrete_q_function(observation_shape, action_size, batch_size,
                                    n_ensembles, n_quantiles, use_batch_norm,
                                    use_quantile_regression):
    q_func = create_discrete_q_function(observation_shape, action_size,
                                        n_ensembles, n_quantiles,
                                        use_batch_norm,
                                        use_quantile_regression)

    if n_ensembles == 1:
        if use_quantile_regression:
            assert isinstance(q_func, QRQFunction)
        else:
            assert isinstance(q_func, DiscreteQFunction)
        assert q_func.head.use_batch_norm == use_batch_norm
    else:
        assert isinstance(q_func, EnsembleDiscreteQFunction)
        for f in q_func.q_funcs:
            assert f.head.use_batch_norm == use_batch_norm
            if use_quantile_regression:
                assert isinstance(f, QRQFunction)
            else:
                assert isinstance(f, DiscreteQFunction)

    x = torch.rand((batch_size, ) + observation_shape)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('n_ensembles', [1, 2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('use_batch_norm', [False, True])
@pytest.mark.parametrize('use_quantile_regression', [False, True])
def test_create_continuous_q_function(observation_shape, action_size,
                                      batch_size, n_ensembles, n_quantiles,
                                      use_batch_norm, use_quantile_regression):
    q_func = create_continuous_q_function(observation_shape, action_size,
                                          n_ensembles, n_quantiles,
                                          use_batch_norm,
                                          use_quantile_regression)

    if n_ensembles == 1:
        if use_quantile_regression:
            assert isinstance(q_func, ContinuousQRQFunction)
        else:
            assert isinstance(q_func, ContinuousQFunction)
        assert q_func.head.use_batch_norm == use_batch_norm
    else:
        assert isinstance(q_func, EnsembleContinuousQFunction)
        for f in q_func.q_funcs:
            if use_quantile_regression:
                assert isinstance(f, ContinuousQRQFunction)
            else:
                assert isinstance(f, ContinuousQFunction)
            assert f.head.use_batch_norm == use_batch_norm

    x = torch.rand((batch_size, ) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)


@pytest.mark.parametrize('n_ensembles', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('reduction', ['min', 'max', 'mean', 'none'])
def test_reduce_ensemble(n_ensembles, batch_size, reduction):
    y = torch.rand(n_ensembles, batch_size, 1)
    ret = reduce_ensemble(y, reduction)
    if reduction == 'min':
        assert ret.shape == (batch_size, 1)
        assert torch.allclose(ret, y.min(dim=0).values)
    elif reduction == 'max':
        assert ret.shape == (batch_size, 1)
        assert torch.allclose(ret, y.max(dim=0).values)
    elif reduction == 'mean':
        assert ret.shape == (batch_size, 1)
        assert torch.allclose(ret, y.mean(dim=0))
    elif reduction == 'none':
        assert ret.shape == (n_ensembles, batch_size, 1)
        assert (ret == y).all()


@pytest.mark.parametrize('n_ensembles', [2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('reduction', ['min', 'max'])
def test_reduce_quantile_ensemble(n_ensembles, n_quantiles, batch_size,
                                  reduction):
    y = torch.rand(n_ensembles, batch_size, n_quantiles)
    ret = reduce_quantile_ensemble(y, reduction)
    mean = y.mean(dim=2)
    if reduction == 'min':
        assert ret.shape == (batch_size, n_quantiles)
        indices = mean.min(dim=0).indices
        assert torch.allclose(ret, y[indices, torch.arange(batch_size)])
    elif reduction == 'max':
        assert ret.shape == (batch_size, n_quantiles)
        indices = mean.max(dim=0).indices
        assert torch.allclose(ret, y[indices, torch.arange(batch_size)])


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
def test_qrq_function(feature_size, action_size, n_quantiles, batch_size,
                      gamma):
    head = DummyHead(feature_size)
    q_func = QRQFunction(head, action_size, n_quantiles)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)

    action = torch.randint(high=action_size, size=(batch_size, ))
    target = q_func.compute_target(x, action)
    quantiles = q_func(x, as_quantiles=True)
    assert target.shape == (batch_size, n_quantiles)
    assert (quantiles[torch.arange(batch_size), action] == target).all()

    # TODO: check quantile huber loss

    # check layer connection
    check_parameter_updates(q_func, (x, ))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_quantiles', [200])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
def test_continuous_qrq_function(feature_size, action_size, n_quantiles,
                                 batch_size, gamma):
    head = DummyHead(feature_size, action_size, concat=True)
    q_func = ContinuousQRQFunction(head, n_quantiles)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)

    target = q_func.compute_target(x, action)
    quantiles = q_func(x, action, as_quantiles=True)
    assert target.shape == (batch_size, n_quantiles)
    assert (target == quantiles).all()

    # TODO: check quantile huber loss

    # check layer connection
    check_parameter_updates(q_func, (x, action))


def ref_huber_loss(a, b):
    abs_diff = np.abs(a - b).reshape((-1, ))
    l2_diff = ((a - b)**2).reshape((-1, ))
    huber_diff = np.zeros_like(abs_diff)
    huber_diff[abs_diff < 1.0] = 0.5 * l2_diff[abs_diff < 1.0]
    huber_diff[abs_diff >= 1.0] = abs_diff[abs_diff >= 1.0] - 0.5
    return np.mean(huber_diff)


def filter_by_action(value, action, action_size):
    act_one_hot = np.identity(action_size)[np.reshape(action, (-1, ))]
    return (value * act_one_hot).sum(axis=1)


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
def test_discrete_q_function(feature_size, action_size, batch_size, gamma):
    head = DummyHead(feature_size)
    q_func = DiscreteQFunction(head, action_size)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = q_func(x)
    assert y.shape == (batch_size, action_size)

    # check compute_target
    action = torch.randint(high=action_size, size=(batch_size, ))
    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, 1)
    assert torch.allclose(y[torch.arange(batch_size), action], target.view(-1))

    # check td calculation
    q_tp1 = np.random.random((batch_size, 1))
    rew_tp1 = np.random.random((batch_size, 1))
    target = rew_tp1 + gamma * q_tp1

    obs_t = torch.rand(batch_size, feature_size)
    act_t = np.random.randint(action_size, size=(batch_size, 1))
    q_t = filter_by_action(q_func(obs_t).detach().numpy(), act_t, action_size)
    ref_loss = ref_huber_loss(q_t.reshape((-1, 1)), target)

    loss = q_func.compute_td(obs_t,
                             torch.tensor(act_t, dtype=torch.int64),
                             torch.tensor(rew_tp1, dtype=torch.float32),
                             torch.tensor(q_tp1, dtype=torch.float32),
                             gamma=gamma)

    assert np.allclose(loss.detach().numpy(), ref_loss)

    # check layer connection
    check_parameter_updates(q_func, (x, ))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('ensemble_size', [5])
@pytest.mark.parametrize('use_quantile_regression', [True, False])
@pytest.mark.parametrize('n_quantiles', [200])
def test_ensemble_discrete_q_function(feature_size, action_size, batch_size,
                                      gamma, ensemble_size,
                                      use_quantile_regression, n_quantiles):
    q_funcs = []
    for _ in range(ensemble_size):
        head = DummyHead(feature_size)
        if use_quantile_regression:
            q_func = QRQFunction(head, action_size, n_quantiles)
        else:
            q_func = DiscreteQFunction(head, action_size)
        q_funcs.append(q_func)
    q_func = EnsembleDiscreteQFunction(q_funcs)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    values = q_func(x, 'none')
    assert values.shape == (ensemble_size, batch_size, action_size)

    # check compute_target
    action = torch.randint(high=action_size, size=(batch_size, ))
    target = q_func.compute_target(x, action)
    if use_quantile_regression:
        assert target.shape == (batch_size, n_quantiles)
    else:
        assert target.shape == (batch_size, 1)
        min_values = values.min(dim=0).values
        assert torch.allclose(min_values[torch.arange(batch_size), action],
                              target.view(-1))

    # check reductions
    assert torch.allclose(values.min(dim=0).values, q_func(x, 'min'))
    assert torch.allclose(values.max(dim=0).values, q_func(x, 'max'))
    assert torch.allclose(values.mean(dim=0), q_func(x, 'mean'))

    # check td computation
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.randint(0,
                          action_size,
                          size=(batch_size, 1),
                          dtype=torch.int64)
    rew_tp1 = torch.rand(batch_size, 1)
    if use_quantile_regression:
        q_tp1 = torch.rand(batch_size, n_quantiles)
    else:
        q_tp1 = torch.rand(batch_size, 1)
    ref_td_sum = 0.0
    for i in range(ensemble_size):
        f = q_func.q_funcs[i]
        ref_td_sum += f.compute_td(obs_t, act_t, rew_tp1, q_tp1, gamma)
    loss = q_func.compute_td(obs_t, act_t, rew_tp1, q_tp1, gamma)
    assert torch.allclose(ref_td_sum, loss)

    # check layer connection
    check_parameter_updates(q_func, (x, 'mean'))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
def test_continuous_q_function(feature_size, action_size, batch_size, gamma):
    head = DummyHead(feature_size, action_size, concat=True)
    q_func = ContinuousQFunction(head)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)

    # check compute_target
    target = q_func.compute_target(x, action)
    assert target.shape == (batch_size, 1)
    assert (target == y).all()

    # check td calculation
    q_tp1 = np.random.random((batch_size, 1))
    rew_tp1 = np.random.random((batch_size, 1))
    target = rew_tp1 + gamma * q_tp1

    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.rand(batch_size, action_size)
    q_t = q_func(obs_t, act_t).detach().numpy()
    ref_loss = ((q_t - target)**2).mean()

    loss = q_func.compute_td(obs_t, act_t,
                             torch.tensor(rew_tp1, dtype=torch.float32),
                             torch.tensor(q_tp1, dtype=torch.float32), gamma)

    assert np.allclose(loss.detach().numpy(), ref_loss)

    # check layer connection
    check_parameter_updates(q_func, (x, action))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('gamma', [0.99])
@pytest.mark.parametrize('ensemble_size', [5])
@pytest.mark.parametrize('use_quantile_regression', [True, False])
@pytest.mark.parametrize('n_quantiles', [200])
def test_ensemble_continuous_q_function(feature_size, action_size, batch_size,
                                        gamma, ensemble_size,
                                        use_quantile_regression, n_quantiles):
    q_funcs = []
    for _ in range(ensemble_size):
        head = DummyHead(feature_size, action_size, concat=True)
        if use_quantile_regression:
            q_func = ContinuousQRQFunction(head, n_quantiles)
        else:
            q_func = ContinuousQFunction(head)
        q_funcs.append(q_func)

    q_func = EnsembleContinuousQFunction(q_funcs)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    values = q_func(x, action, 'none')
    assert values.shape == (ensemble_size, batch_size, 1)

    # check compute_target
    target = q_func.compute_target(x, action)
    if use_quantile_regression:
        assert target.shape == (batch_size, n_quantiles)
    else:
        assert target.shape == (batch_size, 1)
        min_values = values.min(dim=0).values
        assert (target == min_values).all()

    # check reductions
    assert torch.allclose(values.min(dim=0).values, q_func(x, action, 'min'))
    assert torch.allclose(values.max(dim=0).values, q_func(x, action, 'max'))
    assert torch.allclose(values.mean(dim=0), q_func(x, action, 'mean'))

    # check td computation
    obs_t = torch.rand(batch_size, feature_size)
    act_t = torch.rand(batch_size, action_size)
    rew_tp1 = torch.rand(batch_size, 1)
    if use_quantile_regression:
        q_tp1 = torch.rand(batch_size, n_quantiles)
    else:
        q_tp1 = torch.rand(batch_size, 1)
    ref_td_sum = 0.0
    for i in range(ensemble_size):
        f = q_func.q_funcs[i]
        ref_td_sum += f.compute_td(obs_t, act_t, rew_tp1, q_tp1, gamma)
    loss = q_func.compute_td(obs_t, act_t, rew_tp1, q_tp1, gamma)
    assert torch.allclose(ref_td_sum, loss)

    # check layer connection
    check_parameter_updates(q_func, (x, action, 'mean'))
