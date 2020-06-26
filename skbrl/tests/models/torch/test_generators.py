import pytest
import torch
import torch.nn.functional as F

from skbrl.models.torch.generators import create_conditional_vae
from skbrl.models.torch.generators import create_discrete_classifier
from skbrl.models.torch.generators import ConditionalVAE
from skbrl.models.torch.generators import DiscreteClassifier
from skbrl.tests.models.torch.model_test import check_parameter_updates
from skbrl.tests.models.torch.model_test import DummyHead


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('latent_size', [32])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_create_conditional_vae(observation_shape, action_size, latent_size,
                                batch_size, use_batch_norm):
    vae = create_conditional_vae(observation_shape, action_size, latent_size,
                                 use_batch_norm)

    assert isinstance(vae, ConditionalVAE)

    x = torch.rand((batch_size, ) + observation_shape)
    action = torch.rand(batch_size, action_size)
    y = vae(x, action)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('observation_shape', [(4, 84, 84), (100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
@pytest.mark.parametrize('use_batch_norm', [False, True])
def test_create_discrete_classifier(observation_shape, action_size, batch_size,
                                    use_batch_norm):
    classifier = create_discrete_classifier(observation_shape, action_size,
                                            use_batch_norm)

    assert isinstance(classifier, DiscreteClassifier)

    x = torch.rand((batch_size, ) + observation_shape)
    y = classifier(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('latent_size', [32])
@pytest.mark.parametrize('beta', [0.5])
@pytest.mark.parametrize('batch_size', [32])
def test_conditional_vae(feature_size, action_size, latent_size, beta,
                         batch_size):
    encoder_head = DummyHead(feature_size, action_size, True)
    decoder_head = DummyHead(feature_size, latent_size, True)
    vae = ConditionalVAE(encoder_head, decoder_head, beta)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = vae(x, action)
    assert y.shape == (batch_size, action_size)

    # check encode
    dist = vae.encode(x, action)
    assert isinstance(dist, torch.distributions.Normal)
    assert dist.mean.shape == (batch_size, latent_size)

    # check decode
    latent = torch.rand(batch_size, latent_size)
    y = vae.decode(x, latent)
    assert y.shape == (batch_size, action_size)

    # TODO: test vae.compute_likelihood_loss(x, action)

    # check layer connections
    check_parameter_updates(vae, (x, action))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
def test_discrete_classifier(feature_size, action_size, batch_size):
    head = DummyHead(feature_size)
    classifier = DiscreteClassifier(head, action_size)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    y = classifier(x)
    assert torch.allclose(y.exp().sum(dim=1), torch.ones(batch_size))
    y, logits = classifier(x, with_logits=True)
    assert torch.allclose(y, F.log_softmax(logits))

    action = torch.randint(low=0, high=action_size - 1, size=(batch_size, ))
    loss = classifier.compute_cross_entropy(x, action)
    assert torch.allclose(loss, F.nll_loss(y, action))
