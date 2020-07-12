from abc import ABCMeta, abstractmethod
from .base import AlgoBase


class IDDPGImpl(metaclass=ABCMeta):
    @abstractmethod
    def update_critic(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        pass

    @abstractmethod
    def update_actor(self, obs_t):
        pass

    @abstractmethod
    def update_actor_target(self):
        pass

    @abstractmethod
    def update_critic_target(self):
        pass


class DDPG(AlgoBase):
    """ Deep Deterministic Policy Gradients algorithm.

    DDPG is an actor-critic algorithm that trains a Q function parametrized
    with :math:`\\theta` and a policy function parametrized with :math:`\\phi`.

    .. math::

        L(\\theta) = \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D} [(r_{t+1}
            + \gamma Q_{\\theta'}(s_{t+1}, \pi_{\phi'}(s_{t+1}))
            - Q_\\theta(s_t, a_t))^2]

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D} [Q_\\theta(s_t, \pi_\phi(s_t))]

    where :math:`\\theta'` and :math:`\phi` are the target network parameters.
    There target network parameters are updated every iteration.

    .. math::

        \\theta' \\gets \\tau \\theta + (1 - \\tau) \\theta'

        \\phi' \\gets \\tau \\phi + (1 - \\tau) \\phi'

    References:
        * `Silver et al., Deterministic policy gradient algorithms.
          <http://proceedings.mlr.press/v32/silver14.html>`_
        * `Lillicrap et al., Continuous control with deep reinforcement
          learning. <https://arxiv.org/abs/1509.02971>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q function.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        reguralizing_rate (float): reguralizing term for policy function.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function. Available options are
            `['mean', 'qr', 'iqn', 'fqf']`.
        n_epochs (int): the number of epochs to train.
        use_gpu (bool): flag to use GPU.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        impl (d3rlpy.algos.ddpg.IDDPGImpl): algorithm implementation.

    Attributes:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q function.
        batch_size (int): mini-batch size.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        reguralizing_rate (float): reguralizing term for policy function.
        eps (float): :math:`\\epsilon` for Adam optimizer.
        use_batch_norm (bool): flag to insert batch normalization layers.
        q_func_type (str): type of Q function.
        n_epochs (int): the number of epochs to train.
        use_gpu (bool): flag to use GPU.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor.
        impl (d3rlpy.algos.ddpg.IDDPGImpl): algorithm implementation.

    """
    def __init__(self,
                 actor_learning_rate=1e-3,
                 critic_learning_rate=1e-3,
                 batch_size=100,
                 gamma=0.99,
                 tau=0.005,
                 reguralizing_rate=1e-10,
                 eps=1e-8,
                 use_batch_norm=False,
                 q_func_type='mean',
                 n_epochs=1000,
                 use_gpu=False,
                 scaler=None,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size, scaler)
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.reguralizing_rate = reguralizing_rate
        self.eps = eps
        self.use_batch_norm = use_batch_norm
        self.q_func_type = q_func_type
        self.use_gpu = use_gpu
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        from .torch.ddpg_impl import DDPGImpl
        self.impl = DDPGImpl(observation_shape=observation_shape,
                             action_size=action_size,
                             actor_learning_rate=self.actor_learning_rate,
                             critic_learning_rate=self.critic_learning_rate,
                             gamma=self.gamma,
                             tau=self.tau,
                             reguralizing_rate=self.reguralizing_rate,
                             eps=self.eps,
                             use_batch_norm=self.use_batch_norm,
                             q_func_type=self.q_func_type,
                             use_gpu=self.use_gpu,
                             scaler=self.scaler)

    def update(self, epoch, itr, batch):
        critic_loss = self.impl.update_critic(batch.observations,
                                              batch.actions,
                                              batch.next_rewards,
                                              batch.next_observations,
                                              batch.terminals)
        actor_loss = self.impl.update_actor(batch.observations)
        self.impl.update_critic_target()
        self.impl.update_actor_target()
        return critic_loss, actor_loss

    def _get_loss_labels(self):
        return ['critic_loss', 'actor_loss']
