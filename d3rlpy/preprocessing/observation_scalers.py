from dataclasses import field
from typing import Any, Dict, Optional, Sequence, Type

import gym
import numpy as np
import torch
from dataclasses_json import config

from ..dataset import EpisodeBase

__all__ = [
    "ObservationScaler",
    "PixelObservationScaler",
    "MinMaxObservationScaler",
    "StandardObservationScaler",
    "OBSERVATION_SCALER_LIST",
    "register_observation_scaler",
    "create_observation_scaler",
    "make_observation_scaler_field",
]


class ObservationScaler:
    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            episodes: list of episodes.

        """
        raise NotImplementedError

    def fit_with_env(self, env: gym.Env) -> None:
        """Gets scaling parameters from environment.

        Args:
            env: gym environment.

        """
        raise NotImplementedError

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns processed observations.

        Args:
            x: observation.

        Returns:
            processed observation.

        """
        raise NotImplementedError

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reversely transformed observations.

        Args:
            x: observation.

        Returns:
            reversely transformed observation.

        """
        raise NotImplementedError

    @staticmethod
    def get_type() -> str:
        """Returns a scaler type.

        Returns:
            scaler type.

        """
        raise NotImplementedError

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns scaling parameters.

        Args:
            deep: flag to deeply copy objects.

        Returns:
            scaler parameters.

        """
        raise NotImplementedError


class PixelObservationScaler(ObservationScaler):
    """Pixel normalization preprocessing.

    .. math::

        x' = x / 255

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with PixelScaler
        cql = CQL(scaler='pixel')

        cql.fit(dataset.episodes)

    """

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        pass

    def fit_with_env(self, env: gym.Env) -> None:
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.0

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x * 255.0).long()

    @staticmethod
    def get_type() -> str:
        return "pixel"

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {}


class MinMaxObservationScaler(ObservationScaler):
    r"""Min-Max normalization preprocessing.

    .. math::

        x' = (x - \min{x}) / (\max{x} - \min{x})

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with MinMaxScaler
        cql = CQL(scaler='min_max')

        # scaler is initialized from the given transitions
        transitions = []
        for episode in dataset.episodes:
            transitions += episode.transitions
        cql.fit(transitions)

    You can also initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import MinMaxScaler

        # initialize manually
        minimum = observations.min(axis=0)
        maximum = observations.max(axis=0)
        scaler = MinMaxScaler(minimum=minimum, maximum=maximum)

        cql = CQL(scaler=scaler)

    Args:
        min (numpy.ndarray): minimum values at each entry.
        max (numpy.ndarray): maximum values at each entry.

    """
    _minimum: Optional[np.ndarray]
    _maximum: Optional[np.ndarray]

    def __init__(
        self,
        maximum: Optional[np.ndarray] = None,
        minimum: Optional[np.ndarray] = None,
    ):
        self._minimum = None
        self._maximum = None
        if maximum is not None and minimum is not None:
            self._minimum = np.asarray(minimum)
            self._maximum = np.asarray(maximum)

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        if self._minimum is not None and self._maximum is not None:
            return

        maximum = np.zeros(episodes[0].observation_shape)
        minimum = np.zeros(episodes[0].observation_shape)
        for i, episode in enumerate(episodes):
            observations = np.asarray(episode.observations)
            max_observation = np.max(observations, axis=0)
            min_observation = np.min(observations, axis=0)
            if i == 0:
                minimum = min_observation
                maximum = max_observation
            else:
                minimum = np.minimum(minimum, min_observation)
                maximum = np.maximum(maximum, max_observation)

        self._minimum = minimum.reshape((1,) + minimum.shape)
        self._maximum = maximum.reshape((1,) + maximum.shape)

    def fit_with_env(self, env: gym.Env) -> None:
        if self._minimum is not None and self._maximum is not None:
            return

        assert isinstance(env.observation_space, gym.spaces.Box)
        shape = env.observation_space.shape
        low = np.asarray(env.observation_space.low)
        high = np.asarray(env.observation_space.high)
        self._minimum = low.reshape((1,) + shape)
        self._maximum = high.reshape((1,) + shape)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(
            self._minimum, dtype=torch.float32, device=x.device
        )
        maximum = torch.tensor(
            self._maximum, dtype=torch.float32, device=x.device
        )
        return (x - minimum) / (maximum - minimum)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(
            self._minimum, dtype=torch.float32, device=x.device
        )
        maximum = torch.tensor(
            self._maximum, dtype=torch.float32, device=x.device
        )
        return ((maximum - minimum) * x) + minimum

    @staticmethod
    def get_type() -> str:
        return "min_max"

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if self._maximum is not None:
            maximum = self._maximum.copy() if deep else self._maximum
        else:
            maximum = None

        if self._minimum is not None:
            minimum = self._minimum.copy() if deep else self._minimum
        else:
            minimum = None

        return {"maximum": maximum, "minimum": minimum}


class StandardObservationScaler(ObservationScaler):
    r"""Standardization preprocessing.

    .. math::

        x' = (x - \mu) / \sigma

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with StandardScaler
        cql = CQL(scaler='standard')

        # scaler is initialized from the given episodes
        transitions = []
        for episode in dataset.episodes:
            transitions += episode.transitions
        cql.fit(transitions)

    You can initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import StandardScaler

        # initialize manually
        mean = observations.mean(axis=0)
        std = observations.std(axis=0)
        scaler = StandardScaler(mean=mean, std=std)

        cql = CQL(scaler=scaler)

    Args:
        mean (numpy.ndarray): mean values at each entry.
        std (numpy.ndarray): standard deviation at each entry.
        eps (float): small constant value to avoid zero-division.

    """
    _mean: Optional[np.ndarray]
    _std: Optional[np.ndarray]
    _eps: float

    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        eps: float = 1e-3,
    ):
        self._mean = None
        self._std = None
        self._eps = eps
        if mean is not None and std is not None:
            self._mean = np.asarray(mean)
            self._std = np.asarray(std)

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        if self._mean is not None and self._std is not None:
            return

        # compute mean
        total_sum = np.zeros(episodes[0].observation_shape)
        total_count = 0
        for episode in episodes:
            total_sum += np.sum(episode.observations, axis=0)
            total_count += episode.size()
        mean = total_sum / total_count

        # compute stdandard deviation
        total_sqsum = np.zeros(episodes[0].observation_shape)
        expanded_mean = mean.reshape((1,) + mean.shape)
        for episode in episodes:
            observations = np.asarray(episode.observations)
            total_sqsum += np.sum((observations - expanded_mean) ** 2, axis=0)
        std = np.sqrt(total_sqsum / total_count)

        self._mean = mean.reshape((1,) + mean.shape)
        self._std = std.reshape((1,) + std.shape)

    def fit_with_env(self, env: gym.Env) -> None:
        if self._mean is not None and self._std is not None:
            return
        raise NotImplementedError(
            "standard scaler does not support fit_with_env."
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None and self._std is not None
        mean = torch.tensor(self._mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self._std, dtype=torch.float32, device=x.device)
        return (x - mean) / (std + self._eps)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None and self._std is not None
        mean = torch.tensor(self._mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self._std, dtype=torch.float32, device=x.device)
        return ((std + self._eps) * x) + mean

    @staticmethod
    def get_type() -> str:
        return "standard"

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if self._mean is not None:
            mean = self._mean.copy() if deep else self._mean
        else:
            mean = None

        if self._std is not None:
            std = self._std.copy() if deep else self._std
        else:
            std = None

        return {"mean": mean, "std": std, "eps": self._eps}


OBSERVATION_SCALER_LIST: Dict[str, Type[ObservationScaler]] = {}


def register_observation_scaler(cls: Type[ObservationScaler]) -> None:
    """Registers scaler class.

    Args:
        cls: scaler class inheriting ``Scaler``.

    """
    type_name = cls.get_type()
    is_registered = type_name in OBSERVATION_SCALER_LIST
    assert not is_registered, f"{type_name} seems to be already registered"
    OBSERVATION_SCALER_LIST[type_name] = cls


def create_observation_scaler(name: str, **kwargs: Any) -> ObservationScaler:
    """Returns registered scaler object.

    Args:
        name: regsitered scaler type name.
        kwargs: scaler arguments.

    Returns:
        scaler object.

    """
    assert (
        name in OBSERVATION_SCALER_LIST
    ), f"{name} seems not to be registered."
    scaler = OBSERVATION_SCALER_LIST[name](**kwargs)
    assert isinstance(scaler, ObservationScaler)
    return scaler


def _encoder(scaler: Optional[ObservationScaler]) -> Dict[str, Any]:
    if scaler is None:
        return {"type": "none", "params": {}}
    return {"type": scaler.get_type(), "params": scaler.get_params()}


def _decoder(dict_config: Dict[str, Any]) -> Optional[ObservationScaler]:
    if dict_config["type"] == "none":
        return None
    return create_observation_scaler(
        dict_config["type"], **dict_config["params"]
    )


def make_observation_scaler_field() -> Optional[ObservationScaler]:
    return field(
        metadata=config(encoder=_encoder, decoder=_decoder), default=None
    )


register_observation_scaler(PixelObservationScaler)
register_observation_scaler(MinMaxObservationScaler)
register_observation_scaler(StandardObservationScaler)
