import dataclasses
import io
import pickle
from abc import ABCMeta, abstractmethod
from typing import BinaryIO, Optional, Type, TypeVar, Union, cast

import gym
import torch
from gym.spaces import Discrete

from ._version import __version__
from .constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from .dataset import DatasetInfo, ReplayBuffer, Shape
from .logger import LOG, D3RLPyLogger
from .preprocessing import (
    ActionScaler,
    ObservationScaler,
    RewardScaler,
    make_action_scaler_field,
    make_observation_scaler_field,
    make_reward_scaler_field,
)
from .serializable_config import DynamicConfig, generate_config_registration
from .torch_utility import get_state_dict, map_location, set_state_dict

__all__ = [
    "DeviceArg",
    "ImplBase",
    "save_config",
    "dump_learnable",
    "load_learnable",
    "LearnableBase",
    "LearnableConfig",
    "LearnableConfigWithShape",
    "register_learnable",
]


TLearnable = TypeVar("TLearnable", bound="LearnableBase")
DeviceArg = Optional[Union[bool, int, str]]


class ImplBase(metaclass=ABCMeta):
    _observation_shape: Shape
    _action_size: int
    _device: str

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        device: str,
    ):
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._device = device

    def save_model(self, f: BinaryIO) -> None:
        torch.save(get_state_dict(self), f)

    def load_model(self, f: BinaryIO) -> None:
        chkpt = torch.load(f, map_location=map_location(self._device))
        set_state_dict(self, chkpt)

    @property
    def observation_shape(self) -> Shape:
        return self._observation_shape

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def device(self) -> str:
        return self._device


@dataclasses.dataclass()
class LearnableConfig(DynamicConfig):
    batch_size: int = 256
    gamma: float = 0.99
    observation_scaler: Optional[
        ObservationScaler
    ] = make_observation_scaler_field()
    action_scaler: Optional[ActionScaler] = make_action_scaler_field()
    reward_scaler: Optional[RewardScaler] = make_reward_scaler_field()

    def create(self, device: DeviceArg = False) -> "LearnableBase":
        raise NotImplementedError


register_learnable, make_learnable_field = generate_config_registration(
    LearnableConfig
)


@dataclasses.dataclass()
class LearnableConfigWithShape(DynamicConfig):
    observation_shape: Shape
    action_size: int
    config: LearnableConfig = make_learnable_field()

    def create(self, device: DeviceArg = False) -> "LearnableBase":
        algo = self.config.create(device)
        algo.create_impl(self.observation_shape, self.action_size)
        return algo


def save_config(algo: "LearnableBase", logger: D3RLPyLogger) -> None:
    assert algo.impl
    config = LearnableConfigWithShape(
        observation_shape=algo.impl.observation_shape,
        action_size=algo.impl.action_size,
        config=algo.config,
    )
    logger.add_params(config.serialize_to_dict())


def _process_device(value: DeviceArg) -> str:
    """Checks value and returns PyTorch target device.

    Returns:
        str: target device.

    """
    # isinstance cannot tell difference between bool and int
    if isinstance(value, bool):
        return "cuda:0" if value else "cpu:0"
    if isinstance(value, int):
        return f"cuda:{value}"
    if isinstance(value, str):
        return value
    if value is None:
        return "cpu:0"
    raise ValueError("This argument must be bool, int or str.")


def dump_learnable(algo: "LearnableBase", fname: str) -> None:
    assert algo.impl
    with open(fname, "wb") as f:
        torch_bytes = io.BytesIO()
        algo.impl.save_model(torch_bytes)
        config = LearnableConfigWithShape(
            observation_shape=algo.impl.observation_shape,
            action_size=algo.impl.action_size,
            config=algo.config,
        )
        obj = {
            "torch": torch_bytes.getvalue(),
            "config": config.serialize(),
            "version": __version__,
        }
        pickle.dump(obj, f)


def load_learnable(fname: str, device: DeviceArg = None) -> "LearnableBase":
    with open(fname, "rb") as f:
        obj = pickle.load(f)
        if obj["version"] != __version__:
            LOG.warning(
                "There might be incompatibility because of version mismatch.",
                current_version=__version__,
                saved_version=obj["version"],
            )
        config = LearnableConfigWithShape.deserialize(obj["config"])
        algo = config.create(device)
        assert algo.impl
        algo.impl.load_model(io.BytesIO(obj["torch"]))
    return algo


class LearnableBase(metaclass=ABCMeta):
    _config: LearnableConfig
    _device: str
    _impl: Optional[ImplBase]
    _grad_step: int

    def __init__(
        self,
        config: LearnableConfig,
        device: DeviceArg,
        impl: Optional[ImplBase] = None,
    ):
        self._config = config
        self._device = _process_device(device)
        self._impl = impl
        self._grad_step = 0

    def save(self, fname: str) -> None:
        """Saves paired data of neural network parameters and serialized config.

        .. code-block:: python

            algo.save('model.d3')

            # reconstruct everything
            algo2 = d3rlpy.load_learnable("model.d3", device="cuda:0")

        Args:
            fname: destination file path.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        dump_learnable(self, fname)

    def save_model(self, fname: str) -> None:
        """Saves neural network parameters.

        .. code-block:: python

            algo.save_model('model.pt')

        Args:
            fname: destination file path.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        with open(fname, "wb") as f:
            self._impl.save_model(f)

    def load_model(self, fname: str) -> None:
        """Load neural network parameters.

        .. code-block:: python

            algo.load_model('model.pt')

        Args:
            fname: source file path.

        """
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        with open(fname, "rb") as f:
            self._impl.load_model(f)

    @classmethod
    def from_json(
        cls: Type[TLearnable], fname: str, device: DeviceArg = False
    ) -> TLearnable:
        config = LearnableConfigWithShape.deserialize_from_file(fname)
        return config.create(device)  # type: ignore

    def create_impl(self, observation_shape: Shape, action_size: int) -> None:
        """Instantiate implementation objects with the dataset shapes.

        This method will be used internally when `fit` method is called.

        Args:
            observation_shape: observation shape.
            action_size: dimension of action-space.

        """
        if self._impl:
            LOG.warn("Parameters will be reinitialized.")
        self.inner_create_impl(observation_shape, action_size)

    @abstractmethod
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        pass

    def build_with_dataset(self, dataset: ReplayBuffer) -> None:
        """Instantiate implementation object with MDPDataset object.

        Args:
            dataset: dataset.

        """
        dataset_info = DatasetInfo.from_episodes(dataset.episodes)
        observation_shape = dataset.sample_transition().observation_shape
        self.create_impl(observation_shape, dataset_info.action_size)

    def build_with_env(self, env: gym.Env) -> None:
        """Instantiate implementation object with OpenAI Gym object.

        Args:
            env: gym-like environment.

        """
        observation_shape = env.observation_space.shape
        if isinstance(env.action_space, Discrete):
            action_size = cast(int, env.action_space.n)
        else:
            action_size = cast(int, env.action_space.shape[0])
        self.create_impl(observation_shape, action_size)

    def get_action_type(self) -> ActionSpace:
        """Returns action type (continuous or discrete).

        Returns:
            action type.

        """
        raise NotImplementedError

    @property
    def config(self) -> LearnableConfig:
        return self._config

    @property
    def batch_size(self) -> int:
        """Batch size to train.

        Returns:
            int: batch size.

        """
        return self._config.batch_size

    @property
    def gamma(self) -> float:
        """Discount factor.

        Returns:
            float: discount factor.

        """
        return self._config.gamma

    @property
    def observation_scaler(self) -> Optional[ObservationScaler]:
        """Preprocessing observation scaler.

        Returns:
            Optional[ObservationScaler]: preprocessing observation scaler.

        """
        return self._config.observation_scaler

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        """Preprocessing action scaler.

        Returns:
            Optional[ActionScaler]: preprocessing action scaler.

        """
        return self._config.action_scaler

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        """Preprocessing reward scaler.

        Returns:
            Optional[RewardScaler]: preprocessing reward scaler.

        """
        return self._config.reward_scaler

    @property
    def impl(self) -> Optional[ImplBase]:
        """Implementation object.

        Returns:
            Optional[ImplBase]: implementation object.

        """
        return self._impl

    @property
    def observation_shape(self) -> Optional[Shape]:
        """Observation shape.

        Returns:
            Optional[Sequence[int]]: observation shape.

        """
        if self._impl:
            return self._impl.observation_shape
        return None

    @property
    def action_size(self) -> Optional[int]:
        """Action size.

        Returns:
            Optional[int]: action size.

        """
        if self._impl:
            return self._impl.action_size
        return None

    @property
    def grad_step(self) -> int:
        """Total gradient step counter.

        This value will keep counting after ``fit`` and ``fit_online``
        methods finish.

        Returns:
            total gradient step counter.

        """
        return self._grad_step

    def set_grad_step(self, grad_step: int) -> None:
        """Set total gradient step counter.

        This method can be used to restart the middle of training with an
        arbitrary gradient step counter, which has effects on periodic
        functions such as the target update.

        Args:
            grad_step: total gradient step counter.

        """
        self._grad_step = grad_step
