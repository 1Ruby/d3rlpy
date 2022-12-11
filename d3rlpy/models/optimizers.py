from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Tuple, Type

from dataclasses_json import config
from torch import nn
from torch.optim import SGD, Adam, Optimizer, RMSprop

__all__ = [
    "OptimizerFactory",
    "SGDFactory",
    "AdamFactory",
    "RMSpropFactory",
    "OPTIMIZER_LIST",
    "register_optimizer_factory",
    "create_optimizer_factory",
    "make_optimizer_field",
]


@dataclass(frozen=True)
class OptimizerFactory:
    """A factory class that creates an optimizer object in a lazy way.

    The optimizers in algorithms can be configured through this factory class.

    """

    def create(self, params: Iterable[nn.Parameter], lr: float) -> Optimizer:
        """Returns an optimizer object.

        Args:
            params (list): a list of PyTorch parameters.
            lr (float): learning rate.

        Returns:
            torch.optim.Optimizer: an optimizer object.

        """
        raise NotImplementedError

    @staticmethod
    def get_type() -> str:
        """Returns optimizer type.

        Returns:
            optimizer type name.

        """
        raise NotImplementedError


@dataclass(frozen=True)
class SGDFactory(OptimizerFactory):
    """An alias for SGD optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import SGDFactory

        factory = SGDFactory(weight_decay=1e-4)

    Args:
        momentum: momentum factor.
        dampening: dampening for momentum.
        weight_decay: weight decay (L2 penalty).
        nesterov: flag to enable Nesterov momentum.

    """

    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False

    def create(self, params: Iterable[nn.Parameter], lr: float) -> SGD:
        return SGD(
            params,
            lr=lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )

    @staticmethod
    def get_type() -> str:
        return "sgd"


@dataclass(frozen=True)
class AdamFactory(OptimizerFactory):
    """An alias for Adam optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import AdamFactory

        factory = AdamFactory(weight_decay=1e-4)

    Args:
        betas: coefficients used for computing running averages of
            gradient and its square.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        amsgrad: flag to use the AMSGrad variant of this algorithm.

    """

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False

    def create(self, params: Iterable[nn.Parameter], lr: float) -> Adam:
        return Adam(
            params,
            lr=lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

    @staticmethod
    def get_type() -> str:
        return "adam"


@dataclass(frozen=True)
class RMSpropFactory(OptimizerFactory):
    """An alias for RMSprop optimizer.

    .. code-block:: python

        from d3rlpy.optimizers import RMSpropFactory

        factory = RMSpropFactory(weight_decay=1e-4)

    Args:
        alpha: smoothing constant.
        eps: term added to the denominator to improve numerical stability.
        weight_decay: weight decay (L2 penalty).
        momentum: momentum factor.
        centered: flag to compute the centered RMSProp, the gradient is
            normalized by an estimation of its variance.

    """

    alpha: float = 0.95
    eps: float = 1e-2
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = True

    def create(self, params: Iterable[nn.Parameter], lr: float) -> RMSprop:
        return RMSprop(
            params,
            lr=lr,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
        )

    @staticmethod
    def get_type() -> str:
        return "rmsprop"


OPTIMIZER_LIST: Dict[str, Type[OptimizerFactory]] = {}


def register_optimizer_factory(cls: Type[OptimizerFactory]) -> None:
    """Registers optimizer factory class.

    Args:
        cls: Optimizer factory class inheriting ``OptimizerFactory``.

    """
    type_name = cls.get_type()
    is_registered = type_name in OPTIMIZER_LIST
    assert not is_registered, f"{type_name} seems to be already registered"
    OPTIMIZER_LIST[type_name] = cls


def create_optimizer_factory(name: str, **kwargs: Any) -> OptimizerFactory:
    assert name in OPTIMIZER_LIST, "f{name} seems not to be registered"
    factory = OPTIMIZER_LIST[name](**kwargs)
    assert isinstance(factory, OptimizerFactory)
    return factory


def _encoder(optim: OptimizerFactory) -> Dict[str, Any]:
    return {"type": optim.get_type(), "params": asdict(optim)}


def _decoder(dict_config: Dict[str, Any]) -> OptimizerFactory:
    return create_optimizer_factory(
        dict_config["type"], **dict_config["params"]
    )


def make_optimizer_field() -> OptimizerFactory:
    return field(
        metadata=config(encoder=_encoder, decoder=_decoder),
        default=AdamFactory(),
    )


register_optimizer_factory(SGDFactory)
register_optimizer_factory(AdamFactory)
register_optimizer_factory(RMSpropFactory)
