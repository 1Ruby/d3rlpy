from typing import Optional, cast

import torch
import torch.nn as nn

from ..encoders import Encoder, EncoderWithAction
from .base import ContinuousQFunction, DiscreteQFunction
from .utility import (
    compute_quantile_huber_loss,
    compute_reduce,
    pick_quantile_value_by_action,
)


class QRQFunction(nn.Module):  # type: ignore
    _n_quantiles: int

    def __init__(self, n_quantiles: int):
        super().__init__()
        self._n_quantiles = n_quantiles

    def _make_taus(self, h: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(
            self._n_quantiles, dtype=torch.float32, device=h.device
        )
        taus = ((steps + 1).float() / self._n_quantiles).view(1, -1)
        taus_dot = (steps.float() / self._n_quantiles).view(1, -1)
        return (taus + taus_dot) / 2.0

    def _compute_quantile_loss(
        self,
        quantiles_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        ter_tp1: torch.Tensor,
        taus: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        batch_size = rew_tp1.shape[0]
        y = (rew_tp1 + gamma * q_tp1 * (1 - ter_tp1)).view(batch_size, -1, 1)
        quantiles_t = quantiles_t.view(batch_size, 1, -1)
        expanded_taus = taus.view(-1, 1, self._n_quantiles)
        return compute_quantile_huber_loss(quantiles_t, y, expanded_taus)


class DiscreteQRQFunction(QRQFunction, DiscreteQFunction):
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int, n_quantiles: int):
        super().__init__(n_quantiles)
        self._encoder = encoder
        self._action_size = action_size
        self._fc = nn.Linear(
            encoder.get_feature_size(), action_size * n_quantiles
        )

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        h = cast(torch.Tensor, self._fc(h))
        return h.view(-1, self._action_size, self._n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=2)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        ter_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert q_tp1.shape == (obs_t.shape[0], self._n_quantiles)

        # extraect quantiles corresponding to act_t
        h = self._encoder(obs_t)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        quantiles_t = pick_quantile_value_by_action(quantiles, act_t)

        loss = self._compute_quantile_loss(
            quantiles_t=quantiles_t,
            rew_tp1=rew_tp1,
            q_tp1=q_tp1,
            ter_tp1=ter_tp1,
            taus=taus,
            gamma=gamma,
        )

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self._encoder(x)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        if action is None:
            return quantiles
        return pick_quantile_value_by_action(quantiles, action)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ContinuousQRQFunction(QRQFunction, ContinuousQFunction):
    _action_size: int
    _encoder: EncoderWithAction
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction, n_quantiles: int):
        super().__init__(n_quantiles)
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._fc = nn.Linear(encoder.get_feature_size(), n_quantiles)

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        return cast(torch.Tensor, self._fc(h))

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=1, keepdim=True)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        ter_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert q_tp1.shape == (obs_t.shape[0], self._n_quantiles)

        h = self._encoder(obs_t, act_t)
        taus = self._make_taus(h)
        quantiles_t = self._compute_quantiles(h, taus)

        loss = self._compute_quantile_loss(
            quantiles_t=quantiles_t,
            rew_tp1=rew_tp1,
            q_tp1=q_tp1,
            ter_tp1=ter_tp1,
            taus=taus,
            gamma=gamma,
        )

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        h = self._encoder(x, action)
        taus = self._make_taus(h)
        return self._compute_quantiles(h, taus)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder
