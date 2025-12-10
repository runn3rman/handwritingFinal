"""Neural network architectures used by PPO."""

from typing import Iterable, Tuple

import torch
import torch.nn as nn


class MLPBody(nn.Module):
    """Simple feed-forward torso."""

    def __init__(self, input_dim: int, hidden_sizes: Iterable[int]):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """Shared-body actor-critic for continuous control."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes=(256, 256),
        log_std_init: float = -0.5,
        log_std_bounds: Tuple[float, float] = (-2.0, 1.0),
    ):
        super().__init__()
        self.body = MLPBody(obs_dim, hidden_sizes)
        last_dim = hidden_sizes[-1]

        self.mu_head = nn.Linear(last_dim, act_dim)
        self.v_head = nn.Linear(last_dim, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)
        self.log_std_min, self.log_std_max = log_std_bounds

    def forward(self, obs: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = self.body(obs)
        mu = self.mu_head(x)
        value = self.v_head(x).squeeze(-1)
        log_std = self.log_std.clamp(min=self.log_std_min, max=self.log_std_max).expand_as(mu)
        return mu, log_std, value


