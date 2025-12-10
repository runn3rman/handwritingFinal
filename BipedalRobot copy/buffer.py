"""Rollout buffer for PPO trajectories."""

from __future__ import annotations

import torch


class RolloutBuffer:
    def __init__(self, rollout_steps: int, obs_dim: int, act_dim: int, device: torch.device):
        self.rollout_steps = rollout_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.reset_storage()

    def reset_storage(self):
        self.obs = torch.zeros((self.rollout_steps, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.rollout_steps, self.act_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(self.rollout_steps, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.rollout_steps, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(self.rollout_steps, dtype=torch.float32, device=self.device)
        self.logprobs = torch.zeros(self.rollout_steps, dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros(self.rollout_steps, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(self.rollout_steps, dtype=torch.float32, device=self.device)
        self.ptr = 0

    def add(self, obs, action, reward, done, value, logprob):
        if self.ptr >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer overflow. Call reset_storage() after update().")

        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.values[self.ptr] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.logprobs[self.ptr] = torch.as_tensor(logprob, dtype=torch.float32, device=self.device)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.rollout_steps

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        last_value = torch.as_tensor(last_value, dtype=torch.float32, device=self.device)
        if last_value.dim() > 0:
            last_value = last_value.squeeze()

        advantage = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for step in reversed(range(self.rollout_steps)):
            mask = 1.0 - self.dones[step]
            next_value = last_value if step == self.rollout_steps - 1 else self.values[step + 1]
            delta = self.rewards[step] + gamma * next_value * mask - self.values[step]
            advantage = delta + gamma * gae_lambda * mask * advantage
            self.advantages[step] = advantage

        self.returns = self.advantages + self.values

