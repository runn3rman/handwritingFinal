"""PPO agent with policy, buffer, and optimization logic."""

from __future__ import annotations

from typing import Dict

import torch
from torch.distributions import Normal

from buffer import RolloutBuffer
from config import PPOConfig
from models import ActorCritic


class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, config: PPOConfig, device: torch.device):
        self.config = config
        self.device = device

        self.actor_critic = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=config.hidden_sizes,
            log_std_init=config.log_std_init,
            log_std_bounds=(config.log_std_min, config.log_std_max),
        ).to(device)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=config.lr)
        self.buffer = RolloutBuffer(config.rollout_steps, obs_dim, act_dim, device)

    def select_action(self, obs_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, log_std, value = self.actor_critic(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        action = action.squeeze(0)
        logprob = logprob.squeeze(0)
        value = value.squeeze(0)

        action_raw = action.detach()
        logprob = logprob.detach()
        value = value.detach()
        action_clipped = torch.clamp(action_raw, -1.0, 1.0)

        return action_clipped.cpu().numpy(), action_raw.cpu(), value, logprob

    def act_deterministic(self, obs_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, _, _ = self.actor_critic(obs)
        action = torch.clamp(mu.squeeze(0), -1.0, 1.0)
        return action.cpu().numpy()

    def update(self, last_value: torch.Tensor, clip_coef: float, ent_coef: float) -> Dict[str, float]:
        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        num_updates = 0

        for _ in range(self.config.n_epochs):
            indices = torch.randperm(self.config.rollout_steps, device=self.device)
            for start in range(0, self.config.rollout_steps, self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                mb_idx = indices[start:end]

                mb_obs = self.buffer.obs[mb_idx]
                mb_actions = self.buffer.actions[mb_idx]
                mb_old_logprobs = self.buffer.logprobs[mb_idx]
                mb_returns = self.buffer.returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                mu, log_std, values = self.actor_critic(mb_obs)
                std = log_std.exp()
                dist = Normal(mu, std)
                logprobs = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratios = (logprobs - mb_old_logprobs).exp()
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_coef, 1.0 + clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_returns - values).pow(2).mean()

                loss = policy_loss + self.config.vf_coef * value_loss - ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                num_updates += 1

        for key in stats:
            stats[key] /= max(num_updates, 1)

        self.buffer.reset_storage()
        return stats

    def set_learning_rate(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def save(self, path: str):
        torch.save({"model_state": self.actor_critic.state_dict(), "optimizer_state": self.optimizer.state_dict()}, path)

    def load(self, path: str, load_optimizer: bool = True):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["model_state"])
        if load_optimizer and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

