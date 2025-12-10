"""Render or score a trained PPO policy."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


import gymnasium as gym


from config import PPOConfig
from models import ActorCritic
from utils import ObsNormWrapper, env_reset, env_step


def evaluate_checkpoint(
    config: PPOConfig, checkpoint_path: str, episodes: int, device: torch.device, render: bool = False
) -> tuple[float, list]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    render_mode = "human" if render else None
    env = gym.make(config.env_id, render_mode=render_mode)
    if config.use_obs_norm:
        env = ObsNormWrapper(env, clip=config.obs_norm_clip, update_stats=False)
        if config.obs_stats_path and os.path.exists(config.obs_stats_path):
            env.load_stats_file(config.obs_stats_path)
        else:
            print("Warning: observation stats not found; evaluation will be unnormalized.")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = ActorCritic(
        obs_dim,
        act_dim,
        hidden_sizes=config.hidden_sizes,
        log_std_init=config.log_std_init,
        log_std_bounds=(config.log_std_min, config.log_std_max),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    policy.load_state_dict(state_dict)
    policy.eval()

    returns = []
    for _ in range(episodes):
        obs, _ = env_reset(env)
        done = False
        ep_return = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu, _, _ = policy(obs_t)
            action = torch.clamp(mu.squeeze(0), -1.0, 1.0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env_step(env, action)
            done = terminated or truncated
            ep_return += reward
        returns.append(float(ep_return))

    env.close()
    return float(np.mean(returns)), returns


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PPO policy checkpoint.")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint. Defaults to best model.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Enable human rendering.")
    parser.add_argument("--device", type=str, help="Force device (cpu/cuda).")
    return parser.parse_args()


def main():
    args = parse_args()
    config = PPOConfig()
    if args.episodes:
        config.eval_episodes = args.episodes
    checkpoint_path = args.checkpoint or os.path.join(config.checkpoint_dir, config.best_model_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    render_mode = "human" if args.render else None
    env = gym.make(config.env_id, render_mode=render_mode)
    if config.use_obs_norm:
        env = ObsNormWrapper(env, clip=config.obs_norm_clip, update_stats=False)
        if config.obs_stats_path and os.path.exists(config.obs_stats_path):
            env.load_stats_file(config.obs_stats_path)
        else:
            print("Warning: observation stats not found; evaluation will be unnormalized.")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    avg_return, returns = evaluate_checkpoint(config, checkpoint_path, config.eval_episodes, device, render=args.render)
    for idx, ret in enumerate(returns, start=1):
        print(f"Episode {idx}: return={ret:.2f}")
    print(f"Average return over {len(returns)} episodes: {avg_return:.2f}")


if __name__ == "__main__":
    main()


