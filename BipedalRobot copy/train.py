"""Train PPO on BipedalWalker-v3."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
import torch


import gymnasium as gym


from config import PPOConfig
from ppo_agent import PPOAgent
from utils import (
    ObsNormWrapper,
    UprightWalkerWrapper,
    ensure_dir,
    env_reset,
    env_step,
    find_wrapper,
    plot_learning_curve,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PPO trainer for BipedalWalker")
    parser.add_argument("--env-id", type=str, help="Environment id override (e.g., BipedalWalker-v3)")
    parser.add_argument("--total-timesteps", type=int, help="Override total timesteps from config")
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--render", action="store_true", help="Render training environment")
    parser.add_argument("--device", type=str, help="Force device (cpu/cuda)")
    parser.add_argument("--use-upright-wrapper", action="store_true", help="Enable reward shaping wrapper")
    parser.add_argument("--disable-upright-wrapper", action="store_true", help="Disable reward shaping wrapper")
    parser.add_argument("--disable-obs-norm", action="store_true", help="Disable observation normalization")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory for checkpoints")
    parser.add_argument("--obs-stats-path", type=str, help="Path to save/load observation norm stats")
    parser.add_argument("--eval-episodes", type=int, help="Evaluation episodes override")
    return parser.parse_args()


def linear_schedule(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def make_env(config: PPOConfig, render_mode=None, update_obs_stats: bool = True):
    env = gym.make(config.env_id, render_mode=render_mode)
    if config.use_upright_wrapper:
        env = UprightWalkerWrapper(
            env,
            penalty_coef=config.upright_penalty_coef,
            time_penalty=config.time_penalty,
            forward_velocity_coef=config.forward_velocity_coef,
            single_support_bonus=config.single_support_bonus,
            double_support_penalty=config.double_support_penalty,
            step_switch_bonus=config.step_switch_bonus,
            knee_straightness_coef=config.knee_straightness_coef,
            knee_straight_tolerance=config.knee_straight_tolerance,
            hip_phase_bonus=config.hip_phase_bonus,
            angular_velocity_penalty=config.angular_velocity_penalty,
        )
    if config.use_obs_norm:
        env = ObsNormWrapper(env, clip=config.obs_norm_clip, update_stats=update_obs_stats)
        if config.obs_stats_path and os.path.exists(config.obs_stats_path):
            env.load_stats_file(config.obs_stats_path)
        elif not update_obs_stats:
            print("Warning: observation stats not found; proceeding with unnormalized observations.")
    return env


def evaluate_policy(agent: PPOAgent, config: PPOConfig, episodes: int, render: bool = False) -> Tuple[float, list]:
    eval_env = make_env(config, render_mode="human" if render else None, update_obs_stats=False)
    returns = []
    was_training = agent.actor_critic.training
    agent.actor_critic.eval()

    for _ in range(episodes):
        obs, _ = env_reset(eval_env)
        done = False
        ep_return = 0.0
        while not done:
            action = agent.act_deterministic(obs)
            obs, reward, terminated, truncated, _ = env_step(eval_env, action)
            done = terminated or truncated
            ep_return += reward
        returns.append(ep_return)

    if was_training:
        agent.actor_critic.train()
    eval_env.close()
    return float(np.mean(returns)), returns


def train_agent(config: PPOConfig, device: torch.device, render: bool = False) -> Dict[str, object]:
    render_mode = "human" if render else None
    env = make_env(config, render_mode=render_mode, update_obs_stats=config.use_obs_norm)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    set_seed(config.seed, env)

    agent = PPOAgent(obs_dim, act_dim, config, device)
    obs, _ = env_reset(env)

    ensure_dir(config.checkpoint_dir)
    ensure_dir(os.path.dirname(config.plot_path) or ".")

    global_step = 0
    episode_return = 0.0
    returns_history = []
    best_avg_return = -float("inf")
    last_log_step = 0
    last_eval_step = 0

    obs_norm_wrapper = find_wrapper(env, ObsNormWrapper) if config.use_obs_norm else None

    while global_step < config.total_timesteps:
        while not agent.buffer.is_full():
            action_env, action_raw, value, logprob = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env_step(env, action_env)
            done = terminated or truncated

            agent.buffer.add(obs, action_raw, reward, done, value, logprob)

            episode_return += reward
            global_step += 1
            obs = next_obs

            if done:
                returns_history.append(episode_return)
                episode_return = 0.0
                obs, _ = env_reset(env)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, last_value = agent.actor_critic(obs_t)

        progress = min(global_step, config.total_timesteps) / config.total_timesteps

        current_clip = linear_schedule(config.clip_coef, config.clip_coef_final, progress)
        current_ent = linear_schedule(config.ent_coef, config.ent_coef_final, progress)

        if config.anneal_lr:
            current_lr = linear_schedule(config.lr, config.lr_final, progress)
            agent.set_learning_rate(current_lr)
        else:
            current_lr = agent.optimizer.param_groups[0]["lr"]

        update_stats = agent.update(last_value.squeeze(0), clip_coef=current_clip, ent_coef=current_ent)

        if returns_history:
            mean_return = float(np.mean(returns_history[-100:]))
            if mean_return > best_avg_return:
                best_avg_return = mean_return
                agent.save(os.path.join(config.checkpoint_dir, config.best_model_name))
        else:
            mean_return = 0.0

        agent.save(os.path.join(config.checkpoint_dir, config.last_model_name))

        if global_step - last_log_step >= config.log_interval:
            last_log_step = global_step
            print(
                f"step={global_step} "
                f"mean_return(last100)={mean_return:.2f} "
                f"policy_loss={update_stats['policy_loss']:.4f} "
                f"value_loss={update_stats['value_loss']:.4f} "
                f"entropy={update_stats['entropy']:.4f} "
                f"clip={current_clip:.3f} "
                f"ent_coef={current_ent:.4f} "
                f"lr={current_lr:.6f}"
            )

        if config.eval_interval > 0 and global_step - last_eval_step >= config.eval_interval:
            last_eval_step = global_step
            eval_avg, _ = evaluate_policy(agent, config, config.eval_episodes)
            print(f"[eval] step={global_step} avg_return={eval_avg:.2f}")
            if eval_avg >= config.reward_threshold:
                print("Reward threshold reached. Stopping early.")
                break

    env.close()
    if obs_norm_wrapper and config.obs_stats_path:
        obs_norm_wrapper.save_stats(config.obs_stats_path)
    plot_learning_curve(returns_history, config.plot_path)
    return {
        "best_avg_return": best_avg_return,
        "returns_history": returns_history,
        "best_checkpoint": os.path.join(config.checkpoint_dir, config.best_model_name),
        "last_checkpoint": os.path.join(config.checkpoint_dir, config.last_model_name),
    }


def apply_overrides(config: PPOConfig, args) -> PPOConfig:
    cfg = replace(config)
    if args.env_id:
        cfg.env_id = args.env_id
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    if args.seed is not None:
        cfg.seed = args.seed
    if getattr(args, "use_upright_wrapper", False):
        cfg.use_upright_wrapper = True
    if getattr(args, "disable_upright_wrapper", False):
        cfg.use_upright_wrapper = False
    if args.disable_obs_norm:
        cfg.use_obs_norm = False
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
        if not args.obs_stats_path:
            cfg.obs_stats_path = os.path.join(cfg.checkpoint_dir, "obs_stats.npz")
    if args.obs_stats_path:
        cfg.obs_stats_path = args.obs_stats_path
    if args.eval_episodes is not None:
        cfg.eval_episodes = args.eval_episodes
    return cfg


def main():
    args = parse_args()
    base_config = PPOConfig()
    config = apply_overrides(base_config, args)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_agent(config, device, render=args.render)


if __name__ == "__main__":
    main()

