"""Centralized PPO hyperparameters and training config."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PPOConfig:
    # Environment
    env_id: str = "BipedalWalker-v3"
    seed: int = 42

    # Training horizon
    total_timesteps: int = 3_000_000
    rollout_steps: int = 4096
    n_epochs: int = 10
    mini_batch_size: int = 256

    # PPO/GAE hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.15
    clip_coef_final: float = 0.10
    ent_coef: float = 0.02
    ent_coef_final: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    lr_final: float = 1e-4
    anneal_lr: bool = True

    # Network 
    hidden_sizes: Tuple[int, int] = (256, 256)
    log_std_init: float = -0.5
    log_std_min: float = -2.0
    log_std_max: float = 1.0

    # Logging / eval
    log_interval: int = 10_000
    eval_interval: int = 200_000
    eval_episodes: int = 5
    reward_threshold: float = 300.0

    # reward shaping / wrappers
    use_upright_wrapper: bool = True
    upright_penalty_coef: float = 0.5
    time_penalty: float = 0.001
    forward_velocity_coef: float = 0.4
    single_support_bonus: float = 0.08
    double_support_penalty: float = 0.05
    step_switch_bonus: float = 0.15
    knee_straightness_coef: float = 0.08
    knee_straight_tolerance: float = 0.15
    hip_phase_bonus: float = 0.05
    angular_velocity_penalty: float = 0.02

    # Observation normalization
    use_obs_norm: bool = True
    obs_norm_clip: float = 5.0
    obs_stats_path: str = "checkpoints/obs_stats.npz"

    # Paths
    checkpoint_dir: str = "checkpoints"
    best_model_name: str = "ppo_bipedal_best.pt"
    last_model_name: str = "ppo_bipedal_last.pt"
    plot_path: str = "plots/learning_curve.png"


