"""One-command pipeline to train/eval PPO across multiple seeds."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime
from typing import List

import torch

from config import PPOConfig
from eval import evaluate_checkpoint
from train import apply_overrides, train_agent
from utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end PPO pipeline for BipedalWalker.")
    parser.add_argument("--env-id", type=str, default=None, help="Env id override (normal or hardcore).")
    parser.add_argument("--hardcore", action="store_true", help="Shortcut for BipedalWalkerHardcore-v3.")
    parser.add_argument("--total-timesteps", type=int, help="Timesteps per seed.")
    parser.add_argument("--seeds", type=int, default=1, help="Number of sequential seeds to run.")
    parser.add_argument("--seed-list", type=int, nargs="*", help="Explicit list of seeds.")
    parser.add_argument("--device", type=str, help="Device override (cpu/cuda).")
    parser.add_argument("--output-dir", type=str, default="experiments", help="Root dir for all runs.")
    parser.add_argument("--use-upright-wrapper", action="store_true", help="Enable reward shaping for all seeds.")
    parser.add_argument("--disable-obs-norm", action="store_true", help="Disable observation normalization.")
    parser.add_argument(
        "--disable-upright-wrapper", action="store_true", help="Force-disable gait shaping even if config enables it."
    )
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes per deterministic eval.")
    parser.add_argument("--render-training", action="store_true", help="Render during training (slow).")
    parser.add_argument("--render-eval", action="store_true", help="Render best eval run.")
    parser.add_argument("--reward-threshold", type=float, default=300.0, help="Target reward to declare solved.")
    return parser.parse_args()


def build_seed_config(base_config: PPOConfig, seed: int, args) -> PPOConfig:
    cfg = replace(base_config)
    cfg.seed = seed
    run_name = f"{cfg.env_id}_seed{seed}"
    run_dir = os.path.join(args.output_dir, run_name)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    plots_dir = os.path.join(run_dir, "plots")
    ensure_dir(checkpoints_dir)
    ensure_dir(plots_dir)
    cfg.checkpoint_dir = checkpoints_dir
    cfg.best_model_name = "best.pt"
    cfg.last_model_name = "last.pt"
    cfg.plot_path = os.path.join(plots_dir, "learning_curve.png")
    cfg.obs_stats_path = os.path.join(checkpoints_dir, "obs_stats.npz")
    return cfg


def main():
    args = parse_args()
    base_config = PPOConfig()
    if args.hardcore:
        base_config.env_id = "BipedalWalkerHardcore-v3"
    if args.env_id:
        base_config.env_id = args.env_id
    overrides = argparse.Namespace(
        env_id=base_config.env_id,
        total_timesteps=args.total_timesteps,
        seed=None,
        use_upright_wrapper=args.use_upright_wrapper,
        disable_upright_wrapper=args.disable_upright_wrapper,
        disable_obs_norm=args.disable_obs_norm,
        checkpoint_dir=None,
        obs_stats_path=None,
        eval_episodes=args.eval_episodes,
    )
    base_config = apply_overrides(base_config, overrides)
    base_config.reward_threshold = args.reward_threshold

    if args.seed_list:
        seeds: List[int] = args.seed_list
    else:
        seeds = [base_config.seed + i for i in range(args.seeds)]

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    summary = {"runs": [], "env_id": base_config.env_id, "timestamp": datetime.utcnow().isoformat()}

    best_run = None
    best_score = -float("inf")

    for idx, seed in enumerate(seeds, start=1):
        print(f"\n=== Seed {seed} ({idx}/{len(seeds)}) ===")
        cfg = build_seed_config(base_config, seed, args)
        run_info = train_agent(cfg, device, render=args.render_training)
        best_ckpt = run_info["best_checkpoint"]
        avg_return, returns = evaluate_checkpoint(cfg, best_ckpt, args.eval_episodes, device, render=False)
        returns = [float(r) for r in returns]
        solved = avg_return >= args.reward_threshold
        print(
            f"[seed {seed}] deterministic eval avg_return={avg_return:.2f} "
            f"{'(solved!)' if solved else '(needs work)'}"
        )
        if avg_return > best_score:
            best_score = avg_return
            best_run = {
                "seed": seed,
                "checkpoint": best_ckpt,
                "avg_return": avg_return,
                "returns": returns,
                "config": cfg,
            }
        summary["runs"].append(
            {
                "seed": seed,
                "best_checkpoint": best_ckpt,
                "avg_return": avg_return,
                "solved": solved,
                "returns": returns,
            }
        )

    ensure_dir(args.output_dir)
    summary_path = os.path.join(args.output_dir, f"summary_{base_config.env_id}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    if best_run:
        print(
            f"Best seed {best_run['seed']} achieved avg_return={best_run['avg_return']:.2f} "
            f"(checkpoint: {best_run['checkpoint']})"
        )
        if args.render_eval:
            evaluate_checkpoint(best_run["config"], best_run["checkpoint"], args.eval_episodes, device, render=True)
    else:
        print("No successful runs recorded.")


if __name__ == "__main__":
    main()

