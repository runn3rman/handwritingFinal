"""Utility helpers for PPO training scripts."""

from __future__ import annotations

import os
import random
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)


def env_reset(env, seed=None):
    if seed is not None:
        try:
            result = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
            result = env.reset()
    else:
        result = env.reset()

    if isinstance(result, tuple):
        return result
    return result, {}


def env_step(env, action):
    result = env.step(action)
    if len(result) == 5:
        return result
    obs, reward, done, info = result
    truncated = info.get("TimeLimit.truncated", False)
    terminated = done and not truncated
    return obs, reward, terminated, truncated, info


def moving_average(values: Iterable[float], window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return np.asarray(values)
    values = np.asarray(values)
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def plot_learning_curve(returns, path: str, window: int = 10):
    if len(returns) == 0:
        return

    ensure_dir(os.path.dirname(path) or ".")
    plt.figure(figsize=(8, 4))
    plt.plot(returns, label="Episode return", alpha=0.4)
    if len(returns) >= window:
        smoothed = moving_average(returns, window)
        offset = len(returns) - len(smoothed)
        plt.plot(range(offset, offset + len(smoothed)), smoothed, label=f"{window}-ep moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


class UprightWalkerWrapper(gym.Wrapper):
    """Reward shaping to encourage upright, alternating gait."""

    def __init__(
        self,
        env,
        penalty_coef: float = 0.5,
        time_penalty: float = 0.001,
        forward_velocity_coef: float = 0.4,
        single_support_bonus: float = 0.08,
        double_support_penalty: float = 0.05,
        step_switch_bonus: float = 0.15,
        knee_straightness_coef: float = 0.08,
        knee_straight_tolerance: float = 0.15,
        hip_phase_bonus: float = 0.05,
        angular_velocity_penalty: float = 0.02,
    ):
        super().__init__(env)
        self.penalty_coef = penalty_coef
        self.time_penalty = time_penalty
        self.forward_velocity_coef = forward_velocity_coef
        self.single_support_bonus = single_support_bonus
        self.double_support_penalty = double_support_penalty
        self.step_switch_bonus = step_switch_bonus
        self.knee_straightness_coef = knee_straightness_coef
        self.knee_straight_tolerance = knee_straight_tolerance
        self.hip_phase_bonus = hip_phase_bonus
        self.angular_velocity_penalty = angular_velocity_penalty
        self.prev_support = 0  # -1 left, 1 right, 0 none
        self.steps_since_switch = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.prev_support = 0
        self.steps_since_switch = 0
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_arr = np.asarray(obs, dtype=np.float32)
        # Observation index map (Gymnasium BipedalWalker):
        # 0 hull angle, 1 hull angular velocity, 2 forward velocity, 4/6 left hip/knee angles,
        # 8/10 right hip/knee angles, 12/13 foot contact sensors.

        hull_angle = float(obs_arr[0])
        hull_ang_vel = float(obs_arr[1])
        horizontal_speed = float(obs_arr[2])
        hip_left = float(obs_arr[4])
        knee_left = float(obs_arr[6])
        hip_right = float(obs_arr[8])
        knee_right = float(obs_arr[10])
        left_contact = bool(obs_arr[12] > 0.5)
        right_contact = bool(obs_arr[13] > 0.5)

        upright_penalty = -self.penalty_coef * abs(hull_angle)
        forward_bonus = self.forward_velocity_coef * max(0.0, horizontal_speed)
        ang_vel_penalty = -self.angular_velocity_penalty * abs(hull_ang_vel)
        hip_phase_reward = self.hip_phase_bonus * abs(hip_left - hip_right)
        knee_penalty = self.knee_straightness_coef * self._knee_bend_excess(knee_left, knee_right)
        gait_reward = self._gait_reward(left_contact, right_contact)

        shaped_reward = (
            reward
            + upright_penalty
            + forward_bonus
            + hip_phase_reward
            + gait_reward
            + ang_vel_penalty
            - knee_penalty
            - self.time_penalty
        )
        return obs, shaped_reward, terminated, truncated, info

    def _knee_bend_excess(self, knee_left: float, knee_right: float) -> float:
        tolerance = self.knee_straight_tolerance
        excess_left = max(0.0, abs(knee_left) - tolerance)
        excess_right = max(0.0, abs(knee_right) - tolerance)
        return excess_left + excess_right

    def _gait_reward(self, left_contact: bool, right_contact: bool) -> float:
        single_support = left_contact ^ right_contact
        reward = 0.0
        if single_support:
            reward += self.single_support_bonus
            support = -1 if left_contact else 1
            if self.prev_support != 0 and support != self.prev_support:
                reward += self.step_switch_bonus / (1.0 + self.steps_since_switch)
            self.prev_support = support
            self.steps_since_switch = 0
        else:
            if left_contact and right_contact:
                reward -= self.double_support_penalty
            else:
                reward -= 0.5 * self.double_support_penalty
            self.steps_since_switch += 1
        return reward


class RunningMeanStd:
    """Maintain running mean and variance for observation normalization."""

    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = np.expand_dims(x, axis=0)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class ObsNormWrapper(gym.ObservationWrapper):
    """Observation normalization wrapper with optional stat persistence."""

    def __init__(self, env, clip: float = 5.0, eps: float = 1e-8, update_stats: bool = True):
        super().__init__(env)
        self.clip = clip
        self.eps = eps
        self.update_stats = update_stats
        self.rms = RunningMeanStd(self.observation_space.shape)

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if self.update_stats:
            self.rms.update(obs)
        norm_obs = (obs - self.rms.mean) / (self.rms.std + self.eps)
        return np.clip(norm_obs, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.rms.mean, "var": self.rms.var, "count": self.rms.count}

    def load_state_dict(self, state):
        self.rms.mean = state["mean"]
        self.rms.var = state["var"]
        self.rms.count = state["count"]

    def save_stats(self, path: str):
        ensure_dir(os.path.dirname(path) or ".")
        data = self.state_dict()
        np.savez(path, **data)

    def load_stats_file(self, path: str):
        data = np.load(path)
        self.load_state_dict({"mean": data["mean"], "var": data["var"], "count": data["count"]})


def find_wrapper(env, wrapper_type):
    """Return the first wrapper of type wrapper_type in the env stack."""
    current = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, wrapper_type):
            return current
        current = current.env
    return None


