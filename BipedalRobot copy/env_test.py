"""Quick smoke test for BipedalWalker-v3 rendering and API."""



import gymnasium as gym

    

import numpy as np


def main():
    env = gym.make("BipedalWalker-v3", render_mode="human")
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs, info = reset_result, {}
    print("obs shape:", np.shape(obs))
    print("action space:", env.action_space)

    for step in range(1000):
        action = env.action_space.sample()
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done and not info.get("TimeLimit.truncated", False)
            truncated = info.get("TimeLimit.truncated", False)

        if terminated or truncated:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs, info = reset_result, {}
    env.close()
    print("Smoke test finished without crashes.")


if __name__ == "__main__":
    main()

