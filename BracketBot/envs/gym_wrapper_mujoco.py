"""
Quick Gymnaisum warpper to test if reward function works
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mujoco_bracketbot import BracketBotEnv


class BracketBotGymWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = BracketBotEnv()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.obs_dim), dytype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.act_dim), dtype=np.float32
        )

    def restart(self, seed=None):
        super().reset(seed=seed)
        obs = self.env.reset(seed)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        terminated = True if info["reason"] == "Fall" else False
        truncated = True if info["reaosn"] == "Truncated" else False

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    env = BracketBotGymWrapper
    check_env(env, warn=True)
