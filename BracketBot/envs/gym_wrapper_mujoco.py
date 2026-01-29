"""
Quick Gymnaisum warpper to test if reward function works
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mujoco_bracketbot import BracketBotEnv


class BracketBotGymWrapper(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.env = BracketBotEnv()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.act_dim,), dtype=np.float32
        )

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)
        obs = self.env.reset(seed)
        return obs, {}

    def step(self, action) -> tuple:
        obs, reward, done, info = self.env.step(action)

        terminated = True if info["reason"] == "Fall" else False
        truncated = True if info["reason"] == "Truncated" else False

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import mujoco
    import mujoco.renderer
    import os
    import mediapy as media

    env = BracketBotGymWrapper()
    # check_env(env, warn=True)
    # print("passes `check_env()`!")

    # testing if PPO works

    vec_env = DummyVecEnv([lambda: BracketBotGymWrapper()])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # model = PPO(
    #     "MlpPolicy",
    #     vec_env,
    #     verbose=1,
    # )

    # model.learn(total_timesteps=50000)
    # model.save("ppo_mk1")

    # del model

    model = PPO.load("ppo_mk1")

    # rollout
    m_env = vec_env.get_attr("env", indices=0)[0]
    renderer = mujoco.Renderer(m_env.model)
    renderer.update_scene(m_env.data)
    frames = []

    obs = vec_env.reset()
    rewards = []
    episode_reward = 0

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        episode_reward += reward[0]
        m_env = vec_env.get_attr("env", indices=0)[0]
        renderer.update_scene(m_env.data)

        frames.append(renderer.render())

    out_file = "SB_rollout.mp4"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(cur_dir, out_file)
    media.write_video(out_path, frames, fps=27)
