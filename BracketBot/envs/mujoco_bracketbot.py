"""
Following Gymnasium guide (https://gymnasium.farama.org/introduction/create_custom_env/)
"""

import mujoco
import numpy as np
import os
from typing import Optional


class EnvConfig:
    pass


class BracketBotEnv:
    def __init__(self, xml_path="../assets/BracketBot.xml"):
        """
        qpos: [x, y, z, qw, qx, qy, qz, left_wheel_angle, right_wheel_angle]
        qvel: [vx, vy, vz, wx, wy, wz, left_wheel_vel, right_wheel_vel]
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Mujoco Backend params
        self.n_frames = 2

        # Dim Space
        self.obs_dim = 10  # see obs() for details
        self.act_dim = 2  # [left_wheel_vel, right_wheel_vel]

        # Action Space
        self.min_action = -1.0
        self.max_action = 1.0

        self.step_count = 0

        # Env params
        self.rng = np.random.default_rng()  # for seed reset
        self.fall_angle = np.pi / 6  # ~ 60 away upright or 30 from ground
        self.episode_length = 1000

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def restart(self, seed: int = -1) -> np.ndarray:
        if seed != -1:
            self.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        q_min, q_max = -0.5, 0.5
        self.data.qpos[0:2] += self.rng.uniform(size=2, high=q_max, low=q_min)

        qd_min, qd_max = -0.5, 0.5
        self.data.qvel[0:2] += self.rng.uniform(size=2, high=qd_max, low=qd_min)

        self.data.qvel[4] += self.rng.uniform(-0.5, 0.5)  # random pitch

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0

        return self._obs()

    def step(self, action: np.array) -> tuple[np.ndarray, float, bool, dict]:
        # borrowing Gymnaisum return format for later to compare it to using  Gymnasium env approach
        """
        Return format:
            np.ndarray: new observation
            float: reward
            bool: done
            dict: additonal information
        """

        action = np.clip(action, self.min_action, self.max_action)
        self.data.ctrl = action

        for _ in range(self.n_frames):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._obs()
        reward, reward_info = self._reward(obs, action)
        done, reason = self._terminate(obs)

        info = {"step": self.step_count, "reason": reason, **reward_info}

        return obs, reward, done, info

    def _terminate(self, obs) -> tuple[bool, str]:
        """
        Two end conditons:
        1. Pole pitch angle falls past self.fall_angle
        2. 1000 time steps elapses
        """
        pitch = obs[4]
        done = False
        reason = ""

        if np.abs(pitch) > self.fall_angle:
            done = True
            reason = "Fall"
        elif self.step_count >= self.episode_length:
            done = True
            reason = "Truncated"

        return done, reason

    def _roll(self):
        quat = self.data.qpos[3:7]  # w, x, y, z
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    def _pitch(self):
        quat = self.data.qpos[3:7]  # w, x, y, z
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return np.arcsin(2 * (w * y - z * x))

    def _yaw(self):
        quat = self.data.qpos[3:7]  # w, x, y, z
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _reward(self, obs: np.ndarray, action: np.ndarray) -> tuple[float, dict]:
        """
        Return format:
            float: scalar reward
            dict: additonal information
        """

        pitch, pitch_vel = obs[4], obs[6]

        angle_reward = np.cos(3 * pitch)  # zero at pi/4 = 45 degrees

        action_penalty = np.sum(action**2)

        velocity_penalty = pitch_vel**2

        alive = 1.0

        # fall = 1 if pitch > (np.pi / 4) else 0

        angle_weight = 1.0
        action_weight = 0.01
        velocity_weight = 0.1
        alive_weight = 0.1

        reward = (
            (angle_weight * angle_reward)
            + (alive_weight * alive)
            - (velocity_penalty * velocity_weight)
            - (action_weight * action_penalty)
        )

        reward_info = {
            "angle_reward": angle_reward,
            "action_penalty": action_penalty,
            "velocity_penalty": velocity_penalty,
            "pitch_angle": pitch,
            "pitch_angle_degree": np.rad2deg(pitch),
            "reward": reward,
        }

        return reward, reward_info

    def _obs(self):
        """
        Observation Space:

        Idx | Discription
        ----|--------------------
         0  | X position of cart's base
         1  | Y position of cart's base
         2  | X linear velocity of cart's base
         3  | Y linear velocity of cart's base
         4  | Pitch angle of cart in radians
         5  | Yaw angle of cart in radians
         6  | Cart angular velocity along y axis in radians (pitch velocity)
         7  | Cart angular velocity along z axis in radians (yaw velocity)
         8  | left wheel velocity
         9  | right wheel velocity
        """
        q = self.data.qpos
        qd = self.data.qvel

        # will be used later, for now, using a full state discription
        # roll = self._roll()
        pitch = self._pitch()
        yaw = self._yaw()

        # vroll
        pitch_vel = qd[4]  # wy
        yaw_vel = qd[5]  # wz

        base_position = q[0:2]  # x, y
        base_velocity = qd[0:2]  # vx, vy
        wheel_vel = qd[-2:]  # left_wheel_vel, right_wheel_vel

        obs = np.concatenate(
            [
                base_position,  # 0,1
                base_velocity,  # 2,3
                [pitch],  # 4
                [yaw],  # 5
                [pitch_vel],  # 6
                [yaw_vel],  # 7
                wheel_vel,  # 8, 9
            ]
        )
        # dim = 10

        return obs.astype(np.float32)
        # return np.concatenate(q, qd)

    @property
    def _act_dim(self) -> int:
        return self.act_dim

    @property
    def _obs_dim(self) -> int:
        return self.obs_dim


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = BracketBotEnv()

    obs = env.restart(seed=49)

    # test a half episode
    total_reward = 0
    obs = env.restart(seed=0)
    pitch_angles = []
    for i in range(500):
        rand_action = np.random.uniform(size=2)
        # obs, reward, done, info = env.step(np.zeros(2))
        obs, reward, done, info = env.step(rand_action)
        total_reward += reward
        # print(info["pitch_angle"])
        # print(f"{'=' * 10}")
        pitch_angles.append(info["pitch_angle"])

        if done:
            print(f"Terminated: {info['reason']}, pitch: {info['pitch_angle']}")
            print(f"Accululated Reward: {total_reward}")
            break

    y = np.arange(500)
    plt.plot(y, pitch_angles)

    plt.xlabel("Time Step")
    plt.ylabel("Pitch angle (radians)")

    plt.show()

    # testing reset with seed
    env.restart(seed=10)

    obs_1, reward_1, done, info = env.step(np.array([1.0, 1.0]))

    env.restart(seed=10)

    obs_2, reward_2, done, info = env.step(np.array([1.0, 1.0]))

    print(f"obs_1: {obs_1}, reward_1: {reward_1}")
    print(f"obs_2: {obs_2}, reward_2: {reward_2}")
