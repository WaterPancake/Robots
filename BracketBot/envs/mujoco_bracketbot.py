"""
Following Gymnasium guide (https://gymnasium.farama.org/introduction/create_custom_env/)
"""

import mujoco
import numpy as np


class EnvConfig:
    pass


class BracketBotEnv:
    def __init__(self, xml_path="../assets/BracketBot.xml"):
        """
        qpos: [x, y, z, qw, qx, qy, qz, left_wheel_angle, right_wheel_angle]
        qvel: [vx, vy, vz, wx, wy, wz, left_wheel_vel, right_wheel_vel]
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.sys = mujoco.MjData(self.model)

        # Mujoco Backend params
        self.n_frames = 2

        # Dim Space
        self.obs_dim = 10  # see _obs() for details
        self.act_dim = 2  # [left_wheel_vel, right_wheel_vel]

        # Action Space
        self.min_action = -1.0
        self.max_action = 1.0

        self.step_count = 0

        # env params
        self.rng = np.random.default_rng()  # for seed reset
        self.fall_angle = np.pi / 6  # ~ 60 from upright

    def restart(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.sys)

        # randomizing intial conditon

        q_min, q_max = -0.5, 0.5
        self.sys.qpos[0:2] += self.rng.uniform(size=2, high=q_max, low=q_min)

        qd_min, qd_max = -0.5, 0.5

        self.sys.qvel[0:2] += self.rng.uniform(size=2, high=qd_max, low=qd_min)

    def step(
        self, action: np.array
    ) -> tuple[np.ndarray, float, bool, dict]:  # justify why this format
        """
        Return format:
            np.ndarray: new observation
            float: reward
            bool: done
            dict: additonal information
        """

        action = np.clip(action, self.min_action, self.max_action)
        self.sys.ctrl = action

        for _ in range(self.n_frames):
            mujoco.mj_step(self.model, self.sys)

        self.step_count += 1
        obs = self._obs()

        reward, info = self._reward(obs, action)
        # done = self._terminate()

        # either use mj_forward or mj_step??

    def _terminate(self):
        """
        for this version, terminate only after 1000 time steps
        """
        pass

    def _roll(self):
        quat = self.sys.qpos[3:7]  # w, x, y, z
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    def _pitch(self):
        quat = self.sys.qpos[3:7]  # w, x, y, z
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return np.arcsin(2 * (w * y - z * x))

    def _yaw(self):
        quat = self.sys.qpos[3:7]  # w, x, y, z
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _reward(self, obs: np.ndarray, action: np.ndarray) -> tuple[float, dict]:
        """
        Return format:
            float: scalar reward
            dict: additonal information
        """

        pitch, pitch_vel = obs[4], obs[6]

        angle_reward = np.coos(3 * pitch)  # zero at pi/4 = 45 degrees

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

        info = {
            "angle_reward": angle_reward,
            "action_penalty": action_penalty,
            "velocity_penalty": velocity_penalty,
            "pitch_angle": pitch,
            "pitch_angle_degree": np.rad2deg(pitch),
            "reward": reward,
        }

        return reward, info

    @property
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
        q = self.sys.qpos
        qd = self.sys.qvel

        # will be used later, for now, using a full state discription
        # roll = self._roll()
        pitch = self._pitch()
        yaw = self._yaw()

        # vroll
        pitch_vel = qd[4]  # wy
        yaw = qd[5]  # wz

        base_position = q[0:2]  # x, y
        base_velocity = qd[0:2]  # vx, vy
        wheel_vel = qd[-2:]  # left_wheel_vel, right_wheel_vel

        obs = np.concatenate(
            base_position,
            base_velocity,
            [pitch],
            [yaw],
            [pitch_vel],
            [yaw_vel],
            wheel_vel,
        )
        # dim = 10

        return obs.astype(np.float32)
        # return np.concatenate(q, qd)

    @property
    def _action_space(self) -> int:
        return self.act_dim

    @property
    def _observation_space(self) -> int:
        return self.obs_dim
