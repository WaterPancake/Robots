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
        self.obs_dim = 10  # []
        self.act_dim = 2

        # Action Space, limits should be multiple's of 10 for PPO
        self.min_action = -100.0
        self.max_action = 100.0

        self.step_count = 0

        # env params
        self.rng = np.random.default_rng()  # for seed reset

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
        Return
            np.ndarray: new observation
            float: reward
            bool: done
            dict: additonal information
        """

        action = np.clip(action, self.min_action, self.max_action)
        self.sys.ctrl = action

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

    def _reward(self):
        pass

    @property
    def _obs(self):
        q = self.sys.qpos
        qd = self.sys.qvel

        # will be used later, for now, using a full state discription
        roll = self._rol()
        pitch = self._pitch()
        yaw = self._yaw()

        # vroll
        # vpitch
        # vyaw

        base_position = q[0:2]  # x, y
        base_velocity = qd[0:2]  # vx, vy
        wheel_vel = qd[-2:]  # left_wheel_vel, right_wheel_vel

        return np.concatenate(base_position, base_velocity, wheel_vel, [pitch], [yaw])

        return np.concatenate(q, qd)

    @property
    def _action_space(self) -> int:
        return self.act_dim

    @property
    def _observation_space(self) -> int:
        return self.obs_dim
