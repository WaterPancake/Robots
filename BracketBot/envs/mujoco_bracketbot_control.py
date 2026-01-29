import mujoco
import mujoco.renderer
import numpy as np
import os
from typing import Optional
import mediapy as media


class EnvConfig:
    pass


class BracketBotCntrEnv:
    def __init__(
        self, xml_path="../assets/BracketBot.xml", render_width=1280, render_height=720
    ):
        """
        qpos: [x, y, z, qw, qx, qy, qz, left_wheel_angle, right_wheel_angle]
        qvel: [vx, vy, vz, wx, wy, wz, left_wheel_vel, right_wheel_vel]
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Rendering params (lazy initialization)
        self.renderer = mujoco.Renderer(self.model)
        self.renderer.update_scene(self.data)

        self.frames = [self.renderer.render()]

        # Mujoco Backend params
        self.n_frames = 2

        # Dim Space
        self.obs_dim = 14  # see obs() for details
        self.act_dim = 2  # [left_wheel_vel, right_wheel_vel]

        # Action Space
        self.min_action = -1.0
        self.max_action = 1.0

        self._command = np.zeros(2)

        self.step_count = 0

        # Env params
        self.rng = np.random.default_rng()  # for seed reset
        self.fall_angle = np.pi / 3  # ~ 60 away upright or 30 from ground
        self.episode_length = 2000  # total calls of mj_step is n_frams * episode_length

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    # FIX: seeds functionality
    def reset(self, seed: int = -1) -> np.ndarray:
        if seed != -1:
            self.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        q_min, q_max = -0.5, 0.5
        self.data.qpos[0:2] += self.rng.uniform(size=2, high=q_max, low=q_min)

        qd_min, qd_max = -0.5, 0.5
        self.data.qvel[0:2] += self.rng.uniform(size=2, high=qd_max, low=qd_min)

        # TODO: test various randomized inital states
        self.data.qvel[4] += self.rng.uniform(-0.5, 0.5)  # random pitch

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.frames = []
        self.renderer = mujoco.Renderer(self.model)

        return self._obs()

    def control(self, forward_vel, turn_vel):
        self._command[0] = np.clip(
            forward_vel, min=self.min_action, max=self.max_action
        )

        self._command[0] = np.clip(turn, min=self.min_action, max=self.max_action)

    # FIX: time limi?
    def step(self, action: np.array) -> tuple[np.ndarray, float, bool, dict]:
        # borrowing Gymnaisum return format for later to compare it to using  Gymnasium env approach
        """
        Return format:
            np.ndarray: new observation
            float: reward
            bool: done
            dict: additonal information
        """

        action = np.clip(action, self.min_action, self.max_action)  # [-1.0, 1.0]
        self.data.ctrl = action

        for _ in range(self.n_frames):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._obs()
        reward, reward_info = self._reward(obs, action)
        done, reason = self._terminate(obs)

        self.renderer.update_scene(self.data)
        self.frames.append(self.renderer.render())

        info = {"step": self.step_count, "reason": reason, **reward_info}

        return obs, float(reward), done, info

    def _terminate(self, obs) -> tuple[bool, str]:
        """
        Two end conditons:
        1. Pole pitch angle falls past self.fall_angle
        2. 1000 time steps elapses
        """
        pitch = obs[6]

        if np.abs(pitch) > self.fall_angle:
            return True, "Fall"
        elif self.step_count >= self.episode_length:
            return True, "Truncated"
        else:
            return False, ""

    def _roll(self):
        quat = self.data.qpos[3:7]  # w, x, y, z
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    def _pitch(self):
        quat = self.data.qpos[3:7]  # w, x, y, z
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return np.asin(2 * ((w * y) - (z * x)))

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

        cmd_forward = obs[12]
        cmd_turn = obs[13]

        roll, roll_vel = obs[6], obs[9]

        yaw, yaw_vel = obs[5], obs[8]

        # Balancing reward
        angle_reward = np.cos(3 * roll)  # zero at pi/4 = 45 degrees

        velocity_penalty = roll_vel**2

        action_penalty = np.sum(action**2)

        # Command rewards
        vx, vy = obs[2], obs[3]
        forward_vel = vx * np.cos(vx) + vy * np.sin(vy)

        forward_error = (forward_vel - cmd_forward) ** 2
        turn_error = np.exp(yaw_vel - cmd_turn) ** 2

        # evals to 1 when both command and bot vel match
        forward_reward = np.exp(-forward_error)
        turn_reward = np.exp(-turn_error)

        # reward weights
        angle_weight = 1.0
        action_weight = 0.01
        velocity_weight = 0.1
        alive_weight = 0.1
        alive = 1.0

        forward_vel_weight = 1.0
        turn_vel_weight = 0.5

        reward = (
            (angle_weight * angle_reward)
            + (alive_weight * alive)
            - (velocity_penalty * velocity_weight)
            - (action_weight * action_penalty)
            # command
            + (forward_vel_weight * forward_reward)
            + (turn_vel_weight * turn_reward)
        )

        reward_info = {
            "angle_reward": angle_reward,
            "action_penalty": action_penalty,
            "velocity_penalty": velocity_penalty,
            "roll_angle": roll,
            "roll_angle_degree": np.rad2deg(roll),
            "forward_vel": forward_vel,
            "turn_vel": yaw,
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
         6  | Roll angle of cart in radians (what is used by reward function)
         7  | Cart angular velocity along y axis in radians (pitch velocity)
         8  | Cart angular velocity along z axis in radians (yaw velocity)
         9  | Cart angular velocity along the x axis in radian (roll velocity)
         10 | left wheel velocity
         11 | right wheel velocity

         12 | forward command velocity
         13 | turn command velocoty
        """
        q = self.data.qpos
        qd = self.data.qvel

        # will be used later, for now, using a full state discription
        roll = self._roll()
        pitch = self._pitch()
        yaw = self._yaw()

        # vroll
        pitch_vel = qd[4]  # wy
        yaw_vel = qd[5]  # wz
        roll_vel = qd[3]  # wx

        base_position = q[0:2]  # x, y
        base_velocity = qd[0:2]  # vx, vy
        wheel_vel = qd[-2:]  # left_wheel_vel, right_wheel_vel

        obs = np.concatenate(
            [
                base_position,
                base_velocity,
                [pitch],
                [yaw],
                [roll],
                [pitch_vel],
                [yaw_vel],
                [roll_vel],
                wheel_vel,
                self._command,
            ]
        )
        # dim = 12

        return obs.astype(np.float32)

        """Minimized version"""

    @property
    def _act_dim(self) -> int:
        return self.act_dim

    @property
    def _obs_dim(self) -> int:
        return self.obs_dim


#
# for testing
#
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mujoco.renderer

    env = BracketBotCntrEnv()
    renderer = mujoco.Renderer(env.model)
    renderer.update_scene(env.data, camera="profile")

    obs = env.reset(seed=421)

    # test a half episode
    total_reward = 0
    # obs = env.reset(seed=0)
    pitch_angles = []
    yaw_angles = []
    roll_angles = []
    frames = []
    for i in range(1000):
        rand_action = np.random.uniform(size=2)
        # zero_action = np.zeros(2)
        obs, reward, done, info = env.step(rand_action)
        # renderer.update_scene(env.data, camera="profile")
        # frames.append(renderer.render())
        total_reward += reward

        pitch_angles.append(np.rad2deg(obs[4]))
        yaw_angles.append(np.rad2deg(obs[5]))
        roll_angles.append(np.rad2deg(obs[6]))

        if done:
            print(f"Terminated: {info['reason']}, pitch: {info['roll_angle_degree']}")
            print(f"Accululated Reward: {total_reward}")

            break

    """tracking pitch"""

    fig, axs = plt.subplots(3)
    y = np.arange(len(pitch_angles))

    axs[0].plot(y, pitch_angles)
    axs[0].set_title("Pitch")

    axs[1].plot(y, yaw_angles)
    axs[1].set_title("Yaw")

    axs[2].plot(y, roll_angles)
    axs[2].set_title("Roll")
    axs[2].plot(y, [60] * len(y))

    rollout_frames = env.frames
    out_file = "rollout0.mp4"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(cur_dir, out_file)
    media.write_video(out_path, rollout_frames, fps=60)

    """testing seed's reproducability"""
    env.reset(seed=10)

    obs_1, reward_1, done, info = env.step(np.array([1.0, 1.0]))

    env.reset(seed=10)

    obs_2, reward_2, done, info = env.step(np.array([1.0, 1.0]))

    print(f"obs_1: {obs_1}, reward_1: {reward_1}")
    print("\n")
    print(f"obs_2: {obs_2}, reward_2: {reward_2}")

    if np.array_equal(obs_1, obs_2):
        print("Working as intended.")

    plt.show()
