import mujoco
import mujoco.viewer
import numpy as np
from numpy import random
from numpy.random import Generator
import time
from typing import Optional


class TwoAxisInvertedPendulum:
    def __init__(
        self, xml_path: str = "../assets/xml/inverted_two_axis_pendulum_v2.xml"
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.maxAct_val = 1.0
        self.minAct_val = -1.0

    def reset(self, rng: Optional[Generator] = None):
        mujoco.mj_resetData(self.model, self.data)

        if rng is not None:
            pos = random.uniform(0.5, -0.5, 2)
            angle = random.uniform(-0.5, 0.5, 2)

            self.data.qpos[0:2] = pos
            self.data.qpos[2:4] = angle
            # mujoco.mj_step(self.model, self.data)

            # add also random elements to qvel

    def get_obs(self):
        """
        | Idx | Observation                                                |
        |-----|------------------------------------------------------------|
        |  0  | x position of the cart                                     |
        |  1  | y position of the cart                                     |
        |  2  | angle of cart's pole along the x axis expressed in radians |
        |  3  | angle of cart's pole along the y axis expressed in raidans |
        |  4  | x velocity of the cart                                     |
        |  5  | y velocity of the cart                                     |
        |  6  | angular velocity of cart along the x  axis                 |
        |  7  | angular velocity of cart along the y  axis                 |
        """

        x = np.concatenate([self.data.qpos, self.data.qvel])

        return x

    def get_obs_2p(self):
        x = self.get_obs()

        x = [float(round(a, 2)) for a in x]

        return x

    def control(self, action: np.ndarray) -> np.ndarray:
        # self.data.ctrl[:] = np.clip(action, self.minAct_val, self.maxAct_val)

        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)


if __name__ == "__main__":
    sys = TwoAxisInvertedPendulum()

    rng = random.default_rng(1)
    sys.reset(rng)
    x = sys.get_obs_2p()
    print(x)

    # interative viewer?

    with mujoco.viewer.launch_passive(sys.model, sys.data) as viewer:
        while viewer.is_running():
            # rand_action = random.uniform(-1, 1, size=2)
            # sys.control(rand_action)

            viewer.sync()

            time.sleep(0.01)
