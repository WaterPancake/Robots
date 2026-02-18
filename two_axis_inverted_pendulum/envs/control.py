import numpy as np
import scipy
from typing import Optional


class LQR_TwoAxisInvertedPendulum:
    def __init__(
        self,
        m_cart: float = 1.0,
        m_pole: float = 0.25,
        ell: float = 0.6,
        g: float = 9.81,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ):
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.ell = ell
        self.g = g

        self.Inertia = self.m_pole * self.l**2

        self.A, self.B = self.linearize()

        if Q is None:
            self.Q = np.diag([10, 10, 100, 1, 1, 1, 1])

        if R is not None:
            self.R = np.zeros((1, 2))
            self.R[0, 1] = 1
            self.R[1, 1] = 1

    def linearize(self):
        """
        State: [x, y, θ_x, θ_y, ẋ, ẏ, θ̇_x, θ̇_y]
        """

        D = (self.m_cart + self.m_pole) * (self.Inertia + self.m_pole * self.l**2) - (
            self.m_pole * self.l
        ) ** 2

        p = (self.m_cart + self.m_pole) / D  # (m_c + m_p) / D
        q = (self.m_pole**2 * self.l**2 * self.g) / D  # (m_p^2 * l^2  * g) / D)
        r = (
            self.Inertia * self.m_polei * self.g * self.l - self.m_pole**2 * self.l**2
        ) / D  # (I * m_p * g * l - m_p^2 * l^3) / D
        s = self.m_pole * self.l / D  # (m_p * l) / D

        A = np.zeros((8, 8))

        A[0, 4] = 1
        A[1, 5] = 1
        A[2, 6] = 1
        A[3, 7] = 1
        A[4, 2] = q
        A[5, 2] = r

        A[6, 3] = q
        A[7, 3] = r

        B = np.zeros((2, 8))

        B[0, 4] = p
        B[1, 5] = p

        B[0, 6] = s
        B[1, 7] = s

        return A, B

    def solve(self):
        K = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

        return K

    def control(self, cur_state: np.ndarray, K: np.ndarray):
        """
        current_state will be obtained via backend (either mujoco or gymnaisum)
        """


class BangBangTwoAxisInvertedPendulum:
    # 0.35 radians ≈ 20 degrees
    def __init__(self, theta_threshold: float = 0.35):
        self.theta_threshold = theta_threshold

    def control(self, theta_x: float, theta_y: float):
        """ """
        F_x, F_y = 0, 0

        if theta_x > self.theta_threshold:
            F_x = -0.5
        elif theta_x < -self.theta_threshold:
            F_x = 0.5

        if theta_y > self.theta_threshold:
            F_x = -0.5
        elif theta_y < -self.theta_threshold:
            F_x = 0.5

        return np.array([F_x, F_y])


if __name__ == "__main__":
    import mujoco
    import mujoco.viewer
    import time
    from numpy import random
    from interface import TwoAxisInvertedPendulum

    sys = TwoAxisInvertedPendulum()

    cntr = BangBangTwoAxisInvertedPendulum()

    rng = random.default_rng(1)
    sys.reset(rng)
    # x = sys.get_obs_2p()
    # print(x)

    with mujoco.viewer.launch_passive(sys.model, sys.data) as viewer:
        while viewer.is_running():
            x = sys.get_obs()
            theta_x, theta_y = x[2], x[3]

            u = cntr.control(theta_x, theta_y)

            print(u)
            sys.control(u)

            viewer.sync()

            time.sleep(0.2)
