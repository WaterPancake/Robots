import numpy as np
import scipy
from typing import Optional
# import keyboard


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

        self.Inertia = self.m_pole * self.ell**2

        self.A, self.B = self.linearize()

        if Q is not None:
            self.Q = Q
        else:
            self.Q = np.diag([1.0, 1.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0])

        if R is not None:
            self.R = R
        else:
            self.R = np.diag([1.0, 1.0])

        self.K = self.solve()

    def linearize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        State: [x, y, θ_x, θ_y, ẋ, ẏ, θ̇_x, θ̇_y]
        """

        D = (self.m_cart + self.m_pole) * (self.Inertia + self.m_pole * self.ell**2) - (
            self.m_pole * self.ell
        ) ** 2

        p = (self.m_cart + self.m_pole) / D
        q = (self.m_pole**2 * self.ell**2 * self.g) / D
        r = (
            self.Inertia * self.m_pole * self.g * self.ell
            - self.m_pole**2 * self.ell**3 * self.g
        ) / D
        s = self.m_pole * self.ell / D

        A = np.zeros((8, 8))

        A[0, 4] = 1.0
        A[1, 5] = 1.0
        A[2, 6] = 1.0
        A[3, 7] = 1.0
        A[4, 2] = q
        A[5, 2] = r

        A[6, 3] = q
        A[7, 3] = r

        B = np.zeros((8, 2))

        B[4, 0] = p
        B[5, 1] = p

        B[6, 0] = s
        B[7, 1] = s

        return A, B

    def solve(self):
        # K = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R) # for continious time
        P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K

    def control(self, cur_state: np.ndarray):
        """
        current_state will be obtained via backend (either mujoco or gymnaisum)
        """
        return -self.K @ cur_state


class BangBangTwoAxisInvertedPendulum:
    # 0.35 radians ≈ 20 degrees
    # 0.25 radians ≈ 15 degree
    # 0.175 radians ≈ 10 degree
    def __init__(self, theta_threshold: float = 0.35):
        self.theta_threshold = theta_threshold

    # def control(self, theta_x: float, theta_y: float):
    #     """ """
    #     F_x, F_y = 0, 0

    #     if theta_x > self.theta_threshold:
    #         F_x = 0.7
    #     elif theta_x < -self.theta_threshold:
    #         F_x = -0.7

    #     if theta_y > self.theta_threshold:
    #         F_y = -0.7
    #     elif theta_y < -self.theta_threshold:
    #         F_y = 0.7

    #     return np.array([F_x, F_y])

    def control(self, cur_state: np.ndarray):
        """ """
        F_x, F_y = cur_state[2:4]

        if theta_x > self.theta_threshold:
            F_x = 0.7
        elif theta_x < -self.theta_threshold:
            F_x = -0.7

        if theta_y > self.theta_threshold:
            F_y = -0.7
        elif theta_y < -self.theta_threshold:
            F_y = 0.7

        return np.array([F_x, F_y])


if __name__ == "__main__":
    import mujoco
    import mujoco.viewer
    import time
    from numpy import random
    from interface import TwoAxisInvertedPendulum

    sys = TwoAxisInvertedPendulum()

    theta_threshold = 0.25
    BBcntr = BangBangTwoAxisInvertedPendulum(theta_threshold=theta_threshold)
    LQRcntr = LQR_TwoAxisInvertedPendulum()

    rng = random.default_rng(17)
    sys.reset(rng)

    with mujoco.viewer.launch_passive(sys.model, sys.data) as viewer:
        while viewer.is_running():
            x = sys.get_obs()
            theta_x, theta_y = x[2] % np.pi, x[3] % np.pi

            if abs(theta_x) < theta_threshold or abs(theta_y) < theta_threshold:
                u = BBcntr.control(x)
            else:
                u = LQRcntr.control(x)

            angle_x = theta_x
            angle_y = theta_y
            print(f"x: {round(angle_x, 2)}°, y: {round(angle_y, 2)}°, u = {u}")

            sys.control(u)

            viewer.sync()

            time.sleep(0.1)
