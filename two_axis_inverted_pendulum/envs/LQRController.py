import jax
import jax.numpy as jnp
from jax import jit
import scipy.linalg
from typing import Tuple, Optional
import numpy as np


class InvertedTwoAxisPendulum_LQR:
    def __init__(
        self,
        m_cart: float = 1.0,
        m_pole: float = 0.1,
        l: float = 0.6,
        Q: np.ndarray = None,
        R: np.ndarray = None,
    ):
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.l = l
        self.g = 9.81
        self.inertia = 0  # figure out actual value 

        if Q is None:
            Q = np.diag([10, 10, 100, 100, 1, 1, 10, 10], dtype=float)

        if R is None:
            R = np.diag([1, 1], dtype=float)

        self.Q = Q
        self.R = R

    def _linearize(self):
        """
        About the upright fixed point x = [0, 0, pi, pi, 0, 0, 0, 0].
        See README.md for understanding A's construction.
        """
        m_c = self.m_cart
        m_p = self.m_pole
        l = self.l
        g = self.g
        i = self.inertia

        denom = (m_c + m_p) * (i + m_p * l**2) - m_p**2 * l**2
        
        A = np.zeros((8, 8))
        
        a_1 = (m_p**2) * g * (l**2) / denom
        a_2 = -m_p * (m_p + m_c) * g * l / denom

        A[2,4] = a_1
        A[3,5] = a_1
        A[2,6] = a_2
        A[3,6] = a_2

        B = np.zeros((2,8))

        b_1 = i + m_p * l**2 / denom
        b_2 = -m_p * l / denom

        B[0,4] = b_1
        B[1,5] = b_1
        B[0,6] = b_2
        B[1,7] = b_2


    # Solves the Ricatti Equaiton
    def _computer_gain(self):
        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

        K = np.linalg.inv(self.R) @ self.B.T @ P

        return K

    # for controlling system
    def _set_target(self, target_x: float = 0.0, target_y: float = 0.0):
        self.target_state = np.array([target_x, target_y, 0, 0, 0, 0, 0, 0])

        

    def _control(self, cur_state):
        
