"""
BracketBot LQR Controller

This module implements a Linear Quadratic Regulator (LQR) for the BracketBot,
a two-wheeled inverted pendulum robot. The controller can stabilize the robot
in an upright position and move it to desired positions.

Physical System:
- Two-wheeled differential drive robot (like a Segway)
- Tall vertical body with center of mass above wheel axle
- Control inputs: torques to left and right wheels (or single control for coupled wheels)

State Vector (for single control version):
    x = [x_pos, pitch, x_vel, pitch_vel]
where:
    - x_pos: horizontal position of the robot base (m)
    - pitch: pitch angle from vertical (rad, 0 = upright)
    - x_vel: horizontal velocity (m/s)
    - pitch_vel: pitch angular velocity (rad/s)

Control Input:
    u = [tau] (for single control with coupled wheels)
where:
    - tau: motor torque applied to wheels (N·m)
"""

import numpy as np
import scipy.linalg
from typing import Optional, Tuple
import jax.numpy as jnp


class BracketBot_LQR:
    """
    Linear Quadratic Regulator controller for BracketBot.

    This controller is designed for the single-control version where both wheels
    are coupled (move together). It balances the robot while achieving position control.
    """

    def __init__(
        self,
        # Physical parameters from BracketBot.xml
        m_body: float = 12.0,      # Mass of body (upper + lower) in kg
        m_wheel: float = 53.0,      # Total mass of wheels in kg
        l_com: float = 4.1,         # Distance to center of mass in m
        r_wheel: float = 0.4125,    # Wheel radius in m
        I_body: Optional[float] = None,  # Body moment of inertia (computed if None)
        I_wheel: Optional[float] = None, # Wheel moment of inertia (computed if None)
        g: float = 9.81,            # Gravitational acceleration
        motor_gear: float = 200.0,  # Motor gear ratio from XML
        # LQR cost matrices
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ):
        """
        Initialize the LQR controller with physical parameters.

        Parameters from BracketBot_single_control.xml:
        - Body: upper box (0.1 x 0.1 x 4m, 8kg) + lower box (0.1 x 0.1 x 1m, 4kg)
        - Wheels: 2 wheels (radius 0.4125m, width 0.125m, 26.5kg each)
        - Motor gear: 200
        """
        # Physical parameters
        self.m_body = m_body
        self.m_wheel = m_wheel
        self.l_com = l_com  # Distance to center of mass from wheel axle
        self.r_wheel = r_wheel
        self.g = g
        self.motor_gear = motor_gear

        # Compute moments of inertia if not provided
        if I_body is None:
            # Approximate body as a thin rod of length 8m, mass 12kg
            # I = (1/3) * m * L^2 about base (point mass approximation)
            self.I_body = (1/3) * self.m_body * (8.0) ** 2
        else:
            self.I_body = I_body

        if I_wheel is None:
            # Cylinder: I = (1/2) * m * r^2
            self.I_wheel = 0.5 * self.m_wheel * self.r_wheel ** 2
        else:
            self.I_wheel = I_wheel

        # Target state (position, angle, velocities all zero initially)
        self.target_state = np.zeros(4)

        # Design cost matrices if not provided
        if Q is None:
            # State penalties: [x_pos, pitch, x_vel, pitch_vel]
            # Priority: pitch angle (must stay upright) > position > velocities
            Q = np.diag([
                10.0,    # x_pos - moderate penalty on position error
                200.0,   # pitch - CRITICAL: high penalty to keep upright
                1.0,     # x_vel - small penalty on velocity
                50.0,    # pitch_vel - moderate-high penalty on pitch rate
            ])

        if R is None:
            # Control penalty: tau (motor torque)
            # Lower R = more aggressive control, Higher R = smoother control
            R = np.array([[1.0]])  # Single control input

        self.Q = Q
        self.R = R

        # Compute linearized dynamics
        self.A, self.B = self._linearize()

        # Solve for optimal gains
        self.K = self._compute_gain()

        print("BracketBot LQR Controller Initialized")
        print(f"Physical Parameters:")
        print(f"  Body mass: {self.m_body:.2f} kg")
        print(f"  Wheel mass: {self.m_wheel:.2f} kg")
        print(f"  CoM height: {self.l_com:.2f} m")
        print(f"  Wheel radius: {self.r_wheel:.3f} m")
        print(f"  Body inertia: {self.I_body:.2f} kg·m²")
        print(f"  Wheel inertia: {self.I_wheel:.2f} kg·m²")
        print(f"\nControl Gains (K):")
        print(f"  {self.K}")

    def _linearize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the two-wheeled inverted pendulum dynamics about the upright equilibrium.

        Simplified state for a two-wheeled inverted pendulum (cart-pole type):
        State: x = [x_pos, pitch, x_vel, pitch_vel]
        Control: u = [tau]

        The wheel angle is not independently controllable (it's coupled with cart position),
        so we use a 4-state model focusing on position and balance.

        Returns:
            A: State matrix (4x4)
            B: Control matrix (4x1)
        """
        # Shorthand notation
        m_b = self.m_body
        m_w = self.m_wheel
        l = self.l_com
        r = self.r_wheel
        I_b = self.I_body
        g = self.g

        # Simplified model treating wheels as point masses
        # Total mass
        m_total = m_b + m_w

        # Denominator for coupled dynamics
        # From Euler-Lagrange equations for inverted pendulum on cart
        D = I_b * m_total + m_b * m_w * l**2

        # Linearized dynamics around upright position (pitch = 0)
        # ẋ = Ax + Bu

        A = np.zeros((4, 4))

        # Position dynamics: ẋ_pos = x_vel
        A[0, 2] = 1.0

        # Angle dynamics: pitch_dot = pitch_vel
        A[1, 3] = 1.0

        # Acceleration coupling terms (derived from linearized equations of motion)
        # These represent how gravity affects the system when pitched

        # Effect of pitch angle on cart acceleration
        # When pitched forward, gravity pulls the CoM, accelerating the cart
        a_23 = (m_b**2 * g * l**2) / D

        # Effect of pitch angle on pitch acceleration
        # Gravitational restoring torque (negative feedback when upright)
        a_33 = -(m_b * g * l * m_total) / D

        A[2, 1] = a_23  # pitch affects x acceleration
        A[3, 1] = a_33  # pitch affects angular acceleration

        # Control effectiveness matrix
        B = np.zeros((4, 1))

        # Effect of motor torque on system
        # Motor torque applied to wheels creates force F = tau/r on cart

        # Torque to cart acceleration
        # Positive torque -> wheels spin forward -> cart moves forward
        b_2 = (I_b + m_b * l**2) / (D * r)

        # Torque to pitch acceleration
        # Forward acceleration causes cart to lean backward (negative)
        b_3 = -(m_b * l) / (D * r)

        B[2, 0] = b_2  # tau affects x_vel
        B[3, 0] = b_3  # tau affects pitch_vel

        return A, B

    def _compute_gain(self) -> np.ndarray:
        """
        Compute optimal LQR gain matrix by solving the Continuous-Time
        Algebraic Riccati Equation (CARE).

        Returns:
            K: Optimal gain matrix (1x4) for u = -K(x - x_target)
        """
        # Verify controllability before computing gains
        controllability_matrix = self._controllability_matrix()
        rank = np.linalg.matrix_rank(controllability_matrix)

        if rank < self.A.shape[0]:
            print(f"Warning: System may not be fully controllable (rank={rank}/{self.A.shape[0]})")

        # Solve Continuous-Time Algebraic Riccati Equation
        # A^T P + P A - P B R^{-1} B^T P + Q = 0
        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

        # Compute optimal gain: K = R^{-1} B^T P
        K = np.linalg.inv(self.R) @ self.B.T @ P

        return K

    def _controllability_matrix(self) -> np.ndarray:
        """
        Compute controllability matrix [B, AB, A^2B, ..., A^(n-1)B].
        System is controllable if this matrix has full rank.
        """
        n = self.A.shape[0]
        C = self.B.copy()

        for i in range(1, n):
            C = np.hstack([C, np.linalg.matrix_power(self.A, i) @ self.B])

        return C

    def set_target(self, x_target: float = 0.0, velocity_target: float = 0.0):
        """
        Set target position for the robot.

        Args:
            x_target: Desired x position (m)
            velocity_target: Desired velocity (m/s), usually 0
        """
        # Target state: [x_pos, pitch, x_vel, pitch_vel]
        # We want the robot at x_target, upright (pitch=0), at rest
        self.target_state = np.array([
            x_target,
            0.0,  # upright
            velocity_target,
            0.0,  # no pitch velocity
        ])

    def compute_control(self, state: np.ndarray) -> np.ndarray:
        """
        Compute optimal control action using LQR.

        Args:
            state: Current state [x_pos, pitch, x_vel, pitch_vel]

        Returns:
            control: Motor torque [tau]
        """
        # LQR control law: u = -K(x - x_target)
        state_error = state - self.target_state
        control = -self.K @ state_error

        return control

    def compute_control_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-compatible control computation for use with Brax environments.

        Args:
            state: Current state as JAX array

        Returns:
            control: Motor torque as JAX array
        """
        state_error = state - jnp.array(self.target_state)
        K_jax = jnp.array(self.K)
        control = -K_jax @ state_error

        return control

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Allow controller to be called as a function."""
        return self.compute_control(state)

    def get_state_from_pipeline(self, pipeline_state) -> np.ndarray:
        """
        Extract LQR state vector from Brax pipeline state.

        From uni_bracketbot_env.py documentation:
        pipeline_state.q:  [x, y, z, qw, qx, qy, qz, wheel_r, wheel_l]
        pipeline_state.qd: [vx, vy, vz, wx, wy, wz, wheel_r_vel, wheel_l_vel]

        Returns:
            state: [x_pos, pitch, x_vel, pitch_vel]
        """
        # Position
        x_pos = pipeline_state.q[0]

        # Pitch angle (rotation about y-axis)
        quat = pipeline_state.q[3:7]  # [qw, qx, qy, qz]
        pitch = self._pitch_from_quat(quat)

        # Velocity
        x_vel = pipeline_state.qd[0]

        # Pitch velocity (angular velocity about y-axis)
        pitch_vel = pipeline_state.qd[4]  # wy

        state = np.array([x_pos, pitch, x_vel, pitch_vel])

        return state

    def _pitch_from_quat(self, quat):
        """
        Extract pitch angle from quaternion.

        Args:
            quat: Quaternion [w, x, y, z]

        Returns:
            pitch: Pitch angle in radians (numpy or JAX array scalar)
        """
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Pitch (rotation about y-axis)
        # Works with both numpy and jax arrays
        if hasattr(quat, 'device') or str(type(quat).__module__).startswith('jax'):
            pitch = jnp.arcsin(2 * (w * y - z * x))
        else:
            pitch = np.arcsin(2 * (w * y - z * x))
        return pitch


class BracketBot_LQR_TwoControl:
    """
    LQR controller for BracketBot with independent left/right wheel control.

    This version allows differential drive for turning while balancing.

    State: x = [x, y, pitch, yaw, x_vel, y_vel, pitch_vel, yaw_vel, wheel_l, wheel_r, wheel_l_vel, wheel_r_vel]
    Control: u = [tau_left, tau_right]

    Note: This is more complex as it includes both forward/backward motion AND turning.
    """

    def __init__(self):
        """
        Initialize two-control LQR controller.

        TODO: Implement full state-space model for differential drive.
        For now, use single-control version for forward/backward balance.
        """
        raise NotImplementedError(
            "Two-control LQR not yet implemented. Use BracketBot_LQR for single-control version."
        )


if __name__ == "__main__":
    # Test the controller
    print("Testing BracketBot LQR Controller\n")

    # Create controller with default parameters
    controller = BracketBot_LQR()

    print("\n--- Test 1: Stabilization at origin ---")
    controller.set_target(x_target=0.0)

    # Simulate small perturbation
    state = np.array([0.0, 0.1, 0.0, 0.0])  # 0.1 rad pitch error
    control = controller.compute_control(state)
    print(f"State: {state}")
    print(f"Control: {control}")
    print(f"Expected: Negative torque to correct forward lean")

    print("\n--- Test 2: Position control ---")
    controller.set_target(x_target=1.0)

    state = np.array([0.0, 0.0, 0.0, 0.0])  # At origin, need to move to x=1
    control = controller.compute_control(state)
    print(f"State: {state}")
    print(f"Target position: 1.0 m")
    print(f"Control: {control}")
    print(f"Expected: Positive torque to accelerate forward")

    print("\n--- Test 3: Controllability check ---")
    C = controller._controllability_matrix()
    rank = np.linalg.matrix_rank(C)
    print(f"Controllability matrix rank: {rank}/{controller.A.shape[0]}")
    print(f"System is {'fully controllable' if rank == controller.A.shape[0] else 'NOT fully controllable'}")
