# Two Axis Cart Pole
is an extension of the canonical Cart Pole by adding another dimension of traversal and control to the cart, and another degree of freedom to the inverted pendulum.

The model used is Built off the the [inverted pendulum from Brax](https://github.com/google/brax/blob/main/brax/envs/assets/inverted_pendulum.xml) created by Google.

## Environment
The inverted two-axis pendulum can be represented as the following:

$\textbf{x} = \left[ x, y, \theta_x, \theta_y, \dot{x}, \dot y, \dot{\theta_x}, \dot{\theta_y} \right]$


| Variable | Description | minVal | maxVal
| ----- | ----- | --- | --- |
| $x$ | carts position along x-axis | -2 | 2 |
| $y$ | carts position along y-axis | -2 | 2 |
| $\theta_x$ | poles angle along the x axis about the cart in radians | $0$ | $2\pi$ |
| $\theta_y$ | poles angle along the y axis about the cart in radian | $0$ | $2\pi$|
| $\dot x$ | be the linear velocity of cart along x-axis | $-\infty$ | $\infty$ |
| $\dot y$ | be then linear velocity of cart along y-axis | $-\infty$ | $\infty$ |
| $\dot \theta_x$ | the angular velocity of the pole along the x axis about the cart | $-\infty$ | $\infty$ |
| $\dot \theta_y$ | the angular velocity of the pole along the y axis about the car | $-\infty$ | $\infty$ |

The system has control inputs:
$$u = \left[ F_x, F_y \right]^T$$

corresponding to linear force applied to the cart in the $x$ and $y$ axis respectfully.


See `two_axis_inverted_pendulum_env.py` for more details on how the state is obtained from Mujoco XLA.

  

# Learned Swing-up and Balance Controller

The objective of the this controller is to balance the inverted pendulum with two degrees of movement through indirect control of its base.

## Reward function

### Basic swing-up

$$\mathcal{R}_1(s,a) = 2 + \cos(\theta_x) + \cos(\theta_y)$$
  
### Centering and swing-up

In swinging up, the cart will invariably move away from the center as it balances the. This reward function makes the learned controller find a way achieve swing-up, but also make the cart centered. This is achieved by adding penalty proportional to distance to the origin $(0,0)$

$$\mathcal{R}_2(s,a) = 4 \cdot \left(\cos(\theta_x) + \cos(\theta_y)\right) - (x^2 + y^2) $$

  

The $4$ coefficient is to offset the difference in the range of possible values of pole balancing ($[-2,2]$)reward compared to the distance penalty ($[0, -4]$).



## Training


# Linear-Quadradic-Regulator 

**NOTE**: This section uses a different XML model of the inverted two-axis pendulum so that it more accurately represents how the version found in [MIT's Underacuated Robotics](https://underactuated.csail.mit.edu/acrobot.html) with the following parameters for the Cart Pole:

| Parameter | Description | Implemented Value |
| --- | --- | --- |
| $m_c$ | Mass of the cart that moves without friction. | 1.0 kg |
| $m_p$ | Mass of the pole at the end of a rigid, (near) massless pole that move about the cart with two degrees of freedom. | 0.333 kg |
| $\ell$ | The length of the pendulum arm. | 60cm |
| $g$ | Gravity constant | 9.81 $m/s^2$ |

This system makes assumptions like no dampening on force on the cart or friction on the cart 

## Kinematics

  

The position of the pole is in $\mathbb{R}^3$ while carts position in $\mathbb{R}^2$.

$$\textbf{x}_{p} = \begin{bmatrix}

x_p \\ y_p \\ z_p

\end{bmatrix} =\begin{bmatrix}

x_c + \ell \sin(\theta_x)\\

y_c + \ell \sin(\theta_y) \\

\ell \cos(\theta_x) \cos(\theta_y)

\end{bmatrix}$$

  

$$\dot{\textbf{x}}_{p} =

\begin{bmatrix}

\dot x_p \\

\dot y_p \\

\dot z_p

\end{bmatrix}

= \begin{bmatrix}

\dot x_c + \ell \cos(\theta_x) \dot \theta_x\\

\dot y_c + \ell \cos(\theta_y) \dot \theta_y\\

-\ell \sin{(\theta_x)}\cos{(\theta_y)} \dot\theta_x - \ell \sin{(\theta_y)}\cos{(\theta_x)} \dot \theta_y

\end{bmatrix}$$

  

## Equations of Motion

The first step in creating an LQR controller is the linearize the systems equation of motions about a fixed point. We can calculate these equations using the Lagrangian which state that,
$$L = T - V$$ 
where $T$ and $V$ represent the kinetic and potential energy of the system respectively.
  

**Kinetic Energy**


$$T_p = \frac{1}{2} m_p(\dot x_p^2 + \dot y_p^2 + \dot z_p^2)$$


$$T_c = \frac{1}{2} m_c(\dot x_c^2 + \dot y_c^2)$$

The added degree of movement requires that we also consider energy of poles angular movement (rotational kinetic energy) which is:

  
$$T_{r} = \frac{1}{2} I (\dot \theta_x^2 + \theta_y^2)$$

where $I = m_p \ell^2$.

Thus the total kinetic energy defined as: 

$$T = T_p + T_c + T_{r}$$

**Potential Energy**

 The only possible source of potential energy is the pendulum's movement as a result of the gravitational force.

$$V = m_p g \ell \cos(\theta_x) \cos(\theta_y) $$


## Equation of motion

To find the dynamics equations ($Q_i$) for each generalized coordinates ($q_i$) using:

$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot q_i}\right) - \frac{\partial L}{\partial q_i} = Q_i$$

In this system, the generalized coordinates are  $q =\begin{bmatrix} x_c & y_c & \theta_x & \theta_y\end{bmatrix}^T$, but there are variables in $L$ that are not in $q$. Expanding the Lagrangian to contain only the generalized coordinates is: 

$$\begin{align}
L  &= \frac{1}{2}\left(
m_p (\dot x_p^2 + \dot y_p^2 + \dot z_p^2) + I (\dot \theta_x^2 + \dot \theta_y^2) + m_c (\dot x_c^2 + \dot y_c^2) \right)-m_p g \ell \cos(\theta_x) \cos(\theta_y)\notag \\ \notag \\ 
&=\frac{1}{2}\left( m_p\left[ \left(\dot x_c + \ell\cos(\theta_x)\dot \theta_x\right)^2 + \left(\dot y_c +\ell\cos(\theta_y)\dot\theta_y \right)^2 + \left(-\ell \sin{(\theta_x)}\cos{(\theta_y)} \dot\theta_x - \ell \sin{(\theta_y)}\cos{(\theta_x)} \dot \theta_y\right)^2\right] + I(\dot\theta_x^2 + \dot\theta_y^2)  + m_c(\dot x_c^2 +\dot y_c^2)\right) -m_p g\ell\cos(\theta_x)\cos(\theta_y)\notag
\end{align}$$



# Linear equations of motion.
Applying the approximations to the equations of motions produces a pair of coupled equations ($\ddot x_c, \ddot\theta_x$) and ($\ddot y_c,\ddot\theta_y$):

$$\begin{align}
\ddot x_c (m_c+m_p)+m_p\ell\theta_x=0 \\
m_p\ell\ddot x_c+(I +m_p \ell^2)\ddot\theta_x+m_pg\ell\theta_x=0
\end{align}$$


With these equations, the $A$ matrix can be derived, as the matrix which transforms the state $\textbf{x}$ into $\dot{\textbf{x}}$.

This can be done algebraically, solving for each variable of interest and subsitution, though linear algebra makes the process much easier as the coupled equations can be expressed as matrix multiplication as:

$$\underbrace{\begin{bmatrix} (m_c+m_p) & m_p\ell \\ m_p\ell & I + m_p\ell^2 \end{bmatrix}}_M \begin{bmatrix} \ddot x_c \\ \ddot\theta_x\end{bmatrix} = \begin{bmatrix}0 \\-m_pg\ell\theta_x\end{bmatrix}$$

The inverse of the $M$ matrix's is obtained via $\frac{1}{det(M)} \cdot M=M^{-1}$, and is used to solve for $\ddot x_c$ and $\ddot\theta_x$:

$$\begin{bmatrix}\ddot x_c \\ \ddot \theta_x\end{bmatrix} = M^{-1}\begin{bmatrix}0 \\ -m_p g\ell\theta_x\end{bmatrix}=\frac{1}{D}\begin{bmatrix}-m_p^2\ell^2g\theta_x \\-(m_c+m_p)(m_pg\ell\theta_x)\end{bmatrix}$$

Revealing the resulting coefficients to be:
$$\large{\ddot x_c = \theta_x\frac{-m_p^2\ell g}{D}} = \theta_x \cdot p$$

$$\large{\ddot \theta_x = \theta_x\frac{-(m_c+m_p)(m_pg\ell)}{D}}= \theta_x\cdot q$$

By symmetry, the coefficients for the other state variables are:

$$\large{\ddot y_c = \theta_y\frac{-m_p^2\ell g}{D}} = \theta_y \cdot p$$

$$\large{\ddot \theta_y= \theta_y\frac{-(m_c+m_p)(m_pg\ell)}{D}}= \theta_y\cdot q$$


The $A$ matrix is therefore:

$$\begin{bmatrix} \dot x_c \\\dot y_c\\ \dot\theta_x \\ \dot\theta_y\\\ddot x_c \\ \ddot y_c \\ \ddot\theta_x \\ \ddot\theta_y\\\end{bmatrix} =\begin{bmatrix}
0 &0 &0 &0 &1 &0 &0 &0  \\ 
0 &0 &0 &0 &0 &1 &0 &0  \\
0 &0 &0 &0 &0 &0 &1 &0  \\
0 &0 &0 &0 &0 &0 &0 &1  \\
0 &0 &p &0 &0 &0 &0 &0  \\
0 &0 &q &0 &0 &0 &0 &0  \\
0 &0 &0 &p &0 &0 &0 &0  \\
0 &0 &0 &q &0 &0 &0 &0  \\ \end{bmatrix}\cdot
\begin{bmatrix} x \\ y \\ \theta_x \\ \theta_y \\ \dot x_c \\ \dot y_c \\ \dot \theta_x \\ \dot\theta_y\\ \end{bmatrix}$$


## Cost-To-Go


$$J = \int_{0}^{\infty} \left[ x(t)^T Q x(t) + u(t)^T R u(t)\right] dt$$
  

where $Q$ and $R$ are posiive semi-definate and posiive definiate matrices respectively.

  
We must provide the weights for $\mathcal{J}$. I will use the matrices

$Q = diag( \begin{bmatrix} 10 & 10 &100 & 100 &  1 &  1 & 1 & 1 \end{bmatrix})$

and  

$R = \begin{bmatrix} 1 &  1 \end{bmatrix}^T$

Puts emphasis on controlling $\theta_x$ and $\theta_y$ but also concerned with control of the cart to its desired position.

Using out linear approximation $\dot x(t) = Ax(t) + Bu(t)$.

The solution $J$ has the closed form known as the Continuous Algebraic Riccati Equation which can be solved in code using `scipy.linalg.solve_continuous_are(A, B, Q, R)`. The return is a gain matrix $K$ which provides the optimal solution for the system for a discrete time step $-K = u$.
