# Two Axis Cart Pole

is an extention of the canonical Cart Pole by adding another dimension of traversal and control to the cart, and another degree of freedom to the inverted pendulum.

Built off the the [inverted pendulum from Brax ](https://github.com/google/brax/blob/main/brax/envs/assets/inverted_pendulum.xml) by Google.


## Environment

The inverted two-axis penddulum can be represetned as the following:

$\textbf{x} = \left[ x, y, \theta_x, \theta_y, \dot{x}, \dot y, \dot{\theta_x}, \dot{\theta_y} \right]$


| Variable | Discription | minVal | maxVal
| ----- | ----- |  --- | --- |
| $x$ | carts position along x-axis | -2 | 2 |
| $y$ | carts position along y-axis | -2 | 2 |
| $\theta_x$ |  poles angle along the x axis about the cart in radians | $-\infty$ | $\infty$
| $\theta_y$ |  poles angle along the y axis about the cart in radian | $-\infty$ | $\infty$
| $\dot x$ | be the linear velocity of cart along x-axis | $-\infty$ | $\infty$
| $\dot y$ | be then linear velocity of cart along y-axis | $-\infty$ | $\infty$
| $\dot \theta_x$ | the anguluar velocity of the pole along the x axis about the cart | $-\infty$ | $\infty$
| $\dot \theta_y$ | the anguluar velocity of the pole along the y axis about the car | $-\infty$ | $\infty$

The sytem has control input 
with control input 
 $u = \left[ F_x, F_y \right]^T$ corespondng to linear force applied to the cart in the $x$ and $y$ axis respectfully.


 For this my implementation, state variables $x$ and $y$ are limited to $[-2,2]$. Episodes in the enviornment concludes after $1000$ discrete time steps or ig $\theta_x$ or $\theta_y$ exceed $1.57$ radians (~$90$ degrees).

See `two_axis_inverted_pendulum_env.py` for more details on how the state is obtained from mujoco XLA.

# Learned Swing-up and Balance Controller

The objective of the this controller is to balance the inverted pendulum with two degrees of movement through indirect control of its base.

## Reward function

Reward functions is upstream of the behavior of the learned controller. There are several ways to defined a reward function each with 

### Swing-up

$$\mathcal{R}_1(s,a) = 2 + \cos(\theta_x) + \cos(\theta_y)$$



### Centering 
penalty preportional to distance to the origin $(0,0)$
$$\mathcal{R}_2(s,a) = 4 \cdot \left(\cos(\theta_x) + \cos(\theta_y)\right)  - (x^2 + y^2) $$

The $4$ coefficient is to offset the difference in the range of possible values of pole balancing ($[-2,2]$)reward compared to the distance penalty ($[0, -4]$).


## Training 
All methids use PPO (CITE),

# Linear-Quadradic-Regulator
**NOTE**: this section uses a different xml version of the inverted two-axis pendulum so that the model more accurately represents how its represented in [MIT's Underacuated Robotics](https://underactuated.csail.mit.edu/acrobot.html) where the Cart Pole is parametric usin the following: 
| Parameter | discription | value |  
| --- | --- | --- |
| $m_c$ | Mass of the cart that moves without friction along the x-y axis | 1.0 kg
| $m_p$ | Mass of the pole at the end of a rigid, massless pole that move about the cart with two degrees of freedom | 0.333 kg
| $\ell$ | the length of the afore mentioned pole | 60cm
| $g$ | Graviety | 9.81

A differente XML model is used to reflect the changes (most notabily, the mass of the pendulum of being at the end of a massless/near-massless pole).

## Kinematics

The  position of the pole is in $\mathbb{R}^3$ while carts position in $\mathbb{R}^2$.
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

## Legrangian Equation and Equations of Motion

Legraange equation $L = T - V$ where $T$ represents the systems kentici and $V$ represent the system potential energy.

We use these to find the positions of the center of masses of the pole and cast $T_p$, and $T_c$ 

**Kenetic Energy**

$$T_p = \frac{1}{2} m_p(\dot x_p^2 + \dot y_p^2 + \dot z_p^2)$$


$$T_c = \frac{1}{2} m_c(\dot x_c^2 + \dot y_c^2)$$

The added degree of movement requires that we also consider energy of poles angular movement (rotational energy) which is:

$$T_{r} = \frac{1}{2} I (\dot \theta_x^2 + \theta_y^2)$$

Thus 

$$T = T_p + T_c + T_{r}$$
**Potential Energy**
Comes simply from the pendulum's movement as a result of the gravitational force.

$$V = m_p g \ell \cos(\theta_x) \cos(\theta_y) $$   

## Linearizing 

The first step is to create a equation that discribes the forces along the degrees of freedom $q =\begin{bmatrix} x & y & \theta_x & \theta_y\end{bmatrix}^T$


$$\frac{d}{dt} \left( \frac{\partial L}{\partial \dot q_j}\right) - \frac{\partial L}{\partial q_j} = F_j$$

$$L = \frac{1}{2}\left( 
    m_p (\dot x_p^2 + \dot y_p^2 + \dot z_p^2) + I (\dot \theta_x^2 + \dot \theta_y^2) + m_c (\dot x_c^2 + \dot y_c^2) \right)
    - 
    m_p g \ell \cos(\theta_x) \cos(\theta_y)$$


*Note: dropping second order terms for the cart dynamics to make it easier.




**Cart dynamics in x-direction:**

$\frac{\partial L}{\partial \dot x} = \dot x(m_c + m_p) + m_p \ell \cos(\theta_x)\dot \theta_x$

$\frac{d}{dt} \left( \frac{\partial L}{\partial \dot x}\right) = \ddot x_c( m_c  + m_p)+ m_p \ell \left(\cos(\theta_x)\ddot \theta_x - \sin(\theta_x)\dot \theta_x^2 \right)$

**Cart dynamics in y-direction:**

$\frac{\partial L}{\partial \dot y} = \dot y(m_c + m_p) + m_p \ell \cos(\theta_y)\dot \theta_y$

$\frac{d}{dt} \left( \frac{\partial L}{\partial \dot x}\right) = \ddot y_c( m_c  + m_p)+ m_p \ell \left(\cos(\theta_y)\ddot \theta_y - \sin(\theta_y)\dot \theta_y^2 \right)$

**Pole dynamics in x-direction**
$\frac{\partial L}{\partial \dot \theta_y} = m_p \ell \cos(\theta_x) \dot x_c + m_p \ell^2 \dot \theta_x + I\dot \theta_x$

$\frac{d}{dt} \left(\frac{\partial L}{\partial \dot \theta_y}\right) = m_p\ell\cos(\theta_x)\ddot x_c + m_p\ell \sin(\theta_x)\dot\theta_x\dot x_c + (m_[\ell^2 + I)\ddot \theta_x$

### Fixed Point
The fixed points that are of interest are of the form: 
$$\bar{\textbf{x}} = \left[ x, y,\pi, \pi, 0, 0, 0, 0 \right]$$


For the example notebook, the fixed point is specifically $\bar{\textbf{x}} = \left[0, 0, \pi, \pi, 0, 0, 0, 0 \right]^T$ 


We care about moving the cart to the equilibuum point, we must solve for the dynamics of the 

The objective is to appoximate the two axis cart pole dynamics with linear dynamics of the form:
$\dot{\textbf{x}} = A\textbf{x} + Bu$ 

where $A \in \mathbb{R}^{8 \times 8}$ and $B \in \mathbb{R}^{8 \times 2}$

Application of the Legrange equation to solve for each degree of freedom:

To make getting dynamics easier, using the following approximations:
- $\sin(\theta) \approx \theta$
- $\cos(\theta) \approx 1$
- $\sin(\theta)\cos(\theta) \approx 0$
- $\dot \theta^2 \approx 0$

### Cost-To-Go
The linearized dynamics are used in conjuctions with inifnite-horizon 'cost-to-go' function

$$J = \int_{0}^{\infty} \left[ x(t)^T Q x(t) + u(t)^T R u(t)\right] dt$$


where $Q$ and $R$ are posiive semi-definate and posiive definiate matrices respectively.


will solve using `scipy.linalg.solve_continuous_are`

## Cost Funtion 

We must provide the weights for $\mathcal{J}$. I will use the matrices 

$Q = diag \left( \begin{bmatrix} 10 & 10 & 100 & 100 & 1 & 1 & 1 & 1 \end{bmatrix}\right)$

and

$R = [1, 1]^T$

Puts emphesis on controling $\theta_x$ and $\theta_y$ but also concerned with control of the cart to its desired position. 

# Control with LQR

Creating a controller that can move the cart while maintining the pole in an upright posture. 

Acheived by assinging a new cordinate and defining the cost to go function from the systems current state to the the upright state in the new cordinate.

# Controllability Analysis:
- Look at eigen values of the system,