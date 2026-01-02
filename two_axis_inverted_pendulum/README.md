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


 For this my implementation, state variables $x$ and $y$ are limited to $[-2,2]$. Episodes in the enviornment concludes after $1000$ discrete time steps or if the $\theta_x$ or $\theta_y$ exceed $1.57$ radians (~$90$ degrees).

See `two_axis_inverted_pendulum_env.py` for more details on how the state is obtained from mujoco XLA.

# Learned Swing-up and Balance Controller

The objective of the this controller is to 

## Reward function

Reward functions is upstream of the behavior of the learned controller. There are several ways to defined the 

### Swing-up

$$\mathcal{R}_1(s,a) = \cos(\theta_x) + \cos(\theta_y)$$





### Centering 
penalty preportional to distance to the origin $(0,0)$
$$\mathcal{R}_2(s,a) = \cos(\theta_x) + \cos(\theta_y)  - (x^2 + y^2) $$


### Fine Tuned 

## Training 
All methids use PPO (CITE),

# LQR 
Creating a Linear Quadratic Regular requries the following:

1. Derive equations of motions
2. Lineariziation (1) about a fixed point

## Kinematics

The kinematics for the pole position in $\mathbb{R}^3$ 
$$\textbf{x}_{p} = \begin{bmatrix}
x_p \\ y_p \\ z_p
\end{bmatrix} =\begin{bmatrix}
x + l \sin\theta_x \\ 
y + l \sin \theta_y \\
l \cos{\theta_x} \cos{\theta_y}
\end{bmatrix}$$

$$dot{\textbf{x}}_{p} =
\begin{bmatrix}
\dot x_p \\ 
\dot y_p \\ 
\dot z_p
\end{bmatrix} 
= \begin{bmatrix}
\dot x + l \sin(\theta_x) \\ 
\dot y + l sin (\theta_y) \\
-l \sin{(\theta_x)}\cos{(\theta_y)} \dot\theta_x - l \sin{(\theta_y)}\cos{(\theta_x)}\theta_y
\end{bmatrix}$$

Since we don't controll the pole directly, rather indirecly by the cart, the way energy is transfered from the cart to the pole must be defined. 
This translational kenetic energy is given as 

<!-- $$T_p = \frac{1}{2} m_p(\dot x_p^2 + \dot y_p^2 + \dot z_p^2)$$ -->
$$T_p = \frac{1}{2} m_p \textbf{x}_p^2$$


- define `energy` using Lagranians L = T - U
The total kinetic engery for the system is then given as, the carts effect on

Additional details such as the pole length, weight of the cart and mass at the end of the pendulum

## Legrangian

Deriving the equations of motions here...


## Forces
There are two control:
$u = [F_x, F_y]^T$ that move the cart along their respective axies.






## Lineariziation

The fixed points that are of interest are of the form: 
$$\bar{\textbf{x}} = \left[ x, y,\pi, \pi, 0, 0, 0, 0 \right]$$


For the example notebook, the state $\bar{\textbf{x}} = \left[0, 0, \pi, \pi, 0, 0, 0, 0 \right]$


where the cart is stationary anywhere along the x y axis, and the pole is balanced upright.

The objective is to appoximate the two axis cart pole dynamics with linear dynamics of the form:
$\dot{\textbf{x}} = A\textbf{x} + Bu$ 

where $ A \in \mathbb{R}^{8 \times 8}$ and $u \in \mathbb{R}^{2}$

The linearized dynamics are used in conjuctions with inifnite-horizon 'cost-to-go' function

$$J = \int_{0}^{\infty} \left[ x(t)^T Q x(t) + u(t)^T R u(t)\right] dt$$

where $Q$ and $R$ are posiive semi-definate and posiive definiate matrices respectively.

...

The Riccati equations


will solve using `scipy.linalg.solve_continuous_are`


# Control with LQR

LQR can be used to control the cart by minimizing the cost-to-go function paramerizied for a state:
$V(x)$.

##