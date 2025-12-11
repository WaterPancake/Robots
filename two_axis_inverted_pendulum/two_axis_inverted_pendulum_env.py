from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

# from etils import epath
import jax
from jax import numpy as jp


class TwoAxisInvertedPendulum(PipelineEnv):
    """
    ### Description
    This is built off the inverted pendulum enviornment in the Brax githu (https://github.com/google/brax/blob/main/brax/envs/inverted_pendulum.py).

    The change is an added degree of freedom of the cart to move linearly along the x and y axis


    This environemnt currently only works with the MJX backend

    ### Action Space

    The agent takes a 2-element vector for its action that control linear velocity of the cart along the x and y axis. Unlike the original, there is no limit for the actuator, for now...

    | IDX | Action                    | Control Min | Control Max |
    |-----|---------------------------|-------------|-------------|
    |  0  | Linear force along x-axis |    -inf     |      inf    |
    |  1  | Linear force along y-axis |    -inf     |      inf    |

    ### Observation Space

    Refering to values in pipeline_state.q

    | IDX | Observation                                                | Obs Min | Obs Max |
    |-----|------------------------------------------------------------|---------|---------|
    |  0  | x component of the carts position                          |    -1   |    1    |
    |  1  | y component of the carts position                          |    -1   |    1    |
    |  2  | angle of cart's pole along the x axis expressed in radians |  -1.57  |   1.57  |
    |  3  | angle of cart's pole along the y axis expressed in raidans |  -1.57  |   1.57  |

    popeline.qd is the first derivative of these values


    ### Reward
    The objective of the environment is to keep the pole upright as well as keep the cart centered. Since actions are not limited,large actions are diencouraged by an action penalty that increases exponentially relative to the action values

    ### Starting State
    its random...

    ### Episode Termination
    Episodes end (check via 'state.done') or if the pole falls at or below 90 degrees along either axis.

    """

    def __init__(self, backend="mjx", **kwargs):
        path = "inverted_two_axis_pendulum.xml"
        sys = mjcf.load(path)

        n_frames = 2

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        #
        minval, maxval = -0.01, 0.01
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=minval, maxval=maxval
        )
        qd = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=minval, maxval=maxval
        )

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(
            state.pipeline_state, action.astype(jp.float32)
        )
        obs = self._get_obs(pipeline_state)
        reward = self._reward(pipeline_state, action)
        done = self._to_terminate(pipeline_state)

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    @property
    def action_size(self) -> int:
        return 2  # (x_force, y_force)

    def _reward(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        """
        The reward funciton will be
        """
        # position of the cart within its 2d plane of movement
        x_pos, y_pos = pipeline_state.q[:2]
        # angle of pole expressed in radians
        pendulum_x_angle, pendulum_y_angle = pipeline_state.q[2:4]

        pole_reward = jp.cos(pendulum_x_angle) + jp.cos(pendulum_y_angle)
        action_penalty = -0.01 * jp.sum(action**2)
        distance_penalty = -0.01 * (x_pos**2 + y_pos**2)

        return pole_reward + action_penalty + distance_penalty

    def _to_terminate(self, pipeline_state) -> bool:
        """
        state terminates when the pendulum falls at and bellow parallel with the ground
        """
        # angle of pole expressed in radians
        pendulum_x_angle, pendulum_y_angle = pipeline_state.q[2:4]

        done = (
            (jp.abs(pendulum_x_angle) > 1.57)  # ~ 90 degrees
            | (jp.abs(pendulum_y_angle) > 1.57)
        )

        # why as float???
        return done.astype(jp.float32)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        return jp.concatenate([pipeline_state.q, pipeline_state.qd])
