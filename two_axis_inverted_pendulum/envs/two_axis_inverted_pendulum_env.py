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

    | Idx | Action                    | Control Min | Control Max |
    |-----|---------------------------|-------------|-------------|
    |  0  | Linear force along x-axis |     -1      |      1      |
    |  1  | Linear force along y-axis |     -1      |      1      |

    ### Observation Space

    | Idx | Observation                                                | Obs Min | Obs Max | Other access
    |-----|------------------------------------------------------------|---------|---------|------------
    |  0  | x position of the cart                                     |    -2   |    2    | pipeline_state.q
    |  1  | y position of the cart                                     |    -2   |    2    | pipeline_state.q
    |  2  | angle of cart's pole along the x axis expressed in radians |  -2.9   |   2.9   | pipeline_state.q
    |  3  | angle of cart's pole along the y axis expressed in raidans |  -2.9   |   2.9   | pipeline_state.q
    |  4  | x velocity of the cart                                     |  -inf   |   inf   | pipeline_state.qd
    |  5  | y velocity of the cart                                     |  -inf   |   inf   | pipeline_state.qd
    |  6  | angular velocity of cart along the x  axis                 |  -inf   |   inf   | pipeline_state.qd
    |  7  | angular velocity of cart along the y  axis                 |  -inf   |   inf   | pipeline_state.qd

    ### Reward
    The intended behavior of the system is for the cart balance the pole and center of its allowed region. This is acheived with an alive reward component, reward perportional to the angle of the pole in both the x and y axis.

    ### Starting State

    ### Episode Termination
    Episodes end (check via 'state.done') or if the pole falls at or below 90 degrees along either axis.
    """

    def __init__(self, backend="mjx", **kwargs):
        path = "assets/xml/inverted_two_axis_pendulum_v2.xml"
        sys = mjcf.load(path)

        n_frames = 1

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-1.5, maxval=1.5
        )

        q = self.sys.init_q

        qd = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=1, maxval=-1)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jp.zeros(2)
        metrics = {"reward": reward}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        # FIX: verify if the action is clampted correctly
        pipeline_state = self.pipeline_step(
            state.pipeline_state, action.astype(jp.float32)
        )
        obs = self._get_obs(pipeline_state)
        done = self._to_terminate(pipeline_state)

        # reward = self._reward_1(pipeline_state, action)
        reward = self._reward_2(pipeline_state, action)
        metrics = {"reward": reward}

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    @property
    def action_size(self) -> int:
        return 2  # (x_force, y_force)

    def _reward_1(self, pipeline_state: base.State, action: jax.Array):
        theta_x, theta_y = pipeline_state.q[2:4]

        return 2 + jp.cos(theta_x) + jp.cos(theta_y)

    def _reward_2(self, pipeline_state: base.State, action: jax.Array):
        x_pos, y_pos, theta_x, theta_y = pipeline_state.q[:4]
        balance_reward = jp.cos(theta_x) + jp.cos(theta_y)
        distance_penalty = (x_pos**2) + (y_pos**2)

        return 4 * balance_reward - distance_penalty

    def _reward_3(self, pipeline_state: base.State, action: jax.Array):
        pass

    def _to_terminate(self, pipeline_state) -> bool:
        """
        state terminates when the pendulum falls at and bellow parallel with the ground
        """
        # angle of pole expressed in radians
        pendulum_x_angle, pendulum_y_angle = pipeline_state.q[2:4]

        # 1.57 radians ~ 90 degrees
        # done = (jp.abs(pendulum_x_angle) > 1.57) | (jp.abs(pendulum_y_angle) > 1.57)

        # 2.9 radians ~ 170 degrees
        done = (jp.abs(pendulum_x_angle) > 2.9) | (jp.abs(pendulum_y_angle) > 2.9)

        return done.astype(jp.float32)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        return jp.concatenate([pipeline_state.q, pipeline_state.qd])


# for testing
if __name__ == "__main__":
    from brax import envs

    envs.register_environment("TwoAxisInvertedPendulum", TwoAxisInvertedPendulum)
    env = envs.get_environment("TwoAxisInvertedPendulum")

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
