from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

import jax
from jax import numpy as jp

"""
#### Observation Space

#### pipeline_state.geom_xpos[],

| idx | name
|-----|----------
|  0  | world
|  1  | base(upper)
|  2  | base(lower)
|  3  | top
|  4  | left_wheel
|  5  | right_wheel


"""


class Balance_BracketBot(PipelineEnv):
    def __init__(self, backend="mjx", **kwargs):
        path = "BracketBot.xml"
        sys = mjcf.load(path)

        n_frames = 2

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        minval, maxval = -0.1, 0.1

        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=minval, maxval=maxval
        )
        qd = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=minval, maxval=maxval
        )

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        top_pos = pipeline_state.geom_xpos[3]
        top_z = top_pos[2]
        position_reward = top_z * 2.0  # make parametric later

        base_x_velocity = pipeline_state.qd[0]
        velocity_penalty = 1.5 * base_x_velocity

        # done = jp.where(top_z > 1.5, 0.0, 1.0)

        done = 0.0  # hard coded

        alive_reward = 10  # make parameterized later

        reward = (1 - done) * (
            alive_reward + position_reward - velocity_penalty
        )  # Today I learned that early termination state should have a reward of zero

        reward = done * (
            alive_reward + position_reward - velocity_penalty
        )  # Today I learned that early termination state should have a reward of zero

        metrics = {"reward": reward, "z_pos": top_z, "base_vel": base_x_velocity}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state_0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state_0, action)
        obs = self._get_obs(pipeline_state)

        top_pos = pipeline_state.geom_xpos[3]
        top_z = top_pos[2]
        position_reward = top_z * 2.0  # make parametric later

        # base_x_velocity = pipeline_state.qd[0]
        base_x_velocity = (
            pipeline_state.geom_xpos[1][0] - pipeline_state_0.geom_xpos[1][0] / self.dt
        )
        velocity_penalty = 1.5 * base_x_velocity

        # done = jp.where(top_z > 1.5, 0.0, 1.0)
        done = 0.0  #

        alive_reward = 10  # make parameterized later

        reward = (1 - done) * (
            alive_reward + position_reward - velocity_penalty
        )  # Today I learned that early termination state should have a reward of zero

        metrics = {"reward": reward, "z_pos": top_z, "base_vel": base_x_velocity}

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def _reward(self, pipeline_state: base.State, action: jax.Array):
        pass

    """
    
    """

    def _to_terminate(self, pipeline_state: base.State) -> bool:
        # top_pos = pipeline_state.x.pos[2]

        pass

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        base_position = pipeline_state.q[0:3]
        base_velocity = pipeline_state.qd[0:6]

        wheel_vel = pipeline_state.qd[6:8]

        return jp.concatenate([base_position] + [base_velocity] + [wheel_vel])

    @property
    def action_size(self) -> int:
        return 2
