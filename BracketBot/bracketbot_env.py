from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

import jax
from jax import numpy as jp


class BracketBot(PipelineEnv):
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
        reward, done = jp.zeros(2)
        metrics = {}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = state.pipeline_state

        obs = self._get_obs(pipeline_state)
        reward = self._reward(pipeline_state, action)
        done = self._to_terminate(pipeline_state)

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    @property
    def action_size(self) -> int:
        pass

    def _reward(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        pass

    def _to_terminate(self, pipeline_state) -> bool:
        pass

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        pass


if __name__ == "__main__":
    from brax import envs

    envs.register_environment("BracketBot", BracketBot)
    env = envs.get_environment("BracketBot")
    sys = env.sys

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    pipeline_state = state.pipeline_state

    print("Body Indices, Names, and Position:")
    for i, name in enumerate(sys.link_names):
        print(f"ID: {i}, Name: {name}, Position {pipeline_state.x.pos[i]}")
