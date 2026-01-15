from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

import jax
from jax import numpy as jp

"""
## pipeline_state.q
| idx | name
|-----|-----------------------------------
|  0  | base x position
|  1  | base y position
|  2  | base z position
|  3  | W component of base quaternion 
|  4  | X component of base quaternion 
|  5  | Y component of base quaternion
|  6  | Y component of base quaternion
|  7  | Right wheel radians
|  8  | Left wheel radians

## pipeline_state.dq
| idx | name
|-----|-----------------------------------
|  0  | base linear velocity along x axis
|  1  | base linear velocity alongn y axis 
|  2  | base linear velocity alongn z axis
|  3  | base angular velocity along x axis
|  4  | base angular velocity along y axis
|  5  | base angular velocity along z axis
|  6  | Right wheel velocity
|  7  | Left wheel velocity


## pipeline_state.xpos (cordinates in the world)
| idx | name
|-----|-----------------------------------
|  0  | world 
|  1  | cart base
|  3  | right wheel
|  4  | left wheel

## pipeline_state.geom_xpos (local cordinate)
| idx | name
|-----|-----------------------------------
|  0  | world 
|  0  | base
|  0  | upper
|  0  | top 
|  0  | right wheel
|  0  | left wheel
"""


class Balance_BracketBot(PipelineEnv):
    def __init__(self, backend="mjx", target=jp.array([0, 0]), **kwargs):
        path = "xml/BracketBot.xml"
        sys = mjcf.load(path)

        self.MAX_WHEEL_CTR = 3
        self.MIN_WHEEL_CTR = -3

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
        roll, pitch, yaw = (
            self._roll(pipeline_state),
            self._pitch(pipeline_state),
            self._yaw(pipeline_state),
        )
        metrics = {"reward": reward, "roll": roll, "pitch": pitch, "yaw": yaw}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, self.MIN_WHEEL_CTR, self.MAX_WHEEL_CTR)
        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        obs = self._get_obs(pipeline_state)

        reward = self._reward(pipeline_state)

        done = self._to_terminate(pipeline_state)

        roll, pitch, yaw = (
            self._roll(pipeline_state),
            self._pitch(pipeline_state),
            self._yaw(pipeline_state),
        )
        metrics = {"reward": reward, "roll": roll, "pitch": pitch, "yaw": yaw}

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def _reward(self, pipeline_state: base.State):
        """
        Very basic reward for now
        """
        roll = self._roll(pipeline_state)

        survival_reward = 1

        reward = survival_reward + jp.cos(roll)

        return reward

    def _to_terminate(self, pipeline_state: base.State) -> bool:
        cart_roll = self._roll(pipeline_state)

        done = abs(cart_roll) > 1.4  # ~80 degrees forward of backwards
        return done.astype(jp.float32)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        positions = pipeline_state.q
        velocities = pipeline_state.qd

        return jp.concatenate([positions] + [velocities])

    def _roll(self, pipeline_state: base.State):
        quat = pipeline_state.q[3:7]  # w, x, y, z
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return jp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    def _pitch(self, pipeline_state: base.State):
        quat = pipeline_state.q[3:7]  # w, x, y, z
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return jp.arcsin(2 * (w * y - z * x))

    def _yaw(self, pipeline_state: base.State):
        quat = pipeline_state.q[3:7]  # w, x, y, z
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return jp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    @property
    def action_size(self) -> int:
        return 2
