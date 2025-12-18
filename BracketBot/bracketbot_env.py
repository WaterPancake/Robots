from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

import jax
from jax import numpy as jp


class BracketBot(PipelineEnv):
    """
    ### Accessing Geom
    |idx| name
    |---|-----------
    | 0 | world body
    | 1 | upper
    | 2 | lower / base
    | 3 | top
    | 4 | right wheel
    | 5 | left wheel

    The 3d positon are obtained with `.geom_xpos[i]`, a member function of hte

    ### Observation Space
    obtained of `_obs()`

    |idx | obsevation
    |----|----------------------
    | 0  | BracketBot x position (forward-back)
    | 1  | BracketBot y position (left-right)
    | 3  | Bracketbot z position (up-down)
    | 4  | angle of right wheel
    | 5  | angle of left wheel
    | 6  |
    | 7  |
    | 8  |
    | 9  |
    | 10 |
    | 11 |
    | 12 |
    | 13 |
    | 14 |
    | 15 |
    | 16 |

    """

    def __init__(self, backend="mjx", **kwargs):
        path = "BracketBot.xml"
        sys = mjcf.load(path)
        n_frames = 2  # higher so rollout is smoother

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
        reward = self._reward(
            pipeline_state, jp.array([0.0, 0.0])
        )  # make random instead???

        pole_angle = self._pole_angle(pipeline_state)
        metrics = {"pole_angle": pole_angle, "forward_vel": 0.0, "reward": reward}
        done = 0.0

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(
            state.pipeline_state, action.astype(jp.float32)
        )

        obs = self._get_obs(pipeline_state)

        pole_angle_radians = self._pole_angle(pipeline_state)
        pole_angle = (180 / jp.pi) * pole_angle_radians
        # forward_vel = pipeline_state.qd[0]  # x velocity for metrics
        vel = pipeline_state.geom_xpos - pipeline_state0.geom_xpos
        forward_vel = vel[3][0]

        reward = self._reward(pipeline_state, action)

        done = jp.where(pole_angle_radians < 1.4, 0.0, 1.0)  # ~ 80 degrees
        # see https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError for why this notation

        metrics = {
            "pole_angle": pole_angle,
            "forward_vel": forward_vel,
            "reward": reward,
        }

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def _reward(self, pipeline_state: base.State, action: jax.Array):
        """
        Combined reward function that encourages:
        1. Keeping the pole upright (height of top geom)
        2. Forward movement (x-direction velocity)
        3. Energy efficiency (penalize large actions)
        4. Keeping pole angle small (penalize tilting)
        """
        top_height = pipeline_state.geom_xpos[3][2]  # z-coordinate of top
        balance_reward = top_height
        base_vel = pipeline_state.qd[0]  # x velocity of base
        forward_reward = 2.0 * base_vel  # Scale forward movement

        action_penalty = 0.01 * jp.sum(jp.square(action))

        pole_angle = self._pole_angle(pipeline_state)
        angle_penalty = 5.0 * pole_angle

        # Total reward
        total_reward = balance_reward + forward_reward - action_penalty - angle_penalty

        return total_reward

    @property
    def action_size(self) -> int:
        return 2

    """
    Terminates when the top falls over which we will calculate using the `base` and `top` geoms distance,
    the global position of the models geoms (namely their centers) can be obtained via `pipeline_state.geom_xpos[]` and are indexed in order of decleration in the models .xml file.
    """

    def _pole_angle(self, pipeline_state: base.State) -> jp.float32:
        """Calculate the angle of the pole from vertical (in radians)."""
        base = jp.array(pipeline_state.geom_xpos[2])
        top = jp.array(pipeline_state.geom_xpos[3])
        pole_vec = top - base

        z = jp.array([0.0, 0.0, 1.0])
        magnitude = jp.linalg.norm(pole_vec)
        # Angle from vertical (0 when upright, increases as it tilts)
        pole_angle = jp.arccos(jp.clip(jp.dot(pole_vec, z) / magnitude, -1.0, 1.0))

        return pole_angle

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        base_position = pipeline_state.q[0:3]
        base_velocity = pipeline_state.qd[0:6]

        wheel_vel = pipeline_state.qd[6:8]

        qpos = pipeline_state.q
        qvel = pipeline_state.qd
        return jp.concatenate([qpos] + [qvel])
