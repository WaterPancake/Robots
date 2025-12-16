from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

import jax
from jax import numpy as jp


class BracketBot(PipelineEnv):
    """
    ### Accessing Geom

    0 | world body
    1 | upper
    2 | lower / base
    3 | top
    4 | right wheel
    5 | left wheel

    The 3d positon are obtained with `.geom_xpos[i]`, a member function of hte
    """

    def __init__(self, backend="mjx", **kwargs):
        path = "BracketBot.xml"
        sys = mjcf.load(path)

        n_frames = 2  # higher so rollout is smoother
        self._forward_

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
        reward = self._reward(pipeline_state, jp.array([0.0, 0.0]))  # temp

        pole_angle = self._pole_angle(pipeline_state)
        metrics = {"pole_angle": pole_angle, "reward": reward}
        done = 0.0
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(
            state.pipeline_state, action.astype(jp.float32)
        )

        obs = self._get_obs(pipeline_state)
        reward = self._reward(pipeline_state, action)

        pole_angle = self._pole_angle(pipeline_state)

        done = jp.where(
            pole_angle > 3.5, 0.0, 1.0
        )  # see https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError for why this notation

        # done = 0.0

        pole_angle = self._pole_angle(state.pipeline_state)
        metrics = {"pole_angle": pole_angle, "reward": reward}

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def _reward(self, pipeline_state: base.State, action: jax.Array):
        # balance reward, rewards staying up
        return 1.0

    def _forward_reward(self, pipeline_state0: base.State, pipeline_state: base.State):
        vel = pipeline_state.geom_xpos - pipeline_state0.geom_xpos

        base_vel = vel[2] / self.dt

    @property
    def action_size(self) -> int:
        return 2

    """
    Terminates when the top falls over which we will calculate using the `base` and `top` geoms distance,
    the global position of the models geoms (namely their centers) can be obtained via `pipeline_state.geom_xpos[]` and are indexed in order of decleration in the models .xml file.
    """

    # temporarily just returns z cordinate of top
    def _pole_angle(self, pipeline_state) -> jp.float32:
        # base = jp.array(pipeline_state.geom_xpos[2])
        top = jp.array(pipeline_state.geom_xpos[3])
        return top[2]
        # top = top - base

        # z = jp.array([0, 0, 1])
        # magnitude = jp.linalg.norm(top)
        # pole_angle = jp.arccos(jp.dot(top, z) / magnitude)

        # return pole_angle

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        return jp.concatenate([qpos] + [qvel])


# if __name__ == "__main__":
#     import mujoco

# model = mujoco.MjModel.from_xml_path("BracketBot.xml")

# print(f"qpos size: {model.nq}")
# print(f"qvel size: {model.nv}")

# print("Joint Information")
# for i in range(model.njnt):
#     joint_name = model.joint(i).name
#     qpos_idx = model.jnt_qposadr[i]
#     print(f"    {qpos_idx} : {joint_name}")


if __name__ == "__main__":
    from brax import envs

    envs.register_environment("BracketBot", BracketBot)
    env = envs.get_environment("BracketBot")
    sys = env.sys

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    pipeline_state = state.pipeline_state  # this is where we get the position o

    print(len)
