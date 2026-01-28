import numpy as np


class BracketBotVecEnv:
    def __init__(self, env_fn, num_envs):
        self.envs = [env_fn() for i in range(num_envs)]
        self.num_envs = num_envs
        self.obs_dim = self.envs[0].obs_dim

    def reset(self, rng_seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(rng_seed)
        seeds = rng.intergers(100000, -100000)
        obs = np.zeros((self._num_envs, self.obs_dim), dtype=np.float32)
        for i, env in enumerate(self.envs):
            obs[i] = env.reset(seeds[i])

        return obs

    def step(self, action: np.ndarray) -> tuple:
        # vec_obs = [[] * env._num_envs]
        # vec_reward = [[] * env._num_envs]
        # vec_done = [[] * env._num_envs]
        # vec_info = [[] * env._num_envs]

        vec_obs = np.array([[] * self._num_envs])
        vec_reward = np.array([[] * self._num_envs])
        vec_done = np.array([[] * self._num_envs])
        vec_info = np.array([[] * self._num_envs])

        for i, env in enumerate(self.envs):
            vec_obs[i], vec_reward[i], vec_done[i], vec_info[i] = env.step(action[i])

            if vec_done[i]:
                vec_obs[i] = env.reset()

        return vec_obs, vec_reward, vec_done, vec_info
