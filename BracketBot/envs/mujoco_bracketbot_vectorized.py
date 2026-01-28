import numpy as np


def matrixprint(M: np.ndarray):
    M = [[f"{x:.4f}" for x in ob] for ob in M]
    for i, ob in enumerate(M):
        print(f"{i:<2}:    {ob}")


class BracketBotVecEnv:
    def __init__(self, env_fn, num_envs):
        self.envs = [env_fn() for i in range(num_envs)]
        self.num_envs = num_envs
        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].act_dim

    def reset(self, rng_seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(rng_seed)
        seeds = rng.integers(high=100000, low=1, size=len(self.envs))

        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        for i, env in enumerate(self.envs):
            obs[i] = env.reset(seeds[i])

        return obs

    def step(self, action: np.ndarray) -> tuple:
        """
        action expected shape (num_envs, 2)
        """

        # FIX: a check if bad shape

        vec_obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        vec_reward = np.zeros((self.num_envs), dtype=np.float32)
        vec_done = np.zeros((self.num_envs), dtype=bool)
        vec_info = np.array([{}] * self.num_envs)

        for i, env in enumerate(self.envs):
            vec_obs[i], vec_reward[i], vec_done[i], vec_info[i] = env.step(action[i])
            # print(*vec_obs[i])
            # print(vec_reward[i])
            # print(vec_done[i])
            # print(vec_info[i])

            # break
            # if vec_done[i]:
            # vec_obs[i] = env.reset()

        return vec_obs, vec_reward, vec_done, vec_info


if __name__ == "__main__":
    from mujoco_bracketbot import BracketBotEnv
    import numpy as np

    vec_env = BracketBotVecEnv(lambda: BracketBotEnv(), num_envs=16)
    obs = vec_env.reset(49)

    matrixprint(obs)
    print(f"{'=' * 100}")

    action = np.zeros((vec_env.num_envs, vec_env.act_dim))
    for _ in range(50):
        rand_action = np.random.rand(vec_env.num_envs, vec_env.act_dim)
        # print(f"{'=' * 100}")
        obs, rewards, dones, info = vec_env.step(rand_action)

    matrixprint(obs)
