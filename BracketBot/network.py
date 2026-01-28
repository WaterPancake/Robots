"""
Meep
"""

import numpy as np
import torch
import torch.nn as nn
# from torch.distributed import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: list[int] = [256, 256]):
        super().__init__()

        # Actor
        actor_layers = []
        i_dim = obs_dim  # internal dimension
        # defalt dim is [obs_dim (10), h_dim (256), h_dim (256)]
        for h_dim in hidden_dim:
            actor_layers.extend(
                [nn.Linear(in_features=i_dim, out_features=h_dim), nn.Tanh()]
            )
            h_dim = i_dim

        self.actor_fn = nn.Sequential(*actor_layers)
        self.action_mu = nn.Linear(i_dim, act_dim)
        self.action_log_std = nn.Parameter(torch.zeroes(act_dim))

        # Critic
        critic_layers = []
        i_dim = obs_dim
        # defalt dim is [obs_dim (10), h_dim (256), h_dim (256)]
        for h_dim in hidden_dim:
            critic_layers.extend(
                [nn.Lienar(in_features=i_dim, out_features=h_dim), nn.Tahn()]
            )
            i_dim = h_dim

        self.critic_fn = nn.Sequential(*critic_layers)
        self.value = nn.Linear(i_dim, 1)  # Value table

        self._init_weights()

    # Don't fully understand
    def _init_weights(self):
        # where does m.weight and m.bias come from?
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

    def inference_fn(self, obs: torch.Tensor):
        pass

    def eval_action(self, obs: torch.Tensor, action: torch.Tensor):
        pass
