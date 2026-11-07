"""Neural-network building blocks for the baseline MAPPO implementation."""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Shared decentralized policy network.

    Each agent feeds only its local observation through the same actor weights.
    The network emits *logits* rather than probabilities; PyTorch's Categorical
    distribution applies the stable normalization internally.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        
        # Initialize Module's bookkeeping so PyTorch can track layers and weights.
        super().__init__()
        
        # Sequential passes each layer's output into the next layer in order.
        # Linear learns a weighted sum plus bias; Tanh adds a nonlinear transform
        # bounded between -1 and 1. The final Linear emits raw action scores.
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        # Calling Sequential runs all its layers on the last input dimension;
        # any leading batch dimension is preserved.
        return self.net(observations)


class Critic(nn.Module):
    """Centralized value network with one value head per agent.

    A single scalar critic combined with different per-agent rewards creates
    conflicting targets for identical global states.  Returning one value per
    agent keeps the critic centralized while allowing each agent's reward stream
    to have its own value estimate.
    """

    def __init__(self, global_state_dim: int, num_agents: int, hidden_size: int):
        
        # Initialize Module before assigning layers so their parameters are registered.
        super().__init__()
        
        # Sequential chains learned Linear transforms and nonlinear Tanh activations.
        # The final Linear gives one unrestricted value estimate per agent.
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_agents),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        
        # Run the layer chain, producing [num_agents] or [batch, num_agents]
        # values; the PPO update selects the appropriate agent column per sample.
        return self.net(global_state)
