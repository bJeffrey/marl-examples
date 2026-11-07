"""Actor/critic ownership and policy-facing operations."""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from .config import Config
from .networks import Actor, Critic


class MAPPOAgent:
    """Holds the shared actor, centralized critic, and their optimizers."""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        # Module.to moves network weights to the chosen CPU or GPU; inputs must
        # live on that same device for network calculations to work.
        self.actor = Actor(config.obs_dim, config.num_actions, config.hidden_size).to(self.device)
        self.critic = Critic(
            config.global_state_dim, config.num_agents, config.hidden_size
        ).to(self.device)

        # parameters() exposes trainable weights and biases. Adam adjusts them
        # using gradients and running estimates of gradient size and direction.
        # Each network has its own optimizer state and learning rate (lr).
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    # Disable gradient tracking during rollout sampling to avoid storing a
    # computation graph; PPO later re-evaluates these actions with tracking on.
    @torch.no_grad()
    def select_action(self, observation: np.ndarray) -> tuple[int, float]:
        """Sample one action from the current decentralized policy."""
        
        # as_tensor converts the NumPy observation to a PyTorch array of 32-bit
        # floating-point numbers on the network's device.
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        
        # Calling a network runs its forward method, producing one score per action.
        logits = self.actor(obs_tensor)
        
        # Categorical normalizes those scores into probabilities of discrete actions.
        distribution = Categorical(logits=logits)
        
        # sample randomly draws an action according to those probabilities.
        action = distribution.sample()
        
        # log_prob gives the natural log of the chosen action's probability;
        # item() extracts each one-element tensor as a Python number for storage.
        return int(action.item()), float(distribution.log_prob(action).item())

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions under the current PPO policy."""
        
        # Run the actor on a batch, retaining the computation graph so the
        # resulting loss can supply gradients for policy learning.
        logits = self.actor(observations)
        distribution = Categorical(logits=logits)
        
        # log_prob scores stored actions; entropy measures distribution randomness
        # for the exploration reward. Both return one value per batch row.
        return distribution.log_prob(actions), distribution.entropy()

    # Rollout value estimates are fixed targets, so no gradient graph is needed.
    @torch.no_grad()
    def values(self, global_state: np.ndarray) -> np.ndarray:
        """Return one centralized value estimate per agent."""
        
        # Convert the joint state to the same numeric type and device as the critic.
        state_tensor = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        
        # Calling the critic computes values; cpu() brings them to host memory,
        # and numpy() exposes them as an array for the rollout buffer.
        return self.critic(state_tensor).cpu().numpy()
