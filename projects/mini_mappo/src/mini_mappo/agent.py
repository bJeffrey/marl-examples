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
        
        # This actor receives only this agent's local observation: its normalized
        # (row, column) and the target's normalized (row, column). Other agents'
        # positions are reserved for the centralized critic during training.
        # as_tensor converts these NumPy values to 32-bit floating-point numbers
        # on the same CPU/GPU as the actor's weights.
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)

        # Calling the actor invokes Actor.forward(), a fully-connected feedforward neural network,
        # also known as a multilayer perceptron (MLP). It has:
        #   - an input layer of size obs_dim (4)
        #   - a linear hidden layer of size hidden_size (64)
        #   - a Tanh activation function
        #   - a linear 64 --> 64 hidden layer
        #   - a Tanh activation function
        #   - a linear 64 --> num_actions (5) output layer
        #   - action scores (logits) for each of the 5 possible actions: up, down, left, right, stay
        #
        # It passes the observation
        # through its learned Linear layers (weighted sums plus biases) and Tanh
        # activations (nonlinear transforms). The final layer returns a vector
        # of num_actions raw preference scores, called logits.
        # logits[k] corresponds to action index k: up, down, left, right, stay.
        # These scores can be negative and need not sum to one; they are neither
        # predicted rewards nor probabilities that an action will succeed.
        logits = self.actor(obs_tensor)

        # Categorical interprets the logits as a distribution over action indices.
        # Its probabilities are softmax(logits): each exponentiated score divided
        # by their sum, computed stably by PyTorch. Probabilities sum to one;
        # larger logits give an action a larger share of the probability mass.
        distribution = Categorical(logits=logits)

        # Sample one index, favoring actions with higher probabilities rather than
        # always taking the largest score. For example, probability 0.60 for left
        # means a 60% chance of selecting index 2 on this call. Sampling allows
        # exploration: lower-probability alternatives can still generate training
        # experience. This chooses an action; it does not update network weights.
        action = distribution.sample()

        # log_prob(action) is ln(probability assigned to this sampled action).
        # Store it as the rollout policy's reference score: PPO later scores the
        # same action under the updated policy and uses exp(new_log - old_log)
        # to measure how much its selection probability changed.
        # item() extracts each scalar tensor as a Python number; return the action
        # index for env.step() and its log probability for the rollout buffer.
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
