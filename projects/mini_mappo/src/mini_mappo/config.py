"""Central configuration for the educational MAPPO example.

Keeping hyperparameters in one module makes it easy to change the experiment
without hunting through environment, model, and training code.  Later GAT/GART
steps can extend this dataclass with graph dimensions, attention heads, and
memory settings without changing the public training interface.
"""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Config:
    # Environment
    grid_size: int   = 6  # Number of cells along each side of the square grid.
    num_agents: int  = 2  # Agents acting together, all using the shared policy.
    num_actions: int = 5  # Discrete actions per agent: up, down, left, right, stay.
    max_steps: int   = 50  # Maximum joint environment steps per episode.

    # Neural networks
    hidden_size: int = 64  # Neurons in each hidden layer of both actor and critic.

    # PPO / MAPPO
    actor_lr: float     = 3e-4  # Adam learning rate controlling policy weight updates.
    critic_lr: float    = 1e-3  # Adam learning rate controlling value weight updates.
    gamma: float        = 0.99  # Reward discount per step; higher values favor longer horizons.
    # GAE trace decay: higher values give later reward/value prediction errors more weight.
    gae_lambda: float   = 0.95
    clip_epsilon: float = 0.2   # PPO ratio clipping half-width: bounds are 1 +/- this value.
    update_epochs: int  = 4     # Passes over each collected episode's samples during PPO.
    minibatch_size: int = 32    # Maximum (timestep, agent) samples per optimizer step.
    entropy_coef: float = 0.01  # Weight of the policy entropy bonus encouraging exploration.
    value_coef: float   = 0.5   # Multiplier on the critic's mean squared return error.

    # Experiment
    total_episodes: int = 600  # Training episodes to collect and optimize before returning.

    # Training reports
    progress_window: int  = 50          # Episodes in each trailing mean; early windows use available episodes.
    output_directory: str = "outputs"   # Report root, resolved relative to this project by __main__.py.

    @property
    def obs_dim(self) -> int:
        """Local observation: own (row, col) + target (row, col)."""
        return 4

    @property
    def global_state_dim(self) -> int:
        """Global state: every agent position plus the target position."""
        return 2 * self.num_agents + 2

    @property
    def device(self) -> torch.device:
        # is_available checks whether PyTorch can use an NVIDIA CUDA GPU.
        # device describes where tensors and weights should live; it allocates
        # no memory itself. Fall back to CPU when CUDA is unavailable.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
