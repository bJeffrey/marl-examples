"""Small cooperative GridWorld used to expose MAPPO mechanics.

This environment is intentionally simple.  The important distinction is that
an actor receives a local observation, while the centralized critic receives a
global state during training (CTDE: centralized training, decentralized
execution).
"""

from __future__ import annotations

import numpy as np

from .config import Config


class GridWorldEnv:
    """Two-or-more-agent grid world with a shared target location."""

    MOVES = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
        4: (0, 0),   # stay
    }

    def __init__(self, config: Config):
        self.config = config
        self.grid_size = config.grid_size
        self.target_pos: np.ndarray | None = None
        self.agent_positions: list[np.ndarray | None] = [None] * config.num_agents

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Randomize target/agents and return local observations + global state."""
        self.target_pos = np.random.randint(0, self.grid_size, size=2)

        # Avoid starting on the target; agents may still share a starting cell.
        for i in range(self.config.num_agents):
            while True:
                pos = np.random.randint(0, self.grid_size, size=2)
                if not np.array_equal(pos, self.target_pos):
                    self.agent_positions[i] = pos
                    break

        return self.get_observations(), self.get_global_state()

    def _normalizer(self) -> float:
        
        # Coordinates range from 0 to grid_size - 1.  Dividing by that maximum
        # maps the full coordinate range exactly to [0, 1].
        return float(max(self.grid_size - 1, 1))

    def get_observations(self) -> np.ndarray:
        """Return one decentralized observation per agent, shape [N, 4]."""
        
        assert self.target_pos is not None
        observations = []
        scale = self._normalizer()
        
        for pos in self.agent_positions:
            assert pos is not None
            # Each actor sees its own position and the target, both normalized.
            observations.append(np.concatenate([pos / scale, self.target_pos / scale]))
        
        return np.asarray(observations, dtype=np.float32)

    def get_global_state(self) -> np.ndarray:
        """Return centralized state visible only to the critic during training."""
        
        assert self.target_pos is not None
        assert all(pos is not None for pos in self.agent_positions)
        
        scale = self._normalizer()
        
        # Keep agent positions in agent-index order, followed by the target.
        flat_agents = np.concatenate(self.agent_positions) / scale
        state = np.concatenate([flat_agents, self.target_pos / scale])
        
        return np.asarray(state, dtype=np.float32)

    def step(self, actions: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Apply one action per agent and return obs, global state, rewards, done.

        The episode ends when any agent reaches the target.  The original
        single-file example returned this condition from `step` but ignored it
        in the training loop and instead waited for *all* agents; using one
        consistent terminal definition prevents bootstrapping through an episode
        that the environment says is finished.
        """
        if len(actions) != self.config.num_agents:
            raise ValueError(f"Expected {self.config.num_agents} actions, got {len(actions)}")

        assert self.target_pos is not None
        
        rewards = np.zeros(self.config.num_agents, dtype=np.float32)

        for i, action in enumerate(actions):
            if action not in self.MOVES:
                raise ValueError(f"Invalid action {action}")

            pos = self.agent_positions[i]
            assert pos is not None
        
            # Clamp attempted moves at grid boundaries instead of wrapping around.
            new_pos = np.clip(pos + self.MOVES[action], 0, self.grid_size - 1)
            self.agent_positions[i] = new_pos

            # Manhattan distance matches the grid's horizontal/vertical moves.
            distance = np.sum(np.abs(new_pos - self.target_pos))
            if np.array_equal(new_pos, self.target_pos):
                rewards[i] += 10.0
            else:
                # Dense shaping keeps this toy example quick to learn.
                rewards[i] -= 0.1 * distance
                rewards[i] -= 0.01

        done = any(np.array_equal(pos, self.target_pos) for pos in self.agent_positions)
        
        # Rebuild both views from the updated positions for the next policy step.
        return self.get_observations(), self.get_global_state(), rewards, done
