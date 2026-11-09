"""Small regression tests for the educational MAPPO core."""

import unittest
from dataclasses import replace

import numpy as np

from mini_mappo.agent import MAPPOAgent
from mini_mappo import config
from mini_mappo.environment import GridWorldEnv
from mini_mappo.rollout import RolloutBuffer


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(config.load_config(), total_episodes=1, max_steps=4)

    def test_environment_shapes_match_ctde_contract(self):
        env = GridWorldEnv(self.config)
        obs, state = env.reset()
        self.assertEqual(obs.shape, (self.config.num_agents, self.config.obs_dim))
        self.assertEqual(state.shape, (self.config.global_state_dim,))

    def test_actor_and_critic_shapes(self):
        agent = MAPPOAgent(self.config)
        env = GridWorldEnv(self.config)
        obs, state = env.reset()
        action, log_prob = agent.select_action(obs[0])
        values = agent.values(state)
        self.assertTrue(0 <= action < self.config.num_actions)
        self.assertTrue(np.isfinite(log_prob))
        self.assertEqual(values.shape, (self.config.num_agents,))

    def test_gae_preserves_agent_dimension(self):
        buffer = RolloutBuffer(self.config)
        zero_obs = np.zeros((self.config.num_agents, self.config.obs_dim), dtype=np.float32)
        zero_state = np.zeros(self.config.global_state_dim, dtype=np.float32)
        zero_values = np.zeros(self.config.num_agents, dtype=np.float32)
        buffer.add(
            zero_obs,
            zero_state,
            [0] * self.config.num_agents,
            [0.0] * self.config.num_agents,
            np.array([1.0, 2.0], dtype=np.float32),
            zero_values,
            done=True,
        )
        advantages, returns = buffer.compute_gae(zero_values)
        self.assertEqual(advantages.shape, (1, self.config.num_agents))
        np.testing.assert_allclose(returns[0], [1.0, 2.0])

    def test_training_batch_has_writable_agent_ids_in_time_order(self):
        """Single-step and longer rollouts must both support tensor conversion."""
        for time_steps in (1, 2):
            with self.subTest(time_steps=time_steps):
                env    = GridWorldEnv(self.config)
                buffer = RolloutBuffer(self.config)

                # Keep the fixture independent of policy sampling and GPU execution.
                obs, state = env.reset()
                zeros      = np.zeros(self.config.num_agents, dtype=np.float32)
                for _ in range(time_steps):
                    buffer.add(
                        obs, state, [0] * self.config.num_agents,
                        [0.0] * self.config.num_agents, zeros, zeros, done=True,
                    )

                advantages, returns = buffer.compute_gae(zeros)
                batch               = buffer.flatten_for_training(advantages, returns)

                # Flattening must preserve the critic column for each agent row
                # and provide writable storage for torch.as_tensor to share.
                np.testing.assert_array_equal(
                    batch["agent_id"], list(range(self.config.num_agents)) * time_steps,
                )
                self.assertTrue(batch["agent_id"].flags.writeable)


if __name__ == "__main__":
    unittest.main()
