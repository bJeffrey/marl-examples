"""PPO collection boundaries and configurable advantage preprocessing."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mini_mappo import trainer
from mini_mappo.agent import MAPPOAgent
from mini_mappo.config import load_config
from mini_mappo.rollout import RolloutBuffer


class PPOBatchingTests(unittest.TestCase):
    """Use real rollouts and optimizers to verify the update contract."""

    def test_frozen_collection_and_final_partial_update(self):
        """Train every complete collection group and the remaining final episode."""

        cfg = replace(
            load_config(),
            device_name         = "cpu",
            total_episodes      = 5,
            max_steps           = 1,
            episodes_per_update = 2,
            update_epochs       = 1,
        )
        snapshots   = []
        batch_sizes = []
        history     = []
        collect     = trainer.collect_episode
        update      = trainer.ppo_update

        def record_collection(env, agent):
            """Snapshot the actual collection weights before running the environment."""

            # Both networks must remain frozen throughout each collection group.
            weights = []
            for network in (agent.actor, agent.critic):
                weights.extend(parameter.detach().clone() for parameter in network.parameters())
            snapshots.append(weights)

            return collect(env, agent)

        def record_update(agent, batch):
            """Record sample counts while retaining the real optimizer update."""

            batch_sizes.append(len(batch["obs"]))

            return update(agent, batch)

        with patch.object(trainer, "collect_episode", side_effect=record_collection):
            with patch.object(trainer, "ppo_update", side_effect=record_update):
                trainer.train(cfg, history=history)

        self.assertEqual(batch_sizes, [4, 4, 2])
        self.assertEqual([record.ppo_updates_completed for record in history], [0, 1, 1, 2, 3])
        for first, second in ((0, 1), (2, 3)):
            for before, after in zip(snapshots[first], snapshots[second]):
                self.assertTrue(torch.equal(before, after))

    def test_terminal_outcome_difference_drives_policy_update(self):
        """A successful action gains probability relative to a failed action."""

        cfg = replace(
            load_config(),
            device_name             = "cpu",
            gamma                   = 1.0,
            gae_lambda              = 1.0,
            update_epochs           = 1,
            minibatch_size          = 4,
            advantage_normalization = "batch",
            entropy_coef            = 0.0,
            actor_lr                = 0.01,
        )
        torch.manual_seed(cfg.seed)
        agent = MAPPOAgent(cfg)
        with torch.no_grad():
            for parameter in agent.critic.parameters():
                parameter.zero_()

        observations = np.zeros((cfg.num_agents, cfg.obs_dim), dtype=np.float32)
        state        = np.zeros(cfg.global_state_dim, dtype=np.float32)
        batches      = []
        for score, action in ((0.0, 0), (1.0, 2)):
            # Identical states isolate the effect of the different terminal scores.
            actions = torch.full((cfg.num_agents,), action, dtype=torch.long)
            with torch.no_grad():
                log_probs, _ = agent.evaluate_actions(torch.as_tensor(observations), actions)

            buffer = RolloutBuffer(cfg)
            buffer.add(
                observations, state, actions.tolist(), log_probs.tolist(),
                np.zeros(cfg.num_agents), np.zeros(cfg.num_agents), True,
            )
            buffer.set_terminal_team_reward(score)

            advantages, returns = buffer.compute_gae(np.zeros(cfg.num_agents))
            batches.append(buffer.flatten_for_training(advantages, returns))

        combined = {}
        for field in batches[0]:
            combined[field] = np.concatenate([batch[field] for batch in batches], axis=0)

        # Per-episode centering would erase both constant signals; the combined
        # population retains negative failure and positive success advantages.
        normalized = trainer._normalize_advantages(torch.as_tensor(combined["advantage"]))
        torch.testing.assert_close(normalized, torch.tensor([-1.0, -1.0, 1.0, 1.0]))
        obs = torch.as_tensor(observations)
        with torch.no_grad():
            before = agent.actor(obs).clone()
        trainer.ppo_update(agent, combined)
        with torch.no_grad():
            after = agent.actor(obs)

        self.assertTrue(torch.all(after[:, 2] - after[:, 0] > before[:, 2] - before[:, 0]))

    def test_normalization_scope_and_numeric_corner_cases(self):
        """Normalize the chosen sample population, keeping constant inputs finite."""

        cfg = replace(
            load_config(),
            device_name    = "cpu",
            update_epochs  = 2,
            minibatch_size = 2,
        )
        sample_count = 4
        batch = {
            "obs":          np.zeros((sample_count, cfg.obs_dim), dtype=np.float32),
            "state":        np.zeros((sample_count, cfg.global_state_dim), dtype=np.float32),
            "agent_id":     np.array([0, 1, 0, 1]),
            "action":       np.zeros(sample_count, dtype=np.int64),
            "old_log_prob": np.zeros(sample_count, dtype=np.float32),
            "advantage":    np.arange(sample_count, dtype=np.float32),
            "return":       np.ones(sample_count, dtype=np.float32),
        }
        for mode, expected_sizes in (("batch", [4]), ("minibatch", [2, 2, 2, 2]), ("none", [])):
            with self.subTest(mode=mode):
                agent = MAPPOAgent(replace(cfg, advantage_normalization = mode))
                with patch.object(trainer, "_normalize_advantages", wraps=trainer._normalize_advantages) as normalize:
                    trainer.ppo_update(agent, batch)
                self.assertEqual([len(call.args[0]) for call in normalize.call_args_list], expected_sizes)

        for values in (torch.tensor([2.0]), torch.tensor([2.0, 2.0])):
            torch.testing.assert_close(trainer._normalize_advantages(values), torch.zeros_like(values))
