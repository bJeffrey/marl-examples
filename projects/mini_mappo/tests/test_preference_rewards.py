"""Behavioral checks for synthetic outcome preferences and frozen terminal rewards."""

from dataclasses import replace
import unittest

import numpy as np
import torch

from mini_mappo import config, environment, preference_rewards, rollout, trainer


class PreferenceRewardTests(unittest.TestCase):
    """Use a tiny outcome domain and actual PPO components, without human feedback."""

    def test_preference_rules_include_success_priority_and_failure_ties(self):
        """Outcome labels express priorities without assigning reward magnitudes."""
        label = preference_rewards.preference_label
        self.assertEqual(label(True, 50, False, 1), 1.0)
        self.assertEqual(label(False, 1, True, 50), 0.0)
        self.assertEqual(label(True, 3, True, 5), 1.0)
        self.assertEqual(label(True, 5, True, 3), 0.0)
        self.assertEqual(label(True, 3, True, 3), 0.5)
        self.assertEqual(label(False, 1, False, 50), 0.5)

    def test_fit_freezes_model_and_preserves_policy_rng(self):
        """Learned scores order outcomes, including successful lengths held out of fitting."""
        cfg = replace(
            config.load_config("configs/preference_learned.json"),
            max_steps = 6,
            preference_holdout_stride = 3,
        )
        state = torch.random.get_rng_state().clone()
        fit   = preference_rewards.fit_outcome_reward(cfg)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(all(not parameter.requires_grad for parameter in fit.model.parameters()))
        scores = [fit.model.terminal_reward(True, length) for length in range(1, 7)]
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertGreater(scores[-1], fit.model.terminal_reward(False, 6))
        for earlier, later in zip(scores, scores[1:]):
            self.assertGreater(earlier, later)
        self.assertEqual(fit.diagnostics["strict_validation_accuracy"], 1.0)
        self.assertEqual(fit.diagnostics["setup_environment_steps"], 0)
        for pair in fit.comparisons:
            if pair["split"] == "train":
                self.assertFalse(pair["first_success"] and pair["first_steps"] % 3 == 0)
                self.assertFalse(pair["second_success"] and pair["second_steps"] % 3 == 0)

    def test_handcrafted_formula_matches_original_rewards(self):
        """A custom reward configuration affects only the explicit environment formula."""
        cfg = config.load_config()
        env = environment.GridWorldEnv(cfg)
        env.target_pos      = np.array([0, 0])
        env.agent_positions = [np.array([0, 1]), np.array([1, 1])]
        _, _, rewards, success = env.step([2, 4])
        np.testing.assert_allclose(rewards, [10.0, -0.21])
        self.assertTrue(success)

        changed = replace(cfg, goal_reward=20.0, distance_penalty=0.2, step_penalty=0.02)
        env = environment.GridWorldEnv(changed)
        env.target_pos      = np.array([0, 0])
        env.agent_positions = [np.array([0, 1]), np.array([1, 1])]
        _, _, rewards, _ = env.step([2, 4])
        np.testing.assert_allclose(rewards, [20.0, -0.42])

    def test_terminal_rewards_stop_gae_at_success_or_timeout(self):
        """No earlier handcrafted penalties or post-episode bootstrap may affect learned returns."""
        cfg = replace(config.load_config("configs/preference_learned.json"), gae_lambda=1.0)
        for score in (0.0, 0.75):
            with self.subTest(score=score):
                buffer = rollout.RolloutBuffer(cfg)
                for _ in range(2):
                    buffer.add(
                        np.zeros((cfg.num_agents, cfg.obs_dim), dtype=np.float32),
                        np.zeros(cfg.global_state_dim, dtype=np.float32),
                        [4] * cfg.num_agents,
                        [0.0] * cfg.num_agents,
                        np.full(cfg.num_agents, -100.0, dtype=np.float32),
                        np.full(cfg.num_agents, 2.0, dtype=np.float32),
                        done = False,
                    )
                buffer.set_terminal_team_reward(score)
                np.testing.assert_array_equal(buffer.rewards[0], 0.0)
                np.testing.assert_array_equal(buffer.rewards[-1], score)
                self.assertEqual(buffer.masks[-1], 0.0)

                # With gamma=lambda=1 the TD terms telescope to the terminal
                # score. Even an enormous next-state estimate must not leak in.
                _, returns = buffer.compute_gae(np.full(cfg.num_agents, 999.0, dtype=np.float32))
                np.testing.assert_allclose(returns, score)

    def test_mismatched_reward_model_horizon_is_rejected(self):
        """A model fitted for another time limit must not extrapolate its outcome scores."""
        cfg = config.load_config("configs/preference_learned.json")
        model = preference_rewards.OutcomeReward(cfg.max_steps + 1)
        with self.assertRaisesRegex(ValueError, "max_steps must match"):
            trainer.train(cfg, preference_model=model)

    def test_learned_mode_ignores_handcrafted_reward_coefficients(self):
        """With identical seeds, changing only diagnostic reward coefficients cannot change learning."""
        cfg = replace(
            config.load_config("configs/preference_learned.json"),
            total_episodes = 2,
            max_steps = 4,
            preference_holdout_stride = 2,
            device_name = "cpu",
        )
        fit = preference_rewards.fit_outcome_reward(cfg)
        first_history = []
        first = trainer.train(cfg, history=first_history, preference_model=fit.model)
        changed = replace(cfg, goal_reward=1000.0, distance_penalty=5.0, step_penalty=10.0)
        second_history = []
        second = trainer.train(changed, history=second_history, preference_model=fit.model)

        for first_weight, second_weight in zip(first.actor.parameters(), second.actor.parameters()):
            self.assertTrue(torch.equal(first_weight, second_weight))
        self.assertEqual(
            [record.team_return for record in first_history],
            [record.team_return for record in second_history],
        )
        self.assertTrue(all(record.learned_score == record.team_return for record in first_history))
        self.assertNotEqual(
            [record.handcrafted_return for record in first_history],
            [record.handcrafted_return for record in second_history],
        )


if __name__ == "__main__":
    unittest.main()
