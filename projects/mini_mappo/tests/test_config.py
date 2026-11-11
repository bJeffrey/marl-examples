"""Configuration loading and validation behavior."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from mini_mappo.config import PROJECT_ROOT, load_config


class ConfigTests(unittest.TestCase):
    """Exercise mode defaults, resolution, and clear validation failures."""

    def test_loads_both_reward_modes(self):
        handcrafted = load_config()
        learned = load_config("configs/preference_learned.json")

        self.assertEqual(handcrafted.reward_mode, "handcrafted")
        self.assertEqual(handcrafted.gamma, 0.99)
        self.assertEqual(learned.reward_mode, "preference_learned")
        self.assertEqual(learned.gamma, 1.0)
        self.assertEqual(learned.total_episodes, 3000)
        self.assertEqual(learned.episodes_per_update, 8)
        self.assertEqual(learned.advantage_normalization, "batch")
        self.assertEqual(handcrafted.obs_dim, 4)
        self.assertEqual(handcrafted.global_state_dim, 6)

    def test_config_path_is_independent_of_caller_directory(self):
        previous_directory = Path.cwd()
        try:
            os.chdir(tempfile.gettempdir())
            loaded = load_config("configs/handcrafted.json")
        finally:
            os.chdir(previous_directory)

        self.assertEqual(loaded.reward_mode, "handcrafted")

    def test_config_supports_dataclass_overrides(self):
        overridden = replace(load_config(), total_episodes=3)

        self.assertEqual(overridden.total_episodes, 3)
        self.assertEqual(load_config().total_episodes, 600)

    def test_schema_reports_required_unknown_and_wrong_type_fields(self):
        base_path = PROJECT_ROOT / "configs/base.json"
        base = json.loads(base_path.read_text(encoding="utf-8"))
        base.update({"reward_mode": "handcrafted", "gamma": 0.99})
        malformed_documents = (
            ({key: value for key, value in base.items() if key != "seed"}, "seed"),
            ({key: value for key, value in base.items() if key != "episodes_per_update"}, "episodes_per_update"),
            ({key: value for key, value in base.items() if key != "advantage_normalization"}, "advantage_normalization"),
            ({**base, "unexpected": True}, "unexpected"),
            ({**base, "grid_size": "six"}, "grid_size"),
        )
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "configs") as directory:
            config_path = Path(directory) / "invalid.json"
            config_path.write_text(json.dumps(base), encoding="utf-8")
            for document, expected in malformed_documents:
                with self.subTest(expected=expected):
                    config_path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected):
                        load_config(config_path)

    def test_handcrafted_mode_allows_one_step_smoke_config(self):
        raw = json.loads((PROJECT_ROOT / "configs/base.json").read_text(encoding="utf-8"))
        raw.update({"reward_mode": "handcrafted", "gamma": 0.99, "max_steps": 1})
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            config_path = Path(directory) / "short_handcrafted.json"
            config_path.write_text(json.dumps(raw), encoding="utf-8")

            loaded = load_config(config_path)

        self.assertEqual(loaded.max_steps, 1)

    def test_extends_requires_a_string_path(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            config_path = Path(directory) / "bad_extends.json"
            config_path.write_text(json.dumps({"extends": 3}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "bad_extends.json: extends must be a string path"):
                load_config(config_path)

    def test_update_batch_and_advantage_normalization_settings(self):
        base = json.loads((PROJECT_ROOT / "configs/base.json").read_text(encoding="utf-8"))
        base.update({"reward_mode": "handcrafted", "gamma": 0.99})
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            config_path = Path(directory) / "batch_settings.json"
            for normalization in ("batch", "minibatch", "none"):
                with self.subTest(normalization=normalization):
                    configured = {**base, "advantage_normalization": normalization}
                    config_path.write_text(json.dumps(configured), encoding="utf-8")
                    loaded = load_config(config_path)
                    self.assertEqual(loaded.advantage_normalization, normalization)
                    self.assertEqual(loaded.episodes_per_update, 1)

            for invalid in ("episode", "", None):
                with self.subTest(invalid=invalid):
                    configured = {**base, "advantage_normalization": invalid}
                    config_path.write_text(json.dumps(configured), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "advantage_normalization"):
                        load_config(config_path)

            configured = {**base, "episodes_per_update": 0}
            config_path.write_text(json.dumps(configured), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "episodes_per_update"):
                load_config(config_path)

    def test_preference_mode_requires_undiscounted_returns(self):
        raw = json.loads((PROJECT_ROOT / "configs/base.json").read_text(encoding="utf-8"))
        raw.update({"reward_mode": "preference_learned", "gamma": 0.99})
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            config_path = Path(directory) / "wrong_gamma.json"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires gamma == 1.0"):
                load_config(config_path)
