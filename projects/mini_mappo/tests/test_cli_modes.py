"""Exercise both configured reward methods through the public module entry point."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import torch

from mini_mappo import config, preference_rewards


class RewardModeCliTests(unittest.TestCase):
    """Use temporary configurations and actual training/report export, with no desktop UI."""

    def test_both_modes_save_identifiable_results_and_model_weights(self):
        """The CLI works outside the project and produces analysis-ready artifacts per mode."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            process_env = dict(os.environ)
            process_env["MPLBACKEND"] = "Agg"

            for mode in ("handcrafted", "preference_learned"):
                with self.subTest(mode=mode):
                    # These are tiny integration workloads, not performance comparisons.
                    settings = {
                        "extends":                  str(config.PROJECT_ROOT / "configs/base.json"),
                        "reward_mode":              mode,
                        "gamma":                    1.0 if mode == "preference_learned" else 0.99,
                        "total_episodes":           2,
                        "max_steps":                6,
                        "device_name":              "cpu",
                        "show_plots":               False,
                        "visualize_policy":         False,
                        "output_directory":         str(root / "results"),
                    }
                    config_path = root / f"{mode}.json"
                    config_path.write_text(json.dumps(settings), encoding="utf-8")
                    completed = subprocess.run(
                        [sys.executable, "-W", "error", "-m", "mini_mappo", "--config", str(config_path)],
                        cwd            = root,
                        env            = process_env,
                        capture_output = True,
                        text           = True,
                        timeout        = 60,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    runs = list((root / "results").glob(f"*/*_{mode}_seed7"))
                    self.assertEqual(len(runs), 1)
                    run = runs[0]

                    saved_config = json.loads((run / "config.json").read_text())
                    summary      = json.loads((run / "summary.json").read_text())
                    metadata     = json.loads((run / "run.json").read_text())
                    self.assertEqual(saved_config["reward_mode"], mode)
                    self.assertNotIn("extends", saved_config)
                    self.assertEqual(summary["reward_mode"], mode)
                    self.assertEqual(summary["results_version"], 2)
                    self.assertEqual(summary["episodes"], 2)
                    expected_updates = (2 + saved_config["episodes_per_update"] - 1) // saved_config["episodes_per_update"]
                    self.assertEqual(summary["ppo_updates"], expected_updates)
                    self.assertEqual(len(metadata["source_sha256"]), 64)
                    self.assertEqual(len(list((run / "plots").glob("*.png"))), 4)
                    self.assertEqual(list(run.glob("*.png")), [])
                    weights = torch.load(run / "policy.pt", map_location="cpu", weights_only=True)
                    self.assertEqual(set(weights), {"actor", "critic"})

                    with (run / "episodes.csv").open(newline="") as stream:
                        episodes = list(csv.DictReader(stream))
                    self.assertEqual(len(episodes), 2)
                    self.assertIn("handcrafted_return", episodes[0])
                    if mode == "handcrafted":
                        self.assertTrue(all(row["learned_score"] == "" for row in episodes))
                        self.assertFalse((run / "reward_model.pt").exists())
                    else:
                        # Reload the persisted scorer and reproduce every saved
                        # terminal score using the original outcome metadata.
                        model = preference_rewards.OutcomeReward(saved_config["max_steps"])
                        model.load_state_dict(torch.load(run / "reward_model.pt", weights_only=True))
                        for row in episodes:
                            expected = model.terminal_reward(row["success"] == "True", int(row["episode_length"]))
                            self.assertAlmostEqual(float(row["team_return"]), expected)
                            self.assertAlmostEqual(float(row["learned_score"]), expected)
                        fit = json.loads((run / "reward_fit.json").read_text())
                        self.assertEqual(fit["strict_validation_accuracy"], 1.0)
                        self.assertTrue((run / "preference_pairs.csv").is_file())


if __name__ == "__main__":
    unittest.main()
