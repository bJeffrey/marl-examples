"""Check metric meanings, trailing averages, and exported progress reports."""

import csv
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
import numpy as np

from mini_mappo import config, trainer, training_plots


class TrainingProgressTests(unittest.TestCase):
    """Use small real training runs and hand-computed plotting fixtures."""

    def test_training_records_joint_steps_without_changing_return_type(self):
        """Each joint step counts once even when multiple agents act."""
        cfg = replace(config.load_config(), total_episodes=2, max_steps=1)
        history: list[trainer.EpisodeMetrics] = []
        agent = trainer.train(cfg, history=history)

        self.assertIsInstance(agent, trainer.MAPPOAgent)
        self.assertEqual([record.episode for record in history], [1, 2])
        self.assertEqual([record.environment_steps for record in history], [1, 2])
        self.assertEqual([record.episode_length for record in history], [1, 1])
        self.assertGreaterEqual(history[1].elapsed_seconds, history[0].elapsed_seconds)
        self.assertTrue(all(np.isfinite(record.team_return) for record in history))

    def test_report_preserves_raw_data_and_uses_past_episodes_only(self):
        """Two-episode means must use partial startup windows and actual step positions."""
        history = [
            trainer.EpisodeMetrics(1, 2, 2.0, 2, False, 1.0),
            trainer.EpisodeMetrics(2, 3, 4.0, 1, True, 3.0),
            trainer.EpisodeMetrics(3, 6, 8.0, 3, True, 6.0),
        ]
        cfg = replace(config.load_config(), progress_window=2)

        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            figures    = training_plots.save_training_progress(history, cfg, output_dir)
            try:
                expected_names = {
                    "team_return", "target_reaching_rate", "episode_length", "return_vs_time",
                }
                self.assertEqual(set(figures), expected_names)
                for figure in figures.values():
                    self.assertEqual(len(figure.axes), 1)

                return_axis  = figures["team_return"].axes[0]
                success_axis = figures["target_reaching_rate"].axes[0]
                length_axis  = figures["episode_length"].axes[0]
                time_axis    = figures["return_vs_time"].axes[0]
                np.testing.assert_array_equal(return_axis.lines[0].get_xdata(), [2, 3, 6])
                np.testing.assert_array_equal(return_axis.lines[0].get_ydata(), [2, 4, 8])
                np.testing.assert_allclose(return_axis.lines[1].get_ydata(), [2, 3, 6])
                np.testing.assert_allclose(success_axis.lines[1].get_ydata(), [0, 0.5, 1])
                np.testing.assert_allclose(length_axis.lines[1].get_ydata(), [2, 1.5, 2])
                np.testing.assert_array_equal(time_axis.lines[0].get_xdata(), [1, 3, 6])

                # Verify actual artifact formats and original, unsmoothed values.
                plots_dir = output_dir / "plots"
                png_names = {path.stem for path in plots_dir.glob("*.png")}
                self.assertEqual(png_names, expected_names)
                self.assertEqual(list(output_dir.glob("*.png")), [])
                for name in expected_names:
                    png = (plots_dir / f"{name}.png").read_bytes()
                    self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
                with (output_dir / "episodes.csv").open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([float(row["team_return"]) for row in rows], [2, 4, 8])
                self.assertEqual([row["success"] for row in rows], ["False", "True", "True"])
                settings = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(settings["progress_window"], 2)
            finally:
                for figure in figures.values():
                    plt.close(figure)

    def test_empty_history_and_invalid_window_fail_clearly(self):
        """Reject plot requests with no samples or an undefined smoothing width."""
        with self.assertRaisesRegex(ValueError, "completed episode"):
            training_plots.plot_training_progress([], window=2)
        with self.assertRaisesRegex(ValueError, "positive"):
            training_plots.plot_training_progress(
                [trainer.EpisodeMetrics(1, 1, 0.0, 1, False, 0.1)], window=0,
            )


if __name__ == "__main__":
    unittest.main()
