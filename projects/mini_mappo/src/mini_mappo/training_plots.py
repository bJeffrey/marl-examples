"""Training figures and raw reports, separate from optimization and environment rendering."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from . import trainer
from .config import Config


def plot_training_progress(
    history: Sequence[trainer.EpisodeMetrics], window: int,
    *, reward_mode: str | None = None,
) -> dict[str, Figure]:
    """Create four separate figures for a single run, without showing or saving.

    Args:
        history: Completed episodes from one train call, in collection order.
        window: Positive trailing-average width in episodes. Initial points use
            all available episodes; no future values or zero padding are used.
        reward_mode: Reward method to identify on the figures when known.

    Returns:
        Caller-owned figures keyed by report name: team_return,
        target_reaching_rate, episode_length, and return_vs_time.
        No cross-run uncertainty is implied.

    Raises:
        ValueError: If history is empty or window is not positive.
    """
    if not history:
        raise ValueError("Training progress requires at least one completed episode")
    if window < 1:
        raise ValueError("progress_window must be positive")

    steps   = np.array([record.environment_steps for record in history])
    seconds = np.array([record.elapsed_seconds for record in history])
    returns = np.array([record.team_return for record in history])
    success = np.array([record.success for record in history], dtype=float)
    lengths = np.array([record.episode_length for record in history])

    # Compute trailing means in episode space, preserving actual step/time
    # positions. Smoothing over episodes gives every completed attempt equal weight.
    values     = np.column_stack((returns, success, lengths))
    cumulative = np.vstack((np.zeros(3), np.cumsum(values, axis=0)))
    ends       = np.arange(1, len(history) + 1)
    starts     = np.maximum(0, ends - window)
    counts     = (ends - starts)[:, None]
    means      = (cumulative[ends] - cumulative[starts]) / counts

    # Each figure uses the same smoothing definition; the time plot adds a
    # hardware-dependent view without replacing environment-step sample efficiency.
    plots = (
        (
            "team_return", steps, returns, means[:, 0], "Training return",
            "Environment steps", "Training return (mode-specific units)",
        ),
        (
            "target_reaching_rate", steps, success, means[:, 1], "Target-reaching rate",
            "Environment steps", "Fraction of episodes reaching target",
        ),
        (
            "episode_length", steps, lengths, means[:, 2], "Episode length",
            "Environment steps", "Joint steps per episode",
        ),
        (
            "return_vs_time", seconds, returns, means[:, 0], "Return versus training time",
            "Elapsed training time (seconds)", "Training return (mode-specific units)",
        ),
    )
    figures: dict[str, Figure] = {}
    for name, x, raw, smoothed, title, xlabel, ylabel in plots:
        
        # One axes per figure keeps each PNG independently viewable and shareable.
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        fig.suptitle(f"MAPPO — {reward_mode or 'single run'}, training rollouts")
        
        # plot connects raw episode measurements faintly and overlays the trend.
        # This is smoothing within one run, not a confidence interval across seeds.
        ax.plot(x, raw, color="tab:blue", alpha=0.25, linewidth=0.8, label="Per episode")
        ax.plot(x, smoothed, color="tab:blue", label=f"Trailing mean (up to {window} episodes)")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
        figures[name] = fig

    figures["target_reaching_rate"].axes[0].set_ylim(-0.05, 1.05)
    figures["episode_length"].axes[0].set_ylim(bottom=0)
    
    return figures


def save_training_progress(
    history: Sequence[trainer.EpisodeMetrics], config: Config, output_dir: Path,
) -> dict[str, Figure]:
    """Write plots, episode CSV, resolved configuration, and summary for one run.

    Args:
        history: Completed episodes from one train call, in collection order.
        config: Run configuration, including the smoothing window.
        output_dir: Directory for this run. The caller supplies a unique path.

    Returns:
        Caller-owned figures keyed by report name, ready to show in a desktop
        or Python Interactive session. Each is saved as plots/<report_name>.png.

    Raises:
        ValueError: If the history or smoothing window cannot be plotted.
        OSError: If directory creation or report writing fails.
    """
    figures = plot_training_progress(history, config.progress_window, reward_mode=config.reward_mode)
    
    try:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Keep original measurements so smoothing and axes can be changed later.
        with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=asdict(history[0]).keys())
            writer.writeheader()
            for record in history:
                writer.writerow(asdict(record))

        with (output_dir / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(asdict(config), stream, indent=2)
            stream.write("\n")

        # Objective outcomes remain comparable even though the two training
        # reward scales differ. Failed episodes never count as short successes.
        successful_lengths = []
        for record in history:
            if record.success:
                successful_lengths.append(record.episode_length)
        mean_successful_steps = None
        if successful_lengths:
            mean_successful_steps = float(np.mean(successful_lengths))

        summary = {
            "results_version":        2,
            "reward_mode":            config.reward_mode,
            "seed":                   config.seed,
            "episodes":               len(history),
            "environment_steps":      history[-1].environment_steps,
            "success_rate":           len(successful_lengths) / len(history),
            "mean_steps_to_success":  mean_successful_steps,
            "mean_training_return":   float(np.mean([row.team_return for row in history])),
            "mean_handcrafted_return": float(np.mean([row.handcrafted_return for row in history])),
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")

        # savefig creates a standalone bitmap that VS Code can open without a GUI backend.
        for name, fig in figures.items():
            fig.savefig(plots_dir / f"{name}.png")
    except OSError:
        # Ownership transfers only on success; close every figure on I/O failure.
        for fig in figures.values():
            plt.close(fig)
        raise

    return figures
