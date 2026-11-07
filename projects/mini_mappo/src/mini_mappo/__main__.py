"""Entry point for the modular mini-MAPPO example.

The implementation is intentionally split into small modules so each future
step toward graph attention and temporal memory can be introduced without
rewriting an all-in-one training script.
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from mini_mappo import config, environment, trainer, training_plots, visualization


def main() -> None:
    """Train, save progress reports, and display progress and policy demonstrations."""
    cfg = config.Config()
    history: list[trainer.EpisodeMetrics] = []

    # Learn the shared policy before using it in fresh demonstration episodes.
    trained_agent = trainer.train(cfg, history=history)

    # This project is run from its editable src-layout install. Resolve outputs
    # against the project root even when launched from another working directory.
    # Microseconds keep separate runs from overwriting each other.
    run_time     = datetime.now().astimezone()
    project_root = Path(__file__).resolve().parents[2]
    output_dir   = project_root / cfg.output_directory / run_time.strftime("%Y%m%d/%H%M%S_%f")
    training_plots.save_training_progress(history, cfg, output_dir)
    print(f"Training reports: {output_dir}")

    # Display the progress figures without blocking the policy animation. The
    # animation's GUI event loop keeps the windows responsive; PNGs also work in VS Code.
    plt.show(block=False)

    print("\nStarting real-time visualization...")
    # Reuse the training configuration so rendering and policy dimensions agree.
    visualization.visualize_episodes(
        environment.GridWorldEnv(cfg),
        trained_agent,
        num_episodes=5,
        realtime=True,
    )


if __name__ == "__main__":
    main()
