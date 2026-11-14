"""Entry point for the modular mini-MAPPO example.

The implementation is intentionally split into small modules so each future
step toward graph attention and temporal memory can be introduced without
rewriting an all-in-one training script.
"""

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
from time import perf_counter
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from mini_mappo import config, environment, preference_rewards, trainer, training_plots, visualization


def main() -> None:
    """Load a reward-mode configuration, train, and save an independently identifiable run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Config file, relative to the project or absolute")
    args = parser.parse_args()
    cfg  = config.load_config(args.config)
    history: list[trainer.EpisodeMetrics] = []

    # Preserve the date/run hierarchy while making the reward method and seed
    # visible in directory names. Never overwrite another run's artifacts.
    run_time     = datetime.now().astimezone()
    project_root = Path(__file__).resolve().parents[2]
    run_name     = f"{run_time:%H%M%S_%f}_{cfg.reward_mode}_seed{cfg.seed}"
    output_dir   = project_root / cfg.output_directory / run_time.strftime("%Y%m%d") / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    # Reward fitting is a separate, timed phase. Its score remains frozen for all
    # PPO episodes; pairwise fit data never enters the on-policy rollout buffer.
    preference_fit   = None
    preference_model = None
    setup_started    = perf_counter()
    if cfg.reward_mode == "preference_learned":
        preference_fit   = preference_rewards.fit_outcome_reward(cfg)
        preference_model = preference_fit.model
        preference_rewards.save_preference_fit(preference_fit, output_dir)
    
    setup_seconds = perf_counter() - setup_started

    trained_agent = trainer.train(cfg, history=history, preference_model=preference_model)
    figures       = training_plots.save_training_progress(history, cfg, output_dir)

    # Save weights and execution metadata for later analysis, separate from the
    # independent task metrics. Seed recording does not promise identical GPU runs.
    policy_state = {
        "actor":  trained_agent.actor.state_dict(),
        "critic": trained_agent.critic.state_dict(),
    }
    torch.save(policy_state, output_dir / "policy.pt")

    # Fingerprint the source used for this run without requiring Git at runtime.
    source_digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        source_digest.update(source.name.encode("utf-8") + b"\0")
        source_digest.update(source.read_bytes())
    
    run_metadata = {
        "config_source":          str(args.config or Path("configs/handcrafted.json")),
        "reward_setup_seconds":   setup_seconds,
        "device":                 str(cfg.device),
        "torch_version":          torch.__version__,
        "numpy_version":          np.__version__,
        "python_version":         sys.version,
        "source_sha256":          source_digest.hexdigest(),
        "preference_provider":    None,
    }
    
    if preference_fit is not None:
        run_metadata["preference_provider"] = preference_fit.diagnostics["provider"]
    with (output_dir / "run.json").open("w", encoding="utf-8") as stream:
        json.dump(run_metadata, stream, indent=2)
        stream.write("\n")
    print(f"Training reports: {output_dir}")

    # Display the progress figures without blocking the policy animation. The
    # animation's GUI event loop keeps the windows responsive; PNGs also work in VS Code.
    if cfg.show_plots:
        plt.show(block=False)
    else:
        for figure in figures.values():
            plt.close(figure)

    if cfg.visualize_policy:
        print("\nStarting real-time visualization...")
        visualization.visualize_episodes(
            environment.GridWorldEnv(cfg),
            trained_agent,
            num_episodes = cfg.visualization_episodes,
            realtime     = True,
        )
    elif cfg.show_plots:
        plt.show()


if __name__ == "__main__":
    main()
