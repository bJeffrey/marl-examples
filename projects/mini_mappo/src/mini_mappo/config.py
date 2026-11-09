"""Load validated experiment settings for the educational MAPPO example."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import jsonschema
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    """Immutable settings for one MAPPO training and evaluation run."""

    # Configuration identity and experiment intent.
    config_version:            int    # Version of the resolved configuration schema.
    reward_mode:               str    # Handcrafted or preference-learned reward source.
    objective:                 str    # Task objective identifier.
    seed:                      int    # Seed for policy initialization and environment sampling.
    device_name:               str    # auto, cpu, or cuda tensor placement.

    # GridWorld dimensions; positions use row/column cell coordinates.
    grid_size:                 int    # Cells along each side of the square grid.
    num_agents:                int    # Cooperating agents sharing one policy.
    num_actions:               int    # Fixed GridWorld action set: up, down, left, right, stay.
    max_steps:                 int    # Joint environment transitions per episode.

    # Actor/critic and PPO optimization settings.
    hidden_size:               int    # Neurons in each actor and critic hidden layer.
    actor_lr:                  float  # Adam learning rate.
    critic_lr:                 float  # Adam learning rate.
    gamma:                     float  # Reward discount per environment step.
    gae_lambda:                float  # GAE trace decay per environment step.
    clip_epsilon:              float  # PPO ratio clipping half-width.
    update_epochs:             int    # Optimization passes per collected batch.
    minibatch_size:            int    # Samples across timestep and agent dimensions.
    entropy_coef:              float  # Policy entropy bonus weight.
    value_coef:                float  # Critic squared-error loss weight.

    # Experiment length and report destination.
    total_episodes:            int    # Training episodes to collect.
    progress_window:           int    # Episodes in each trailing report mean.
    output_directory:          str    # Absolute path or relative to the project root.

    # Reward parameters and preference model fitting.
    goal_reward:               float  # Bonus for each agent that reaches the target.
    distance_penalty:          float  # Reward penalty per cell of target distance.
    step_penalty:              float  # Reward penalty per environment step.
    preference_epochs:         int    # Preference model optimization passes.
    preference_lr:             float  # Preference model optimizer learning rate.
    preference_weight_decay:   float  # L2 regularization strength.
    preference_holdout_stride: int    # Every nth successful length is held out of fitting.

    # Optional interactive outputs.
    show_plots:                bool   # Display training figures after the run.
    visualize_policy:          bool   # Render policy demonstrations.
    visualization_episodes:    int    # Evaluation episodes to render.

    @property
    def obs_dim(self) -> int:
        """Return the four-value local observation size (two positions)."""

        return 4

    @property
    def global_state_dim(self) -> int:
        """Return the joint state size, in position coordinates."""

        return 2 * self.num_agents + 2

    @property
    def device(self) -> torch.device:
        """Select the configured compute device, resolving ``auto`` at runtime."""

        # CUDA availability is checked only for auto; an explicit choice is honored.
        if self.device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.device_name

        return torch.device(device_name)


def load_config(path: str | Path | None = None) -> Config:
    """Load a JSON config, resolve one base layer, and validate its contract.

    Args:
        path: Config path, relative to the project root when not absolute.
            ``None`` selects ``configs/handcrafted.json``.

    Returns:
        A frozen configuration populated from the resolved JSON document.

    Raises:
        OSError: If a config or schema file cannot be read.
        ValueError: If JSON parsing, schema validation, or domain constraints fail.
    """
    config_path = Path(path) if path is not None else Path("configs/handcrafted.json")
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()

    raw_config = _read_json(config_path)
    if "extends" in raw_config:
        extends_path = raw_config["extends"]
        if not isinstance(extends_path, str):
            raise ValueError(f"{config_path}: extends must be a string path")
        base_path = (config_path.parent / extends_path).resolve()
        base_config = _read_json(base_path)
        if "extends" in base_config:
            raise ValueError(f"{base_path}: base config cannot extend another file")
        overrides = {key: value for key, value in raw_config.items() if key != "extends"}
        raw_config = {**base_config, **overrides}

    schema_path = PROJECT_ROOT / "configs/schema.json"
    schema = _read_json(schema_path)
    try:
        jsonschema.validate(instance=raw_config, schema=schema)
    except jsonschema.ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(f"{config_path}: {location}: {error.message}") from error

    _validate_domain(raw_config, config_path)

    return Config(**raw_config)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and retain the source path in parse errors.

    Args:
        path: JSON file to read.

    Returns:
        The decoded object mapping.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If JSON is malformed or the document is not an object.
    """
    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return document


def _validate_domain(values: dict[str, Any], path: Path) -> None:
    """Check cross-field constraints that JSON Schema cannot express here.

    Args:
        values: Schema-validated resolved configuration.
        path: Source config path for actionable failure context.

    Raises:
        ValueError: If preference-mode discount or episode length is inconsistent.
    """
    if values["reward_mode"] == "preference_learned":
        if values["gamma"] != 1.0:
            raise ValueError(f"{path}: preference_learned reward mode requires gamma == 1.0")
        if values["max_steps"] < values["preference_holdout_stride"] + 1:
            raise ValueError(f"{path}: max_steps must be at least preference_holdout_stride + 1")
