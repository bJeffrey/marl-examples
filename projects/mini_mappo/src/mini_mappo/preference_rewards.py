"""Learn a frozen terminal reward from synthetic preferences over episode outcomes.

This first example uses two outcome features, not trajectory clips or human
feedback. The preference provider can later be replaced without changing PPO.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from . import config


def preference_label(
    first_success: bool, first_steps: int, second_success: bool, second_steps: int,
) -> float:
    """Return 1 if the first outcome is preferred, 0 for the second, or 0.5 for a tie.

    Args:
        first_success: Whether the first attempt reached the target.
        first_steps: Joint steps in the first attempt.
        second_success: Whether the second attempt reached the target.
        second_steps: Joint steps in the second attempt.

    Returns:
        Synthetic preference: success beats failure, shorter successful attempts
        are preferred, and failed attempts tie regardless of length.
    """
    if first_success != second_success:
        return float(first_success)
    if not first_success or first_steps == second_steps:
        return 0.5

    return float(first_steps < second_steps)


class OutcomeReward(nn.Module):
    """Linear outcome scorer; its two weights are fitted from preference labels.

    Features are success and successful time saved relative to the episode
    limit. Failed attempts have zero features and therefore score zero. This
    fixes the arbitrary score offset; fastest success is normalized to one.
    """

    def __init__(self, max_steps: int):
        """Create a CPU scorer for a task with the supplied joint-step limit."""
        super().__init__()
        self.max_steps = max_steps
        self.linear    = nn.Linear(2, 1, bias=False)

        # Pairwise logistic regression starts with no preference. Scale is a
        # frozen calibration buffer, saved alongside learned weights.
        nn.init.zeros_(self.linear.weight)
        self.register_buffer("score_scale", torch.ones(()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Score [..., 2] outcome features, returning one scalar per outcome."""
        return self.linear(features).squeeze(-1) / self.score_scale

    @torch.no_grad()
    def terminal_reward(self, success: bool, steps: int) -> float:
        """Return the frozen score for one completed episode.

        Args:
            success: Whether any agent reached the target.
            steps: Actual joint steps before success or the finite episode limit.

        Returns:
            Learned outcome score; failures score zero by the feature convention.
        """
        # Multiplying time saved by success prevents different failures from
        # acquiring a preference that the configured objective never supplied.
        achieved   = float(success)
        time_saved = achieved * (1.0 - steps / self.max_steps)
        features   = torch.tensor([achieved, time_saved], dtype=torch.float32)

        return float(self(features).item())


@dataclass
class PreferenceFit:
    """Frozen model plus reproducible comparison data and fit diagnostics."""

    model:       OutcomeReward          # CPU model used only to assign terminal rewards.
    comparisons: list[dict]             # Synthetic pair labels with train/validation split.
    diagnostics: dict                   # Loss, accuracy, data provenance, and learned weights.


def fit_outcome_reward(cfg: config.Config) -> PreferenceFit:
    """Fit pairwise logistic scores, holding out successful episode lengths.

    Args:
        cfg: Validated settings for reward fitting and the finite grid-world task.

    Returns:
        Frozen reward model, all comparison labels, and diagnostic measurements.
        Training uses hypothetical outcome summaries, not extra environment steps.

    Raises:
        RuntimeError: If the fitted model cannot represent the supplied ordering.
    """
    # Each possible successful length is an outcome category, not a claim that
    # an episode with that length was executed. One failure category suffices
    # because all failures are tied. Hold out lengths, rather than overlapping pairs.
    outcomes = [(False, cfg.max_steps)]
    outcomes.extend((True, steps) for steps in range(1, cfg.max_steps + 1))
    features = []
    for success, steps in outcomes:
        achieved = float(success)
        features.append([achieved, achieved * (1.0 - steps / cfg.max_steps)])
    feature_tensor = torch.tensor(features, dtype=torch.float32)

    train_pairs      = []
    validation_pairs = []
    comparisons      = []
    for first, (first_success, first_steps) in enumerate(outcomes):
        for second in range(first, len(outcomes)):
            second_success, second_steps = outcomes[second]
            
            first_held_out  = first_success and first_steps % cfg.preference_holdout_stride == 0
            second_held_out = second_success and second_steps % cfg.preference_holdout_stride == 0
            split           = "train"
            if first_held_out or second_held_out:
                split = "validation"

            label = preference_label(first_success, first_steps, second_success, second_steps)
            pair  = (first, second, label)
            if split == "train":
                train_pairs.append(pair)
            else:
                validation_pairs.append(pair)

            comparison = {
                "first_success":  first_success,
                "first_steps":    first_steps,
                "second_success": second_success,
                "second_steps":   second_steps,
                "preference":     label,
                "split":          split,
            }
            comparisons.append(comparison)

    # Zero initialization removes seed dependence of this convex fit. Preserve
    # the caller's CPU RNG because constructing a Linear layer initially draws weights.
    with torch.random.fork_rng(devices=[]):
        model = OutcomeReward(cfg.max_steps)

    first_indices  = torch.tensor([pair[0] for pair in train_pairs], dtype=torch.long)
    second_indices = torch.tensor([pair[1] for pair in train_pairs], dtype=torch.long)
    labels         = torch.tensor([pair[2] for pair in train_pairs], dtype=torch.float32)
    optimizer      = torch.optim.Adam(
        model.parameters(),
        lr           = cfg.preference_lr,
        weight_decay = cfg.preference_weight_decay,
    )

    for _ in range(cfg.preference_epochs):
        # sigmoid(score_A - score_B) models the probability that A is preferred.
        # BCEWithLogits combines that sigmoid with cross-entropy stably; 0.5 labels
        # express ties. This learns numeric scores without a target reward magnitude.
        scores = model(feature_tensor)
        logits = scores[first_indices] - scores[second_indices]
        loss   = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Pairwise preferences do not identify a physical reward scale. Normalize once
    # using fastest success, and require the intended ordering before freezing.
    with torch.no_grad():
        scores       = model(feature_tensor)
        final_logits = scores[first_indices] - scores[second_indices]
        loss         = F.binary_cross_entropy_with_logits(final_logits, labels)
        fastest      = scores[1]
        success_gaps = scores[1:-1] - scores[2:]
        valid_order  = bool(torch.all(success_gaps > 0)) and bool(scores[-1] > scores[0])
        if not bool(torch.isfinite(scores).all()) or not valid_order:
            raise RuntimeError("Preference fit did not learn success-first, shorter-success ordering")
        model.score_scale.copy_(fastest)

    model.eval()
    model.requires_grad_(False)
    strict_validation = []
    for first, second, label in validation_pairs:
        if label != 0.5:
            correct = float((scores[first] > scores[second]).item()) == label
            strict_validation.append(float(correct))

    diagnostics = {
        "provider":                   "synthetic_outcome_preferences_v1",
        "objective":                  cfg.objective,
        "training_pairs":             len(train_pairs),
        "validation_pairs":           len(validation_pairs),
        "validation_split":           "pairs with at least one held-out successful length",
        "final_training_loss":        float(loss.item()),
        "loss_space":                 "raw scores before terminal-reward normalization",
        "strict_validation_accuracy": sum(strict_validation) / len(strict_validation),
        "learned_weights":            model.linear.weight.detach().tolist()[0],
        "score_scale":                float(model.score_scale.item()),
        "setup_environment_steps":    0,
    }

    return PreferenceFit(model, comparisons, diagnostics)


def save_preference_fit(fit: PreferenceFit, output_dir: Path) -> None:
    """Save model weights, pair labels, and diagnostics into an existing run directory.

    Args:
        fit: Frozen outcome scorer and its training evidence.
        output_dir: Run directory already created by the caller.

    Raises:
        OSError: If an artifact cannot be written.
    """
    torch.save(fit.model.state_dict(), output_dir / "reward_model.pt")
    with (output_dir / "preference_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fit.comparisons[0].keys())
        writer.writeheader()
        writer.writerows(fit.comparisons)
    with (output_dir / "reward_fit.json").open("w", encoding="utf-8") as stream:
        json.dump(fit.diagnostics, stream, indent=2)
        stream.write("\n")
