"""MAPPO optimization and the high-level training loop."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from .agent import MAPPOAgent
from . import config as configuration
from . import preference_rewards
from .environment import GridWorldEnv
from .rollout import RolloutBuffer


@dataclass(frozen=True)
class EpisodeMetrics:
    """One completed training episode; these are rollout, not evaluation, metrics."""

    episode:               int                 # One-based episode index within this training call.
    environment_steps:     int                 # Cumulative joint env.step calls, not individual agent actions.
    team_return:           float               # Training objective: summed handcrafted reward or shared learned score.
    episode_length:        int                 # Joint environment steps in this episode.
    success:               bool                # Any agent reached the target, including on the last step.
    elapsed_seconds:       float               # Collection and optimization wall time since training began.
    handcrafted_return:    float        = 0.0   # Original reward formula, recorded in both modes for diagnostics.
    learned_score:         float | None = None  # Terminal outcome score; absent in handcrafted mode.
    ppo_updates_completed: int          = 0     # PPO update calls completed by this episode's reporting time.


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Center and scale one configured population of advantage samples.

    Args:
        advantages: Nonempty, one-dimensional tensor of fixed GAE estimates.

    Returns:
        Centered estimates divided by population standard deviation. The small
        numerical safeguard and unbiased=False keep constant/single-row inputs finite.
    """

    # mean averages samples; population std measures their spread without a
    # single-sample division by zero. The epsilon also handles constant inputs.
    centered = advantages - advantages.mean()
    scale    = advantages.std(unbiased=False) + 1e-8

    return centered / scale


def ppo_update(agent: MAPPOAgent, batch: dict[str, np.ndarray]) -> dict[str, float]:
    """Run configured PPO epochs over a batch collected before any weight updates.

    Args:
        agent: Actor, critic, optimizers, and validated experiment settings.
        batch: Aligned sample arrays from one or more complete episodes, with
            fixed old-policy log probabilities, GAE advantages, and return targets.

    Returns:
        Final minibatch actor and critic losses, not episode-averaged diagnostics.
    """
    
    cfg    = agent.config
    device = agent.device

    # as_tensor converts NumPy arrays to PyTorch tensors on the CPU or GPU.
    # float32 holds continuous data; long holds 64-bit action and agent indices.
    obs           = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    states        = torch.as_tensor(batch["state"], dtype=torch.float32, device=device)
    agent_ids     = torch.as_tensor(batch["agent_id"], dtype=torch.long, device=device)
    actions       = torch.as_tensor(batch["action"], dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(batch["old_log_prob"], dtype=torch.float32, device=device)
    advantages    = torch.as_tensor(batch["advantage"], dtype=torch.float32, device=device)
    returns       = torch.as_tensor(batch["return"], dtype=torch.float32, device=device)

    # GAE targets stay fixed throughout the PPO update. Batch normalization is
    # performed once across all collected episodes, preserving outcome differences
    # that centering each terminal-reward episode separately would erase.
    if cfg.advantage_normalization == "batch":
        advantages = _normalize_advantages(advantages)

    sample_count     = obs.shape[0]
    last_actor_loss  = 0.0
    last_critic_loss = 0.0

    for _ in range(cfg.update_epochs):
        # randperm returns each row index exactly once in random order. Indexing
        # every field with the same indices keeps shuffled samples aligned.
        permutation = torch.randperm(sample_count, device=device)

        for start in range(0, sample_count, cfg.minibatch_size):
            idx = permutation[start : start + cfg.minibatch_size]

            # Minibatch mode deliberately repeats normalization for each shuffled
            # optimizer slice in every epoch. In none mode raw GAE is unchanged.
            minibatch_advantages = advantages[idx]
            if cfg.advantage_normalization == "minibatch":
                minibatch_advantages = _normalize_advantages(minibatch_advantages)

            # Compare the current policy with the policy that collected each action.
            new_log_probs, entropy = agent.evaluate_actions(obs[idx], actions[idx])
            
            # exp applies e**x element by element, converting the difference of
            # log probabilities into the new/old action probability ratio.
            ratio = torch.exp(new_log_probs - old_log_probs[idx])

            # Clip the surrogate objective to limit incentives for large policy changes.
            unclipped = ratio * minibatch_advantages
            
            # clamp bounds each ratio between the supplied minimum and maximum.
            clipped = torch.clamp(
                ratio, 1.0 - cfg.clip_epsilon,
                1.0 + cfg.clip_epsilon
            ) * minibatch_advantages
            
            # min chooses the smaller objective per sample; mean averages across
            # the minibatch. Negation turns the objective into a loss to minimize.
            actor_loss = -torch.min(unclipped, clipped).mean() - cfg.entropy_coef * entropy.mean()

            # Each row belongs to one agent; select its value head from the
            # centralized critic before comparing with that agent's return.
            all_values = agent.critic(states[idx])
            
            # unsqueeze makes indices [batch, 1]; gather selects that agent's
            # column in each row; squeeze removes the size-one column dimension.
            selected_values = all_values.gather(1, agent_ids[idx].unsqueeze(1)).squeeze(1)
            
            # mse_loss averages squared prediction errors against the return targets.
            critic_loss = cfg.value_coef * F.mse_loss(selected_values, returns[idx])

            # zero_grad clears previous derivatives, which PyTorch otherwise
            # accumulates. backward computes loss derivatives for each weight;
            # step asks Adam to update the weights using those derivatives.
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            # Apply the same gradient/update sequence to the separate value network.
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            # item extracts a scalar tensor as a Python number for reporting.
            last_actor_loss  = float(actor_loss.item())
            last_critic_loss = float(critic_loss.item())

    return {"actor_loss": last_actor_loss, "critic_loss": last_critic_loss}


def collect_episode(
    env: GridWorldEnv, agent: MAPPOAgent
) -> tuple[RolloutBuffer, float, bool, np.ndarray]:
    """Collect one on-policy episode and return its rollout."""
    
    cfg = agent.config
    
    # Start a fresh trajectory with matching local observations and global state.
    obs, state     = env.reset()
    buffer         = RolloutBuffer(cfg)
    episode_reward = 0.0
    done           = False

    # each iteration of this loop corresponds to one joint environment step, not one agent action, on this episode
    for _ in range(cfg.max_steps):
        actions:   list[int] = []
        log_probs: list[float] = []

        for i in range(cfg.num_agents):
            # All agents share policy weights but act from their own observations.

            # action:   sampled from the policy distribution.
            #           In this case, it is specified in environment.py, class GridWorldEnv: MOVES
            # log_prob: the natural log probability the actor assigned to the selected action.
            #           later, the PPO update uses this log probability to compute the ratio of new to old probabilities
            #           for the selected action, which is used in the surrogate loss function to update the policy.  
            action, log_prob = agent.select_action(obs[i])
            actions.append(action)
            log_probs.append(log_prob)

        # The centralized critic evaluates the pre-action global state.
        values = agent.values(state)
        
        # Apply the joint action to obtain rewards and the next observations.
        next_obs, next_state, rewards, done = env.step(actions)

        # Pair pre-action inputs with their resulting rewards and terminal flag.
        buffer.add(obs, state, actions, log_probs, rewards, values, done)
        episode_reward += float(np.sum(rewards))

        obs, state = next_obs, next_state
        if done:
            break

    return buffer, episode_reward, done, state


def train(
    config:           configuration.Config | None           = None,
    *,
    history:          list[EpisodeMetrics] | None            = None,
    preference_model: preference_rewards.OutcomeReward | None = None,
) -> MAPPOAgent:
    """Train the baseline MAPPO policy, optionally recording episode metrics.

    Args:
        config: Validated settings; load configs/handcrafted.json when omitted.
        history: Caller-owned list to append one immutable record per episode.
            Counters restart for each call. Timing excludes setup and reporting.
        preference_model: Frozen, previously fitted outcome scorer. Required
            for preference_learned mode and unused in handcrafted mode.

    Returns:
        The trained agent; plotting and file output remain the caller's responsibility.

    Raises:
        ValueError: If preference_learned mode has no supplied reward model.
    """

    cfg = config or configuration.load_config()
    if cfg.reward_mode == "preference_learned":
        if preference_model is None:
            raise ValueError("preference_learned mode requires a fitted preference_model")
        if preference_model.max_steps != cfg.max_steps:
            raise ValueError("preference_model max_steps must match the training episode limit")

    # Both runs start from the configured policy/environment seed. Fitting the
    # CPU reward scorer is separate and does not consume this rollout RNG stream.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env   = GridWorldEnv(cfg)
    agent = MAPPOAgent(cfg)

    print(f"Using device: {cfg.device}")
    print(f"Starting {cfg.reward_mode} training (seed {cfg.seed})...")
    
    reward_history:  list[float]                 = []
    pending_batches: list[dict[str, np.ndarray]] = []
    environment_steps                            = 0
    ppo_updates_completed                        = 0
    training_started                             = perf_counter()

    for episode in range(1, cfg.total_episodes + 1):
        
        # Collect with the current policy before any optimizer steps change it.
        buffer, episode_reward, done, final_state = collect_episode(env, agent)

        handcrafted_return = episode_reward
        learned_score      = None
        if cfg.reward_mode == "preference_learned":

            # The objective concerns this complete finite attempt, including
            # timeout as a failed outcome. Never bootstrap beyond that boundary.
            # Each agent learns the same team score; report it once for the team.
            learned_score = preference_model.terminal_reward(done, len(buffer.rewards))
            buffer.set_terminal_team_reward(learned_score)
            episode_reward = learned_score

        # Do not bootstrap beyond a terminal state. If max_steps truncated the
        # handcrafted rollout instead, preserve the original critic bootstrap.
        last_values = (
            np.zeros(cfg.num_agents, dtype=np.float32)
            if done or cfg.reward_mode == "preference_learned"
            else agent.values(final_state)
        )
        
        # Compute GAE separately within each episode, including its own success
        # or timeout boundary. The critic is unchanged while this batch is collected.
        advantages, returns = buffer.compute_gae(last_values)
        batch               = buffer.flatten_for_training(advantages, returns)
        pending_batches.append(batch)

        # Keep all trajectories on-policy: update only after collection finishes,
        # then discard that batch before sampling with the newly updated actor.
        # Flush a final smaller batch so no completed episodes are silently dropped.
        batch_is_full = len(pending_batches) == cfg.episodes_per_update
        final_episode = episode == cfg.total_episodes
        if batch_is_full or final_episode:

            combined_batch = {}
            for field in pending_batches[0]:
                episode_arrays        = [episode_batch[field] for episode_batch in pending_batches]
                combined_batch[field] = np.concatenate(episode_arrays, axis=0)

            ppo_update(agent, combined_batch)
            ppo_updates_completed += 1
            pending_batches.clear()

        # Count joint transitions once regardless of agent count. The optimizer
        # reads scalar losses back to CPU before returning, completing its GPU work.
        episode_length     = len(buffer.rewards)
        environment_steps += episode_length
        if history is not None:

            metrics = EpisodeMetrics(
                episode               = episode,
                environment_steps     = environment_steps,
                team_return           = episode_reward,
                episode_length        = episode_length,
                success               = done,
                elapsed_seconds       = perf_counter() - training_started,
                handcrafted_return    = handcrafted_return,
                learned_score         = learned_score,
                ppo_updates_completed = ppo_updates_completed,
            )
            history.append(metrics)

        reward_history.append(episode_reward)
        if episode % cfg.progress_window == 0:
            average_reward = float(np.mean(reward_history[-cfg.progress_window:]))
            print(f"Episode {episode}: average reward = {average_reward:.2f}")

    print("Training complete.")
    return agent
