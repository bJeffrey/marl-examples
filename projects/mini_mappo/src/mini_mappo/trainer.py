"""MAPPO optimization and the high-level training loop."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from .agent import MAPPOAgent
from .config import Config
from .environment import GridWorldEnv
from .rollout import RolloutBuffer


@dataclass(frozen=True)
class EpisodeMetrics:
    """One completed training episode; these are rollout, not evaluation, metrics."""

    episode:           int      # One-based episode index within this training call.
    environment_steps: int      # Cumulative joint env.step calls, not individual agent actions.
    team_return:       float    # Undiscounted rewards summed across timesteps and agents.
    episode_length:    int      # Joint environment steps in this episode.
    success:           bool     # True if any agent reached the target, including on the last step.
    elapsed_seconds:   float    # Wall time for collection and optimization since training began.


def ppo_update(agent: MAPPOAgent, batch: dict[str, np.ndarray]) -> dict[str, float]:
    """Run clipped PPO updates over one on-policy trajectory batch."""
    
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

    # Generalized Advantage Estimation (GAE) advantages are fixed rollout targets.
    # Normalize once rather than recomputing them from the changing critic during every PPO epoch.
    # mean() averages all entries; std(unbiased=False) uses population standard
    # deviation, which remains defined even for one sample.
    # add small constant in denominator for numerical stability
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    sample_count     = obs.shape[0]
    last_actor_loss  = 0.0
    last_critic_loss = 0.0

    for _ in range(cfg.update_epochs):
        # randperm returns each row index exactly once in random order. Indexing
        # every field with the same indices keeps shuffled samples aligned.
        permutation = torch.randperm(sample_count, device=device)

        for start in range(0, sample_count, cfg.minibatch_size):
            idx = permutation[start : start + cfg.minibatch_size]

            # Compare the current policy with the policy that collected each action.
            new_log_probs, entropy = agent.evaluate_actions(obs[idx], actions[idx])
            
            # exp applies e**x element by element, converting the difference of
            # log probabilities into the new/old action probability ratio.
            ratio = torch.exp(new_log_probs - old_log_probs[idx])

            # Clip the surrogate objective to limit incentives for large policy changes.
            unclipped = ratio * advantages[idx]
            
            # clamp bounds each ratio between the supplied minimum and maximum.
            clipped = torch.clamp(
                ratio, 1.0 - cfg.clip_epsilon,
                1.0 + cfg.clip_epsilon
            ) * advantages[idx]
            
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
    config: Config | None = None,
    *,
    history: list[EpisodeMetrics] | None = None,
) -> MAPPOAgent:
    """Train the baseline MAPPO policy, optionally recording episode metrics.

    Args:
        config: Training settings; use Config defaults when omitted.
        history: Caller-owned list to append one immutable record per episode.
            Counters restart for each call. Timing excludes setup and reporting.

    Returns:
        The trained agent; plotting and file output remain the caller's responsibility.
    """
    cfg   = config or Config()
    env   = GridWorldEnv(cfg)
    agent = MAPPOAgent(cfg)

    print(f"Using device: {cfg.device}")
    print("Starting training...")
    reward_history: list[float] = []
    environment_steps = 0
    training_started  = perf_counter()

    for episode in range(1, cfg.total_episodes + 1):
        
        # Collect with the current policy before any optimizer steps change it.
        buffer, episode_reward, done, final_state = collect_episode(env, agent)

        # Do not bootstrap beyond a terminal state. If max_steps truncated the
        # rollout instead, bootstrap from the critic at the final state.
        last_values = (
            np.zeros(cfg.num_agents, dtype=np.float32)
            if done
            else agent.values(final_state)
        )
        
        # Compute temporal targets before flattening time and agent dimensions.
        advantages, returns = buffer.compute_gae(last_values)
        batch               = buffer.flatten_for_training(advantages, returns)
        
        # Reuse this on-policy batch for the configured PPO epochs, then discard it.
        ppo_update(agent, batch)

        # Count joint transitions once regardless of agent count. The optimizer
        # reads scalar losses back to CPU before returning, completing its GPU work.
        episode_length     = len(buffer.rewards)
        environment_steps += episode_length
        if history is not None:

            metrics = EpisodeMetrics(
                episode           = episode,
                environment_steps = environment_steps,
                team_return       = episode_reward,
                episode_length    = episode_length,
                success           = done,
                elapsed_seconds   = perf_counter() - training_started,
            )
            history.append(metrics)

        reward_history.append(episode_reward)
        if episode % 50 == 0:
            average_reward = float(np.mean(reward_history[-50:]))
            print(f"Episode {episode}: average reward = {average_reward:.2f}")

    print("Training complete.")
    return agent
