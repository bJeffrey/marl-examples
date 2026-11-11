# Configuration

`mini_mappo` loads `configs/handcrafted.json` by default. A relative config path
passed to `load_config()` is resolved from the project root. Config files may
extend one base file with `extends`; the child values override the base.
The resolved document is checked against `configs/schema.json` before it is
used. Numeric controls below are subject to the ranges in that schema.

## Training and policy controls

`episodes_per_update` controls collection size. The trainer runs complete
episodes with one frozen policy and critic until it has collected this many,
then performs one PPO update. It flushes a final partial collection at the end
of training. Thus, comparing eight episodes per update with one changes the
number of policy updates under the same episode budget; it does not hold update
count constant.

`advantage_normalization` controls policy advantage preprocessing. `batch`
centers and scales advantages once over the combined rollout before PPO epochs.
`minibatch` does so independently for each shuffled minibatch immediately before
each optimizer step in every epoch. `none` leaves advantages raw. This setting
is separate from any normalization used to train the preference reward model.

For terminal-only rewards, centering a single episode can remove its shared
outcome signal. Collecting multiple episodes and using `batch` normalization
retains differences between outcomes. A minibatch containing only identical
advantages still loses that signal when centered; very small minibatches make
this more likely. `none` preserves the raw signal but makes actor gradients more
sensitive to reward scale.

The learned-preference preset uses eight episodes per update, 3,000 total
episodes, minibatches of 128, and a GAE lambda of 0.98. These are starting
experiment settings, not a claim that the preset is optimal. The handcrafted
preset retains one episode per update and 600 total episodes. Both use batch
advantage normalization by default.

## Parameter reference

| Parameter | Control and effect |
|---|---|
| `config_version` | Configuration contract version. Change only with an intentional schema revision. |
| `reward_mode` | `handcrafted` uses environment rewards; `preference_learned` trains on the fitted reward model. |
| `objective` | Fixed ordering: success outranks failure, shorter successful attempts rank higher, and failed attempts tie. |
| `seed` | Seed for policy initialization and environment sampling. A new seed changes the stochastic run. |
| `device_name` | `auto`, `cpu`, or `cuda` tensor execution. Hardware affects runtime and may affect numerical results. |
| `grid_size` | Cells along each side. Larger grids increase travel distances and usually make coordination harder. |
| `num_agents` | Cooperating agents sharing a policy. More agents increase joint state and action complexity, but also provide more chances for any agent to reach the target. |
| `num_actions` | Fixed at five: up, down, left, right, and stay. |
| `max_steps` | Maximum transitions per episode. More steps allow longer attempts; preference fitting also needs enough successful lengths for its holdout stride. |
| `hidden_size` | Neurons in each actor and critic hidden layer. Larger networks can represent more complex policies, at higher compute and data cost. |
| `actor_lr` | Adam actor learning rate. Larger values make policy changes more aggressive and can destabilize learning. |
| `critic_lr` | Adam critic learning rate. Larger values change value estimates faster and can destabilize advantages. |
| `gamma` | Reward discount per step. Larger values emphasize later rewards. Preference-learned mode requires `1.0` for undiscounted trajectory returns. |
| `gae_lambda` | GAE trace decay. Larger values include more distant TD residuals, reducing bias while often increasing variance. |
| `clip_epsilon` | PPO probability-ratio clipping half-width. Larger values allow more change before clipping removes the incentive; this is not a hard bound on actual policy changes. |
| `update_epochs` | PPO passes over each collected rollout. More passes reuse data but can overfit it or move policy too far. |
| `minibatch_size` | Maximum samples per optimizer step. Larger slices mean fewer, less noisy gradient steps per epoch. |
| `episodes_per_update` | Complete frozen-policy episodes collected before one PPO update. Larger batches mean fewer updates per fixed episode budget; the final partial batch is flushed. |
| `advantage_normalization` | `batch` normalizes once over the combined rollout; `minibatch` normalizes each shuffled optimizer slice each epoch; `none` uses raw advantages. This is distinct from reward-model normalization. |
| `entropy_coef` | Policy entropy bonus weight. Increasing it favors exploration and can slow decisive convergence. |
| `value_coef` | Critic squared-error loss weight. Increasing it scales critic loss and gradients only; actor and critic use separate optimizers. With Adam this does not translate directly into a proportional learning-rate change. |
| `total_episodes` | Number of training episodes. More experience costs more runtime and does not guarantee improvement. |
| `progress_window` | Episodes in trailing report averages. Larger windows smooth variation but react more slowly. |
| `output_directory` | Report root. Absolute paths are direct; relative paths resolve from the project root. |
| `goal_reward` | Handcrafted bonus for each agent reaching the target. Increasing it makes success more valuable relative to penalties. |
| `distance_penalty` | Handcrafted penalty per cell of target distance. Increasing it favors moving toward the target more strongly. |
| `step_penalty` | Handcrafted penalty per transition. Increasing it favors shorter episodes, including unsuccessful ones. |
| `preference_epochs` | Optimization passes for the preference reward model. More passes cost compute and may overfit. |
| `preference_lr` | Preference-model optimizer learning rate. Larger values fit faster but can overshoot or destabilize fitting. |
| `preference_weight_decay` | L2 regularization strength for the preference model. Increasing it regularizes the fit more strongly. |
| `preference_holdout_stride` | Every nth successful trajectory length is held out of preference fitting. A smaller stride holds out more successful lengths, leaving fewer for fitting. Validation pairs touch a held-out length; failure remains an available baseline partner. |
| `show_plots` | Whether training figures are displayed interactively. `false` skips plot display. |
| `visualize_policy` | Whether policy demonstrations are rendered. `false` skips rendering. |
| `visualization_episodes` | Demonstration episodes when visualization is enabled. More episodes show more behavior and take longer. |
