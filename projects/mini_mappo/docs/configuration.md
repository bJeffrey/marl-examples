# Configuration

`mini_mappo` loads `configs/handcrafted.json` by default. A relative config path
passed to `load_config()` is resolved from the project root. Config files may
extend one base file with `extends`; the child values override the base.
The resolved document is checked against `configs/schema.json` before it is
used. [schema.json](../configs/schema.json) is the authoritative reference for
parameter types, ranges, meanings, and effects of changing them. Experiment
values remain in the base and method JSON files.

## Editing in VS Code

Open the `marl-examples` repository folder in VS Code. The workspace settings
associate config JSON files with [editor.schema.json](../configs/editor.schema.json).
Hover over a field name to read its description, or press Ctrl+Space for field
and allowed-value suggestions. Unknown fields, invalid types, and out-of-range
values receive editor diagnostics. No additional extension is required.

The editor schema reuses the field definitions in `schema.json`, including
`extends`, and allows partial files such as `base.json` and method overrides.
It checks only the values written in the current file. When the program starts,
the loader merges the base and overrides, then checks the complete document
against `schema.json` and applies the cross-field constraints. The editor does
not resolve inheritance or check those runtime relationships.

For maintainers: update descriptions and constraints in
`schema.json` under `$defs.source.properties`; both editor and runtime validation
reuse them. Add new training fields to its root `required` list and the Python
`Config` dataclass. Keep parameter-by-parameter explanations in the schema and
use this guide for interactions and experiment design.

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

The current presets are defined in
[handcrafted.json](../configs/handcrafted.json),
[preference_learned.json](../configs/preference_learned.json), and their shared
[base.json](../configs/base.json). They are starting experiment settings rather
than proven optima. For comparisons, match task settings, seeds, and training
budgets deliberately; batching changes the number of updates under a fixed
episode budget. Compare success rates and successful completion lengths because
the two reward modes use different return scales.
