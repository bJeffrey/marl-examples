# MARL examples

Educational multi-agent reinforcement learning projects. Each project owns its
source code, dependencies, tests, documentation, generated outputs, and license.

## Projects

- [mini_mappo](projects/mini_mappo/README.md): a small cooperative GridWorld
  demonstrating Multi-Agent PPO with centralized training and decentralized execution.

## Repository layout

```text
projects/
└── mini_mappo/
    ├── README.md
    ├── LICENSE
    ├── pyproject.toml
    ├── src/mini_mappo/
    ├── tests/
    ├── docs/assets/
    └── outputs/<date>/<run>/
```

See each project's README for setup and run instructions. Virtual environments
and generated outputs are local artifacts and are excluded from Git. Licensing
terms are specified in each project's `LICENSE`; mini_mappo retains its upstream
MIT copyright and permission notice.
