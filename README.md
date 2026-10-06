# HiveBoard RL

Reinforcement and imitation learning on the [isaaclab-hiveboard](https://github.com/EESC-LabRoM/isaaclab-hiveboard)
environments: PPO teachers, distilled / PPO students and the cuRobo expert banks they track (ANYmal ball valve,
small valve, M30 thread, circuit breaker), plus robomimic behaviour cloning and DAgger.

The core environments are a git submodule (`dependencies/isaaclab-hiveboard`), installed editable. This repo only
adds the `isaaclab_hiveboard_rl` package, which registers the `*-RL-v0` / `*-RL-Play-v0` tasks on top of the core
ones, and the training scripts.

## Installation

```bash
git clone https://github.com/EESC-LabRoM/hiveboard-rl.git && cd hiveboard-rl
just setup           # core submodule + the nested ones it needs, then uv sync --extra imitation
just sync-assets     # copy generated USD assets from ../isaaclab-hiveboard (or pass core=<path>)
just list-envs
```

USD assets are gitignored in the core repo, so a fresh submodule has none: either copy them from a core checkout
that built them (`just sync-assets`) or build them inside the submodule with its `generate-newton-usd` /
`generate-anymal-usd` recipes.

`pyproject.toml` mirrors the core repo's `[tool.uv]` indexes, sources and overrides (uv only reads them from the
top-level project). Keep the two in sync when bumping the submodule.

## Usage

```bash
just rl-bank                          # build the cuRobo expert bank -> logs/expert_bank/
just rl-teacher                       # PPO teacher on privileged state
just rl-student <teacher model_*.pt>  # distil onto deployable observations
just rl-student-ppo                   # student trained directly with PPO
just rl-eval <model_*.pt>             # success rate / stage metrics
just rl-play <model_*.pt>             # Newton viewer + TorchScript/ONNX export

RL_TOOL=SmallValve just rl-teacher    # BallValve (default), SmallValve, M30Thread, CircuitBreaker

just il-collect                       # scripted-expert demos (robomimic layout)
just il-train <dataset.hdf5>          # behaviour cloning
just il-dagger <dataset.hdf5>         # DAgger rounds
just il-eval --checkpoint <ckpt>
```

## Updating the core

```bash
git -C dependencies/isaaclab-hiveboard pull origin master
just test && just list-envs
git add dependencies/isaaclab-hiveboard && git commit -m "Bump isaaclab-hiveboard"
```

The RL tasks import core modules directly (`mdp.commands.sequential_pose_command`, `assets.anymal.bench`,
`tasks.anymal.*` scenes, `mdp.events`, `mdp.actions`, `mdp.recorders`, `utils.command_setup`), so run the tests
after every bump.

## Repository structure

```
source/isaaclab_hiveboard_rl/isaaclab_hiveboard_rl/
  __init__.py          # registers the RL tasks (and, via import, the core ones)
  tasks/anymal/        # ball_valve_rl, small_valve_rl, m30_thread_rl, circuit_breaker_rl
  imitation/           # scripted expert, datasets, robomimic policy, rollouts
scripts/rl/            # train, evaluate, play, expert-bank tools
scripts/imitation/     # collect_demos, train_bc, dagger, eval_policy
logs/                  # expert banks, rsl_rl runs, imitation datasets (gitignored)
```
