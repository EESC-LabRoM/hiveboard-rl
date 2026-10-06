# HiveBoard RL

Reinforcement and imitation learning on the [isaaclab-hiveboard](https://github.com/EESC-LabRoM/isaaclab-hiveboard)
environments (ANYmal ball valve, small valve, M30 thread, circuit breaker).
**The current working solution is the PPO student trained with a cuRobo expert trajectory bank.**
Teacher training, student distillation, robomimic behaviour cloning and DAgger remain available as alternative
workflows, but are not the current working solution.

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

The working workflow is **trajectory bank → PPO student → evaluation / playback**. No PPO teacher or teacher
checkpoint is used. The student learns through PPO rewards: the bank supplies episode resets, expert-tracking
rewards and deviation terminations, rather than supervised action targets. The actor uses deployable
observations; the critic uses privileged simulator state, including expert-reference errors.

```bash
just rl-bank                          # build the cuRobo expert bank -> logs/expert_bank/
just rl-student-ppo                   # working solution: bank-guided PPO, no teacher checkpoint
just rl-eval <model_*.pt>             # success rate / stage metrics
just rl-play <model_*.pt>             # Newton viewer + TorchScript/ONNX export

# Select the same task for bank generation, training and evaluation:
RL_TOOL=SmallValve just rl-bank       # BallValve (default), SmallValve, M30Thread, CircuitBreaker
RL_TOOL=SmallValve just rl-student-ppo
```

The following alternative workflows remain in the repository. Teacher checkpoints are required only for
student distillation; these commands are not steps in the working PPO-student workflow above.

```bash
just rl-teacher                       # PPO teacher on privileged state
just rl-student <teacher-model.pt>    # distil teacher actions onto deployable observations
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
