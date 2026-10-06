# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared argument parsing and environment setup for the imitation scripts."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import add_launcher_args
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

DEFAULT_TASK = "Isaac-HiveBoard-Spot-Lamp-v0"
DEFAULT_DATASET_DIR = os.path.join("logs", "imitation", "datasets")
DEFAULT_RUN_DIR = os.path.join("logs", "imitation", "runs")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the task/simulation arguments every imitation script shares."""
    parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel.")
    parser.add_argument("--seed", type=int, default=None, help="Environment reset seed.")
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help="Override the task's episode length in seconds. Raise it when the scripted sequence "
        "needs longer than the task default to finish.",
    )
    parser.add_argument(
        "--disable_events",
        default=None,
        help="Comma-separated event terms to switch off, e.g. 'valve_joint_parameters'. Use this "
        "when a domain-randomization term defeats the scripted expert and would otherwise make "
        "every episode a rejected demonstration.",
    )
    parser.add_argument(
        "--setup",
        default=None,
        help="Command and TCP settings saved by scripts/command_edit.py. "
        "Default: configs/<task>.json, or the base task's file for a -Play-v0 variant - "
        "the same setup scripts/play.py runs, so the expert here is the sequence you tuned there.",
    )
    parser.add_argument(
        "--no_setup",
        action="store_true",
        help="Ignore the task's configs/<task>.json and run the task as coded.",
    )
    add_launcher_args(parser)


def parse_with_presets(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse CLI args, applying this repository's Newton defaults.

    Mirrors ``scripts/play.py``: Hydra overrides are split out of ``argv`` and
    the Newton MJWarp physics preset is selected unless the caller picked one.

    Args:
        parser: Parser already populated by :func:`add_common_args`.

    Returns:
        The parsed arguments. Hydra tokens are moved into ``sys.argv`` so that
        :func:`resolve_task_config` picks them up.
    """
    args, hydra_args = setup_preset_cli(parser)
    if not any(token.startswith(("physics=", "presets=")) for token in hydra_args):
        hydra_args.append("physics=newton_mjwarp")
    if args.visualizer is None and not getattr(args, "visualizer_explicit", False):
        # Data collection and DAgger rounds are throughput-bound, so no viewer
        # unless one is asked for - the opposite of play.py, where watching is
        # the point. This must be an empty list rather than ["none"]: the
        # simulation context treats any named type as a request it has to
        # satisfy, and "none" is not a visualizer it can build.
        args.visualizer = []
    sys.argv = [sys.argv[0], *hydra_args]
    return args


def build_env_cfg(args: argparse.Namespace):
    """Resolve the task config and apply the shared CLI overrides.

    The saved command setup is applied exactly as ``scripts/play.py`` applies
    it. The scripted expert *is* the task's command sequence, so a dataset
    collected without the setup would transcribe a different expert from the
    one the task was tuned and watched with.

    Args:
        args: Parsed arguments from :func:`parse_with_presets`.

    Returns:
        The resolved environment configuration.
    """
    env_cfg, _ = resolve_task_config(args.task, "")
    _apply_command_setup(env_cfg, args)
    # The recorder stamps this into the dataset's env_args, which is how a
    # dataset (and anything merged from it) stays traceable to its task.
    env_cfg.env_name = args.task
    env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
    if getattr(args, "episode_length_s", None) is not None:
        env_cfg.episode_length_s = args.episode_length_s
    for term in _split_csv(getattr(args, "disable_events", None)):
        if not hasattr(env_cfg.events, term):
            available = sorted(name for name in vars(env_cfg.events) if not name.startswith("_"))
            raise SystemExit(f"Task '{args.task}' has no event term '{term}'. Available: {available}")
        setattr(env_cfg.events, term, None)
        print(f"[INFO] Disabled event term: {term}")
    return env_cfg


def _apply_command_setup(env_cfg, args: argparse.Namespace) -> None:
    """Apply the task's saved command setup, mirroring ``scripts/play.py``.

    Args:
        env_cfg: Freshly resolved environment configuration.
        args: Parsed arguments, supplying ``setup`` and ``no_setup``.

    Raises:
        SystemExit: If both ``--setup`` and ``--no_setup`` are given.
    """
    from isaaclab_hiveboard.utils.command_setup import apply_setup, load_setup, resolve_setup_path

    setup = getattr(args, "setup", None)
    if getattr(args, "no_setup", False):
        if setup:
            raise SystemExit("--setup and --no_setup are mutually exclusive.")
        return
    if not setup:
        resolved = resolve_setup_path(args.task)
        if resolved is None:
            return
        setup = str(resolved)
    print(f"[INFO] Using saved setup {setup} (pass --no_setup to skip).")
    apply_setup(env_cfg, load_setup(setup), task=args.task)


def _split_csv(value: str | None) -> list[str]:
    """Split a comma-separated option into stripped, non-empty entries."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def attach_recorder(env_cfg, recorder_cfg, dataset_dir: str, dataset_name: str):
    """Point a freshly built recorder config at ``dataset_dir/dataset_name``.

    The config is rebuilt by the caller rather than mutated in place because
    Hydra round-trips class types to strings, which would leave
    ``dataset_file_handler_class_type`` and every term's ``class_type``
    uncallable - the same fix ``scripts/play.py`` applies.

    Args:
        env_cfg: Environment configuration to attach the recorder to.
        recorder_cfg: A live recorder manager config instance.
        dataset_dir: Directory for the HDF5 output.
        dataset_name: Filename stem, without the ``.hdf5`` extension.

    Returns:
        The absolute path the dataset will be written to.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    recorder_cfg.dataset_export_dir_path = dataset_dir
    recorder_cfg.dataset_filename = dataset_name
    env_cfg.recorders = recorder_cfg
    return os.path.join(dataset_dir, f"{dataset_name}.hdf5")


def require_missing(path: str) -> None:
    """Refuse to clobber an existing dataset.

    Args:
        path: Path that must not already exist.

    Raises:
        FileExistsError: If ``path`` exists.
    """
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite an existing dataset: {path}")
