#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained RSL-RL checkpoint on a HiveBoard task (Isaac Lab's unified entrypoint).

Also exports the policy to ``<run>/exported`` (TorchScript/ONNX) for deployment::

    uv run python scripts/rl/play.py --task Isaac-HiveBoard-Anymal-BallValve-RL-Play-v0 \\
        --checkpoint logs/rsl_rl/anymal_ball_valve_teacher/<run>/model_<it>.pt --viz newton

Use ``--agent rsl_rl_distillation_cfg_entry_point`` for a student checkpoint.
For success-rate statistics use ``scripts/rl/evaluate.py``.
"""

import sys

import isaaclab_hiveboard_rl  # noqa: F401  (registers the HiveBoard tasks)
from train import with_newton_default

from isaaclab_rl import run_play_cli

if __name__ == "__main__":
    sys.exit(run_play_cli(["--rl_library", "rsl_rl", *with_newton_default(sys.argv[1:])]) or 0)
