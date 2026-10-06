# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL (teacher/student) configuration for the fixed-base ANYmal + DynaArm small gate valve.

Same MDP as the ball valve (``tasks/anymal/ball_valve_rl``): time-indexed
tracking of a cuRobo expert bank, a privileged ``teacher`` group and a
deployable ``policy`` group. What changes is the valve
(:data:`SMALL_VALVE_TASK`): the expert clamps the handwheel's hub, with the
TCP on the stem axis, and turns it a quarter turn (``RevoluteJoint`` 0 ->
-pi/2) by rolling the wrist. The joint is continuous, so there is no end stop
at open. The stem's ``PrismaticJoint`` is held by a stiff drive and is not
randomized.

The bank is built with::

    uv run python scripts/rl/build_expert_bank.py --task Isaac-HiveBoard-Anymal-SmallValve-RL-v0 \\
        --num_envs 512 --num_trajectories 5000 --output logs/expert_bank/anymal_small_valve_bank_5000.pt
"""

import math
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.small_valve.configs.scene import SmallValveSceneCfg

from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import expert_bank, mdp, valve_dynamics
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.env import (  # noqa: F401  (re-exported for the bank builder)
    ARM_FLIP_PROB,
    ARM_POSTURES,
    ARM_RANGE,
    ARM_WRIST_FLIP,
    GRIPPER_CLOSE_RAD,
    HOLD,
    REPO_ROOT,
    SUCCESS_TOLERANCE_RAD,
    AnymalBallValveRLEnvCfg,
    configure_play,
    force_limited_gripper,
)
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.expert_diversity import EXPERT_DIVERSITY_RANGES as _BALL_DIVERSITY

VALVE_NAME = "small_valve"
SMALL_VALVE_TASK = mdp.ValveTaskCfg(
    asset_name=VALVE_NAME,
    joint_name="RevoluteJoint",
    closed_rad=0.0,
    open_rad=-math.pi / 2,
    # The expert's engage goal: on the stem axis, 12 mm above the handwheel.
    grasp_frame="rotate_frame",
    # The TCP sits on rotate_frame when the expert starts turning (the bank
    # builder's grasp_in_frame: median within 2 mm, about the stem either way
    # up, spread by the sampled grasp roll and shift).
    grasp_offset_pos=(0.0, 0.0, 0.0),
    grasp_offset_quat=(0.0, 0.0, 0.0, 1.0),
    # The handwheel joint is continuous.
    closed_end_stop=False,
)

EXPERT_TASK = "Isaac-HiveBoard-Anymal-SmallValve-v0"
EXPERT_BANK_PATH = os.path.join(REPO_ROOT, "logs", "expert_bank", "anymal_small_valve_bank_5000.pt")
BANK_PREFIX = "anymal_small_valve"

VALVE_POSE_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.06, 0.06),
    "z": (-0.05, 0.05),
    "roll": (-math.radians(15.0), math.radians(15.0)),
    "pitch": (-math.pi, math.pi),
    "yaw": (-math.radians(15.0), math.radians(15.0)),
}
"""Valve root pose offset [m, rad], in the valve's frame. The stem runs along the valve body's Y axis, so
pitch turns the handwheel's spokes through a full turn about it; roll and yaw tilt the panel."""
VALVE_ANGLE_RANGE = (-0.4, 0.0)
"""Initial handwheel angle offset [rad]: some episodes start part-open."""

GRIP_TORQUE_NM = 20.0
"""Finger drive torque limit [N.m] (the ball valve's: ~100 N per pad)."""

VALVE_DYNAMICS_RANGES = {
    "friction": (0.02, 1.0),
    "damping": (0.0, 0.1),
    # No spring or seat torque: the handwheel has no end stop at closed, and
    # any torque toward closed spun it past closed (up to 63 rad/s before the
    # expert arrived). A hard-to-turn valve is a high friction instead.
    "spring": (0.0, 0.0),
    "breakaway": (0.0, 0.0),
    "armature": (0.0005, 0.005),
}
"""Handwheel dynamics per episode (see ``valve_dynamics.py``). The default handwheel (friction 0.0002 N.m)
spins freely: struck by a finger it reached -138 rad/s."""
STUCK_PROB = 0.0
STUCK_BREAKAWAY_RANGE = (0.0, 0.0)
EXPERT_DIVERSITY_RANGES = {
    **_BALL_DIVERSITY,
    # Along the stem: how deep the pads close over the hub [m].
    "grasp_shift": (-0.005, 0.005),
    # About the stem [rad]. Clocked further round the handwheel, the pads
    # slipped: kept 96% below 0.15 rad, 19% at 0.20-0.25 (128-env test bank).
    "grasp_roll": (-0.12, 0.12),
}
EXPERT_GRASP_AXES = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
"""Grasp shift and roll both about the grasp goal's X axis, the stem: deeper or shallower over the hub, and
the hand clocked around it."""
EXPERT_OVERSHOOT = (0.0, 0.05)
"""No end stop holds the handwheel at open, so the arc does not run past it for a free valve; under load it
runs on by the handwheel's lag (0.05 rad per N.m, as on the ball valve)."""
EXPERT_GRIP_S = 0.3
"""The command setup closes the gripper for 1.2 s, sized for the scripted task's slow close; the bank's
gripper closes at 2 rad/s, onto the hub in ~0.25 s."""


@configclass
class AnymalSmallValveRLEnvCfg(AnymalBallValveRLEnvCfg):
    """ANYmal + DynaArm opens the HiveBoard small gate valve (handwheel 0 -> -90 deg) with RL."""

    scene: SmallValveSceneCfg = SmallValveSceneCfg(num_envs=1024, env_spacing=3.0)  # type: ignore
    valve_task: mdp.ValveTaskCfg = SMALL_VALVE_TASK

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = force_limited_gripper(self.scene.robot, GRIP_TORQUE_NM)
        valve = SceneEntityCfg(VALVE_NAME)
        self.events.valve_gravcomp.params["asset_cfg"] = valve
        self.events.valve_physics_material.params["asset_cfg"] = valve
        self.events.valve_dynamics = valve_dynamics.randomize_valve_dynamics_cfg(
            VALVE_DYNAMICS_RANGES, stuck_prob=STUCK_PROB, stuck_breakaway=STUCK_BREAKAWAY_RANGE
        )
        self.events.reset_from_bank = expert_bank.reset_from_expert_bank_cfg(
            EXPERT_BANK_PATH, mid_start_prob=0.5, expert_task=EXPERT_TASK
        )
        self.actions.valve_load.asset_name = VALVE_NAME
        self.viewer.asset_name = VALVE_NAME
        self.viewer.body_name = "eixo_trans"


@configclass
class AnymalSmallValveRLEnvCfg_PLAY(AnymalSmallValveRLEnvCfg):
    """Evaluation: nominal physics, bank starts (randomized like training), registration error kept."""

    def __post_init__(self):
        super().__post_init__()
        configure_play(self)
