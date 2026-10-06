# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL (teacher/student) configuration for the fixed-base ANYmal + DynaArm M30 thread.

Same MDP as the ball valve (``tasks/anymal/ball_valve_rl``): time-indexed
tracking of a cuRobo expert bank, a privileged ``teacher`` group and a
deployable ``policy`` group. What changes is the part (:data:`M30_THREAD_TASK`):
the M30 nut starts two turns out and the expert runs it down until seated
(``RevoluteJoint`` 0 -> -4 pi, ``PrismaticJoint`` 7 mm -> 0), regrasping it
every 120 deg (``tasks/anymal/screw.py``): grasp, turn, let go, back off 3 cm,
grasp again. The nut's travel follows its angle through the thread's mimic
constraint, so resets write both joints (``ValveTaskCfg.coupled_joint_name``),
and the nut counts as held while both pads touch it (the hand turns away from
the fixed grasp pose with the nut).

The bank is built with::

    RL_TOOL=M30Thread just rl-bank 512 4000
"""

import math
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import MechanismGoalCfg, register_screw_joint_mimic
from isaaclab_hiveboard.tasks.anymal.m30_thread.env import AnymalM30ThreadSceneCfg
from isaaclab_hiveboard.tasks.anymal.screw import screw_commands
from isaaclab_hiveboard.tasks.scenes.threads import M30_SPEC

from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import expert_bank, mdp, valve_dynamics
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.env import (  # noqa: F401  (re-exported for the bank builder)
    ARM_FLIP_PROB,
    ARM_POSTURES,
    ARM_RANGE,
    ARM_WRIST_FLIP,
    GRIPPER_CLOSE_RAD,
    HOLD,
    REPO_ROOT,
    AnymalBallValveRLEnvCfg,
    configure_play,
    force_limited_gripper,
)
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.expert_diversity import EXPERT_DIVERSITY_RANGES as _BALL_DIVERSITY

PART_NAME = "thread"
M30_THREAD_TASK = mdp.ValveTaskCfg(
    asset_name=PART_NAME,
    joint_name="RevoluteJoint",
    closed_rad=0.0,
    # Seated: the start's two turns, tightening negative about the thread.
    open_rad=-2.0 * math.pi * M30_SPEC.turns,
    # The pinch goal 36 mm above the plate, on the board (the nut's flats meet
    # it again every 60 deg). The TCP stops 2 cm short of it along the
    # approach when the turn starts (the bank builder's grasp_in_frame median,
    # 32-env test bank; the jaws either way up).
    grasp_frame="grasp",
    grasp_offset_pos=(-0.02, 0.0, 0.0),
    grasp_offset_quat=(0.0, 0.0, 0.0, 1.0),
    # Unscrewing has no stop short of the end of the rod.
    closed_end_stop=False,
    coupled_joint_name="PrismaticJoint",
    coupled_ratio=M30_SPEC.pitch / (2.0 * math.pi),
    coupled_offset=M30_SPEC.start,
    hold_by_contact=True,
)

# Seated within the protocol's tolerance (screw_terminations): the nut at most
# seated_tolerance [m] off the plate, as an angle short of seated.
SUCCESS_TOLERANCE_RAD = 2.0 * math.pi * M30_SPEC.seated_tolerance / M30_SPEC.pitch

EXPERT_TASK = "Isaac-HiveBoard-Anymal-M30Thread-v0"
EXPERT_BANK_PATH = os.path.join(REPO_ROOT, "logs", "expert_bank", "anymal_m30_thread_bank_4000.pt")
BANK_PREFIX = "anymal_m30_thread"

VALVE_POSE_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.06, 0.06),
    "z": (-0.05, 0.05),
    "roll": (-math.pi, math.pi),
    "pitch": (-math.radians(15.0), math.radians(15.0)),
    "yaw": (-math.radians(15.0), math.radians(15.0)),
}
"""Board pose offset [m, rad], in the part's frame. The thread runs along its X axis, so roll clocks the
nut's flats (and the jaws' grasp) through a full turn about it; pitch and yaw tilt the board."""
VALVE_ANGLE_RANGE = (0.0, 0.0)
"""Initial nut angle offset [rad]. None: the expert's grasp goal is fixed on the board, aligned with the
flats at the start angle, so a nut clocked off them would be pinched on its corners."""

GRIP_TORQUE_NM = 10.0
"""Finger drive torque limit [N.m]: ~50 N per pad, ample for the thread friction (<= 0.3 N.m)."""
GRIPPER_PART_CONTACT = {"shape_regex": "/robotiq_2f_140/|/Thread/", "ke": 4.0e4, "kd": 400.0, "solimp": (0.95, 0.99, 0.001)}
"""Stiff pad-nut contacts: at MJWarp's default softness the squeeze sinks the pads into the nut."""

VALVE_DYNAMICS_RANGES = {
    # Thread friction, around the task's 0.1 N.m.
    "friction": (0.05, 0.3),
    # On top of the asset's drive damping (0.05 N.m.s/rad), which with its
    # 0.001 kg.m2 armature keeps a released nut from being flicked round.
    "damping": (0.0, 0.1),
    "spring": (0.0, 0.0),
    "breakaway": (0.0, 0.0),
    "armature": (0.001, 0.003),
}
"""Nut dynamics per episode (``valve_dynamics.py``). No spring or seat torque: nothing stops the nut at its
start, so a torque toward it would unscrew the nut."""
STUCK_PROB = 0.0
STUCK_BREAKAWAY_RANGE = (0.0, 0.0)
EXPERT_DIVERSITY_RANGES = {
    **_BALL_DIVERSITY,
    # Around the authored 0.6 rad/s: the six 120-deg turns take 25 s at the
    # slowest, 17 s at the fastest.
    "turn_rate": (0.5, 0.75),
    # The regrasps keep the authored goal; offsetting only the first grasp
    # would make it differ from the other six.
    "grasp_shift": (0.0, 0.0),
    "grasp_roll": (0.0, 0.0),
}
EXPERT_GRASP_AXES = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
EXPERT_OVERSHOOT = (0.0, 0.0)
"""Each arc runs to the nut's goal angle, at most 120 deg (no end stop to press against)."""
EXPERT_GRIP_S = 0.5
"""The command setup closes for 0.9 s, sized for the scripted task's 0.7 rad/s close; the bank's
force-limited gripper closes at 2 rad/s."""


def configure_expert(env_cfg) -> None:
    """Bank builder hook: skip the spare regrasps once the nut is seated.

    The scripted sequence has one more grasp than the turns need, to make up
    slip. A grasp, grip and turn whose nut is already within 0.1 rad of seated
    ends at once; the release and backing off still run, so the expert lets go
    and clears the nut the same way after its last turn.
    """
    seated = MechanismGoalCfg(goal_tolerance=0.1)
    for command in env_cfg.commands.pose_command.commands:
        if getattr(command, "phase", None) in ("engage", "grip", "actuate"):
            command.until = seated


@configclass
class AnymalM30ThreadRLEnvCfg(AnymalBallValveRLEnvCfg):
    """ANYmal + DynaArm runs the HiveBoard M30 nut down two turns until seated, with RL."""

    scene: AnymalM30ThreadSceneCfg = AnymalM30ThreadSceneCfg(num_envs=1024, env_spacing=3.0)  # type: ignore
    valve_task: mdp.ValveTaskCfg = M30_THREAD_TASK

    def __post_init__(self):
        super().__post_init__()
        # Six grasp-turn-release cycles of ~6 s: at the authored 0.6 rad/s the
        # expert seats the nut at ~42.5 s and has backed off by ~49.5 s; the
        # slowest turn rate adds ~4 s.
        self.episode_length_s = 57.0
        self.scene.robot = force_limited_gripper(self.scene.robot, GRIP_TORQUE_NM)
        # The thread's mimic constraint sets the travel; a drive anchored to
        # the start would hold the nut there (see m30_thread/env.py).
        self.scene.thread.actuators["advance"].stiffness = 0.0
        self.scene.thread.actuators["advance"].damping = 0.0
        coupling = screw_commands(PART_NAME, M30_SPEC).pose_command.screw_coupling
        register_screw_joint_mimic(coupling, revolute_pos_at_reset=0.0, prismatic_pos_at_reset=M30_SPEC.start)
        part = SceneEntityCfg(PART_NAME)
        self.events.valve_gravcomp.params["asset_cfg"] = part
        self.events.valve_physics_material.params["asset_cfg"] = part
        self.events.gripper_valve_contacts.params = dict(GRIPPER_PART_CONTACT)
        self.events.valve_dynamics = valve_dynamics.randomize_valve_dynamics_cfg(
            VALVE_DYNAMICS_RANGES, stuck_prob=STUCK_PROB, stuck_breakaway=STUCK_BREAKAWAY_RANGE
        )
        self.events.reset_from_bank = expert_bank.reset_from_expert_bank_cfg(
            EXPERT_BANK_PATH, mid_start_prob=0.5, expert_task=EXPERT_TASK
        )
        self.actions.valve_load.asset_name = PART_NAME
        # The expert lets go after every turn: the reference gripper opens too.
        self.observations.teacher.expert_gripper.params["follow_release"] = True
        # The cuRobo reach to the approach point is fast (forearm and wrist
        # targets up to 2.6 rad/s, past the action's 2 rad/s): replaying the
        # expert's own targets on the reference timeline left the arm up to
        # 0.23 rad behind it, and over 0.3 rad in 3 of 8 episodes with the
        # randomized command latency.
        self.terminations.expert_drift.params["max_error"] = 0.5
        # The regrasps add ~0.5 s per grasp of lag a 0.3 rad window at 0.6 rad/s does not allow.
        self.terminations.expert_valve_lag.params["max_error"] = 0.5
        # Side view of the hand at the nut, in environment 0's frame (the nut
        # sits at MECHANISM_SPAWN_POS, 1 m ahead of the robot).
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.asset_name = PART_NAME
        self.viewer.eye = (0.45, 0.65, 1.0)
        self.viewer.lookat = (1.0, 0.0, 0.7)


@configclass
class AnymalM30ThreadRLEnvCfg_PLAY(AnymalM30ThreadRLEnvCfg):
    """Evaluation: nominal physics, bank starts (randomized like training), registration error kept."""

    def __post_init__(self):
        super().__post_init__()
        configure_play(self)
