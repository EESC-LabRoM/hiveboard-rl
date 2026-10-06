# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL (teacher/student) configuration for the fixed-base ANYmal + DynaArm circuit breaker.

Same MDP as the ball valve (``tasks/anymal/ball_valve_rl``): time-indexed
tracking of a cuRobo expert bank, a privileged ``teacher`` group and a
deployable ``policy`` group. What changes is the mechanism
(:data:`CIRCUIT_BREAKER_TASK`): the breaker's toggle lever is too small to
pinch (2.3 cm long, 2 cm thick; the 2F-140 bottoms out ~24 mm apart), so
the expert closes its hand into a fist, comes in 7 cm below the pivot and
pushes straight up past it to 7 cm above, which flips the lever from its down
stop (``RevoluteJoint`` +30 deg) to its up stop (-30 deg), then backs off to
the approach point. The pushed lever runs ahead of the hand to its stop, so
it counts as switched once a pad has touched it (``ValveTaskCfg.push``).

The bank is built with::

    RL_TOOL=CircuitBreaker just rl-bank 512 5000
"""

import math
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import GripperCommand
from isaaclab_hiveboard.tasks.anymal.circuit_breaker.configs.events import CircuitBreakerEventCfg
from isaaclab_hiveboard.tasks.anymal.circuit_breaker.configs.scene import CircuitBreakerSceneCfg

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
)
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.expert_diversity import EXPERT_DIVERSITY_RANGES as _BALL_DIVERSITY

BREAKER_NAME = "circuit_breaker"
# HiveBoard limits are [-pi/6, +pi/6] about the lever's Y axis; positive tips
# the lever down.
LEVER_DOWN_RAD = math.pi / 6
LEVER_UP_RAD = -math.pi / 6
CIRCUIT_BREAKER_TASK = mdp.ValveTaskCfg(
    asset_name=BREAKER_NAME,
    joint_name="RevoluteJoint",
    closed_rad=LEVER_DOWN_RAD,
    open_rad=LEVER_UP_RAD,
    # 2 cm out from the pivot, on the breaker body (not the lever): where the
    # fist crosses the lever. The expert has no grasp pose, so this is only
    # the target the privileged observations point at and that "clear" is
    # measured from.
    grasp_frame="rotate_frame",
    grasp_offset_pos=(0.0, 0.0, 0.0),
    grasp_offset_quat=(0.0, 0.0, 0.0, 1.0),
    # The joint limit stops the lever at down.
    closed_end_stop=True,
    push=True,
)

GRIPPER_BREAKER_CONTACT = {
    "shape_regex": "/robotiq_2f_140/|/CircuitBreaker/",
    "ke": 4.0e4,
    "kd": 400.0,
    "solimp": (0.95, 0.99, 0.001),
}
"""Stiff hand-lever contacts, as on the valves."""

SUCCESS_TOLERANCE_RAD = 0.1
"""Switched: the lever within 6 deg of its up stop. The HiveBoard lever is a free hinge, with no toggle spring
to snap it home, and before its stops were damped (``breaker_end_stops``) a flicked lever rebounded a few
degrees (the ball valve's 2 deg failed 5.5% of the expert's flips that stayed up)."""
PUSH_DONE_RAD = 0.035
"""The push ends once the lever is this close to its up stop [rad]."""

EXPERT_TASK = "Isaac-HiveBoard-Anymal-CircuitBreaker-v0"
EXPERT_BANK_PATH = os.path.join(REPO_ROOT, "logs", "expert_bank", "anymal_circuit_breaker_bank_5000.pt")
BANK_PREFIX = "anymal_circuit_breaker"

VALVE_POSE_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.06, 0.06),
    "z": (-0.05, 0.05),
    "roll": (-math.radians(15.0), math.radians(15.0)),
    "pitch": (-math.radians(15.0), math.radians(15.0)),
    "yaw": (-math.radians(15.0), math.radians(15.0)),
}
"""Breaker pose offset [m, rad]: board placement and tilt. Breaker panels are mounted upright, so no full
turn about the panel normal."""
VALVE_ANGLE_RANGE = (0.0, 0.0)
"""Initial lever angle offset from down [rad]: on its stop. Started 0.05 rad off it, the detent's torque
slammed the lever into the stop at up to 15 rad/s on the first steps."""

VALVE_DYNAMICS_RANGES = {
    # The scripted task's range.
    "friction": (0.01, 0.1),
    "damping": (0.0, 0.005),
    # Off, as on the ball valve: a released lever must stay up, and MJWarp's
    # soft friction creeps it back under any spring torque.
    "spring": (0.0, 0.0),
    # A light detent holding the lever down over its first 0.15 rad.
    "breakaway": (0.0, 0.1),
    "armature": (0.0005, 0.002),
}
"""Lever dynamics per episode (``valve_dynamics.py``), as stiff as a fist push on the DynaArm manages.

The push loads the soft wrist (flexion 40 N.m/rad: ~1 cm of give per 10 N at the fingertips), so the hand
yields out and over the lever's tip instead of driving it on. With detents up to 0.5 N.m, a quarter of the
levers stuck at 0.5-1.5 N.m and armatures up to 0.005 kg.m2, the expert flipped 27% of them (50% in the
lightest third of detents, 3% in the stiffest); with these ranges, 94%. A detent left the lever 0.05 rad
off its stop slammed it in at up to 15 rad/s.
"""
STUCK_PROB = 0.0
STUCK_BREAKAWAY_RANGE = (0.1, 0.1)
EXPERT_DIVERSITY_RANGES = {
    **_BALL_DIVERSITY,
    # The push's hand speed [m/s], around the authored 0.15.
    "turn_rate": (0.1, 0.2),
    # Across the lever's 4 cm width [m].
    "grasp_shift": (-0.01, 0.01),
    # The fist rolled about the approach axis [rad].
    "grasp_roll": (-0.3, 0.3),
}
EXPERT_GRASP_AXES = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
"""Push goals' (shift, roll) axes: the goal frames' X points into the board, Y across the lever."""
EXPERT_OVERSHOOT = (0.0, 0.0)
"""No arc: the push itself runs past the lever."""
EXPERT_GRIP_S = None
PUSH_DEPTH = 0.01
"""How much deeper than the scene's push goals the expert pushes [m], toward the board. The goals put the
fingertips over the last ~3 mm of the lever's tip, which slipped past them as the lever rose: pushed from
its down stop the lever stopped between +0.18 and -0.2 rad. 1 and 1.5 cm deeper both flipped it; a push
that also went deeper as it rose (1 -> 3 cm, 2 -> 4 cm) flipped fewer (19%, 4%). With the goals raised,
0.5 cm (nearer the lever's edge) flipped as many but left more levers bouncing back off the stop: 80.9%
complete against 88.3% at 1 cm (512 expert episodes each)."""
GOAL_RAISE = 0.02
"""How much higher than the scene's frames every go-to goal of the expert is [m]. Watching student v1, the
user saw its push start too low and not rise high enough. Raised 2 cm, the expert flipped 94.7% of the
levers (93.2% unraised; 91.8% at 1 cm with the 0.5 cm depth), 88.3% complete either way."""
EXPERT_MAX_OPEN_S = 5.0
"""Latest flip the bank keeps [s]: the expert flips 95% of the levers by 4.35 s; the 3% past 5 s are pushes
stalled for up to 4.75 s under a lever creeping to its stop."""
PRESS_S = 0.3
"""How long the fist holds the lever at its up stop after the push [s]."""
RETREAT_OUT = 0.06
"""How far out from the push's top goal the retreat first goes [m]: 7 cm above the pivot, 10 cm out from it."""


def configure_expert(env_cfg) -> None:
    """Bank builder hook: the lever starts down, the push goes deeper and ends at the stop, and the expert
    neither idles nor opens its hand.

    The scripted task starts the lever at 0 rad, half way, where nothing holds
    it; a breaker rests at a stop, and from there the authored push slips off
    the lever (:data:`PUSH_DEPTH`). Deeper, the fist ends up under the lever
    held at its up stop, short of the push's goal above it, so the push ends
    once the lever is up (:data:`PUSH_DONE_RAD`) and holds it there for
    :data:`PRESS_S`. The authored sequence also waits 0.3 s before the push
    with the fist already closed, and opens the hand once it has backed off,
    which a push does not need; its retreat is replaced by
    :data:`RETREAT_OUT`'s up-and-out leg and then the approach point.
    """
    env_cfg.scene.circuit_breaker.init_state.joint_pos = {"RevoluteJoint": LEVER_DOWN_RAD}
    command = env_cfg.commands.pose_command
    # The fist is closed from the approach on: no grip wait and no release.
    # After the push it holds the lever at its stop for PRESS_S: let go at
    # once, a lever flicked into the stop bounced back to -0.42 rad in 14% of
    # the episodes.
    press = next(c for c in command.commands if isinstance(c, GripperCommand) and c.phase == "actuate")
    command.commands = [c for c in command.commands if not isinstance(c, GripperCommand) or c is press]
    press.duration_s = PRESS_S
    for c in command.commands:
        if isinstance(c, GripperCommand):
            continue
        # The goal frames' X points into the board, their Z up.
        depth = PUSH_DEPTH if c.phase in ("engage", "actuate") else 0.0
        c.target_offset_pos = (depth, 0.0, GOAL_RAISE)
        if c.phase == "actuate":
            c.done_when_joint = (BREAKER_NAME, "RevoluteJoint", -1.0, LEVER_UP_RAD + PUSH_DONE_RAD)
    # Back off up and out before coming down to the approach point. The push
    # ends on the lever's angle with the soft arm lagging its plan, so the hand
    # keeps rising past the lever's tip; a retreat that moved down (straight to
    # the approach point, or first dropping under the pivot) came down on the
    # lever and turned it back (7% and 34% of the episodes).
    engage = next(c for c in command.commands if c.phase == "engage")
    retreat = next(i for i, c in enumerate(command.commands) if c.phase == "retreat")
    clear = engage.replace(
        phase="retreat", target_frame_name="lever_pivot_above", target_offset_pos=(-RETREAT_OUT, 0.0, GOAL_RAISE)
    )
    command.commands = command.commands[:retreat] + [clear] + command.commands[retreat:]


@configclass
class AnymalCircuitBreakerRLEnvCfg(AnymalBallValveRLEnvCfg):
    """ANYmal + DynaArm flips the HiveBoard circuit breaker up (+30 -> -30 deg) with RL."""

    scene: CircuitBreakerSceneCfg = CircuitBreakerSceneCfg(num_envs=1024, env_spacing=3.0)  # type: ignore
    valve_task: mdp.ValveTaskCfg = CIRCUIT_BREAKER_TASK

    def __post_init__(self):
        super().__post_init__()
        # The expert has flipped the lever by 4.2 s and backed off by 6.8 s
        # (median; 90% by 6.9 s, the slowest reaches included). Pushes that
        # stall under a lever creeping to its stop ran to 12 s; the bank, one
        # second longer than this, drops them as not backed off.
        self.episode_length_s = 9.0
        self.scene.circuit_breaker.init_state.joint_pos = {"RevoluteJoint": LEVER_DOWN_RAD}
        breaker = SceneEntityCfg(BREAKER_NAME)
        self.events.valve_gravcomp.params["asset_cfg"] = breaker
        self.events.valve_physics_material.params["asset_cfg"] = breaker
        self.events.gripper_valve_contacts.params = dict(GRIPPER_BREAKER_CONTACT)
        self.events.breaker_end_stops = CircuitBreakerEventCfg().breaker_end_stops
        self.events.valve_dynamics = valve_dynamics.randomize_valve_dynamics_cfg(
            VALVE_DYNAMICS_RANGES, stuck_prob=STUCK_PROB, stuck_breakaway=STUCK_BREAKAWAY_RANGE
        )
        self.events.reset_from_bank = expert_bank.reset_from_expert_bank_cfg(
            EXPERT_BANK_PATH, mid_start_prob=0.5, expert_task=EXPERT_TASK
        )
        self.actions.valve_load.asset_name = BREAKER_NAME
        # The fist closes on the way in and stays closed.
        self.observations.teacher.expert_gripper.params["follow_reach"] = True
        # The lever flips in a fraction of a second at up to ~3 rad/s, ahead of
        # the hand; its speed is not the teacher's to set.
        self.rewards.valve_overspeed = None
        # The flick covers 1 rad in ~0.3 s, so a step or two off the expert's
        # timing is 0.3 rad: replaying the expert's own actions ended 8 of 32
        # episodes at the default 0.3 rad.
        self.terminations.expert_valve_lag.params["max_error"] = 0.6
        self.viewer.asset_name = BREAKER_NAME
        self.viewer.body_name = "lever_pivot"


@configclass
class AnymalCircuitBreakerRLEnvCfg_PLAY(AnymalCircuitBreakerRLEnvCfg):
    """Evaluation: nominal physics, bank starts (randomized like training), registration error kept."""

    def __post_init__(self):
        super().__post_init__()
        configure_play(self)
