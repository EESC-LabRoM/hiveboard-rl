# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode variations of the scripted cuRobo expert, for a diverse expert bank.

A reset event (:class:`sample_expert_diversity`) samples, per environment,
the values in :data:`EXPERT_DIVERSITY` and writes them into the
``pose_command`` handlers before they plan:

* ``turn_rate`` - the valve arc's angular speed [rad/s] (a pushing expert's
  hand speed along the push [m/s]);
* ``reach_scale`` - a factor on the approach and grasp segments' speeds;
* ``grasp_shift`` - where along the lever the TCP grasps [m];
* ``grasp_roll`` - the grasp's rotation about the lever's long axis [rad],
  i.e. the side the hand comes from around the bar;
* ``turn_overshoot`` - not sampled: how far past 90 deg the arc runs [rad].
  Under load the lever lags the hand by ~0.02 rad per N·m (mostly the arm's
  compliance), so the expert turns further the harder the valve
  (:func:`turn_overshoot`) and the valve's end stop holds the lever at open.
  Without it the expert opened 38% of the 3-4 N·m valves; with it 85%.

With ``wrist_branch_nearest`` the cuRobo reach and retreat plan to the IK
branch nearest the arm's current joints: the usual one (wrist flexion
negative) or its wrist-flipped twin with the same hand pose (forearm turned by
pi, wrist flexion negated, wrist rotation by pi; the DynaArm has no other
branch within its limits). Which one the expert takes then follows from the
start posture (some start flipped, see ``mdp.reset_joints_from_postures``),
which the student observes; a randomly drawn branch would not be learnable.

The grasp offset is applied in the grasp goal's frame: the shift along
``grasp_axes[0]`` and the roll about ``grasp_axes[1]``. On the ball valve both
are the goal's ``y`` axis, which runs along the lever (the goal sits 6 cm out
along it from the valve axis). The approach goals before the grasp get the
same offset expressed from their own frame, so the final approach keeps its
direction relative to the grasp.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

#: Order of :attr:`sample_expert_diversity.values` and the bank's ``expert_diversity``.
EXPERT_DIVERSITY = ("turn_rate", "reach_scale", "grasp_shift", "grasp_roll", "turn_overshoot")
#: Sampling ranges. The authored expert turns at 0.3 rad/s with ``reach_scale``
#: 1 and no grasp offset.
EXPERT_DIVERSITY_RANGES = {
    "turn_rate": (0.25, 0.5),
    "reach_scale": (0.75, 1.3),
    "grasp_shift": (-0.015, 0.015),
    "grasp_roll": (-0.25, 0.25),
}
#: Arc overshoot [rad] for a free valve, and per N·m of torque at open.
OVERSHOOT_RAD = 0.03
OVERSHOOT_RAD_PER_NM = 0.05
#: Lever axis in the grasp goal's frame.
LEVER_AXIS = (0.0, 1.0, 0.0)
#: The ball valve's (shift, roll) axes of the grasp offset, in the grasp goal's frame: both along the lever.
LEVER_GRASP_AXES = (LEVER_AXIS, LEVER_AXIS)
#: The DynaArm's (roll, pitch, roll) wrist, for the wrist flip.
WRIST_FLIP_JOINTS = ("dynaarm_forearm_rotation", "dynaarm_wrist_flexion", "dynaarm_wrist_rotation")


def turn_overshoot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    overshoot: tuple[float, float] = (OVERSHOOT_RAD, OVERSHOOT_RAD_PER_NM),
) -> torch.Tensor:
    """Arc overshoot [rad] for ``env_ids`` from the torque their valve needs at open (``valve_dynamics.py``).

    ``overshoot`` is (free valve [rad], per N·m at open [rad/(N·m)]).
    """
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import mdp

    base, per_nm = overshoot
    term = getattr(env, "valve_dynamics_term", None)
    if term is None:
        return torch.full((len(env_ids),), base, device=env.device)
    task = mdp.valve_task(env)
    friction, _, spring, _, _ = term.params[env_ids].unbind(dim=-1)
    torque = friction + spring * abs(task.open_rad - task.closed_rad)
    return base + per_nm * torque


class sample_expert_diversity(ManagerTermBase):
    """Reset event: sample :data:`EXPERT_DIVERSITY` per environment and apply it to the scripted expert.

    The ``pose_command`` handlers are found on the first call (the command
    manager is built after the event manager): the rotate segments (one per
    grasp when the expert regrasps), the go-to segments before the first one
    (the reach), the last of those (the first grasp) and the cuRobo go-to
    segments after the last rotate (the retreat). Only the first grasp is
    offset; a regrasping expert's later grasps keep their authored goals.
    ``values`` holds every environment's current sample, ``(num_envs, 5)``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.values = torch.zeros(env.num_envs, len(EXPERT_DIVERSITY), device=env.device)
        self._rotate = None
        env.expert_diversity_term = self

    def _resolve(self, env: ManagerBasedEnv, command_name: str) -> None:
        from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
            _CuroboPlannedGoToFrameHandler,
            _GoToFrameHandler,
            _RotateFrameHandler,
        )

        handlers = env.command_manager.get_term(command_name)._command_handlers
        rotates = [i for i, h in enumerate(handlers) if isinstance(h, _RotateFrameHandler)]
        # An expert that pushes (a circuit breaker's lever) has no rotate: its
        # actuation is the go-to segments tagged "actuate", and turn_rate is
        # their hand speed [m/s].
        pushes = [] if rotates else [
            i for i, h in enumerate(handlers) if isinstance(h, _GoToFrameHandler) and h.cfg.phase == "actuate"
        ]
        if not rotates and not pushes:
            raise ValueError("Expert diversity needs a rotate segment or a go-to segment tagged 'actuate'.")
        actuate = rotates or pushes
        self._rotate = [handlers[i] for i in rotates]
        self._push = [handlers[i] for i in pushes]
        self._reach = [h for h in handlers[: actuate[0]] if isinstance(h, _GoToFrameHandler)]
        self._grasp = self._reach[-1]
        self._retreat = [h for h in handlers[actuate[-1] + 1 :] if isinstance(h, _CuroboPlannedGoToFrameHandler)]
        # Each earlier goal in the grasp goal's frame, K, from the frames'
        # authored offsets. Reading the frame transformer here would latch its
        # pre-reset poses for this step, and the command would then plan to
        # where the lever was in the previous episode. A push's goals move with
        # the grasp (its start) too, so the push keeps its direction.
        grasp = self._goal_in_body(self._grasp, env.device)
        self._moved_with_grasp = self._reach[:-1] + self._push
        self._approach_in_grasp = [
            math_utils.subtract_frame_transforms(*grasp, *self._goal_in_body(h, env.device))
            for h in self._moved_with_grasp
        ]

    @staticmethod
    def _goal_in_body(handler, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A go-to handler's goal (frame offset, then its target offset) in the frame's body, ``(1, 3)``/``(1, 4)`` xyzw."""
        cfg = handler._frame.cfg
        frame = next(f for f in cfg.target_frames if f.name == handler.cfg.target_frame_name)
        if frame.prim_path != cfg.target_frames[0].prim_path:
            raise ValueError("Expert diversity needs the reach goals to be offsets of one body.")
        return math_utils.combine_frame_transforms(
            torch.tensor([frame.offset.pos], device=device),
            torch.tensor([frame.offset.rot], device=device),
            torch.tensor([handler.cfg.target_offset_pos], device=device),
            torch.tensor([handler.cfg.target_offset_rot], device=device),
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        ranges: dict[str, tuple[float, float]],
        wrist_branch_nearest: bool = False,
        command_name: str = "pose_command",
        grasp_axes: tuple[tuple[float, float, float], tuple[float, float, float]] = LEVER_GRASP_AXES,
        overshoot: tuple[float, float] = (OVERSHOOT_RAD, OVERSHOOT_RAD_PER_NM),
    ) -> None:
        if self._rotate is None:
            self._resolve(env, command_name)
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
        n = len(ids)
        values = torch.empty(n, len(EXPERT_DIVERSITY), device=env.device)
        for i, name in enumerate(EXPERT_DIVERSITY[:-1]):
            values[:, i].uniform_(*ranges[name])
        # The valve dynamics event runs first.
        values[:, -1] = turn_overshoot(env, ids, overshoot)
        self.values[ids] = values
        turn_rate, reach_scale, shift, roll, overshoot = values.unbind(dim=-1)

        for rotate in self._rotate:
            rotate.speed_scale[ids] = turn_rate / rotate.cfg.angular_velocity
            rotate.angle_extra_rad[ids] = overshoot
        for push in self._push:
            push.speed_scale[ids] = turn_rate / push.cfg.velocity
        for handler in self._reach:
            handler.speed_scale[ids] = reach_scale
        if wrist_branch_nearest:
            from isaaclab_hiveboard.mdp.commands.sequential_pose_command import WRIST_BRANCH_NEAREST

            for handler in self._reach + self._retreat:
                if getattr(handler.cfg, "wrist_flip_joints", None) is None:
                    raise ValueError("wrist_branch_nearest needs the cuRobo go-to commands' wrist_flip_joints.")
                handler.wrist_branch[ids] = WRIST_BRANCH_NEAREST
                handler.forearm_winding[ids] = 0.0

        shift_axis = torch.tensor(grasp_axes[0], device=env.device).expand(n, 3)
        roll_axis = torch.tensor(grasp_axes[1], device=env.device).expand(n, 3)
        grasp_pos = shift_axis * shift[:, None]
        grasp_quat = math_utils.quat_from_angle_axis(roll, roll_axis)
        self._grasp.env_offset_pos[ids] = grasp_pos
        self._grasp.env_offset_rot[ids] = grasp_quat
        # Earlier goals move rigidly with the grasp: with K the earlier goal in
        # the grasp goal's frame, its offset is K^-1 * E * K.
        for handler, (k_pos, k_quat) in zip(self._moved_with_grasp, self._approach_in_grasp):
            k_pos, k_quat = k_pos.expand(n, 3), k_quat.expand(n, 4)
            moved = math_utils.combine_frame_transforms(grasp_pos, grasp_quat, k_pos, k_quat)
            handler.env_offset_pos[ids], handler.env_offset_rot[ids] = math_utils.subtract_frame_transforms(
                k_pos, k_quat, *moved
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass


def sample_expert_diversity_cfg(
    ranges: dict[str, tuple[float, float]] | None = None,
    wrist_branch_nearest: bool = False,
    grasp_axes: tuple[tuple[float, float, float], tuple[float, float, float]] = LEVER_GRASP_AXES,
    overshoot: tuple[float, float] = (OVERSHOOT_RAD, OVERSHOOT_RAD_PER_NM),
) -> EventTermCfg:
    """Reset event sampling the expert's variations from ``ranges`` (default :data:`EXPERT_DIVERSITY_RANGES`).

    ``grasp_axes`` are the grasp offset's (shift, roll) axes in the grasp goal's frame; ``overshoot`` is
    the arc's overshoot for a free valve [rad] and per N·m at open (:func:`turn_overshoot`).
    """
    return EventTermCfg(
        func=sample_expert_diversity,
        mode="reset",
        params={
            "ranges": dict(ranges or EXPERT_DIVERSITY_RANGES),
            "wrist_branch_nearest": wrist_branch_nearest,
            "grasp_axes": grasp_axes,
            "overshoot": overshoot,
        },
    )
