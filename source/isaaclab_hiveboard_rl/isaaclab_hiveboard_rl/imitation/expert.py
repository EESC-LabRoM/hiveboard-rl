# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The scripted expert that labels states for behaviour cloning and DAgger."""

from __future__ import annotations

import torch


class ScriptedExpert:
    """Turn the task's sequential pose command into an action vector.

    The HiveBoard tasks are already solved by :class:`SequentialPoseCommand`,
    whose waypoint handlers advance on *measured* TCP error rather than on a
    step counter. That property is what makes this class usable as a DAgger
    expert and not merely as a demonstration replayer: query it at a state the
    learner drifted into and the command term, having never seen its waypoint
    reached, keeps pointing at the pose the learner failed to achieve.

    One caveat worth knowing before trusting the labels blindly: the rotating
    handlers (``RotateFrame``, and the screw coupling) integrate their target
    against wall-clock ``step_dt``, not against measured progress. If a learner
    stalls mid-turn, those targets keep advancing and the label degrades into
    "catch up to where the bulb should be by now". The stall watchdog on the
    command term reports when this happens.

    Args:
        env: The environment (wrapped or unwrapped).
        command_name: Name of the sequential pose command term.
    """

    #: Action-term layouts this expert can fill in pose mode, keyed by
    #: ``(term names, term dims)``. There the command packs as
    #: ``[gripper, pos(3), quat(4)]`` and a pose arm action wants ``[pos, quat]``.
    _POSE_LAYOUTS = {
        (("arm_action", "gripper_action"), (7, 1)),
        (("gripper_action", "arm_action"), (1, 7)),
    }

    def __init__(self, env, command_name: str = "pose_command"):
        self._env = env.unwrapped
        if command_name not in self._env.command_manager.active_terms:
            raise ValueError(
                f"Task has no '{command_name}' command term; the scripted expert needs one. "
                f"Active terms: {list(self._env.command_manager.active_terms)}."
            )
        self._command = self._env.command_manager.get_term(command_name)

        # Terms without actions (e.g. one applying a valve's load) do not
        # change the action layout.
        layout = [
            (term, int(dim))
            for term, dim in zip(self._env.action_manager.active_terms, self._env.action_manager.action_term_dim)
            if int(dim) > 0
        ]
        terms = tuple(term for term, _ in layout)
        dims = tuple(dim for _, dim in layout)
        self._terms = terms
        self._action_dim = sum(dims)

        # A command term configured with ``output_joint_positions`` publishes
        # cuRobo's joint waypoints as ``[q_arm, gripper]``, which is already the
        # action vector a JointPositionAction + binary gripper consumes. Pose
        # tasks instead publish ``[gripper, pos, quat]`` and need reordering.
        self._joint_mode = bool(getattr(self._command.cfg, "output_joint_positions", False))
        if self._joint_mode:
            if terms != ("arm_action", "gripper_action"):
                raise ValueError(
                    f"Joint-position command with unsupported action terms {terms}. Expected "
                    "('arm_action', 'gripper_action') so the command's [q_arm, gripper] layout "
                    "lines up with the action vector."
                )
            if dims[-1] != 1:
                raise ValueError(
                    f"Expected a 1-D gripper action to match the command's trailing gripper bit, got {dims[-1]}."
                )
        elif (terms, dims) not in self._POSE_LAYOUTS:
            raise ValueError(
                f"Unsupported action layout terms={terms}, dims={dims}. In pose mode the scripted "
                "expert emits an absolute TCP pose plus a gripper bit, so it needs a 7-D pose arm "
                "action and a 1-D gripper action. Tasks using relative actions need their own expert."
            )

    @property
    def action_dim(self) -> int:
        """Width of the action vector this expert produces."""
        return self._action_dim

    def compute(self) -> torch.Tensor:
        """Return the expert action for the environment's current state.

        Returns:
            Actions of shape ``(num_envs, action_dim)`` ordered to match the
            task's action terms.

        Raises:
            ValueError: If the command width does not match the action width,
                which otherwise reaches the action manager as a shape error with
                no indication of which side is wrong.
        """
        command = self._command.command
        width = int(command.shape[-1])

        if self._joint_mode:
            if width != self._action_dim:
                raise ValueError(
                    f"Joint command is {width}-D but the action space is {self._action_dim}-D. The "
                    "command's robot_joint_names must match the arm action's joint_names."
                )
            return command

        arm_action = command[:, 1:8]
        gripper_action = command[:, 0:1]
        if self._terms[0] == "arm_action":
            return torch.cat((arm_action, gripper_action), dim=-1)
        return torch.cat((gripper_action, arm_action), dim=-1)

    def is_done(self) -> torch.Tensor:
        """Per-environment flag for whether the scripted sequence finished."""
        return self._command.is_done()
