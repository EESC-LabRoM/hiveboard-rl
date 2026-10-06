# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode valve dynamics: how hard the lever (or handwheel) is to turn.

Each reset samples five parameters per environment (:data:`VALVE_DYNAMICS`):

* ``friction`` - Coulomb friction [N·m], MJWarp's ``dof_frictionloss``;
* ``damping`` - viscous friction [N·m·s/rad], MJWarp's ``dof_damping``;
* ``spring`` - a return spring toward closed [N·m/rad];
* ``breakaway`` - an extra seat torque [N·m] that holds the lever toward
  closed, full at the seat and fading to zero over
  :data:`BREAKAWAY_WIDTH_RAD` of opening (a ball valve's seat is tightest
  where it starts to open); applied by :class:`ValveLoadAction`;
* ``armature`` - added rotor inertia [kg·m²].

With probability ``stuck_prob`` an episode's valve is instead stuck closed:
the lever starts seated at closed and ``breakaway`` comes from
:data:`STUCK_BREAKAWAY_RANGE`, so it takes several N·m to unseat before it
turns like the others.

Friction, damping and armature are simulator joint properties written on
reset. The spring and seat torques depend on the angle and are applied by
:class:`ValveLoadAction` every env step (writing the valve drive's
stiffness has no effect on Newton). Newton runs the decimation loop itself, so
the torque is held for the step's physics substeps; the lever moves ~0.02 rad
in that time.

Expert trajectories are recorded under the sampled values and store them
(``valve_dynamics`` in the bank); RL episodes restore them with the
trajectory so the time-indexed references stay feasible.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import BaseArticulation
from isaaclab.managers import ActionTerm, ActionTermCfg, EventTermCfg, ManagerTermBase
from isaaclab.utils.configclass import configclass

from .mdp import _opening_sign, valve_task

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

#: Parameter order in :attr:`randomize_valve_dynamics.params` and the bank.
VALVE_DYNAMICS = ("friction", "damping", "spring", "breakaway", "armature")
#: Sampling ranges. The defaults (``friction`` 0.02 N·m, the rest 0) turn with
#: almost no effort; these reach a stiff valve (several N·m at open) that the
#: ~100 N force-limited grip (``env.py``) still holds.
VALVE_DYNAMICS_RANGES = {
    "friction": (0.05, 2.0),
    "damping": (0.0, 0.5),
    # Off: the policy lets go once the valve is open, and a released lever
    # must stay open. MJWarp's friction is soft, so any spring torque left at
    # open creeps the lever shut (~0.03 rad/s), whatever the friction.
    "spring": (0.0, 0.0),
    "breakaway": (0.0, 2.0),
    "armature": (0.001, 0.02),
}
#: Seat torque [N·m] of a valve stuck closed, and how often that happens.
STUCK_BREAKAWAY_RANGE = (2.0, 5.0)
STUCK_PROB = 0.25


def dynamics_scale(ranges: dict[str, tuple[float, float]], stuck_breakaway: tuple[float, float]) -> tuple[float, ...]:
    """:func:`valve_dynamics_obs` scale: each parameter's largest sampled value (1 for a parameter fixed at 0)."""
    return tuple(
        max(ranges[name][1], stuck_breakaway[1] if name == "breakaway" else 0.0) or 1.0 for name in VALVE_DYNAMICS
    )


#: The ball valve's :func:`dynamics_scale`.
VALVE_DYNAMICS_SCALE = dynamics_scale(VALVE_DYNAMICS_RANGES, STUCK_BREAKAWAY_RANGE)
#: Opening [rad] over which the seat torque fades from ``breakaway`` to 0.
BREAKAWAY_WIDTH_RAD = 0.15
#: Opening [rad] up to which the spring may not overcome friction. A lever
#: reset part-open (down to -0.4 rad) then only creeps until it is grasped
#: (MJWarp's friction is a soft constraint: 0.15 N·m held below friction
#: creeps at ~0.03 rad/s whatever the friction), instead of snapping shut
#: from under the expert's planned grasp.
SPRING_STATIC_RAD = 0.4


def closing_torque(
    angle: torch.Tensor,
    spring: torch.Tensor,
    breakaway: torch.Tensor,
    closed_rad: float = 0.0,
    opening_sign: float = -1.0,
) -> torch.Tensor:
    """Spring plus seat torque [N·m] at valve ``angle`` [rad], signed to turn the valve toward ``closed_rad``.

    ``opening_sign`` is the direction of opening (-1: negative rotation opens, as on the HiveBoard valves).
    """
    opening = (opening_sign * (angle - closed_rad)).clamp(min=0.0)
    magnitude = spring * opening + breakaway * (1.0 - opening / BREAKAWAY_WIDTH_RAD).clamp(min=0.0)
    return -opening_sign * magnitude


class randomize_valve_dynamics(ManagerTermBase):
    """Reset event: sample :data:`VALVE_DYNAMICS` per environment and write them to the valve.

    ``params`` holds every environment's current values, ``(num_envs, 5)``, and
    ``scale`` each one's largest possible value; :meth:`write` sets given
    values instead (the expert-bank reset uses it). The valve is the
    environment's ``valve_task`` (``mdp.ValveTaskCfg``).
    The spring is capped so that it cannot overcome friction up to
    :data:`SPRING_STATIC_RAD` of opening. A stuck valve's lever is moved to
    closed, so this event runs after the valve joint's reset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        task = valve_task(env)
        if not task.closed_end_stop:
            ranges, stuck = cfg.params["ranges"], cfg.params.get("stuck_prob", 0.0)
            if ranges["spring"][1] > 0.0 or ranges["breakaway"][1] > 0.0 or stuck > 0.0:
                raise ValueError(
                    f"'{task.asset_name}' has no end stop at closed: a spring or seat torque would spin it past"
                    " closed. Set the spring and breakaway ranges and stuck_prob to 0."
                )
        self.valve: BaseArticulation = env.scene[task.asset_name]
        self.joint_ids = self.valve.find_joints(task.joint_name)[0]
        self.closed_rad = task.closed_rad
        self.params = torch.zeros(env.num_envs, len(VALVE_DYNAMICS), device=env.device)
        self.scale = torch.tensor(
            dynamics_scale(cfg.params["ranges"], cfg.params.get("stuck_breakaway", STUCK_BREAKAWAY_RANGE)),
            device=env.device,
        )
        # The seat-torque action and the observation terms find the values here.
        env.valve_dynamics_term = self

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        ranges: dict[str, tuple[float, float]],
        stuck_prob: float = 0.0,
        stuck_breakaway: tuple[float, float] = STUCK_BREAKAWAY_RANGE,
    ) -> None:
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
        values = torch.empty(len(ids), len(VALVE_DYNAMICS), device=env.device)
        for i, name in enumerate(VALVE_DYNAMICS):
            values[:, i].uniform_(*ranges[name])
        spring_cap = values[:, 0] / SPRING_STATIC_RAD
        values[:, 2] = torch.minimum(values[:, 2], spring_cap)
        stuck = torch.rand(len(ids), device=env.device) < stuck_prob
        if stuck.any():
            values[stuck, 3] = torch.empty(int(stuck.sum()), device=env.device).uniform_(*stuck_breakaway)
            closed = torch.full((int(stuck.sum()), 1), self.closed_rad, device=env.device)
            self.valve.write_joint_position_to_sim_index(position=closed, joint_ids=self.joint_ids, env_ids=ids[stuck])
            self.valve.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(closed), joint_ids=self.joint_ids, env_ids=ids[stuck]
            )
        self.write(ids, values)

    def write(self, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        """Write ``values`` ``(len(env_ids), 5)`` in :data:`VALVE_DYNAMICS` order to the simulator."""
        self.params[env_ids] = values
        joint = self.joint_ids
        self.valve.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=values[:, 0:1].contiguous(),
            joint_viscous_friction_coeff=values[:, 1:2].contiguous(),
            joint_ids=joint,
            env_ids=env_ids,
        )
        self.valve.write_joint_armature_to_sim_index(armature=values[:, 4:5].contiguous(), joint_ids=joint, env_ids=env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass


def randomize_valve_dynamics_cfg(
    ranges: dict[str, tuple[float, float]] | None = None,
    stuck_prob: float = STUCK_PROB,
    stuck_breakaway: tuple[float, float] = STUCK_BREAKAWAY_RANGE,
) -> EventTermCfg:
    """Reset event sampling the valve dynamics from ``ranges`` (default :data:`VALVE_DYNAMICS_RANGES`)."""
    return EventTermCfg(
        func=randomize_valve_dynamics,
        mode="reset",
        params={"ranges": dict(ranges or VALVE_DYNAMICS_RANGES), "stuck_prob": stuck_prob, "stuck_breakaway": stuck_breakaway},
    )


def valve_dynamics_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """The episode's valve dynamics, each divided by its largest sampled value, ``(N, 5)`` (privileged)."""
    term = env.valve_dynamics_term
    return term.params / term.scale


class ValveLoadAction(ActionTerm):
    """Applies the valve's spring and seat torque (:func:`closing_torque`); takes no actions.

    An action term because action terms are applied every env step before the
    physics runs; the policy's action vector does not change.
    """

    cfg: ValveLoadActionCfg

    def __init__(self, cfg: ValveLoadActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        task = valve_task(env)
        if task.asset_name != cfg.asset_name:
            raise ValueError(f"ValveLoadActionCfg.asset_name '{cfg.asset_name}' is not the valve '{task.asset_name}'.")
        self._joint_ids = self._asset.find_joints(task.joint_name)[0]
        self._closed_rad = task.closed_rad
        self._opening_sign = _opening_sign(env)
        self._empty = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        term = getattr(self._env, "valve_dynamics_term", None)
        if term is None:
            return
        angle = self._asset.data.joint_pos.torch[:, self._joint_ids[0]]
        torque = closing_torque(angle, term.params[:, 2], term.params[:, 3], self._closed_rad, self._opening_sign)
        self._asset.set_joint_effort_target_index(target=torque[:, None], joint_ids=self._joint_ids)


@configclass
class ValveLoadActionCfg(ActionTermCfg):
    """Configuration for :class:`ValveLoadAction`."""

    class_type: type = ValveLoadAction
    asset_name: str = "ball_valve"
