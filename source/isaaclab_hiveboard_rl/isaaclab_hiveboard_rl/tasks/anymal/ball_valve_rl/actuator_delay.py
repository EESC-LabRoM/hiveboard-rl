# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode command latency on the arm's joint drives.

On the robot the policy's joint targets reach the DynaArm's drives some tens
of milliseconds late (policy -> ROS 2 -> EtherCAT -> drive). The arm's PD
actuators become :class:`~isaaclab.actuators.DelayedPDActuatorCfg`, which on
Newton runs as a native ``Delay`` inside the physics step's CUDA graph: the
drive tracks the target it received ``delay_steps`` physics steps ago.

Newton applies ``max_delay`` to every environment (``min_delay`` is ignored),
so :class:`randomize_actuator_delay` writes each environment's own lag into
the delay's per-DOF ``delay_steps`` at reset, drawn from a range in seconds.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg
from isaaclab.actuators.newton.adapter import read_group_parameter, write_group_parameter
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

#: Arm command latency [s] sampled per episode.
ARM_DELAY_RANGE_S = (0.0, 0.04)
#: The arm's actuator groups.
ARM_ACTUATOR_GROUPS = ("dynaarm_shoulder", "dynaarm_elbow", "dynaarm_forearm", "dynaarm_wrist_flex", "dynaarm_wrist_rot")


def delayed_actuators(robot: ArticulationCfg, groups: tuple[str, ...], max_delay_s: float, physics_dt: float):
    """``robot`` with the PD actuator ``groups`` delayed by up to ``max_delay_s`` (sized in physics steps)."""
    max_delay = math.ceil(round(max_delay_s / physics_dt, 6))
    actuators = dict(robot.actuators)
    for name in groups:
        cfg = actuators[name]
        if not isinstance(cfg, IdealPDActuatorCfg):
            raise TypeError(f"Actuator group '{name}' is a {type(cfg).__name__}, not an explicit PD actuator.")
        fields = cfg.to_dict()
        for key in ("class_type", "min_delay", "max_delay"):
            fields.pop(key, None)
        actuators[name] = DelayedPDActuatorCfg(**fields, min_delay=0, max_delay=max_delay)
    return robot.replace(actuators=actuators)


def randomize_actuator_delay(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    delay_range_s: tuple[float, float],
    groups: tuple[str, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset event: give each environment one command lag for all ``groups``, uniform in ``delay_range_s``.

    The lag is rounded to whole physics steps and capped at the groups' ``max_delay``.
    """
    asset = env.scene[asset_cfg.name]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    dt = env.physics_dt
    low = math.floor(round(delay_range_s[0] / dt, 6))
    high = math.ceil(round(delay_range_s[1] / dt, 6))
    steps = torch.randint(low, high + 1, (len(ids),), device=env.device)
    for name in groups:
        num_joints = read_group_parameter(asset.actuators, name, "delay", "delay_steps").shape[1]
        values = steps.clamp(max=asset.cfg.actuators[name].max_delay)[:, None].expand(-1, num_joints)
        write_group_parameter(asset.actuators, name, "delay", "delay_steps", values, env_ids=ids)


def randomize_actuator_delay_cfg(
    delay_range_s: tuple[float, float] = ARM_DELAY_RANGE_S, groups: tuple[str, ...] = ARM_ACTUATOR_GROUPS
) -> EventTermCfg:
    """Reset event drawing each episode's arm command latency from ``delay_range_s`` [s]."""
    return EventTermCfg(
        func=randomize_actuator_delay, mode="reset", params={"delay_range_s": delay_range_s, "groups": groups}
    )
