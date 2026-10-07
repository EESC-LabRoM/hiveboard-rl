"""RL valve tasks: the ball-valve, small-valve, M30-thread and circuit-breaker configurations and the valve terms' conventions.

Config-only checks (no simulation)::

    uv run --with pytest python -m pytest scripts/test_valve_rl_tasks.py
"""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import gymnasium as gym
import isaaclab_hiveboard_rl  # noqa: F401
import pytest
import torch
from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl import valve_dynamics

_TASKS = {
    "Isaac-HiveBoard-Anymal-BallValve-RL-v0": ("ball_valve", "Isaac-HiveBoard-Anymal-BallValve-v0"),
    "Isaac-HiveBoard-Anymal-SmallValve-RL-v0": ("small_valve", "Isaac-HiveBoard-Anymal-SmallValve-v0"),
    "Isaac-HiveBoard-Anymal-M30Thread-RL-v0": ("thread", "Isaac-HiveBoard-Anymal-M30Thread-v0"),
    "Isaac-HiveBoard-Anymal-CircuitBreaker-RL-v0": ("circuit_breaker", "Isaac-HiveBoard-Anymal-CircuitBreaker-v0"),
}


def _cfg(task: str):
    module, name = gym.spec(task).kwargs["env_cfg_entry_point"].split(":")
    return importlib.import_module(module), getattr(importlib.import_module(module), name)()


@pytest.mark.parametrize("task", list(_TASKS))
@pytest.mark.parametrize("play", [False, True])
def test_every_valve_reference_names_the_task_valve(task, play):
    module, cfg = _cfg(task.replace("-RL-v0", "-RL-Play-v0") if play else task)
    asset, expert_task = _TASKS[task]
    assert cfg.valve_task.asset_name == asset
    assert getattr(cfg.scene, asset) is not None
    assert cfg.events.valve_gravcomp.params["asset_cfg"].name == asset
    assert cfg.events.valve_physics_material is None or cfg.events.valve_physics_material.params["asset_cfg"].name == asset
    assert cfg.actions.valve_load.asset_name == asset
    assert cfg.viewer.asset_name == asset
    assert cfg.events.reset_from_bank.params["expert_task"] == expert_task == module.EXPERT_TASK
    # The grasp frame exists in the scene and sits on a body of the valve.
    frames = {f.name: f for f in cfg.scene.target_frame.target_frames}
    assert cfg.valve_task.grasp_frame in frames
    assert frames[cfg.valve_task.grasp_frame].prim_path.startswith(getattr(cfg.scene, asset).prim_path + "/")
    # The bank builder's settings.
    for name in ("VALVE_POSE_RANGE", "VALVE_ANGLE_RANGE", "ARM_POSTURES", "VALVE_DYNAMICS_RANGES",
                 "EXPERT_DIVERSITY_RANGES", "EXPERT_GRASP_AXES", "EXPERT_OVERSHOOT", "EXPERT_GRIP_S", "BANK_PREFIX"):
        assert hasattr(module, name), name
    assert cfg.events.reset_from_bank.params["mid_start_prob"] == (0.0 if play else 0.5)


def test_ball_valve_dynamics_unchanged():
    _, cfg = _cfg("Isaac-HiveBoard-Anymal-BallValve-RL-v0")
    params = cfg.events.valve_dynamics.params
    assert params["ranges"] == valve_dynamics.VALVE_DYNAMICS_RANGES
    scale = valve_dynamics.dynamics_scale(params["ranges"], params["stuck_breakaway"])
    assert scale == (2.0, 0.5, 1.0, 5.0, 0.02)


def test_integrated_target_cannot_wind_up_against_blocked_joint():
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.mdp import IntegratedJointPositionAction

    action = object.__new__(IntegratedJointPositionAction)
    measured = torch.tensor([[0.0, 0.95]])
    action.cfg = SimpleNamespace(smoothing=1.0, max_position_error=0.1)
    action._joint_ids = [0, 1]
    action._target = measured.clone()
    action._filtered = torch.zeros_like(measured)
    action._processed_actions = torch.full_like(measured, 0.1)
    action._lower = torch.full_like(measured, -1.0)
    action._upper = torch.full_like(measured, 1.0)
    writes = []
    action._asset = SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=measured)),
        set_joint_position_target_index=lambda **kwargs: writes.append(kwargs['target'].clone()))
    for _ in range(100):
        action._fresh = True
        action.apply_actions()
        action.apply_actions()  # Physics substeps must not integrate twice.
    assert torch.allclose(action._target, torch.tensor([[0.1, 1.0]]))
    assert torch.equal(writes[-1], writes[-2])
    # Reverse direction immediately: there must be no hidden accumulated target.
    action._processed_actions.fill_(-0.1)
    action._fresh = True
    action.apply_actions()
    assert torch.allclose(action._target, torch.tensor([[0.0, 0.9]]))


def test_error_limit_is_enabled_only_for_ball_valve():
    for task in _TASKS:
        _, cfg = _cfg(task)
        assert cfg.actions.arm_action.max_position_error == (0.1 if 'BallValve' in task else None)


def test_expert_velocity_uses_idle_shift_and_stops_at_end():
    from isaaclab_hiveboard_rl.tasks.anymal.ball_valve_rl.expert_bank import ExpertBank

    bank = object.__new__(ExpertBank)
    bank.dt = 0.05
    bank.grasp_step = torch.tensor([2])
    bank.idle_shift = torch.tensor([2])
    bank.reach = torch.tensor([[[0.0], [0.05], [0.1]]])
    # Two idle samples removed from the recorded reach; turn begins at bank step 4.
    bank.q = torch.tensor([[[0.0], [0.0], [0.0], [0.05], [0.1], [0.15], [0.2]]])
    for step in (0, 1, 2, 3):
        assert bank.velocity_reference(torch.tensor([0]), torch.tensor([step])).item() == pytest.approx(1.0)
    for step in (4, 5, 100):
        assert bank.velocity_reference(torch.tensor([0]), torch.tensor([step])).item() == 0.0


@pytest.mark.parametrize("closed, opening_sign", [(0.0, -1.0), (0.0, 1.0), (0.3, -1.0)])
def test_closing_torque_turns_toward_closed(closed, opening_sign):
    spring, breakaway = torch.tensor([1.0]), torch.tensor([2.0])
    seated = valve_dynamics.closing_torque(torch.tensor([closed]), spring, breakaway, closed, opening_sign)
    part_open = torch.tensor([closed + opening_sign * 0.5])
    opened = valve_dynamics.closing_torque(part_open, spring, breakaway, closed, opening_sign)
    # Full seat torque at closed, spring only past the seat, both pushing against opening.
    assert seated.item() == pytest.approx(-opening_sign * 2.0)
    assert opened.item() == pytest.approx(-opening_sign * 0.5)
    # Past closed (the wrong way) only the seat torque, still toward closed.
    beyond = valve_dynamics.closing_torque(torch.tensor([closed - opening_sign * 0.2]), spring, breakaway, closed, opening_sign)
    assert beyond.item() == pytest.approx(-opening_sign * 2.0)


def test_small_valve_turns_a_quarter_turn_open_negative():
    _, cfg = _cfg("Isaac-HiveBoard-Anymal-SmallValve-RL-v0")
    assert cfg.valve_task.open_rad == pytest.approx(-math.pi / 2) and cfg.valve_task.closed_rad == 0.0


def test_m30_thread_runs_the_nut_down_two_turns():
    module, cfg = _cfg("Isaac-HiveBoard-Anymal-M30Thread-RL-v0")
    task = cfg.valve_task
    assert task.open_rad == pytest.approx(-4.0 * math.pi) and task.closed_rad == 0.0
    # The nut's travel is 7 mm at the start and 0 when seated.
    assert task.coupled_offset == pytest.approx(0.007)
    assert task.coupled_offset + task.coupled_ratio * task.open_rad == pytest.approx(0.0, abs=1e-9)
    assert task.hold_by_contact and not task.closed_end_stop
    # The thread's mimic constraint moves the nut, not a drive anchored at the start.
    assert cfg.scene.thread.actuators["advance"].stiffness == 0.0
    # Seated within the protocol's 1 mm.
    assert module.SUCCESS_TOLERANCE_RAD * task.coupled_ratio == pytest.approx(0.001)
    assert cfg.observations.teacher.expert_gripper.params["follow_release"]


def test_circuit_breaker_pushes_the_lever_up_from_its_down_stop():
    module, cfg = _cfg("Isaac-HiveBoard-Anymal-CircuitBreaker-RL-v0")
    task = cfg.valve_task
    assert task.push and task.closed_end_stop and not task.hold_by_contact
    # Down (+30 deg) to up (-30 deg), the joint's limits.
    assert task.closed_rad == pytest.approx(math.pi / 6) and task.open_rad == pytest.approx(-math.pi / 6)
    assert cfg.scene.circuit_breaker.init_state.joint_pos["RevoluteJoint"] == pytest.approx(task.closed_rad)
    # The fist closes during the reach, and the flick's speed is not penalized.
    assert cfg.observations.teacher.expert_gripper.params["follow_reach"]
    assert cfg.rewards.valve_overspeed is None
    # No spring: a released lever stays up.
    assert module.VALVE_DYNAMICS_RANGES["spring"] == (0.0, 0.0)
    # The end stops are damped, so a flicked lever does not bounce back down.
    from isaaclab_hiveboard.tasks.scenes.circuit_breaker import BREAKER_LIMIT_KD

    assert cfg.events.breaker_end_stops.params["kd"] == BREAKER_LIMIT_KD


def test_circuit_breaker_expert_pushes_without_idling():
    from isaaclab_hiveboard.mdp.commands.sequential_pose_command import GripperCommand

    module, _ = _cfg("Isaac-HiveBoard-Anymal-CircuitBreaker-RL-v0")
    m, c = gym.spec(module.EXPERT_TASK).kwargs["env_cfg_entry_point"].split(":")
    expert = getattr(importlib.import_module(m), c)()
    module.configure_expert(expert)
    commands = expert.commands.pose_command.commands
    # The only gripper segment left is the press that holds the lever up; the fist stays closed throughout.
    assert [cmd.phase for cmd in commands] == ["approach", "engage", "actuate", "actuate", "retreat", "retreat"]
    assert [type(cmd) is GripperCommand for cmd in commands] == [False, False, False, True, False, False]
    assert all(not getattr(cmd, "gripper_open", False) and not getattr(cmd, "open_gripper", False) for cmd in commands)
    # The push ends on the lever's angle near its up stop.
    push = commands[2]
    assert push.done_when_joint[:2] == ("circuit_breaker", "RevoluteJoint")
    assert push.done_when_joint[3] == pytest.approx(module.LEVER_UP_RAD + module.PUSH_DONE_RAD)
    assert expert.scene.circuit_breaker.init_state.joint_pos["RevoluteJoint"] == pytest.approx(module.LEVER_DOWN_RAD)
