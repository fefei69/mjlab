"""The trained Go1 velocity policy, driven by externally planned commands.

A `Tracker` is one mjlab environment plus the trained policy. Velocity commands
come from the caller, not from mjlab's random command sampler. Trackers built
from the same scene can exchange simulation state. The box-push demo uses that
to plan: it copies the real world's state into every world of a second tracker
and rolls candidate command sequences forward there.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

TASK_ID = "Mjlab-Velocity-Flat-Unitree-Go1"
# The baseline policy, kept in the repository so the demos run after a clone.
DEFAULT_CHECKPOINT = (
  Path(__file__).parent / "checkpoints" / "go1_flat_baseline" / "model_9999.pt"
)

# Command ranges the policy finished training on: forward, sideways, yaw rate.
CMD_MIN = (-1.5, -1.0, -0.7)
CMD_MAX = (2.0, 1.0, 0.7)
# Wider ranges the policy still tracks without falling, found by sweeping commands
# (sweep_commands.py). Above 3 m/s forward or 1.5 rad/s it starts to fall.
CMD_MIN_FAST = (-1.5, -1.0, -1.5)
CMD_MAX_FAST = (3.0, 1.0, 1.5)

MOTOR_FIELDS = ("actuator_forcerange", "actuator_gainprm", "actuator_biasprm")

Policy = Callable[[TensorDict], torch.Tensor]
Recorder = Callable[["Tracker"], dict[str, torch.Tensor]]
DrawFn = Callable[[DebugVisualizer], None]


def make_tracker_cfg(num_envs: int, task_id: str = TASK_ID) -> ManagerBasedRlEnvCfg:
  """The Go1 flat-terrain play config, set up for externally planned commands.

  `task_id` picks the policy's task, which sets what the policy observes (the
  fault-trained policies also have a fault event, switched off here; the demos
  apply their own faults).
  """
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = num_envs
  if "motor_fault" in cfg.events:
    cfg.events["motor_fault"].params["fault_prob"] = 0.0

  # Commands are written by the planner, so switch off everything that would
  # overwrite them: resampling, heading control, standing and forward-only envs.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.resampling_time_range = (1e9, 1e9)
  twist.debug_vis = False  # The demo videos show the command in their own panel.

  # Drop the startup randomization (friction, encoder bias, centre of mass) so
  # every world, real or planning, has the same nominal physics.
  cfg.events = {k: v for k, v in cfg.events.items() if v.mode != "startup"}
  # Start exactly at the configured initial pose.
  cfg.events["reset_base"].params["pose_range"] = {}
  # No terminations: a fallen robot stays fallen instead of being teleported back,
  # so the caller can detect the fall and rollouts stay continuous.
  cfg.terminations = {}
  # Keep sensor debug rays out of the videos.
  for sensor in cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      setattr(sensor, "debug_vis", False)  # noqa: B010  (not on the base class)
  return cfg


@dataclass
class SimState:
  """What one world needs to continue from where another one is."""

  qpos: torch.Tensor
  qvel: torch.Tensor
  qacc_warmstart: torch.Tensor
  ctrl: torch.Tensor
  action: torch.Tensor
  prev_action: torch.Tensor


class _Overlay:
  def __init__(self, draw: DrawFn) -> None:
    self._draw = draw

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    self._draw(visualizer)


def record_base(tracker: Tracker) -> dict[str, torch.Tensor]:
  """Default rollout recorder: base pose and how upright the base is."""
  return {"pose": tracker.pose(), "upright": tracker.upright()}


class Tracker:
  def __init__(
    self,
    cfg: ManagerBasedRlEnvCfg,
    device: str,
    policy: Policy | None = None,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    render: bool = False,
    task_id: str = TASK_ID,
  ) -> None:
    self.env = ManagerBasedRlEnv(
      cfg=cfg, device=device, render_mode="rgb_array" if render else None
    )
    agent_cfg = load_rl_cfg(task_id)
    self._vec_env = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
    if policy is None:
      runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
      runner = runner_cls(self._vec_env, asdict(agent_cfg), device=device)
      runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
      )
      policy = runner.get_inference_policy(device=device)
    assert policy is not None
    self.policy: Policy = policy

    # Give every world its own motor parameters, so faults can differ per robot.
    # Unexpanded fields share one copy: a write to one world changes them all.
    self.env.sim.expand_model_fields(MOTOR_FIELDS)

    self.robot: Entity = self.env.scene["robot"]
    self._trunk = self.robot.find_bodies("trunk")[0]
    twist = self.env.command_manager.get_term("twist")
    assert isinstance(twist, UniformVelocityCommand)
    self._twist = twist

  @property
  def num_envs(self) -> int:
    return self.env.num_envs

  @property
  def step_dt(self) -> float:
    """Duration of one policy step in seconds."""
    return self.env.step_dt

  @property
  def device(self) -> str:
    return self.env.device

  # State.

  def pose(self) -> torch.Tensor:
    """Base x, y, yaw in the world frame, shape (num_envs, 3)."""
    pos = self.robot.data.root_link_pos_w
    return torch.cat([pos[:, :2], self.robot.data.heading_w[:, None]], dim=-1)

  def upright(self) -> torch.Tensor:
    """1 when the base is level, 0 when it lies on its side, shape (num_envs,)."""
    return -self.robot.data.projected_gravity_b[:, 2]

  def get_state(self, env_idx: int = 0) -> SimState:
    data = self.env.sim.data
    actions = self.env.action_manager
    return SimState(
      qpos=data.qpos[env_idx].clone(),
      qvel=data.qvel[env_idx].clone(),
      qacc_warmstart=data.qacc_warmstart[env_idx].clone(),
      ctrl=data.ctrl[env_idx].clone(),
      action=actions.action[env_idx].clone(),
      prev_action=actions.prev_action[env_idx].clone(),
    )

  def set_state(self, state: SimState) -> None:
    """Put every world of this tracker into `state`."""
    data = self.env.sim.data
    data.qpos[:] = state.qpos
    data.qvel[:] = state.qvel
    data.qacc_warmstart[:] = state.qacc_warmstart
    data.ctrl[:] = state.ctrl
    actions = self.env.action_manager
    actions.action[:] = state.action
    actions.prev_action[:] = state.prev_action
    # Refresh kinematics and sensors, which the next observation reads.
    self.env.sim.forward()
    self.env.sim.sense()

  # Disturbance.

  def pull_trunk(self, force: torch.Tensor) -> None:
    """Pull on the trunk with a horizontal force, in newtons.

    `force` is (2,) or (num_envs, 2): forward and to the left, in the frame that
    turns with the robot's heading. MuJoCo holds the force fixed in the world
    until the next call, so call this before every step to keep it turning with
    the robot.
    """
    force = force.expand(self.num_envs, 2)
    yaw = self.robot.data.heading_w
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    world = torch.stack(
      [
        cos * force[:, 0] - sin * force[:, 1],
        sin * force[:, 0] + cos * force[:, 1],
        torch.zeros_like(yaw),
      ],
      dim=-1,
    )
    self.robot.write_external_wrench_to_sim(
      world[:, None], torch.zeros_like(world[:, None]), body_ids=self._trunk
    )

  def weaken_joints(self, pattern: str, torque: float, gain: float = 1.0) -> list[str]:
    """Simulate a motor fault: scale the joints' torque limit and PD gains.

    `pattern` is a regular expression on joint names, such as "FR_calf_joint" or
    "FR_.*" for a whole leg. `torque` and `gain` scale the healthy values, so 1
    is healthy and 0 is a dead motor. Applies to every world, and lasts until it
    is called again with 1. Returns the joints it changed.
    """
    model = self.env.sim.model
    default_force = self.env.sim.get_default_field("actuator_forcerange")
    default_gain = self.env.sim.get_default_field("actuator_gainprm")
    default_bias = self.env.sim.get_default_field("actuator_biasprm")
    changed = []
    for act in self.robot.actuators:
      for i, name in enumerate(act.target_names):
        if re.fullmatch(pattern, name) is None:
          continue
        ctrl = act.global_ctrl_ids[i]
        model.actuator_forcerange[:, ctrl] = default_force[ctrl] * torque
        model.actuator_gainprm[:, ctrl, 0] = default_gain[ctrl, 0] * gain
        model.actuator_biasprm[:, ctrl, 1:3] = default_bias[ctrl, 1:3] * gain
        changed.append(name)
    if not changed:
      raise ValueError(f"No joint matches {pattern!r}.")
    return changed

  def set_motor_strength(self, joint: torch.Tensor, strength: torch.Tensor) -> None:
    """Give each world one weak motor: scale joint `joint[i]`'s torque limit.

    `joint` (num_envs,) indexes `robot.joint_names`; `strength` (num_envs,) is the
    torque scale, 1 for healthy. Every other joint gets its healthy limit back.
    """
    ctrl = torch.full((self.robot.num_joints,), -1, dtype=torch.long)
    for act in self.robot.actuators:
      ctrl[act.target_ids.cpu()] = act.global_ctrl_ids.cpu().long()
    ctrl = ctrl.to(self.device)
    default = self.env.sim.get_default_field("actuator_forcerange")[ctrl]
    scale = torch.ones(self.num_envs, len(ctrl), device=self.device)
    scale[torch.arange(self.num_envs, device=self.device), joint] = strength
    self.env.sim.model.actuator_forcerange[:, ctrl] = default * scale[..., None]

  # Stepping.

  def _observe(self, command: torch.Tensor) -> TensorDict:
    self._twist.vel_command_b[:] = command
    # Force a recompute: mjlab caches the last observation, which predates the new
    # command and any state copy. Dropping the cache and computing without
    # updating history refreshes every term that has no history (the command
    # among them) and leaves history buffers as the last step left them. A policy
    # with history would also need those buffers copied in `set_state`.
    manager = self.env.observation_manager
    manager._obs_buffer = None  # noqa: SLF001  (no public way to drop the cache)
    obs = manager.compute(update_history=False)
    return TensorDict(obs, batch_size=[self.num_envs])

  @torch.no_grad()
  def step(self, command: torch.Tensor, hold: int = 1) -> None:
    """Track `command` for `hold` policy steps.

    `command` is (3,) or (num_envs, 3): forward, sideways and yaw-rate velocity.
    """
    obs = self._observe(command)
    for _ in range(hold):
      obs, _, _, _ = self._vec_env.step(self.policy(obs))

  @torch.no_grad()
  def rollout(
    self, commands: torch.Tensor, hold: int, record: Recorder = record_base
  ) -> dict[str, torch.Tensor]:
    """Roll every world forward under its own command sequence.

    `commands` is (num_envs, H, 3). Each command is held for `hold` policy steps.
    Returns what `record` reports after each of the H commands, stacked to
    (num_envs, H, ...).
    """
    horizon = commands.shape[1]
    frames: list[dict[str, torch.Tensor]] = []
    obs = self._observe(commands[:, 0])
    for h in range(horizon):
      for i in range(hold):
        action = self.policy(obs)
        # The observation returned by this step is the one the policy acts on
        # next, so it must already carry the next command.
        if i == hold - 1 and h + 1 < horizon:
          self._twist.vel_command_b[:] = commands[:, h + 1]
        obs, _, _, _ = self._vec_env.step(action)
      frames.append(record(self))
    return {k: torch.stack([f[k] for f in frames], dim=1) for k in frames[0]}

  # Rendering.

  def set_overlay(self, draw: DrawFn) -> None:
    """Draw extra shapes (track, goal, planned path) into every rendered frame."""
    visualizers: dict[str, Any] = self.env.manager_visualizers
    visualizers["mppi_overlay"] = _Overlay(draw)

  def render(self) -> np.ndarray:
    frame = self.env.render()
    assert frame is not None, "Tracker was created with render=False."
    return frame

  def close(self) -> None:
    self.env.close()
