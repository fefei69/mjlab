"""The trained Go1 velocity policy, driven by externally planned commands.

A `Tracker` is one mjlab environment plus the trained policy. Velocity commands
come from the caller, not from mjlab's random command sampler. Trackers built
from the same scene can exchange simulation state. The box-push demo uses that
to plan: it copies the real world's state into every world of a second tracker
and rolls candidate command sequences forward there.
"""

from __future__ import annotations

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
DEFAULT_CHECKPOINT = Path("logs/rsl_rl/go1_velocity/2026-11-10_17-04-19/model_9999.pt")

# Command ranges the policy finished training on: forward, sideways, yaw rate.
CMD_MIN = (-1.5, -1.0, -0.7)
CMD_MAX = (2.0, 1.0, 0.7)
# Wider ranges the policy still tracks without falling, found by sweeping commands
# (sweep_commands.py). Above 3 m/s forward or 1.5 rad/s it starts to fall.
CMD_MIN_FAST = (-1.5, -1.0, -1.5)
CMD_MAX_FAST = (3.0, 1.0, 1.5)

Policy = Callable[[TensorDict], torch.Tensor]
Recorder = Callable[["Tracker"], dict[str, torch.Tensor]]
DrawFn = Callable[[DebugVisualizer], None]


def make_tracker_cfg(num_envs: int) -> ManagerBasedRlEnvCfg:
  """The Go1 flat-terrain play config, set up for externally planned commands."""
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = num_envs

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
  ) -> None:
    self.env = ManagerBasedRlEnv(
      cfg=cfg, device=device, render_mode="rgb_array" if render else None
    )
    agent_cfg = load_rl_cfg(TASK_ID)
    self._vec_env = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
    if policy is None:
      runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
      runner = runner_cls(self._vec_env, asdict(agent_cfg), device=device)
      runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
      )
      policy = runner.get_inference_policy(device=device)
    assert policy is not None
    self.policy: Policy = policy

    self.robot: Entity = self.env.scene["robot"]
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

  # Stepping.

  def _observe(self, command: torch.Tensor) -> TensorDict:
    self._twist.vel_command_b[:] = command
    # Force a recompute. mjlab caches the last observation, which predates the new
    # command and any state copy. The policy has no observation history, so
    # recomputing has no side effects. A policy with history would also need its
    # history buffers copied in `set_state`.
    obs = self.env.observation_manager.compute(update_history=True)
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
