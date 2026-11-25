"""The jump command: run, then take off with a commanded speed, then stop.

The command is ``(vx, vy, wz, vz)``: the planar velocity and turn rate to run
at, and a vertical speed. ``vz`` is 0 while running; above 0 it means take off
now with that vertical speed, and it stays set until touch-down. In training
a scripted coach plays the planner: it steers along the course, triggers the
jump just before the edge, and picks the take-off speed so that a ballistic
flight under the episode's true gravity lands on the box. After touch-down it
commands a stop. With ``coach=False`` a planner sets the run-up speed, the
trigger point and the take-off speed instead, through `set_plan`.

The term also tracks each episode's phase (run, take-off, flight, landed), the
velocity at lift-off, and whether the robot came to rest on the landing box.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.box_jump.course import BoxJumpCourseCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

__all__ = ["JumpCommand", "JumpCommandCfg", "JumpPhase"]

NOMINAL_GRAVITY = 9.81


class JumpPhase(IntEnum):
  RUN = 0
  TAKEOFF = 1
  """The jump is triggered; the feet have not all left the ground yet."""
  FLIGHT = 2
  LANDED = 3


class JumpCommand(CommandTerm):
  cfg: JumpCommandCfg

  def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    n, device = self.num_envs, self.device
    self.cmd = torch.zeros(n, 4, device=device)
    self.phase = torch.zeros(n, dtype=torch.long, device=device)
    self.run_speed = torch.zeros(n, device=device)
    self.landing_offset = torch.zeros(n, device=device)
    self.takeoff_cmd = torch.zeros(n, 2, device=device)
    """Commanded forward and vertical speed at lift-off."""
    self.takeoff_vel = torch.zeros(n, 3, device=device)
    """World-frame trunk velocity at lift-off."""
    self.takeoff_x = torch.zeros(n, device=device)
    self.landing_x = torch.zeros(n, device=device)
    self.landing_vel = torch.zeros(n, 3, device=device)
    self.trigger_x = torch.zeros(n, device=device)
    """Where the jump was triggered, along +x from the spawn point."""
    self.liftoff_now = torch.zeros(n, dtype=torch.bool, device=device)
    self.landed_now = torch.zeros(n, dtype=torch.bool, device=device)
    self.success = torch.zeros(n, dtype=torch.bool, device=device)
    """Landed and on the landing box, as of the last step."""
    self._airborne = torch.zeros(n, dtype=torch.long, device=device)
    self._first_air_vel = torch.zeros(n, 3, device=device)
    self.plan_trigger_x = torch.full((n,), torch.inf, device=device)
    self.plan_vz = torch.zeros(n, device=device)

    # The course of every terrain row, relative to its spawn point.
    terrain = env.scene.terrain
    generator = terrain.cfg.terrain_generator if terrain is not None else None
    rows = generator.num_rows if generator is not None else 1
    layouts = [cfg.course.layout(r / max(rows - 1, 1)) for r in range(rows)]
    self._edge = torch.tensor([lay.edge for lay in layouts], device=device)
    self._pad_start = torch.tensor([lay.pad_start for lay in layouts], device=device)
    self._pad_end = torch.tensor([lay.pad_end for lay in layouts], device=device)
    self._pad_half_width = layouts[0].pad_half_width

    for name in ("jumped", "takeoff_vz_error", "on_pad"):
      self.metrics[name] = torch.zeros(n, device=device)

  @property
  def command(self) -> torch.Tensor:
    return self.cmd

  # Course geometry for each environment.

  def _levels(self) -> torch.Tensor:
    terrain = self._env.scene.terrain
    levels = getattr(terrain, "terrain_levels", None)
    if levels is None:
      return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    return levels

  def course(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edge, landing-box start and end, along +x from each spawn point."""
    levels = self._levels()
    return self._edge[levels], self._pad_start[levels], self._pad_end[levels]

  def position(self) -> torch.Tensor:
    """Trunk position relative to the spawn point (the top of the long box)."""
    return self.robot.data.root_link_pos_w - self._env.scene.env_origins

  def set_plan(
    self,
    run_speed: torch.Tensor,
    trigger_x: torch.Tensor,
    vz: torch.Tensor,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    """The planner's run-up speed, trigger point (along +x from the spawn point)
    and take-off speed. Takes effect while the robot is still running."""
    ids = slice(None) if env_ids is None else env_ids
    self.run_speed[ids] = run_speed
    self.plan_trigger_x[ids] = trigger_x
    self.plan_vz[ids] = vz

  def on_pad(self) -> torch.Tensor:
    """Trunk above the landing box and not below its top."""
    _, pad_start, pad_end = self.course()
    pos = self.position()
    return (
      (pos[:, 0] > pad_start)
      & (pos[:, 0] < pad_end)
      & (pos[:, 1].abs() < self._pad_half_width)
      & (pos[:, 2] > 0.1)
    )

  def _gravity(self) -> torch.Tensor:
    name = self.cfg.physics_event
    events = self._env.event_manager
    if name is not None and any(
      name in names for names in events.active_terms.values()
    ):
      term = events.get_term_cfg(name).func
      gravity = getattr(term, "gravity", None)
      if gravity is not None:
        return gravity
    return torch.full((self.num_envs,), NOMINAL_GRAVITY, device=self.device)

  def _flight_distance(self) -> torch.Tensor:
    """Trunk travel from lift-off to touch-down that the coach aims for."""
    edge, pad_start, _ = self.course()
    takeoff = edge - self.cfg.takeoff_before_edge
    return pad_start + self.landing_offset - takeoff

  # Command term interface.

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """A new episode: back to running, with a new speed and landing point."""
    cfg = self.cfg
    n = len(env_ids)
    r = torch.rand(n, device=self.device)
    self.phase[env_ids] = JumpPhase.RUN
    lo, hi = cfg.landing_offset
    self.landing_offset[env_ids] = lo + (hi - lo) * torch.rand(n, device=self.device)
    # Fast enough that the jump needs at most the top take-off speed.
    longest = 2 * cfg.vz_range[1] / NOMINAL_GRAVITY
    slowest = self._flight_distance()[env_ids] / longest
    lo, hi = cfg.run_speed
    self.run_speed[env_ids] = torch.maximum(lo + (hi - lo) * r, slowest).clamp(max=hi)
    if not cfg.coach:  # Stand still until the planner sets a plan.
      self.run_speed[env_ids] = 0.0
      self.plan_trigger_x[env_ids] = torch.inf
      self.plan_vz[env_ids] = 0.0
    self.takeoff_cmd[env_ids] = 0.0
    self.takeoff_vel[env_ids] = 0.0
    self._airborne[env_ids] = 0
    self._first_air_vel[env_ids] = 0.0
    self.liftoff_now[env_ids] = False
    self.landed_now[env_ids] = False
    self.success[env_ids] = False

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    jumped = self.phase[env_ids] >= JumpPhase.FLIGHT
    self.metrics["jumped"][env_ids] = jumped.float()
    self.metrics["on_pad"][env_ids] = self.success[env_ids].float()
    error = (self.takeoff_vel[env_ids, 2] - self.takeoff_cmd[env_ids, 1]).abs()
    self.metrics["takeoff_vz_error"][env_ids] = torch.where(jumped, error, 0.0)
    return super().reset(env_ids)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    cfg = self.cfg
    if env_ids is not None:
      # Reset path: freshly reset environments start running from rest.
      self.cmd[env_ids] = 0.0
      return
    pos = self.position()
    vel = self.robot.data.root_link_lin_vel_w
    edge, _, _ = self.course()
    sensor: ContactSensor = self._env.scene[cfg.contact_sensor]
    assert sensor.data.found is not None
    in_air = (sensor.data.found == 0).all(dim=-1)
    self.liftoff_now[:] = False
    self.landed_now[:] = False

    # Run -> take-off when the trunk reaches the trigger point.
    running = self.phase == JumpPhase.RUN
    speed = vel[:, 0].clamp(min=0.5)
    if cfg.coach:
      trigger = edge - cfg.takeoff_before_edge - speed * cfg.push_time
    else:
      trigger = self.plan_trigger_x
    start = running & (pos[:, 0] >= trigger)
    if start.any():
      if cfg.coach:
        flight_time = self._flight_distance() / speed
        vz = self._gravity() * flight_time / 2
        vz = vz * (1 + cfg.vz_noise * (2 * torch.rand_like(vz) - 1))
      else:
        vz = self.plan_vz
      vz = vz.clamp(*cfg.vz_range)
      self.trigger_x[start] = pos[start, 0]
      self.takeoff_cmd[start, 0] = speed[start]
      self.takeoff_cmd[start, 1] = vz[start]
      self.phase[start] = JumpPhase.TAKEOFF

    # Take-off -> flight once every foot has been off the ground for a few steps
    # and the trunk left rising: a jump, not the short flight of a running gait.
    # The take-off velocity is the one at the first step in the air.
    self._airborne = torch.where(in_air, self._airborne + 1, 0)
    first = self._airborne == 1
    self._first_air_vel[first] = vel[first]
    lift = (
      (self.phase == JumpPhase.TAKEOFF)
      & (self._airborne >= cfg.liftoff_steps)
      & (self._first_air_vel[:, 2] >= cfg.liftoff_vz)
    )
    self.liftoff_now[:] = lift
    self.takeoff_vel[lift] = self._first_air_vel[lift]
    self.takeoff_x[lift] = pos[lift, 0]
    self.phase[lift] = JumpPhase.FLIGHT

    # Flight -> landed at the first touch-down.
    land = (self.phase == JumpPhase.FLIGHT) & ~in_air & ~lift
    self.landed_now[:] = land
    self.landing_x[land] = pos[land, 0]
    self.landing_vel[land] = vel[land]
    self.phase[land] = JumpPhase.LANDED
    self.success[:] = (self.phase == JumpPhase.LANDED) & self.on_pad()

    # The command for each phase.
    t = self._env.episode_length_buf.float() * self._env.step_dt
    run = torch.minimum(self.run_speed, cfg.run_accel * t)
    jumping = (self.phase == JumpPhase.TAKEOFF) | (self.phase == JumpPhase.FLIGHT)
    landed = self.phase == JumpPhase.LANDED
    vx = torch.where(jumping, self.takeoff_cmd[:, 0], run)
    vx = torch.where(landed, 0.0, vx)
    heading = self.robot.data.heading_w
    self.cmd[:, 0] = vx
    self.cmd[:, 1] = (-cfg.steer_lateral * pos[:, 1]).clamp(-0.4, 0.4)
    self.cmd[:, 2] = (-cfg.steer_yaw * heading).clamp(-0.6, 0.6)
    self.cmd[:, 3] = torch.where(jumping, self.takeoff_cmd[:, 1], 0.0)


@dataclass(kw_only=True)
class JumpCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  course: BoxJumpCourseCfg
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  """One command plan per episode."""
  run_speed: tuple[float, float] = (1.5, 3.0)
  """Run-up speed, raised where needed so the jump stays within `vz_range`."""
  run_accel: float = 2.0
  """How fast the commanded run-up speed ramps up from rest, in m/s^2."""
  vz_range: tuple[float, float] = (0.5, 2.0)
  vz_noise: float = 0.1
  """Relative noise on the commanded take-off speed, so the policy learns to
  follow the command rather than the course."""
  push_time: float = 0.12
  """The coach's guess of the time from the trigger to lift-off, in s."""
  takeoff_before_edge: float = 0.05
  """Where the coach wants the trunk at lift-off, in metres before the edge."""
  landing_offset: tuple[float, float] = (0.05, 0.3)
  """Where the coach aims the trunk at touch-down, in metres past the start of
  the landing box."""
  liftoff_steps: int = 3
  liftoff_vz: float = 0.25
  """A lift-off: every foot off the ground for `liftoff_steps` policy steps, and
  the trunk rising at least this fast (m/s) when they left."""
  coach: bool = True
  """False: a planner sets the plan through `JumpCommand.set_plan`."""
  steer_lateral: float = 1.0
  steer_yaw: float = 1.5
  physics_event: str | None = "physics_change"
  """Event whose `gravity` the coach uses. Nominal gravity without one."""
  contact_sensor: str = "feet_ground_contact"

  def build(self, env: ManagerBasedRlEnv) -> JumpCommand:
    return JumpCommand(self, env)
