"""Observations, rewards, terminations and curriculum of the box-jump task.

They read the jump command's phase and course geometry (`jump_command.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.box_jump.mdp.jump_command import (
  NOMINAL_GRAVITY,
  JumpCommand,
  JumpPhase,
)
from mjlab.tasks.box_jump.mdp.physics_change import physics_change

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "base_height",
  "course_ahead",
  "fell_off_course",
  "jump_phase",
  "landed_on_pad",
  "physics_params",
  "takeoff_push",
  "takeoff_velocity",
  "terrain_levels_jump",
  "track_jump_velocity",
  "track_jump_yaw_rate",
]


def _command(env: ManagerBasedRlEnv, name: str) -> JumpCommand:
  term = env.command_manager.get_term(name)
  assert isinstance(term, JumpCommand)
  return term


# Observations.


def course_ahead(env: ManagerBasedRlEnv, command_name: str = "jump") -> torch.Tensor:
  """Distance ahead to the edge and to the start and end of the landing box, m."""
  cmd = _command(env, command_name)
  edge, pad_start, pad_end = cmd.course()
  x = cmd.position()[:, 0]
  return torch.stack([edge - x, pad_start - x, pad_end - x], dim=-1)


def base_height(env: ManagerBasedRlEnv, command_name: str = "jump") -> torch.Tensor:
  """Trunk height above the box tops, m. Privileged: for the critic."""
  return _command(env, command_name).position()[:, 2:3]


def jump_phase(env: ManagerBasedRlEnv, command_name: str = "jump") -> torch.Tensor:
  """One-hot phase of the jump. Privileged: for the critic."""
  phase = _command(env, command_name).phase
  return torch.nn.functional.one_hot(phase, len(JumpPhase)).float()


def physics_params(
  env: ManagerBasedRlEnv, event_name: str = "physics_change"
) -> torch.Tensor:
  """True gravity (relative to nominal) and foot friction. Privileged."""
  term = env.event_manager.get_term_cfg(event_name).func
  assert isinstance(term, physics_change)
  return torch.stack([term.gravity / NOMINAL_GRAVITY, term.friction], dim=-1)


# Rewards.


def track_jump_velocity(
  env: ManagerBasedRlEnv, std: float, command_name: str = "jump"
) -> torch.Tensor:
  """Track the commanded planar velocity; on the ground, also no vertical motion.

  In the air and while pushing off, vertical motion is the point, so only the
  planar velocity counts.
  """
  cmd = _command(env, command_name)
  vel = cmd.robot.data.root_link_lin_vel_b
  error = (vel[:, :2] - cmd.command[:, :2]).square().sum(dim=-1)
  grounded = (cmd.phase == JumpPhase.RUN) | (cmd.phase == JumpPhase.LANDED)
  error = error + torch.where(grounded, vel[:, 2].square(), 0.0)
  return torch.exp(-error / std**2)


def track_jump_yaw_rate(
  env: ManagerBasedRlEnv, std: float, command_name: str = "jump"
) -> torch.Tensor:
  cmd = _command(env, command_name)
  wz = cmd.robot.data.root_link_ang_vel_b[:, 2]
  return torch.exp(-(wz - cmd.command[:, 2]).square() / std**2)


def takeoff_velocity(
  env: ManagerBasedRlEnv,
  std_vz: float,
  std_vx: float,
  command_name: str = "jump",
) -> torch.Tensor:
  """Once, at lift-off: how close the take-off was to the commanded speeds."""
  cmd = _command(env, command_name)
  vz_error = (cmd.takeoff_vel[:, 2] - cmd.takeoff_cmd[:, 1]).square() / std_vz**2
  vx_error = (cmd.takeoff_vel[:, 0] - cmd.takeoff_cmd[:, 0]).square() / std_vx**2
  return cmd.liftoff_now.float() * torch.exp(-vz_error - vx_error)


def takeoff_push(env: ManagerBasedRlEnv, command_name: str = "jump") -> torch.Tensor:
  """While pushing off: upward speed as a fraction of the commanded one."""
  cmd = _command(env, command_name)
  vz = cmd.robot.data.root_link_lin_vel_w[:, 2]
  share = (vz / cmd.takeoff_cmd[:, 1].clamp(min=0.1)).clamp(0.0, 1.0)
  return (cmd.phase == JumpPhase.TAKEOFF).float() * share


def landed_on_pad(env: ManagerBasedRlEnv, command_name: str = "jump") -> torch.Tensor:
  return _command(env, command_name).success.float()


# Terminations.


def fell_off_course(
  env: ManagerBasedRlEnv, min_height: float = -0.1, command_name: str = "jump"
) -> torch.Tensor:
  """The trunk dropped below the box tops: into the gap or off a box."""
  return _command(env, command_name).position()[:, 2] < min_height


# Curriculum.


def terrain_levels_jump(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str = "jump"
) -> dict[str, torch.Tensor]:
  """Harder course after landing on the box, easier after anything else."""
  terrain = env.scene.terrain
  assert terrain is not None
  success = _command(env, command_name).success[env_ids]
  move_up, move_down = success, ~success
  if env.common_step_counter == 0:  # The first reset: nothing to judge yet.
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)
  terrain.update_env_origins(env_ids, move_up, move_down)
  levels = terrain.terrain_levels.float()
  return {"mean": levels.mean(), "max": levels.max()}
