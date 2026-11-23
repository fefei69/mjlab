"""A planar motion model of the robot under its tracking policy.

The model predicts the base velocity over the next plan step from the velocity
over the last one and the commanded velocity:

  v_next = A v + B u + G s(u) + c

v and u are (forward, sideways, yaw rate) in the body frame. A captures how
slowly the robot responds, B how much of a command it delivers, and c any steady
drift. s(u) holds the nine products u_i |u_j|, so G captures how the response to
one command weakens as any command grows, which matters near the robot's speed
limits. Everything comes from one least-squares fit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(angle), torch.cos(angle))


def body_velocity(
  pose_prev: torch.Tensor, pose: torch.Tensor, dt: float
) -> torch.Tensor:
  """Average body-frame velocity that moves `pose_prev` to `pose` in `dt`.

  Poses are (..., 3) tensors of x, y, yaw. The displacement is expressed in the
  body frame at the midpoint heading, which is the inverse of `integrate_pose`.
  """
  dyaw = wrap_angle(pose[..., 2] - pose_prev[..., 2])
  mid = pose_prev[..., 2] + 0.5 * dyaw
  dx = pose[..., 0] - pose_prev[..., 0]
  dy = pose[..., 1] - pose_prev[..., 1]
  cos, sin = torch.cos(mid), torch.sin(mid)
  return torch.stack([cos * dx + sin * dy, -sin * dx + cos * dy, dyaw], dim=-1) / dt


def integrate_pose(pose: torch.Tensor, vel: torch.Tensor, dt: float) -> torch.Tensor:
  """Move `pose` with the body-frame velocity `vel` for `dt`."""
  mid = pose[..., 2] + 0.5 * vel[..., 2] * dt
  cos, sin = torch.cos(mid), torch.sin(mid)
  return torch.stack(
    [
      pose[..., 0] + (cos * vel[..., 0] - sin * vel[..., 1]) * dt,
      pose[..., 1] + (sin * vel[..., 0] + cos * vel[..., 1]) * dt,
      pose[..., 2] + vel[..., 2] * dt,
    ],
    dim=-1,
  )


def speed_terms(cmd: torch.Tensor) -> torch.Tensor:
  """The nine products u_i |u_j| of a (..., 3) command, shape (..., 9)."""
  return (cmd[..., :, None] * cmd[..., None, :].abs()).flatten(start_dim=-2)


class MotionModel(Protocol):
  """Anything that predicts the velocity over the next plan step."""

  dt: float

  def next_velocity(self, vel: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor: ...


def rollout_model(
  model: MotionModel, pose: torch.Tensor, vel: torch.Tensor, commands: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Predict poses and velocities for command sequences of shape (K, H, 3).

  `pose` and `vel` are the current pose and the velocity over the last plan
  step, each (3,) or (K, 3). Returns poses and velocities of shape (K, H, 3),
  where entry h is the state after the h-th command.
  """
  k, horizon, _ = commands.shape
  pose = pose.expand(k, 3)
  vel = vel.expand(k, 3)
  poses, vels = [], []
  for h in range(horizon):
    vel = model.next_velocity(vel, commands[:, h])
    pose = integrate_pose(pose, vel, model.dt)
    poses.append(pose)
    vels.append(vel)
  return torch.stack(poses, dim=1), torch.stack(vels, dim=1)


@dataclass
class PlanarModel:
  a: torch.Tensor
  """(3, 3) dependence of the next velocity on the current velocity."""
  b: torch.Tensor
  """(3, 3) dependence of the next velocity on the command."""
  c: torch.Tensor
  """(3,) constant offset."""
  dt: float
  """Duration of one plan step in seconds."""
  g: torch.Tensor | None = None
  """(3, 9) dependence on the speed terms. None for a purely linear model."""
  cmd_min: tuple[float, ...] | None = None
  cmd_max: tuple[float, ...] | None = None
  """Command range the model was fitted on. It is only valid inside it."""

  @classmethod
  def ideal(cls, dt: float, device: str | torch.device = "cpu") -> PlanarModel:
    """The naive model: the robot moves exactly as commanded, with no lag."""
    return cls(
      a=torch.zeros(3, 3, device=device),
      b=torch.eye(3, device=device),
      c=torch.zeros(3, device=device),
      dt=dt,
    )

  @classmethod
  def fit(
    cls,
    vel: torch.Tensor,
    cmd: torch.Tensor,
    vel_next: torch.Tensor,
    dt: float,
    linear: bool = False,
    cmd_min: tuple[float, ...] | None = None,
    cmd_max: tuple[float, ...] | None = None,
  ) -> PlanarModel:
    """Least-squares fit from (N, 3) samples of velocity, command, next velocity.

    With `linear`, the speed terms are left out.
    """
    vel, cmd = vel.double().cpu(), cmd.double().cpu()
    columns = [vel, cmd] if linear else [vel, cmd, speed_terms(cmd)]
    ones = torch.ones(vel.shape[0], 1, dtype=torch.float64)
    x = torch.cat([*columns, ones], dim=-1)
    w = torch.linalg.lstsq(x, vel_next.double().cpu()).solution.float()
    return cls(
      a=w[:3].T,
      b=w[3:6].T,
      c=w[-1],
      dt=dt,
      g=None if linear else w[6:15].T,
      cmd_min=cmd_min,
      cmd_max=cmd_max,
    )

  def to(self, device: str | torch.device) -> PlanarModel:
    return PlanarModel(
      a=self.a.to(device),
      b=self.b.to(device),
      c=self.c.to(device),
      dt=self.dt,
      g=None if self.g is None else self.g.to(device),
      cmd_min=self.cmd_min,
      cmd_max=self.cmd_max,
    )

  def next_velocity(self, vel: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    out = vel @ self.a.T + cmd @ self.b.T + self.c
    if self.g is not None:
      out = out + speed_terms(cmd) @ self.g.T
    return out

  def rollout(
    self, pose: torch.Tensor, vel: torch.Tensor, commands: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """See `rollout_model`."""
    return rollout_model(self, pose, vel, commands)

  def save(self, path: Path, **extra: object) -> None:
    payload = {
      "a": self.a.tolist(),
      "b": self.b.tolist(),
      "c": self.c.tolist(),
      "g": None if self.g is None else self.g.tolist(),
      "dt": self.dt,
      "cmd_min": self.cmd_min,
      "cmd_max": self.cmd_max,
      **extra,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")

  @classmethod
  def load(cls, path: Path, device: str | torch.device = "cpu") -> PlanarModel:
    payload = json.loads(path.read_text())
    g = payload.get("g")
    return cls(
      a=torch.tensor(payload["a"], device=device),
      b=torch.tensor(payload["b"], device=device),
      c=torch.tensor(payload["c"], device=device),
      dt=float(payload["dt"]),
      g=None if g is None else torch.tensor(g, device=device),
      cmd_min=tuple(payload["cmd_min"]) if payload.get("cmd_min") else None,
      cmd_max=tuple(payload["cmd_max"]) if payload.get("cmd_max") else None,
    )
