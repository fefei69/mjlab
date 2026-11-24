"""An online correction to the planar motion model.

The fitted `PlanarModel` stays fixed as the nominal model. A residual sits on top
of it, and its coefficients are re-fitted during the run from how the robot
actually moved:

  v_next = nominal(v, u) + s * (F(v, u) a)

F(v, u) is a (3, P) matrix of features and a holds the P coefficients. With a = 0
this is the nominal model, which is the healthy robot. The features are a
replaceable part: the hand-chosen ones below, or later a network trained offline.

Both sides are scaled so that one setting suits every axis and feature. The
features see the velocity and the command divided by the command range, so their
inputs are of order one. s is the nominal model's usual error on the healthy
robot, per axis, so a coefficient of 1 means "one usual error".

The update is the one in Neural-Fly (O'Connell et al., Science Robotics 2022,
supplement eqs. 15 to 19) without its tracking-error term: a Kalman filter on the
coefficients, with a leak that returns them to zero when the data says nothing.
Because of the scaling by s, its measurement noise is the identity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from mppi.planar_model import PlanarModel, rollout_model

Features = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""Maps scaled velocity and command, each (..., 3), to a (..., 3, P) matrix."""


def _eye(cmd: torch.Tensor) -> torch.Tensor:
  eye = torch.eye(3, dtype=cmd.dtype, device=cmd.device)
  return eye.expand(*cmd.shape[:-1], 3, 3)


def offset_features(vel: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
  """One offset per axis (P = 3). This is a disturbance observer."""
  del vel
  return _eye(cmd)


def gain_offset_features(vel: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
  """A gain on each axis's own command and an offset (P = 6).

  The coefficients are (gain_x, gain_y, gain_yaw, offset_x, offset_y, offset_yaw).
  """
  del vel
  return torch.cat([torch.diag_embed(cmd), _eye(cmd)], dim=-1)


def command_features(vel: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
  """Each axis responds to all three commands and has an offset (P = 12).

  The coefficients, reshaped to (3, 4), hold one row per axis: its gains on the
  forward, sideways and yaw commands, then its offset. Unlike the gain per axis,
  this can express a robot that veers more the faster it is told to go.
  """
  del vel
  phi = torch.cat([cmd, torch.ones_like(cmd[..., :1])], dim=-1)
  return (_eye(cmd)[..., :, :, None] * phi[..., None, None, :]).flatten(-2)


FEATURE_SETS: dict[str, Features] = {
  "offset": offset_features,
  "gain+offset": gain_offset_features,
  "command": command_features,
}


@dataclass(kw_only=True)
class ResidualCfg:
  meas_std: tuple[float, float, float] = (0.10, 0.06, 0.06)
  """One-step error of the nominal model on the healthy robot, per axis.

  The default is what `fit_planar_model` reports for the baseline policy.
  """
  drift_std: float = 0.6
  """How far a coefficient may move per square root of a second.

  This sets the speed: higher reacts faster to a change and follows noise more.
  """
  leak: float = 0.1
  """Rate, per second, at which coefficients return to zero without evidence."""
  gain_drift_ratio: float = 1 / 3
  """Gains drift at this fraction of `drift_std`; offsets at the full rate.

  While the commands are held, a gain and an offset explain the same error. With
  equal rates the filter splits a steady push between them, and the planner then
  brakes to shrink a drift that does not depend on speed. Slower gains let the
  offset take such an error first; gains grow once varied commands show a real
  dependence on them. 1/3 gave no falls in ten runs of the pull test.
  """


class ResidualModel:
  """The nominal model plus a residual whose coefficients are fitted online.

  With `num_envs` set, it keeps one set of coefficients per robot: `update` takes
  (num_envs, 3) inputs, and `residual` and `rollout` take inputs whose leading
  dimension is the robot.
  """

  def __init__(
    self,
    nominal: PlanarModel,
    features: Features,
    cfg: ResidualCfg | None = None,
    num_envs: int | None = None,
  ) -> None:
    self.nominal = nominal
    self.num_envs = num_envs
    # The filter runs in double precision: the covariance update loses symmetry
    # in single precision. One robot's filter is a few tiny matrices, which the
    # CPU does faster; many robots run on the nominal model's device.
    self._device = torch.device("cpu") if num_envs is None else nominal.a.device
    self.features = features
    self.cfg = cfg or ResidualCfg()
    self.dt = nominal.dt
    if nominal.cmd_min is not None and nominal.cmd_max is not None:
      limits = zip(nominal.cmd_min, nominal.cmd_max, strict=True)
      reach = [max(abs(lo), abs(hi)) for lo, hi in limits]
    else:
      reach = [1.0, 1.0, 1.0]
    self._in_scale = torch.tensor(reach).to(nominal.a)
    self._out_scale = torch.tensor(self.cfg.meas_std).to(nominal.a)
    # Gains are the coefficients whose features vanish at rest, with zero velocity
    # and command. Offsets are the rest.
    at_rest = self._features(self._in_scale * 0, self._in_scale * 0)
    self.is_gain = (at_rest.abs().sum(dim=-2) == 0).cpu()
    self.num_coef = self.is_gain.numel()
    rate = torch.full((self.num_coef,), self.cfg.drift_std, dtype=torch.float64)
    rate[self.is_gain] *= self.cfg.gain_drift_ratio
    self._drift_var = torch.diag(rate**2).to(self._device)
    batch = 1 if num_envs is None else num_envs
    p = self.num_coef
    self._coef = torch.zeros(batch, p, dtype=torch.float64, device=self._device)
    self._cov = torch.zeros(batch, p, p, dtype=torch.float64, device=self._device)
    self.reset()

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Forget everything: the model is the nominal model again.

    With `env_ids`, only for those robots.
    """
    ids = slice(None) if env_ids is None else env_ids.to(self._device)
    self._coef[ids] = 0.0
    self._cov[ids] = 0.0
    self._publish()

  def _publish(self) -> None:
    # The copy the rollouts use, on the nominal model's device: (P,) for one
    # robot, (num_envs, P) for many.
    coef = self._coef.to(self.nominal.a)
    self.coef = coef[0] if self.num_envs is None else coef

  @property
  def coef_std(self) -> torch.Tensor:
    """(P,) or (num_envs, P): how uncertain the filter is about each coefficient."""
    std = self._cov.diagonal(dim1=-2, dim2=-1).sqrt().to(self.nominal.a)
    return std[0] if self.num_envs is None else std

  # Prediction.

  def _features(self, vel: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    return self.features(vel / self._in_scale, cmd / self._in_scale)

  def residual(self, vel: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    """The correction to the nominal model's next velocity, shape (..., 3)."""
    feat = self._features(vel, cmd)
    if self.num_envs is None:
      return self._out_scale * (feat @ self.coef)
    return self._out_scale * torch.einsum("b...ip,bp->b...i", feat, self.coef)

  def next_velocity(self, vel: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    return self.nominal.next_velocity(vel, cmd) + self.residual(vel, cmd)

  def rollout(
    self, pose: torch.Tensor, vel: torch.Tensor, commands: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """See `rollout_model`."""
    return rollout_model(self, pose, vel, commands)

  # Fitting.

  @torch.no_grad()
  def update(
    self, vel: torch.Tensor, cmd: torch.Tensor, vel_next: torch.Tensor
  ) -> torch.Tensor:
    """Take in one plan step of motion: velocity, command, resulting velocity.

    Each is (3,), or (num_envs, 3) for many robots. Returns the nominal model's
    error on this sample, same shape, which is what the residual is fitted to.
    """
    error = vel_next - self.nominal.next_velocity(vel, cmd)
    dev, f64 = self._device, torch.float64
    feat = self._features(vel, cmd).to(dev, f64).reshape(-1, 3, self.num_coef)
    err = (error / self._out_scale).to(dev, f64).reshape(-1, 3, 1)
    eye = torch.eye(self.num_coef, dtype=f64, device=dev)

    # Between samples: leak toward zero, and allow for drift since the last one.
    keep = 1.0 - self.cfg.leak * self.dt
    coef = keep * self._coef[..., None]  # (B, P, 1)
    cov = keep**2 * self._cov + self._drift_var * self.dt

    # Correct by what the current coefficients failed to predict.
    feat_t = feat.transpose(-1, -2)
    innovation_cov = feat @ cov @ feat_t + torch.eye(3, dtype=f64, device=dev)
    gain = torch.linalg.solve(innovation_cov, feat @ cov).transpose(-1, -2)
    self._coef = (coef + gain @ (err - feat @ coef))[..., 0]
    rest = eye - gain @ feat
    self._cov = rest @ cov @ rest.transpose(-1, -2) + gain @ gain.transpose(-1, -2)
    self._publish()
    return error
