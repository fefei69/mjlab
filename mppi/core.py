"""MPPI over a short sequence of velocity commands.

The planner does not know how a command sequence is scored. The caller passes a
function that maps K candidate sequences to K costs, so the same planner works
with a cheap planar motion model (planar_model.py) and with full simulator
rollouts (tracker.py).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

CostFn = Callable[[torch.Tensor], torch.Tensor]
"""Maps commands of shape (K, H, nu) to costs of shape (K,)."""


@dataclass(kw_only=True)
class MppiCfg:
  horizon: int
  """Plan steps in the horizon. Each plan step holds one command."""
  num_samples: int
  """Candidate command sequences scored per iteration."""
  u_min: tuple[float, ...]
  u_max: tuple[float, ...]
  noise_sigma: tuple[float, ...]
  """Standard deviation of the sampling noise, per command dimension."""
  num_knots: int = 4
  """Noise is drawn at this many points along the horizon and interpolated
  linearly in between, which keeps the sampled sequences smooth."""
  temperature: float = 0.3
  """Softmax temperature, applied to costs divided by their standard deviation."""
  iterations: int = 1
  """Sample-and-update rounds per call to `plan`."""


class Mppi:
  def __init__(self, cfg: MppiCfg, device: str | torch.device) -> None:
    self.cfg = cfg
    self.device = torch.device(device)
    self._u_min = torch.tensor(cfg.u_min, device=self.device)
    self._u_max = torch.tensor(cfg.u_max, device=self.device)
    self._sigma = torch.tensor(cfg.noise_sigma, device=self.device)
    self.mean = torch.zeros(cfg.horizon, len(cfg.u_min), device=self.device)

  def set_limits(self, u_min: tuple[float, ...], u_max: tuple[float, ...]) -> None:
    """Change the command limits mid-run. The plan is pulled inside them."""
    self._u_min = torch.tensor(u_min, device=self.device)
    self._u_max = torch.tensor(u_max, device=self.device)
    self.mean = self.clamp(self.mean)

  def clamp(self, commands: torch.Tensor) -> torch.Tensor:
    return torch.clamp(commands, self._u_min, self._u_max)

  def sample(self) -> torch.Tensor:
    """Draw candidate sequences around the current plan, shape (K, H, nu).

    The samples are not clamped to the command limits. Clamp them before scoring
    and update with the raw samples (see `plan`).
    """
    cfg = self.cfg
    nu = self.mean.shape[1]
    knots = torch.randn(cfg.num_samples, nu, cfg.num_knots, device=self.device)
    knots = knots * self._sigma[None, :, None]
    if cfg.num_knots == 1:
      noise = knots.expand(-1, -1, cfg.horizon)
    else:
      noise = F.interpolate(knots, size=cfg.horizon, mode="linear", align_corners=True)
    raw = self.mean[None] + noise.transpose(1, 2)
    raw[0] = self.mean  # Always score the current plan itself.
    return raw

  def update(self, raw: torch.Tensor, costs: torch.Tensor) -> None:
    """Move the plan toward the low-cost samples."""
    z = (costs - costs.min()) / (costs.std() + 1e-6)
    weights = torch.softmax(-z / self.cfg.temperature, dim=0)
    mean = (weights[:, None, None] * raw).sum(dim=0)
    # The plan is kept unclamped so that it can sit on a command limit: averaging
    # clamped samples would always land inside the limit. Bound how far past the
    # limit it may drift, where the cost no longer changes.
    self.mean = torch.clamp(mean, self._u_min - self._sigma, self._u_max + self._sigma)

  def plan(self, cost_fn: CostFn) -> torch.Tensor:
    """Improve the plan and return it clamped to the limits, shape (H, nu)."""
    for _ in range(self.cfg.iterations):
      raw = self.sample()
      self.update(raw, cost_fn(self.clamp(raw)))
    return self.clamp(self.mean)

  def shift(self, steps: int = 1) -> None:
    """Drop the first `steps` commands after they were executed."""
    tail = self.mean[-1:].expand(steps, -1)
    self.mean = torch.cat([self.mean[steps:], tail], dim=0)
