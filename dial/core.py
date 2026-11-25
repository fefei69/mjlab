"""DIAL-MPC's planner: MPPI with diffusion-style annealing.

After the authors' `dial_core.py` (github.com/LeCAR-Lab/dial-mpc). A plan is a
few control nodes spread over the horizon, interpolated to every control step.
Each control step runs a few rounds of:

- draw samples around the plan, with noise that grows along the horizon and
  shrinks by `traj_diffuse_factor` every round; the first node, the control
  applied now, is not perturbed;
- roll every sample out and score it by its mean per-step reward;
- move the plan to the reward-weighted mean, with weights from a softmax of the
  rewards relative to the current plan's, divided by their spread.

After a control step the plan is shifted by one step. The first call runs more
rounds (`diffuse_steps_init`). One difference from the authors' code: nodes are
joined linearly, not by a quadratic spline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

RolloutFn = Callable[[torch.Tensor], torch.Tensor]
"""Maps controls (K, H, nu) in [-1, 1] to mean per-step rewards (K,)."""


@dataclass(kw_only=True)
class DialCfg:
  """Defaults from the authors' sequential-jump example (unitree_go2_seq_jump)."""

  num_samples: int = 2048
  horizon_steps: int = 20
  """Hsample: control steps in the horizon."""
  num_nodes: int = 5
  """Hnode: the plan has num_nodes + 1 nodes, at both ends of the horizon."""
  diffuse_steps: int = 2
  diffuse_steps_init: int = 10
  temperature: float = 0.05
  horizon_diffuse_factor: float = 0.9
  traj_diffuse_factor: float = 0.5
  sigma_scale: float = 1.0
  ctrl_dt: float = 0.02


class Dial:
  def __init__(self, cfg: DialCfg, action_dim: int, device: str = "cpu") -> None:
    self.cfg = cfg
    self.device = device
    horizon = cfg.horizon_steps * cfg.ctrl_dt
    self.node_t = torch.linspace(0.0, horizon, cfg.num_nodes + 1, device=device)
    self.step_t = torch.arange(cfg.horizon_steps, device=device) * cfg.ctrl_dt
    # Later nodes get more noise: factor^(Hnode - j) for node j.
    powers = torch.arange(cfg.num_nodes + 1, device=device).flip(0)
    self.sigma = cfg.horizon_diffuse_factor**powers * cfg.sigma_scale
    self.nodes = torch.zeros(cfg.num_nodes + 1, action_dim, device=device)
    self.started = False
    self.last_rewards: torch.Tensor | None = None
    self.last_controls: torch.Tensor | None = None

  def controls(self, nodes: torch.Tensor, times: torch.Tensor | None = None):
    """Nodes (..., Hnode + 1, nu) to controls (..., T, nu) at `times` (default:
    every control step of the horizon), held flat past the last node."""
    t = self.step_t if times is None else times
    t = t.clamp(max=float(self.node_t[-1]))
    spacing = float(self.node_t[1] - self.node_t[0])
    f = t / spacing
    i0 = f.floor().long().clamp(max=self.cfg.num_nodes - 1)
    w = (f - i0)[:, None]
    return nodes[..., i0, :] * (1 - w) + nodes[..., i0 + 1, :] * w

  def improve(self, rollout: RolloutFn) -> None:
    """One control step's rounds of annealed sampling."""
    cfg = self.cfg
    rounds = cfg.diffuse_steps if self.started else cfg.diffuse_steps_init
    self.started = True
    for i in range(rounds):
      scale = self.sigma[:, None] * cfg.traj_diffuse_factor**i
      noise = torch.randn(cfg.num_samples, *self.nodes.shape, device=self.device)
      samples = self.nodes + noise * scale
      samples[:, 0] = self.nodes[0]  # The control applied now stays.
      samples = samples.clamp(-1.0, 1.0)
      # The current plan is scored last, as in the authors' code.
      samples = torch.cat([samples, self.nodes[None]], dim=0)
      controls = self.controls(samples)
      rewards = rollout(controls)
      logits = (rewards - rewards[-1]) / (rewards.std() + 1e-6) / cfg.temperature
      weights = torch.softmax(logits, dim=0)
      self.nodes = (weights[:, None, None] * samples).sum(dim=0)
      self.last_rewards, self.last_controls = rewards, controls

  def action(self) -> torch.Tensor:
    """The control to apply now, (nu,) in [-1, 1]."""
    return self.nodes[0].clamp(-1.0, 1.0)

  def shift(self) -> None:
    """Move the plan one control step on; the last node is held."""
    self.nodes = self.controls(self.nodes, self.node_t + self.cfg.ctrl_dt)
