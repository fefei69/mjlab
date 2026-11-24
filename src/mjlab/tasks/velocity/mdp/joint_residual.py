"""An online residual per joint, to notice a weak or dead motor.

A nominal model, fitted once on the healthy robot, predicts how much each joint's
velocity changes over one policy step from the state of all joints: positions,
velocities and how far each is from its target. It is a small network, because
a linear model per joint explains only 20 to 50% of the change (contacts and the
other joints matter), against about 97% for the network.

A weak motor shows up as a joint that keeps moving toward its target less than
predicted. The score is that shortfall, in units of the model's usual error,
averaged over the last half second:

  z = -sign(e) * (dqd - nominal) / noise,   e = target - q
  score <- score + (dt / tau) * (z - score)

It stays near 0 on a healthy joint. Impacts and other surprises push the error
both ways and average out; a motor that cannot deliver its torque pushes it one
way only. The threshold is set on healthy robots, so that few of them ever cross
it. Everything is batched over robots, so it also runs inside a training
environment, as the observation term `joint_fault_score`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg


class JointModel(nn.Module):
  """Nominal model: velocity change of every joint over one policy step."""

  def __init__(self, names: list[str], hidden: int = 256) -> None:
    super().__init__()
    self.names = list(names)
    j = len(names)
    self.net = nn.Sequential(
      nn.Linear(3 * j, hidden),
      nn.ELU(),
      nn.Linear(hidden, hidden),
      nn.ELU(),
      nn.Linear(hidden, j),
    )
    # Input and output scaling, the model's usual error per joint, and the score's
    # threshold and averaging time. Buffers, so they are saved with the weights.
    self.q_mean: torch.Tensor
    self.in_mean: torch.Tensor
    self.in_std: torch.Tensor
    self.out_std: torch.Tensor
    self.noise: torch.Tensor
    self.error_std: torch.Tensor
    self.threshold: torch.Tensor
    self.tau: torch.Tensor
    for name, size in (
      ("q_mean", j),
      ("in_mean", 3 * j),
      ("in_std", 3 * j),
      ("out_std", j),
      ("noise", j),
      ("error_std", j),
    ):
      self.register_buffer(name, torch.zeros(size))
    self.register_buffer("threshold", torch.tensor(1.0))
    self.register_buffer("tau", torch.tensor(0.5))

  def inputs(self, q: torch.Tensor, qd: torch.Tensor, target: torch.Tensor):
    return torch.cat([q - self.q_mean, qd, target - q], dim=-1)

  def forward(
    self, q: torch.Tensor, qd: torch.Tensor, target: torch.Tensor
  ) -> torch.Tensor:
    x = (self.inputs(q, qd, target) - self.in_mean) / self.in_std
    return self.net(x) * self.out_std

  def save(self, path: Path, **extra: object) -> None:
    torch.save({"names": self.names, "state": self.state_dict(), **extra}, path)

  @classmethod
  def load(cls, path: Path, device: torch.device | str = "cpu") -> JointModel:
    data = torch.load(path, map_location=device)
    model = cls(data["names"]).to(device)
    model.load_state_dict(data["state"])
    return model.eval()


def fit_joint_model(
  q: torch.Tensor,
  qd: torch.Tensor,
  target: torch.Tensor,
  qd_next: torch.Tensor,
  names: list[str],
  steps: int = 4000,
  seed: int = 0,
) -> JointModel:
  """Fit on (N, J) samples from healthy robots."""
  torch.manual_seed(seed)
  model = JointModel(names).to(q.device)
  model.q_mean.copy_(q.mean(dim=0))
  x = model.inputs(q, qd, target)
  y = qd_next - qd
  model.in_mean.copy_(x.mean(dim=0))
  model.in_std.copy_(x.std(dim=0).clamp(min=1e-6))
  model.out_std.copy_(y.std(dim=0))
  model.error_std.copy_((target - q).std(dim=0))
  opt = torch.optim.Adam(model.net.parameters(), lr=1e-3)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
  for _ in range(steps):
    i = torch.randint(len(q), (4096,), device=q.device)
    err = (model(q[i], qd[i], target[i]) - y[i]) / model.out_std
    loss = err.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
  with torch.no_grad():
    model.noise.copy_((model(q, qd, target) - y).std(dim=0))
  return model.eval()


@dataclass
class JointResidual:
  """Running fault score per robot and joint."""

  model: JointModel
  num_envs: int
  dt: float
  clip: float = 3.0

  def __post_init__(self) -> None:
    device = self.model.noise.device
    j = len(self.model.names)
    self.score = torch.zeros(self.num_envs, j, device=device)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    self.score[slice(None) if env_ids is None else env_ids] = 0.0

  @torch.no_grad()
  def update(
    self,
    q: torch.Tensor,
    qd: torch.Tensor,
    target: torch.Tensor,
    qd_next: torch.Tensor,
  ) -> torch.Tensor:
    """One policy step. q, qd before it, the target during it, qd after it."""
    r = (qd_next - qd - self.model(q, qd, target)) / self.model.noise
    z = (-torch.sign(target - q) * r).clamp(-self.clip, self.clip)
    self.score += (self.dt / float(self.model.tau)) * (z - self.score)
    return self.score

  @property
  def alarm(self) -> torch.Tensor:
    """(B,) whether any joint's score is above the threshold."""
    return (self.score > self.model.threshold).any(dim=-1)


class joint_fault_score:
  """Observation: every joint's fault score, divided by the threshold.

  About 0 on a healthy joint; 1 or more means the joint is flagged. Keeps the
  joint state from the previous step, so it updates once per environment step
  however often observations are computed.

  Params:
    model_path: The fitted `JointModel` (see `mppi/fit_joint_model.py`).
    asset_name: The robot entity.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self._asset: Entity = env.scene[cfg.params.get("asset_name", "robot")]
    model = JointModel.load(Path(cfg.params["model_path"]), env.device)
    if model.names != list(self._asset.joint_names):
      raise ValueError("The joint model was fitted on other joints.")
    self.residual = JointResidual(model, env.num_envs, env.step_dt)
    self._q = self._asset.data.joint_pos.clone()
    self._qd = self._asset.data.joint_vel.clone()
    self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    self._step = -1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.residual.score[ids] = 0.0
    # The first update after a reset mixes two episodes, so it is dropped.
    self._fresh[ids] = True

  def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
    if env.common_step_counter != self._step:
      self._step = env.common_step_counter
      data = self._asset.data
      self.residual.update(self._q, self._qd, data.joint_pos_target, data.joint_vel)
      self.residual.score[self._fresh] = 0.0
      self._fresh[:] = False
      self._q, self._qd = data.joint_pos.clone(), data.joint_vel.clone()
    return self.residual.score / self.residual.model.threshold
