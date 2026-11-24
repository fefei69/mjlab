"""Learn the planar residual's features offline, as Neural-Fly does.

The residual is `F(v, u) a`: features F shared by every condition the robot can
be in, and coefficients a that the Kalman filter fits online for the condition
it is in now. This script learns F from data instead of choosing it by hand.

Many robots each run under one condition for the whole run (healthy, a weak
motor on one joint, or a steady pull on the trunk) under random commands. F is
a small network, trained so that for every robot one set of coefficients, fitted
by ridge regression on part of its data, predicts the nominal model's error on
the rest (Neural-Fly's DAIML without its adversarial term). The result is then
scored the way the planner uses it: the online filter runs over held-out robots,
including faults on a joint never seen in training, and its prediction error is
compared with the hand-chosen features.

  uv run python -m mppi.learn_features
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import nn

from mppi.fit_planar_model import sample_commands
from mppi.planar_model import PlanarModel, body_velocity
from mppi.residual_model import FEATURE_SETS, ResidualCfg, ResidualModel
from mppi.tracker import (
  CMD_MAX,
  CMD_MIN,
  DEFAULT_CHECKPOINT,
  TASK_ID,
  Tracker,
  make_tracker_cfg,
)

HELD_OUT_JOINT = "FL_calf_joint"  # As in the fault-trained policies' held-out set.


class LearnedFeatures(nn.Module):
  """F(v, u): scaled velocity and command, each (..., 3), to a (..., 3, P) matrix."""

  def __init__(self, num_coef: int = 6, hidden: int = 64) -> None:
    super().__init__()
    self.num_coef = num_coef
    self.net = nn.Sequential(
      nn.Linear(6, hidden),
      nn.Tanh(),
      nn.Linear(hidden, hidden),
      nn.Tanh(),
      nn.Linear(hidden, 3 * num_coef),
    )

  def forward(self, vel: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    weight = next(self.parameters())
    out = self.net(torch.cat([vel, cmd], dim=-1).to(weight))
    return out.reshape(*out.shape[:-1], 3, self.num_coef).to(vel)

  def save(self, path: Path, **extra: object) -> None:
    torch.save({"num_coef": self.num_coef, "state": self.state_dict(), **extra}, path)

  @classmethod
  def load(cls, path: Path, device: torch.device | str = "cpu") -> LearnedFeatures:
    data = torch.load(path, map_location=device)
    features = cls(data["num_coef"]).to(device)
    features.load_state_dict(data["state"])
    return features.eval()


def draw_conditions(
  n: int, names: list[str], held_out: bool, device: str
) -> dict[str, torch.Tensor]:
  """One condition per robot: 20% healthy, 50% a weak motor, 30% a pull."""
  u = torch.rand(n, device=device)
  kind = torch.where(u < 0.2, 0, torch.where(u < 0.7, 1, 2))
  allowed = [i for i, name in enumerate(names) if (name == HELD_OUT_JOINT) == held_out]
  choice = torch.tensor(allowed, device=device)
  joint = choice[torch.randint(len(choice), (n,), device=device)]
  strength = torch.where(kind == 1, 0.5 * torch.rand(n, device=device), 1.0)
  angle = 2 * math.pi * torch.rand(n, device=device)
  size = torch.where(kind == 2, 10 + 30 * torch.rand(n, device=device), 0.0)
  force = size[:, None] * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
  if held_out:  # Held-out robots all have the held-out fault.
    kind = torch.ones_like(kind)
    strength = 0.5 * torch.rand(n, device=device)
    force = torch.zeros_like(force)
  return {"kind": kind, "joint": joint, "strength": strength, "force": force}


def collect(
  tracker: Tracker,
  cond: dict[str, torch.Tensor],
  windows: int,
  lo: torch.Tensor,
  hi: torch.Tensor,
  hold: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Poses (W+1, N, 3), commands (W, N, 3), and whether still upright (W+1, N)."""
  n, device = tracker.num_envs, tracker.device
  tracker.set_motor_strength(cond["joint"], cond["strength"])
  cmd = sample_commands(n, lo, hi)
  left = torch.zeros(n, dtype=torch.long, device=device)
  poses, cmds, ok = [tracker.pose()], [], [tracker.upright() > 0.5]
  for _ in range(windows):
    expired = left <= 0
    cmd = torch.where(expired[:, None], sample_commands(n, lo, hi), cmd)
    left = torch.where(expired, torch.randint(2, 31, (n,), device=device), left) - 1
    for _ in range(hold):
      tracker.pull_trunk(cond["force"])
      tracker.step(cmd)
    poses.append(tracker.pose())
    cmds.append(cmd.clone())
    ok.append(ok[-1] & (tracker.upright() > 0.5))
  return torch.stack(poses), torch.stack(cmds), torch.stack(ok)


def transitions(
  poses: torch.Tensor, cmds: torch.Tensor, ok: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Velocity, command, next velocity (T, N, 3) and validity (T, N)."""
  vels = body_velocity(poses[:-1], poses[1:], dt)
  return vels[:-1], cmds[1:], vels[1:], ok[2:]


def train(
  model: ResidualModel,
  features: LearnedFeatures,
  data: tuple[torch.Tensor, ...],
  steps: int,
  ridge: float = 1.0,
) -> None:
  """Meta-train F: coefficients fitted on half a robot's data predict the rest."""
  vel, cmd, vel_next, ok = data
  t, n, _ = vel.shape
  # Targets and inputs in the residual model's scaled units.
  err = (vel_next - model.nominal.next_velocity(vel, cmd)) / model._out_scale
  xv, xu = vel / model._in_scale, cmd / model._in_scale
  opt = torch.optim.Adam(features.parameters(), lr=1e-3)
  eye = torch.eye(features.num_coef, device=vel.device)
  window = 30  # 3 s of support and 3 s of query per robot, from one stretch.
  for _ in range(steps):
    robots = torch.randint(n, (128,), device=vel.device)
    start = torch.randint(t - 2 * window, (128,), device=vel.device)
    idx = start[:, None] + torch.arange(2 * window, device=vel.device)
    pick = (idx, robots[:, None])
    phi = features(xv[pick], xu[pick])  # (R, 2W, 3, P)
    y, mask = err[pick], ok[pick].float()[..., None, None]
    phi_s, y_s, m_s = phi[:, :window], y[:, :window], mask[:, :window]
    a_mat = (m_s * phi_s.transpose(-1, -2) @ phi_s).sum(dim=1) + ridge * eye
    b_vec = (m_s * phi_s.transpose(-1, -2) @ y_s[..., None]).sum(dim=1)
    coef = torch.linalg.solve(a_mat, b_vec)  # (R, P, 1)
    pred = phi[:, window:] @ coef[:, None]
    m_q = mask[:, window:, :, 0]
    loss = (m_q * (pred[..., 0] - y[:, window:]).pow(2)).sum() / m_q.sum().clamp(min=1)
    opt.zero_grad()
    loss.backward()
    opt.step()


@torch.no_grad()
def online_error(
  nominal: PlanarModel,
  features,
  data: tuple[torch.Tensor, ...],
  warmup: int,
) -> torch.Tensor:
  """RMS one-step error per axis of the filter's prediction, after `warmup` steps."""
  vel, cmd, vel_next, ok = data
  model = ResidualModel(nominal, features, ResidualCfg(), num_envs=vel.shape[1])
  sq, count = torch.zeros(3, device=vel.device), torch.zeros((), device=vel.device)
  for t in range(vel.shape[0]):
    if t >= warmup:
      miss = vel_next[t] - model.next_velocity(vel[t], cmd[t])
      sq += (miss.pow(2) * ok[t, :, None]).sum(dim=0)
      count += ok[t].sum()
    model.update(vel[t], cmd[t], vel_next[t])
  return (sq / count).sqrt()


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  p.add_argument("--task", default=TASK_ID)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--duration", type=float, default=40.0, help="Seconds per robot.")
  p.add_argument("--num-coef", type=int, default=6)
  p.add_argument("--steps", type=int, default=3000, help="Training steps.")
  p.add_argument("--cmd-min", type=float, nargs=3, default=CMD_MIN)
  p.add_argument("--cmd-max", type=float, nargs=3, default=CMD_MAX)
  p.add_argument("--out", type=Path, default=None)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  torch.manual_seed(args.seed)
  dev = args.device
  lo, hi = (
    torch.tensor(args.cmd_min, device=dev),
    torch.tensor(args.cmd_max, device=dev),
  )
  nominal = PlanarModel.load(args.checkpoint.parent / "planar_model.json", dev)

  runs = {}
  for name, held_out in (("train", False), ("held-out joint", True)):
    tracker = Tracker(
      make_tracker_cfg(args.num_envs, args.task),
      dev,
      checkpoint=args.checkpoint,
      task_id=args.task,
    )
    names = list(tracker.robot.joint_names)
    cond = draw_conditions(args.num_envs, names, held_out, dev)
    hold = round(nominal.dt / tracker.step_dt)
    windows = round(args.duration / nominal.dt)
    runs[name] = transitions(*collect(tracker, cond, windows, lo, hi, hold), nominal.dt)
    tracker.close()

  # Robots 0 to 80% train F; the rest of the same conditions are a held-out test.
  data = runs["train"]
  split = int(0.8 * args.num_envs)
  fit = tuple(x[:, :split] for x in data)
  test = {
    "held-out robots": tuple(x[:, split:] for x in data),
    "held-out joint": runs["held-out joint"],
  }
  template = ResidualModel(nominal, FEATURE_SETS["offset"])
  features = LearnedFeatures(args.num_coef).to(dev)
  train(template, features, fit, args.steps)
  features.eval()

  warmup = round(2.0 / nominal.dt)
  print(
    f"\n[features] {args.num_envs} robots per set x {args.duration:.0f} s,"
    f" commands {args.cmd_min} to {args.cmd_max}."
  )
  print(
    "[features] One-step RMS error of the online model after 2 s"
    " (vx m/s, vy m/s, wz rad/s):"
  )
  report: dict[str, object] = {}
  candidates = {
    "nominal only": None,
    **{f"hand, {k}": f for k, f in FEATURE_SETS.items()},
    f"learned, {args.num_coef} coefficients": features,
  }
  for test_name, test_data in test.items():
    print(f"  {test_name}:")
    for name, feats in candidates.items():
      if feats is None:
        vel, cmd, vel_next, ok = test_data
        miss = (vel_next - nominal.next_velocity(vel, cmd))[warmup:]
        valid = ok[warmup:, :, None].float()
        rms = ((miss.pow(2) * valid).sum(dim=(0, 1)) / valid.sum()).sqrt()
      else:
        rms = online_error(nominal, feats, test_data, warmup)
      print(f"    {name:28s} {rms.cpu().numpy().round(3)}")
      report[f"{test_name}: {name}"] = rms.tolist()

  out = args.out or args.checkpoint.parent / "planar_features.pt"
  features.save(out, checkpoint=str(args.checkpoint), report=report)
  print(f"\n[features] Saved the features to {out}")


if __name__ == "__main__":
  main()
