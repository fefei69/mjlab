"""Fit the planar motion model to the trained policy.

Drives many copies of the policy with random, piecewise-constant velocity
commands and records how the base actually moves. It then fits
`v_next = A v + B u + c` by least squares and reports how well the fit predicts
held-out motion, next to the naive model that assumes the robot does exactly
what it is told.

  uv run python -m mppi.fit_planar_model
  uv run python -m mppi.fit_planar_model --cmd-max 2.0 1.0 0.7   # Trained range only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mppi.planar_model import PlanarModel, body_velocity, wrap_angle
from mppi.tracker import (
  CMD_MAX_FAST,
  CMD_MIN_FAST,
  DEFAULT_CHECKPOINT,
  TASK_ID,
  Tracker,
  make_tracker_cfg,
)


def sample_commands(num: int, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
  """Random commands between `lo` and `hi`, with many near-pure motions."""
  device = lo.device
  cmd = lo + (hi - lo) * torch.rand(num, 3, device=device)
  # Zero each axis 30% of the time so pure forward, sideways and turning motions
  # are well covered, and shrink half of the commands to cover low speeds.
  cmd = cmd * (torch.rand(num, 3, device=device) > 0.3)
  shrink = torch.where(
    torch.rand(num, 1, device=device) < 0.5,
    torch.rand(num, 1, device=device),
    torch.ones(num, 1, device=device),
  )
  return cmd * shrink


def collect(
  tracker: Tracker, windows: int, hold: int, lo: torch.Tensor, hi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Returns poses (W+1, N, 3), commands (W, N, 3) and upright flags (W+1, N)."""
  n, device = tracker.num_envs, tracker.device
  cmd = sample_commands(n, lo, hi)
  left = torch.zeros(n, dtype=torch.long, device=device)
  poses, cmds = [tracker.pose()], []
  ok = [tracker.upright() > 0.5]
  for _ in range(windows):
    # Each command is held for 0.2 to 3 s (2 to 30 windows of 0.1 s).
    expired = left <= 0
    cmd = torch.where(expired[:, None], sample_commands(n, lo, hi), cmd)
    left = torch.where(expired, torch.randint(2, 31, (n,), device=device), left) - 1
    tracker.step(cmd, hold=hold)
    cmds.append(cmd.clone())
    poses.append(tracker.pose())
    # A robot that fell stays excluded for the rest of the run.
    ok.append(ok[-1] & (tracker.upright() > 0.5))
  return torch.stack(poses), torch.stack(cmds), torch.stack(ok)


def multi_step_error(
  model: PlanarModel,
  poses: torch.Tensor,
  cmds: torch.Tensor,
  vels: torch.Tensor,
  ok: torch.Tensor,
  horizon: int,
) -> tuple[float, float, float]:
  """Open-loop prediction error after `horizon` windows.

  Returns the mean position error in m, the mean heading error in rad, and the
  mean distance the robot actually travelled, for scale.
  """
  pos_err, yaw_err, dist = [], [], []
  for s in range(1, cmds.shape[0] - horizon, horizon):
    valid = ok[s + horizon]
    pred, _ = model.rollout(
      poses[s], vels[s - 1], cmds[s : s + horizon].transpose(0, 1)
    )
    end, true = pred[:, -1], poses[s + horizon]
    pos_err.append((end[:, :2] - true[:, :2]).norm(dim=-1)[valid])
    yaw_err.append(wrap_angle(end[:, 2] - true[:, 2]).abs()[valid])
    dist.append((true[:, :2] - poses[s][:, :2]).norm(dim=-1)[valid])
  return (
    torch.cat(pos_err).mean().item(),
    torch.cat(yaw_err).mean().item(),
    torch.cat(dist).mean().item(),
  )


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  p.add_argument("--task", default=TASK_ID, help="Task the checkpoint was trained on.")
  p.add_argument("--num-envs", type=int, default=512)
  p.add_argument("--duration", type=float, default=60.0, help="Seconds per env.")
  p.add_argument("--plan-dt", type=float, default=0.1, help="Model step in seconds.")
  p.add_argument(
    "--cmd-min", type=float, nargs=3, default=CMD_MIN_FAST, help="Lowest vx vy wz."
  )
  p.add_argument(
    "--cmd-max", type=float, nargs=3, default=CMD_MAX_FAST, help="Highest vx vy wz."
  )
  p.add_argument(
    "--linear", action="store_true", help="Fit without the speed-dependent terms."
  )
  p.add_argument("--out", type=Path, default=None)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  torch.manual_seed(args.seed)
  cmd_min, cmd_max = tuple(args.cmd_min), tuple(args.cmd_max)
  lo = torch.tensor(cmd_min, device=args.device)
  hi = torch.tensor(cmd_max, device=args.device)

  tracker = Tracker(
    make_tracker_cfg(args.num_envs, args.task),
    args.device,
    checkpoint=args.checkpoint,
    task_id=args.task,
  )
  hold = round(args.plan_dt / tracker.step_dt)
  windows = round(args.duration / args.plan_dt)
  poses, cmds, ok = collect(tracker, windows, hold, lo, hi)
  tracker.close()

  # vels[k] is the velocity over window k, which ran under cmds[k].
  vels = body_velocity(poses[:-1], poses[1:], args.plan_dt)
  n_train = int(0.8 * args.num_envs)
  train, test = slice(0, n_train), slice(n_train, None)

  def samples(envs: slice) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = ok[2:, envs]
    return (
      vels[:-1, envs][mask],
      cmds[1:, envs][mask],
      vels[1:, envs][mask],
    )

  model = PlanarModel.fit(
    *samples(train),
    dt=args.plan_dt,
    linear=args.linear,
    cmd_min=cmd_min,
    cmd_max=cmd_max,
  ).to(args.device)
  ideal = PlanarModel.ideal(args.plan_dt, args.device)

  v, u, v_next = samples(test)
  fell = int((~ok[-1]).sum())
  print(f"\n[fit] Commands from {cmd_min} to {cmd_max} (vx, vy, wz).")
  print(f"[fit] {args.num_envs} robots x {args.duration:.0f} s, {fell} fell.")
  print(f"[fit] {v.shape[0]} held-out samples from {args.num_envs - n_train} robots.")
  torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)
  print("[fit] A (dependence on current velocity):\n", model.a.cpu())
  print("[fit] B (dependence on command):\n", model.b.cpu())
  print("[fit] c (offset):", model.c.cpu())

  report: dict[str, object] = {"robots_fallen": fell}
  print("\n[fit] One-step velocity RMSE on held-out data (vx m/s, vy m/s, wz rad/s):")
  for name, m in (("fitted", model), ("ideal", ideal)):
    rmse = (m.next_velocity(v, u) - v_next).pow(2).mean(dim=0).sqrt()
    print(f"  {name:7s} {rmse.cpu().numpy().round(3)}")
    report[f"one_step_rmse_{name}"] = rmse.tolist()

  print("\n[fit] Open-loop prediction error on held-out data:")
  for seconds in (1.0, 2.0):
    horizon = round(seconds / args.plan_dt)
    for name, m in (("fitted", model), ("ideal", ideal)):
      pos, yaw, dist = multi_step_error(
        m, poses[:, test], cmds[:, test], vels[:, test], ok[:, test], horizon
      )
      print(
        f"  {seconds:.0f} s ahead, {name:7s} position {pos:.3f} m, heading"
        f" {yaw:.3f} rad (robot travelled {dist:.2f} m on average)"
      )
      report[f"open_loop_{seconds:.0f}s_{name}"] = {"pos_m": pos, "yaw_rad": yaw}

  out = args.out or args.checkpoint.parent / "planar_model.json"
  model.save(out, checkpoint=str(args.checkpoint), report=report)
  print(f"\n[fit] Saved the model to {out}")


if __name__ == "__main__":
  main()
