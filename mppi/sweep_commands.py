"""Measure what the policy does with commands outside its trained ranges.

One robot per command, on a grid of forward speed and yaw rate. Each command is
ramped in over 2 s and then held. The script reports the velocity each robot
actually reached and which robots fell. The result is what sets the command
limits MPPI may use (CMD_MAX_FAST in tracker.py).

  uv run python -m mppi.sweep_commands
  uv run python -m mppi.sweep_commands --vy 1.0   # With a sideways command added.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from mppi.planar_model import body_velocity
from mppi.tracker import DEFAULT_CHECKPOINT, Tracker, make_tracker_cfg


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  p.add_argument("--vx-max", type=float, default=4.0, help="Highest forward command.")
  p.add_argument("--wz-max", type=float, default=2.0, help="Highest yaw-rate command.")
  p.add_argument("--vy", type=float, default=0.0, help="Sideways command for all.")
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  vx = torch.linspace(0.0, args.vx_max, 17)
  wz = torch.linspace(0.0, args.wz_max, 11)
  grid = torch.cartesian_prod(vx, wz)  # (N, 2), forward speed varying slowest.
  commands = torch.stack(
    [grid[:, 0], torch.full((len(grid),), args.vy), grid[:, 1]], dim=-1
  ).to(args.device)
  tracker = Tracker(
    make_tracker_cfg(len(commands)), args.device, checkpoint=args.checkpoint
  )

  # Ramp for 2 s, hold for 6 s, and average the velocity over the last 3 s.
  dt = 0.1
  hold = round(dt / tracker.step_dt)
  ramp_steps, total_steps, averaged_steps = 20, 80, 30
  upright = torch.ones(len(commands), dtype=torch.bool, device=args.device)
  vels = []
  prev = tracker.pose()
  for k in range(total_steps):
    tracker.step(commands * min(1.0, (k + 1) / ramp_steps), hold=hold)
    pose = tracker.pose()
    vels.append(body_velocity(prev, pose, dt))
    prev = pose
    upright &= tracker.upright() > 0.5
  tracker.close()

  vel = torch.stack(vels[-averaged_steps:]).mean(dim=0).cpu()
  speed = vel[:, :2].norm(dim=-1).reshape(len(vx), len(wz))
  yaw_rate = vel[:, 2].reshape(len(vx), len(wz))
  ok = upright.cpu().reshape(len(vx), len(wz))

  def table(title: str, values: torch.Tensor) -> None:
    print(f"\n[sweep] {title}. Rows: forward command (m/s). Columns: yaw command.")
    print("        " + "".join(f"{w:6.1f}" for w in wz))
    for i, x in enumerate(vx):
      cells = [f"{values[i, j]:6.2f}" if ok[i, j] else "  fell" for j in range(len(wz))]
      print(f"  {x:5.2f} " + "".join(cells))

  print(f"\n[sweep] Sideways command {args.vy} m/s. {int((~ok).sum())} robots fell.")
  table("Speed reached (m/s)", speed)
  table("Yaw rate reached (rad/s)", yaw_rate)

  print("\n[sweep] Fastest steady circling found, by circle radius:")
  turn_radius = speed / yaw_rate.clamp(min=1e-3)
  for radius in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
    near = ok & ((turn_radius - radius).abs() < 0.1 * radius)
    if not near.any():
      print(f"  {radius:.1f} m: no command on the grid turns at this radius")
      continue
    best = int(torch.where(near, speed, torch.zeros_like(speed)).argmax())
    i, j = divmod(best, len(wz))
    print(
      f"  {radius:.1f} m: {speed[i, j]:.2f} m/s, one lap in"
      f" {2 * math.pi / yaw_rate[i, j]:.1f} s, with command vx {vx[i]:.2f},"
      f" wz {wz[j]:.1f}"
    )


if __name__ == "__main__":
  main()
