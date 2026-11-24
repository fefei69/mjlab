"""Fit the per-joint nominal model and set the fault score's threshold.

Drives many copies of the policy with random commands. Half of the robots stay
healthy: most of them train the nominal model, the rest set the threshold so
that only a few healthy robots ever cross it. The other half get a motor fault on
a random joint partway through, which measures how fast and how often the score
catches it.

  uv run python -m mppi.fit_joint_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mjlab.tasks.velocity.mdp.joint_residual import JointResidual, fit_joint_model
from mppi.fit_planar_model import sample_commands
from mppi.tracker import (
  CMD_MAX_FAST,
  CMD_MIN_FAST,
  DEFAULT_CHECKPOINT,
  TASK_ID,
  Tracker,
  make_tracker_cfg,
)


def collect(
  tracker: Tracker,
  steps: int,
  lo: torch.Tensor,
  hi: torch.Tensor,
  fault_at: int,
  joint: torch.Tensor,
  strength: torch.Tensor,
  standing: float = 0.15,
) -> dict[str, torch.Tensor]:
  """Run every robot for `steps` policy steps; robots with strength < 1 fail.

  A fraction `standing` of the commands is zero.
  """
  n, device = tracker.num_envs, tracker.device
  robot = tracker.robot
  cmd = sample_commands(n, lo, hi)
  left = torch.zeros(n, dtype=torch.long, device=device)
  hold = round(0.1 / tracker.step_dt)
  q, qd, target, upright = [], [], [], []
  for t in range(steps):
    if t == fault_at:
      tracker.set_motor_strength(joint, strength)
    if t % hold == 0:
      # Each command is held for 0.2 to 3 s, as in fit_planar_model.
      expired = left <= 0
      # Some robots stand still, as in training.
      new = sample_commands(n, lo, hi) * (torch.rand(n, 1, device=device) > standing)
      cmd = torch.where(expired[:, None], new, cmd)
      new_left = torch.randint(2, 31, (n,), device=device)
      left = torch.where(expired, new_left, left) - 1
    q.append(robot.data.joint_pos.clone())
    qd.append(robot.data.joint_vel.clone())
    upright.append(tracker.upright() > 0.5)
    tracker.step(cmd)
    target.append(robot.data.joint_pos_target.clone())
  qd.append(robot.data.joint_vel.clone())
  ok = torch.stack(upright).cumprod(dim=0).bool()  # Fallen robots stay excluded.
  return {
    "q": torch.stack(q),
    "qd": torch.stack(qd[:-1]),
    "qd_next": torch.stack(qd[1:]),
    "target": torch.stack(target),
    "ok": ok,
  }


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  p.add_argument("--task", default=TASK_ID, help="Task the checkpoint was trained on.")
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--duration", type=float, default=20.0, help="Seconds per robot.")
  p.add_argument("--fault-at", type=float, default=8.0)
  p.add_argument("--cmd-min", type=float, nargs=3, default=CMD_MIN_FAST)
  p.add_argument("--cmd-max", type=float, nargs=3, default=CMD_MAX_FAST)
  p.add_argument(
    "--healthy-alarms",
    type=float,
    default=0.05,
    help="Fraction of healthy robots allowed to cross the threshold in a run.",
  )
  p.add_argument("--tau", type=float, default=0.5, help="Score averaging, in s.")
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

  tracker = Tracker(
    make_tracker_cfg(args.num_envs, args.task),
    dev,
    checkpoint=args.checkpoint,
    task_id=args.task,
  )
  n, dt = args.num_envs, tracker.step_dt
  names = list(tracker.robot.joint_names)
  num_joints = len(names)
  faulty = torch.arange(n, device=dev) >= n // 2
  joint = torch.randint(num_joints, (n,), device=dev)
  strength = torch.where(faulty, 0.5 * torch.rand(n, device=dev), 1.0)
  steps, fault_at = round(args.duration / dt), round(args.fault_at / dt)
  data = collect(tracker, steps, lo, hi, fault_at, joint, strength)
  tracker.close()

  healthy = (~faulty).nonzero().squeeze(-1)
  fit_ids, val_ids = (
    healthy[: int(0.7 * len(healthy))],
    healthy[int(0.7 * len(healthy)) :],
  )
  warm = round(1.0 / dt)

  def samples(ids: torch.Tensor) -> list[torch.Tensor]:
    keep = data["ok"][warm:, ids]
    return [data[k][warm:, ids][keep] for k in ("q", "qd", "target", "qd_next")]

  model = fit_joint_model(*samples(fit_ids), names=names)
  model.tau.fill_(args.tau)
  q, qd, target, qd_next = samples(val_ids)
  with torch.no_grad():
    y = qd_next - qd
    r2 = 1 - (model(q, qd, target) - y).var(dim=0) / y.var(dim=0)

  # Run the score over every robot, from the start.
  residual = JointResidual(model, n, dt)
  scores = []
  for t in range(steps):
    residual.update(*(data[k][t] for k in ("q", "qd", "target", "qd_next")))
    scores.append(residual.score.clone())
  score = torch.stack(scores)  # (T, N, J)
  score = torch.where(data["ok"][..., None], score, 0.0)

  peak = score[warm:, val_ids].amax(dim=(0, 2))
  threshold = torch.quantile(peak, 1 - args.healthy_alarms).item()
  model.threshold.fill_(threshold)

  print(
    f"\n[joint] {n} robots x {args.duration:.0f} s, commands {args.cmd_min}"
    f" to {args.cmd_max}."
  )
  print(
    f"[joint] Nominal model R2 on held-out healthy robots, per joint: "
    f"{r2.cpu().numpy().round(2)}"
  )
  print(
    f"[joint] Threshold {threshold:.2f}: {100 * args.healthy_alarms:.0f}% of"
    f" healthy robots cross it within {args.duration:.0f} s."
  )
  print("[joint] Motor faults after the change (detected, median delay, right joint):")
  report: dict[str, object] = {"threshold": threshold, "r2": r2.tolist()}
  ids = faulty.nonzero().squeeze(-1)
  after = score[fault_at:, ids]  # (T', F, J)
  above = (after > threshold).any(dim=-1)
  found = above.any(dim=0)
  first = above.float().argmax(dim=0)
  flagged = after[first, torch.arange(len(ids), device=dev)].argmax(dim=-1)
  right = flagged == joint[ids]
  for lo_s, hi_s in ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5)):
    band = (strength[ids] >= lo_s) & (strength[ids] < hi_s)
    hit = found & band
    delay = (first[hit].float() * dt).median().item() if hit.any() else float("nan")
    rate = found[band].float().mean().item()
    right_rate = right[hit].float().mean().item() if hit.any() else float("nan")
    print(
      f"  torque {lo_s:.2f} to {hi_s:.2f}: {100 * rate:.0f}% detected, median"
      f" {delay:.2f} s, right joint {100 * right_rate:.0f}%"
    )
    report[f"torque_{lo_s}_{hi_s}"] = {
      "detected": rate,
      "median_delay_s": delay,
      "right_joint": right_rate,
    }
  out = args.out or args.checkpoint.parent / "joint_model.pt"
  model.save(out, checkpoint=str(args.checkpoint), report=report)
  print(f"\n[joint] Saved the model to {out}")


if __name__ == "__main__":
  main()
