"""Demo: race around a circle. MPPI plans velocity commands, the RL policy tracks.

MPPI scores candidate command sequences with the planar motion model, which is
cheap enough to re-plan every 0.1 s. Fit the model once per checkpoint first.
MPPI may command anything inside the range the model was fitted on, which by
default is wider than the range the policy was trained on.

  uv run python -m mppi.fit_planar_model
  uv run python -m mppi.demo_circle_race
  uv run python -m mppi.demo_circle_race --model ideal   # Naive model, to compare.
  # Stay inside the trained command range (slower):
  uv run python -m mppi.demo_circle_race --cmd-min -1.5 -1 -0.7 --cmd-max 2 1 0.7

A steady pull on the trunk makes the fitted model wrong. `--residual` lets the
planner correct its model online from how the robot actually moves:

  uv run python -m mppi.demo_circle_race --pull 0 -30                    # Fixed model.
  uv run python -m mppi.demo_circle_race --pull 0 -30 --residual command

`--fault` weakens one or more motors mid-run instead, for example the front right
knee at 30% of its torque:

  uv run python -m mppi.demo_circle_race --fault FR_calf_joint --residual command
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from functools import partial
from pathlib import Path
from typing import Any

import mediapy as media
import numpy as np
import torch

from mjlab.viewer import ViewerConfig
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mppi.core import Mppi, MppiCfg
from mppi.planar_model import PlanarModel, body_velocity, wrap_angle
from mppi.residual_model import FEATURE_SETS, ResidualCfg, ResidualModel
from mppi.tracker import (
  CMD_MAX_FAST,
  CMD_MIN_FAST,
  DEFAULT_CHECKPOINT,
  Tracker,
  make_tracker_cfg,
)
from mppi.viz import PLAN_COLOR, TRAIL_COLOR, Hud, Rgba, draw_path, draw_rollouts

TRACK_COLOR = (0.95, 0.95, 0.95, 1.0)
PULL_COLOR = (1.0, 0.33, 0.62, 1.0)
PULL_ARROW = 0.035  # Metres of arrow per newton of pull.


def race_cost(
  poses: torch.Tensor,
  commands: torch.Tensor,
  pose0: torch.Tensor,
  radius: float,
  half_width: float,
  facing_weight: float,
  centre_weight: float,
) -> torch.Tensor:
  """Cost of predicted poses (K, H, 3) on a counter-clockwise circular track.

  Rewards distance covered along the circle. Penalizes leaving the track, not
  facing along the track, jerky commands and, optionally, leaving the centre line.
  """
  x, y, yaw = poses.unbind(dim=-1)
  angle = torch.atan2(y, x)
  angle0 = torch.atan2(pose0[1], pose0[0]).expand(angle.shape[0], 1)
  steps = wrap_angle(torch.diff(torch.cat([angle0, angle], dim=1), dim=1))
  progress = steps.sum(dim=1) * radius  # Metres along the track.

  off_centre = torch.hypot(x, y) - radius
  off_track = torch.relu(off_centre.abs() - half_width)
  facing = 1.0 - torch.cos(yaw - (angle + 0.5 * math.pi))
  jerk = torch.diff(commands, dim=1).pow(2).sum(dim=-1)
  return (
    -progress
    + centre_weight * off_centre.pow(2).mean(dim=1)
    + 200.0 * off_track.pow(2).mean(dim=1)
    + facing_weight * facing.mean(dim=1)
    + 0.2 * jerk.mean(dim=1)
  )


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  p.add_argument("--model", choices=("fitted", "ideal"), default="fitted")
  p.add_argument(
    "--model-file",
    type=Path,
    default=None,
    help="Fitted planar model. Default: planar_model.json next to the checkpoint.",
  )
  p.add_argument(
    "--cmd-min",
    type=float,
    nargs=3,
    default=None,
    help="Lowest vx vy wz MPPI may command. Default: the range the model was fitted on.",
  )
  p.add_argument("--cmd-max", type=float, nargs=3, default=None)
  p.add_argument("--no-video", action="store_true", help="Skip rendering.")
  p.add_argument("--radius", type=float, default=2.5, help="Track radius in m.")
  p.add_argument("--half-width", type=float, default=0.4, help="Track half-width.")
  p.add_argument("--duration", type=float, default=30.0, help="Seconds to race.")
  p.add_argument("--horizon", type=float, default=1.0, help="Planning horizon in s.")
  p.add_argument("--samples", type=int, default=2048)
  p.add_argument("--iterations", type=int, default=4, help="MPPI rounds per plan.")
  p.add_argument("--temperature", type=float, default=0.1)
  p.add_argument(
    "--facing-weight",
    type=float,
    default=10.0,
    help="Cost on not facing along the track. 0 lets the robot run sideways.",
  )
  p.add_argument(
    "--centre-weight",
    type=float,
    default=0.0,
    help="Cost on leaving the centre line. 0 lets the robot use the whole track.",
  )
  p.add_argument(
    "--rollouts-shown", type=int, default=64, help="Sampled plans drawn in the video."
  )
  p.add_argument(
    "--pull",
    type=float,
    nargs=2,
    default=None,
    metavar=("FORWARD", "LEFT"),
    help="Steady force on the trunk in N, forward and to the left of the robot.",
  )
  p.add_argument(
    "--fault",
    default=None,
    metavar="JOINTS",
    help="Weaken these motors, a regular expression on joint names.",
  )
  p.add_argument(
    "--fault-torque", type=float, default=0.3, help="Torque limit left, 0 to 1."
  )
  p.add_argument("--fault-gain", type=float, default=1.0, help="PD gains left, 0 to 1.")
  p.add_argument(
    "--change-at",
    "--pull-at",
    type=float,
    default=8.0,
    help="When the pull or the fault starts, in s.",
  )
  p.add_argument(
    "--residual",
    choices=("off", *FEATURE_SETS),
    default="off",
    help="Correct the model online from the robot's motion, with these features.",
  )
  p.add_argument("--drift-std", type=float, default=ResidualCfg.drift_std)
  p.add_argument("--leak", type=float, default=ResidualCfg.leak)
  p.add_argument(
    "--gain-drift-ratio",
    type=float,
    default=ResidualCfg.gain_drift_ratio,
    help="How fast gains may change, relative to offsets. 1 treats them alike.",
  )
  p.add_argument("--out", type=Path, default=Path("logs/mppi/circle_race.mp4"))
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  torch.manual_seed(args.seed)
  radius, half_width = args.radius, args.half_width

  if args.model == "fitted":
    model_path = args.model_file or args.checkpoint.parent / "planar_model.json"
    if not model_path.exists():
      raise SystemExit(
        f"{model_path} not found. Run `uv run python -m mppi.fit_planar_model`."
      )
    model = PlanarModel.load(model_path, args.device)
  else:
    model = PlanarModel.ideal(0.1, args.device)
  # The model the planner uses: the fitted one, or the fitted one plus a residual
  # that is re-fitted during the run.
  residual: ResidualModel | None = None
  if args.residual != "off":
    residual = ResidualModel(
      model,
      FEATURE_SETS[args.residual],
      ResidualCfg(
        drift_std=args.drift_std,
        leak=args.leak,
        gain_drift_ratio=args.gain_drift_ratio,
      ),
    )
  plan_model: PlanarModel | ResidualModel = residual or model
  pull = torch.tensor(args.pull or (0.0, 0.0), device=args.device)
  pulling = False
  faulted: list[str] = []
  fault_text = (
    f"{args.fault}: torque {args.fault_torque:.0%}, gains {args.fault_gain:.0%}"
    if args.fault
    else ""
  )
  pull_text = ", ".join(
    f"{abs(size):.0f} N {positive if size > 0 else negative}"
    for size, positive, negative in (
      (float(pull[0]), "forward", "backward"),
      (float(pull[1]), "to the left", "to the right"),
    )
    if size != 0
  )

  # Start on the track, facing along it, and watch from a fixed overhead camera.
  cfg = make_tracker_cfg(num_envs=1)
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.init_state.pos = (radius, 0.0, robot_cfg.init_state.pos[2])
  robot_cfg.init_state.rot = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    # Look a little to the side so the track clears the text panel.
    lookat=(-0.55 * radius, 0.0, 0.0),
    distance=2.5 * radius + 1.5,
    elevation=-50.0,
    azimuth=90.0,
    width=1280,
    height=720,
  )
  tracker = Tracker(
    cfg, args.device, checkpoint=args.checkpoint, render=not args.no_video
  )
  hold = round(model.dt / tracker.step_dt)

  # MPPI stays inside the command range the model was fitted on.
  cmd_min = tuple(args.cmd_min or model.cmd_min or CMD_MIN_FAST)
  cmd_max = tuple(args.cmd_max or model.cmd_max or CMD_MAX_FAST)

  mppi = Mppi(
    MppiCfg(
      horizon=round(args.horizon / model.dt),
      num_samples=args.samples,
      u_min=cmd_min,
      u_max=cmd_max,
      noise_sigma=tuple(
        0.15 * (hi - lo) for lo, hi in zip(cmd_min, cmd_max, strict=True)
      ),
      temperature=args.temperature,
      iterations=args.iterations,
    ),
    args.device,
  )

  # Overlay: track edges, the path driven so far, the sampled plans colored by
  # cost, and the chosen plan on top.
  ring = np.linspace(0.0, 2.0 * math.pi, 90, endpoint=False)
  edges = [
    np.stack([r * np.cos(ring), r * np.sin(ring), np.full_like(ring, 0.01)], axis=1)
    for r in (radius - half_width, radius + half_width)
  ]
  shown = torch.linspace(0, args.samples - 1, args.rollouts_shown).long()
  fan_xy = np.zeros((0, 0, 2))
  fan_costs = np.zeros(0)
  plan_xy = np.zeros((0, 2))
  trail: list[np.ndarray] = []

  def draw(vis: DebugVisualizer) -> None:
    for edge in edges:
      for point in edge:
        vis.add_sphere(point, 0.025, TRACK_COLOR)
    if len(trail) > 1:
      draw_path(vis, np.stack(trail[-50:]), 0.008, 0.008, TRAIL_COLOR)
    if len(fan_costs) > 0:
      draw_rollouts(vis, fan_xy, fan_costs, 0.02, 0.006)
    if len(plan_xy) > 0:
      draw_path(vis, plan_xy, 0.04, 0.016, PLAN_COLOR)
    if pulling:
      # Read the pose now: the robot moves several frames per plan step.
      x, y, yaw = tracker.pose()[0].tolist()
      forward, left = (PULL_ARROW * pull).tolist()
      tip = (
        x + math.cos(yaw) * forward - math.sin(yaw) * left,
        y + math.sin(yaw) * forward + math.cos(yaw) * left,
      )
      vis.add_arrow(
        np.array([x, y, 0.45]), np.array([*tip, 0.45]), PULL_COLOR, width=0.04
      )

  tracker.set_overlay(draw)
  legend: list[tuple[str, Rgba | str]] = [
    ("Sampled plans (light = low cost)", "ramp"),
    ("Chosen plan", PLAN_COLOR),
    ("Path driven", TRAIL_COLOR),
  ]
  title = "MPPI circle race"
  if args.pull is not None:
    legend.append(("Pull on the trunk", PULL_COLOR))
    title = "Model corrected online" if residual is not None else "Fixed model"
  hud = Hud(
    title,
    f"Planar model · {args.samples} samples · {args.horizon:.1f} s horizon",
    legend,
  )

  # The last batch of sampled plans that MPPI scored, kept for drawing.
  scored: dict[str, torch.Tensor] = {}

  def plan_cost(
    commands: torch.Tensor, pose: torch.Tensor, vel: torch.Tensor
  ) -> torch.Tensor:
    poses, _ = plan_model.rollout(pose, vel, commands)
    costs = race_cost(
      poses,
      commands,
      pose,
      radius,
      half_width,
      args.facing_weight,
      args.centre_weight,
    )
    scored["xy"], scored["costs"] = poses[..., :2], costs
    return costs

  pose = tracker.pose()[0]
  prev_pose = pose.clone()
  vel = torch.zeros(3, device=args.device)
  command = torch.zeros(3, device=args.device)
  log: dict[str, list[np.ndarray]] = {
    "pose": [],
    "cmd": [],
    "vel": [],
    "model_error": [],
    "nominal_error": [],
    "correction": [],
    "changed": [],
  }
  plan_ms: list[float] = []
  speeds: list[float] = []
  laps: list[float] = []
  travelled = 0.0  # Radians around the circle.
  fell_at: float | None = None

  args.out.parent.mkdir(parents=True, exist_ok=True)
  fps = round(1.0 / tracker.step_dt)
  with contextlib.ExitStack() as stack:
    video = None
    if not args.no_video:
      video = stack.enter_context(
        media.VideoWriter(args.out, shape=(720, 1280), fps=fps)
      )
    for k in range(round(args.duration / model.dt)):
      prev_vel, vel = vel, body_velocity(prev_pose, pose, model.dt)
      # How wrong the planner's model was about the step just taken. Taken
      # before the residual sees that step, so it is a true prediction error.
      # The nominal model's error is what the residual is fitted to, and the
      # correction is the residual's prediction of it.
      model_error = torch.zeros_like(vel)
      nominal_error = torch.zeros_like(vel)
      correction = torch.zeros_like(vel)
      if k > 0:
        model_error = vel - plan_model.next_velocity(prev_vel, command)
        nominal_error = vel - model.next_velocity(prev_vel, command)
        if residual is not None:
          correction = residual.residual(prev_vel, command)
          residual.update(prev_vel, command, vel)
      log["model_error"].append(model_error.cpu().numpy())
      log["nominal_error"].append(nominal_error.cpu().numpy())
      log["correction"].append(correction.cpu().numpy())
      changed = k * model.dt >= args.change_at
      pulling = args.pull is not None and changed
      if args.fault is not None and changed and not faulted:
        faulted = tracker.weaken_joints(args.fault, args.fault_torque, args.fault_gain)
      has_change = args.pull is not None or args.fault is not None
      log["changed"].append(np.array(changed and has_change))

      torch.cuda.synchronize()
      t0 = time.perf_counter()
      plan = mppi.plan(partial(plan_cost, pose=pose, vel=vel))
      torch.cuda.synchronize()
      plan_ms.append(1e3 * (time.perf_counter() - t0))
      command = plan[0].clone()
      mppi.shift()

      # Paths to draw, each starting at the robot.
      start = pose[:2]
      plan_poses = plan_model.rollout(pose, vel, plan[None])[0][0, :, :2]
      plan_xy = torch.cat([start[None], plan_poses]).cpu().numpy()
      fan = scored["xy"][shown]
      fan_xy = torch.cat([start.expand(len(shown), 1, 2), fan], dim=1).cpu().numpy()
      fan_costs = scored["costs"][shown].cpu().numpy()

      speeds.append(float(vel[:2].norm()))
      speed = float(np.mean(speeds[-5:]))  # Averaged over 0.5 s to hide the gait.
      lap_text = f"{len(laps) + 1}"
      if laps:
        lap_text += f"  (last lap {np.diff([0.0, *laps])[-1]:.1f} s)"
      readouts = [
        ("Time", f"{k * model.dt:.1f} s"),
        ("Speed", f"{speed:.2f} m/s"),
        ("Lap", lap_text),
        (
          "Command",
          f"vx {command[0]:+.2f}  vy {command[1]:+.2f}  wz {command[2]:+.2f}",
        ),
      ]
      if args.pull is not None or args.fault is not None or residual is not None:
        if args.pull is not None:
          readouts.append(("Pull", pull_text if pulling else "none yet"))
        if args.fault is not None:
          readouts.append(("Fault", fault_text if faulted else "none yet"))
        # Both averaged over 1 s. The model error covers forward and sideways.
        recent = np.stack(log["model_error"][-10:])[:, :2]
        off = [float(np.hypot(*p[:2])) - radius for p in log["pose"][-10:]]
        readouts += [
          ("Correction", f"{args.residual} features" if residual else "none"),
          ("Off centre", f"{np.mean(off):+.2f} m" if off else ""),
          ("Model error", f"{np.sqrt((recent**2).sum(axis=1).mean()):.2f} m/s"),
        ]
      for _ in range(hold):
        if pulling:
          tracker.pull_trunk(pull)
        tracker.step(command)
        if video is not None:
          video.add_image(hud.draw(tracker.render(), readouts))

      prev_pose, pose = pose, tracker.pose()[0]
      trail.append(pose[:2].cpu().numpy())
      log["pose"].append(pose.cpu().numpy())
      log["cmd"].append(command.cpu().numpy())
      log["vel"].append(vel.cpu().numpy())
      before = travelled
      travelled += wrap_angle(
        torch.atan2(pose[1], pose[0]) - torch.atan2(prev_pose[1], prev_pose[0])
      ).item()
      # A small step backward at the start must not count as a lap.
      if travelled // (2 * math.pi) > max(before, 0.0) // (2 * math.pi):
        laps.append((k + 1) * model.dt)
      if tracker.upright()[0] < 0.5:
        fell_at = (k + 1) * model.dt
        break

  tracker.close()
  arrays = {name: np.stack(values) for name, values in log.items()}
  payload: dict[str, Any] = {"dt": model.dt, **arrays}
  np.savez(args.out.with_suffix(".npz"), **payload)

  seconds = len(log["pose"]) * model.dt
  off_centre = np.hypot(arrays["pose"][:, 0], arrays["pose"][:, 1]) - radius
  lap_times = np.diff([0.0, *laps])
  print(f"\n[race] Model: {args.model}. Track radius {radius} m, +-{half_width} m.")
  print(f"[race] Command limits (vx, vy, wz): {cmd_min} to {cmd_max}.")
  if fell_at is not None:
    print(f"[race] The robot FELL at {fell_at:.1f} s.")
  print(
    f"[race] {travelled * radius:.1f} m along the centre line in {seconds:.1f} s:"
    f" average {travelled * radius / seconds:.2f} m/s."
  )
  ground = np.linalg.norm(np.diff(arrays["pose"][:, :2], axis=0), axis=1).sum()
  print(f"[race] Ground speed: average {ground / seconds:.2f} m/s.")
  print(f"[race] Laps: {len(laps)}, lap times (s): {np.round(lap_times, 2).tolist()}")
  print(
    f"[race] Distance from the centre line: mean {np.abs(off_centre).mean():.3f} m,"
    f" max {np.abs(off_centre).max():.3f} m."
  )
  track_angle = np.arctan2(arrays["pose"][:, 1], arrays["pose"][:, 0])
  off_heading = arrays["pose"][:, 2] - (track_angle + 0.5 * math.pi)
  off_heading = np.degrees(np.abs(np.arctan2(np.sin(off_heading), np.cos(off_heading))))
  print(
    f"[race] Heading away from the track direction: mean {off_heading.mean():.0f} deg."
  )
  print(
    f"[race] Mean command: vx {arrays['cmd'][:, 0].mean():.2f} m/s,"
    f" vy {arrays['cmd'][:, 1].mean():.2f} m/s, wz {arrays['cmd'][:, 2].mean():.2f} rad/s."
  )
  print(
    f"[race] Planning time: median {np.median(plan_ms):.1f} ms per plan"
    f" ({args.samples} samples, {args.iterations} iterations), budget {1e3 * model.dt:.0f} ms."
  )
  if args.fault is not None:
    print(
      f"[race] Fault from {args.change_at:.1f} s on {', '.join(faulted)}: {fault_text}."
    )
  if args.pull is not None or args.fault is not None:
    error = np.sqrt((arrays["model_error"][:, :2] ** 2).sum(axis=1))
    change = arrays["changed"]
    for name, part in (("Before", ~change), ("After", change)):
      if not part.any():
        continue
      beyond = np.maximum(np.abs(off_centre[part]) - half_width, 0.0)
      print(
        f"[race] {name} the change: {np.abs(off_centre[part]).mean():.2f} m from the"
        f" centre line on average, off the track {100 * (beyond > 0).mean():.0f}%"
        f" of the time and {100 * beyond.mean():.1f} cm beyond its edge on average,"
        f" model error {np.sqrt((error[part] ** 2).mean()):.2f} m/s."
      )
  if not args.no_video:
    print(f"[race] Video: {args.out}")


if __name__ == "__main__":
  main()
