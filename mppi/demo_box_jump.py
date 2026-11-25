"""Demo: the box-jump course, with the velocity policy running the run-up.

The robot spawns near the back of the long box and is driven straight at the gap
with a fixed speed ramp; there is no planner and no jump skill yet, so this shows
the scene, the run-up, and what the current policy does at the edge. Gravity and
foot friction change when the robot crosses the yellow line.

  uv run python -m mppi.demo_box_jump
  uv run python -m mppi.demo_box_jump --gravity 12 --friction 1.0 --speed 3
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import mediapy as media
import numpy as np
import torch

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.box_jump.course import (
  DEMO_CHANGE_BEFORE_EDGE,
  DEMO_COURSE,
  box_jump_terrain,
)
from mjlab.tasks.box_jump.mdp import physics_change
from mjlab.viewer import ViewerConfig
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mppi.tracker import DEFAULT_CHECKPOINT, TASK_ID, Tracker, make_tracker_cfg
from mppi.viz import Hud, Rgba

FEET = tuple(f"{leg}_foot_collision" for leg in ("FR", "FL", "RR", "RL"))
LINE_COLOR: Rgba = (1.0, 0.85, 0.25, 0.9)
PAD_COLOR: Rgba = (0.93, 0.58, 0.20, 1.0)


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  p.add_argument("--task", default=TASK_ID)
  p.add_argument("--speed", type=float, default=2.5, help="Run-up speed, m/s.")
  p.add_argument("--accel", type=float, default=1.5, help="Speed ramp, m/s^2.")
  p.add_argument("--gravity", type=float, default=7.0, help="After the change.")
  p.add_argument("--friction", type=float, default=0.4, help="After the change.")
  p.add_argument(
    "--change-before",
    type=float,
    default=DEMO_CHANGE_BEFORE_EDGE,
    help="Where gravity and friction change: metres before the edge.",
  )
  p.add_argument("--duration", type=float, default=5.0, help="Seconds.")
  p.add_argument("--no-video", action="store_true")
  p.add_argument("--out", type=Path, default=Path("logs/mppi/box_jump_scene.mp4"))
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  course = DEMO_COURSE
  layout = course.layout(0.0)
  onset = layout.edge - args.change_before

  cfg = make_tracker_cfg(num_envs=1, task_id=args.task)
  cfg.scene.terrain = box_jump_terrain(course)
  cfg.events["physics_change"] = EventTermCfg(
    func=physics_change,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FEET),
      "gravity_range": (args.gravity, args.gravity),
      "friction_range": (args.friction, args.friction),
      "onset_x": (onset, onset),
    },
  )
  # Follow the robot from its left side, so it runs left to right.
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    entity_name="robot",
    distance=4.0,
    elevation=-12.0,
    azimuth=90.0,
    width=1280,
    height=720,
  )
  tracker = Tracker(
    cfg,
    args.device,
    checkpoint=args.checkpoint,
    render=not args.no_video,
    task_id=args.task,
  )
  change = tracker.env.event_manager.get_term_cfg("physics_change").func
  assert isinstance(change, physics_change)
  origin = tracker.env.scene.env_origins[0].cpu().numpy()

  def draw(vis: DebugVisualizer) -> None:
    line = origin + np.array([onset, 0.0, 0.005])
    size = np.array([0.03, layout.runup_half_width, 0.005])
    vis.add_box(line, size, np.eye(3), LINE_COLOR)

  tracker.set_overlay(draw)
  hud = Hud(
    "Box-jump course",
    "Velocity policy, fixed speed ramp, no jump skill",
    [("Gravity and friction change", LINE_COLOR), ("Landing box", PAD_COLOR)],
  )

  dt = tracker.step_dt
  robot = tracker.robot
  changed_at: float | None = None
  left_at: float | None = None
  fell_at: float | None = None
  top_speed = 0.0
  args.out.parent.mkdir(parents=True, exist_ok=True)
  with contextlib.ExitStack() as stack:
    video = None
    if not args.no_video:
      video = stack.enter_context(
        media.VideoWriter(args.out, shape=(720, 1280), fps=round(1.0 / dt))
      )
    for k in range(round(args.duration / dt)):
      t = k * dt
      pos = robot.data.root_link_pos_w[0].cpu().numpy() - origin
      yaw = tracker.pose()[0, 2].item()
      speed = robot.data.root_link_lin_vel_b[0, 0].item()
      top_speed = max(top_speed, speed)
      # Straight at the gap: ramp the speed, hold the centre line and heading.
      vx = min(args.speed, args.accel * t)
      vy = float(np.clip(-1.0 * pos[1], -0.5, 0.5))
      wz = float(np.clip(-2.0 * yaw, -0.7, 0.7))
      tracker.step(torch.tensor([vx, vy, wz], device=tracker.device))

      if changed_at is None and bool(change.changed[0]):
        changed_at = t
      if left_at is None and pos[0] > layout.edge:
        left_at = t
      # Trunk below the box tops: off the course.
      if fell_at is None and pos[2] < 0.1:
        fell_at = t
      if video is not None:
        readouts = [
          ("Time", f"{t:4.1f} s"),
          ("Speed", f"{speed:4.2f} m/s"),
          ("Command", f"vx {vx:+.2f}"),
          ("To the edge", f"{layout.edge - pos[0]:+5.2f} m"),
          ("Gravity", f"{change.gravity[0].item():5.2f} m/s^2"),
          ("Foot friction", f"{change.friction[0].item():4.2f}"),
        ]
        video.add_image(hud.draw(tracker.render(), readouts))
      if fell_at is not None and t > fell_at + 1.0:
        break

  final = robot.data.root_link_pos_w[0].cpu().numpy() - origin
  on_pad = layout.pad_start <= final[0] <= layout.pad_end and final[2] > 0.1
  print(
    f"\n[box] Course: run-up {course.runup_length} m, gap"
    f" {layout.pad_start - layout.edge:.2f} m, landing box"
    f" {layout.pad_end - layout.pad_start:.2f} x {2 * layout.pad_half_width:.2f} m,"
    f" tops {layout.height} m high. Spawn {layout.edge:.1f} m before the edge."
  )
  print(
    f"[box] Gravity {args.gravity} m/s^2 and foot friction {args.friction} from"
    f" {onset:.1f} m: changed at "
    + (f"{changed_at:.2f} s." if changed_at is not None else "never.")
  )
  print(
    f"[box] Top speed {top_speed:.2f} m/s. Past the edge at "
    + (f"{left_at:.2f} s" if left_at is not None else "never")
    + ("; fell off the course at " + f"{fell_at:.2f} s." if fell_at else ".")
  )
  print(f"[box] Ends {final[0]:.2f} m along, {'on' if on_pad else 'not on'} the box.")
  if video is not None:
    print(f"[box] Video: {args.out}")
  tracker.close()


if __name__ == "__main__":
  main()
