"""Demo: push a box to goals. MPPI plans velocity commands, the RL policy tracks.

There is no hand-written model of the box. MPPI scores candidate command
sequences by rolling them out in the simulator: the real world's state is copied
into every world of a second, parallel simulation, each world runs the policy
under one candidate sequence, and the box's predicted path gives the cost. The
real world is paused while planning, so this demo is not real time.

  uv run python -m mppi.demo_box_push
  uv run python -m mppi.demo_box_push --goals 3.5 -1.0   # A single goal.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import mediapy as media
import mujoco
import numpy as np
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.viewer import ViewerConfig
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mppi.core import Mppi, MppiCfg
from mppi.tracker import CMD_MAX, CMD_MIN, DEFAULT_CHECKPOINT, Tracker, make_tracker_cfg
from mppi.viz import PLAN_COLOR, Hud, draw_path, draw_rollouts

BOX_HALF_SIZE = (0.25, 0.25, 0.2)
BOX_MASS = 2.0
BOX_FRICTION = 0.5
# Distance from the box centre at which the robot stands to push it.
STANDOFF = 0.65

GOAL_COLOR = (0.1, 0.8, 0.2, 0.6)
DONE_COLOR = (0.5, 0.5, 0.5, 0.4)


def box_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box")
  body.add_freejoint(name="box_joint")
  body.add_geom(
    name="box_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=BOX_HALF_SIZE,
    mass=BOX_MASS,
    rgba=(0.85, 0.3, 0.2, 1.0),
    friction=(BOX_FRICTION, 0.005, 0.0001),
    # Priority 1 makes contacts use the box's friction, not the larger of the two
    # geoms', so the box slides on the floor as configured.
    priority=1,
    group=2,
  )
  return spec


def make_cfg(num_envs: int, box_xy: tuple[float, float]) -> ManagerBasedRlEnvCfg:
  cfg = make_tracker_cfg(num_envs)
  cfg.scene.entities["box"] = EntityCfg(
    spec_fn=box_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(*box_xy, BOX_HALF_SIZE[2])),
  )
  cfg.events["reset_box"] = EventTermCfg(
    func=envs_mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("box"),
    },
  )
  cfg.sim.njmax = 500  # Room for the box's contacts.
  return cfg


def record(tracker: Tracker) -> dict[str, torch.Tensor]:
  box = tracker.env.scene["box"]
  return {
    "pose": tracker.pose(),
    "upright": tracker.upright(),
    "box": box.data.root_link_pos_w[:, :2].clone(),
  }


def push_cost(
  traj: dict[str, torch.Tensor], commands: torch.Tensor, goal: torch.Tensor
) -> torch.Tensor:
  """Cost of simulated rollouts, each (K, H, ...), for pushing the box to `goal`.

  The box's distance to the goal is the objective. Two shaping terms get the robot
  into position when it is not touching the box yet: stand behind the box on the
  line to the goal, and face the box. Falling over is penalized heavily.
  """
  robot_xy, yaw = traj["pose"][..., :2], traj["pose"][..., 2]
  box = traj["box"]
  to_goal = goal - box
  dist = to_goal.norm(dim=-1)
  push_dir = to_goal / dist.clamp(min=1e-3)[..., None]
  behind = box - STANDOFF * push_dir
  approach = (robot_xy - behind).norm(dim=-1)
  to_box = box - robot_xy
  facing = 1.0 - torch.cos(yaw - torch.atan2(to_box[..., 1], to_box[..., 0]))
  fallen = torch.relu(0.8 - traj["upright"])
  jerk = torch.diff(commands, dim=1).pow(2).sum(dim=-1)
  return (
    10.0 * dist[:, -1]
    + 3.0 * dist.mean(dim=1)
    + 1.0 * approach.mean(dim=1)
    + 0.5 * facing.mean(dim=1)
    + 100.0 * fallen.mean(dim=1)
    + 0.1 * jerk.mean(dim=1)
  )


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  p.add_argument("--box", type=float, nargs=2, default=(1.5, 0.5), help="Box x y.")
  p.add_argument(
    "--goals",
    type=float,
    nargs="+",
    default=(3.5, -1.0, 3.5, 1.5, 1.0, 1.5),
    help="Goal positions for the box, visited in order: x1 y1 x2 y2 ...",
  )
  p.add_argument("--tolerance", type=float, default=0.1, help="Goal radius in m.")
  p.add_argument("--duration", type=float, default=40.0, help="Time limit in s.")
  p.add_argument("--horizon", type=float, default=2.0, help="Planning horizon in s.")
  p.add_argument("--samples", type=int, default=384, help="Planning worlds.")
  p.add_argument(
    "--rollouts-shown", type=int, default=64, help="Rollouts drawn in the video."
  )
  p.add_argument("--out", type=Path, default=Path("logs/mppi/box_push.mp4"))
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  torch.manual_seed(args.seed)
  box_xy = (float(args.box[0]), float(args.box[1]))
  if len(args.goals) % 2 != 0:
    raise SystemExit("--goals needs an even number of values: x1 y1 x2 y2 ...")
  goals_np = np.array(args.goals, dtype=np.float64).reshape(-1, 2)
  goals = torch.tensor(goals_np, dtype=torch.float32, device=args.device)
  goal_idx = 0

  # The real world: one robot and the box, rendered from a fixed camera that
  # frames the robot's start, the box and all goals.
  cfg = make_cfg(1, box_xy)
  points = np.concatenate([np.zeros((1, 2)), np.array([box_xy]), goals_np])
  centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
  extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    lookat=(float(centre[0]), float(centre[1]), 0.0),
    distance=1.1 * extent + 2.0,
    elevation=-40.0,
    azimuth=110.0,
    width=1280,
    height=720,
  )
  real = Tracker(cfg, args.device, checkpoint=args.checkpoint, render=True)
  # The planning worlds: same scene, one world per candidate command sequence.
  planner = Tracker(make_cfg(args.samples, box_xy), args.device, policy=real.policy)

  plan_dt = 0.1
  hold = round(plan_dt / real.step_dt)
  replan_every = 2  # Plan steps executed between plans.
  mppi = Mppi(
    MppiCfg(
      horizon=round(args.horizon / plan_dt),
      num_samples=args.samples,
      u_min=CMD_MIN,
      u_max=CMD_MAX,
      noise_sigma=(0.5, 0.3, 0.4),
      temperature=0.1,
    ),
    args.device,
  )

  box = real.env.scene["box"]
  shown = torch.linspace(0, args.samples - 1, args.rollouts_shown).long()
  # Paths to draw: the sampled rollouts and the best one, for the robot and the box.
  paths: dict[str, np.ndarray] = {}

  def cost_fn(commands: torch.Tensor) -> torch.Tensor:
    planner.set_state(real.get_state())
    traj = planner.rollout(commands, hold, record)
    costs = push_cost(traj, commands, goals[goal_idx])

    def from_start(start: torch.Tensor, xy: torch.Tensor) -> np.ndarray:
      """Prepend the current position to (N, H, 2) predicted paths."""
      first = start.expand(xy.shape[0], 1, 2)
      return torch.cat([first, xy], dim=1).cpu().numpy()

    robot_xy, box_xy = real.pose()[0, :2], box.data.root_link_pos_w[0, :2]
    best = costs.argmin().reshape(1)
    paths["robot_fan"] = from_start(robot_xy, traj["pose"][shown, :, :2])
    paths["box_fan"] = from_start(box_xy, traj["box"][shown])
    paths["costs"] = costs[shown].cpu().numpy()
    paths["robot_best"] = from_start(robot_xy, traj["pose"][best, :, :2])[0]
    paths["box_best"] = from_start(box_xy, traj["box"][best])[0]
    return costs

  def draw(vis: DebugVisualizer) -> None:
    for i, g in enumerate(goals_np):
      vis.add_cylinder(
        np.array([g[0], g[1], 0.0]),
        np.array([g[0], g[1], 0.01]),
        BOX_HALF_SIZE[0],
        GOAL_COLOR if i >= goal_idx else DONE_COLOR,
      )
    if paths:
      draw_rollouts(vis, paths["robot_fan"], paths["costs"], 0.02, 0.005)
      draw_rollouts(vis, paths["box_fan"], paths["costs"], 0.02, 0.011)
      draw_path(vis, paths["robot_best"], 0.04, 0.014, PLAN_COLOR)
      draw_path(vis, paths["box_best"], 0.04, 0.014, PLAN_COLOR, markers=True)

  real.set_overlay(draw)
  hud = Hud(
    "MPPI box push",
    f"Simulator rollouts · {args.samples} worlds · {args.horizon:.1f} s horizon",
    [
      ("Sampled rollouts (light = low cost)", "ramp"),
      ("Best rollout (box path dotted)", PLAN_COLOR),
    ],
    note="Thin lines: robot paths. Thick lines: box paths.",
  )

  def readouts(t: float, command: torch.Tensor) -> list[tuple[str, str]]:
    error = float((box.data.root_link_pos_w[0, :2] - goals[goal_idx]).norm())
    return [
      ("Time", f"{t:.1f} s"),
      ("Goal", f"{goal_idx + 1} of {len(goals)}"),
      ("Box to goal", f"{error:.2f} m"),
      (
        "Command",
        f"vx {command[0]:+.2f}  vy {command[1]:+.2f}  wz {command[2]:+.2f}",
      ),
    ]

  log: dict[str, list[np.ndarray]] = {"pose": [], "box": [], "cmd": []}
  plan_s: list[float] = []
  reached_at: list[float] = []
  fell_at: float | None = None
  t = 0.0

  args.out.parent.mkdir(parents=True, exist_ok=True)
  fps = round(1.0 / real.step_dt)
  with media.VideoWriter(args.out, shape=(720, 1280), fps=fps) as video:
    while t < args.duration:
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      plan = mppi.plan(cost_fn)
      torch.cuda.synchronize()
      plan_s.append(time.perf_counter() - t0)

      for command in plan[:replan_every].clone():
        panel = readouts(t, command)
        for _ in range(hold):
          real.step(command)
          video.add_image(hud.draw(real.render(), panel))
        t += plan_dt
        log["pose"].append(real.pose()[0].cpu().numpy())
        log["box"].append(box.data.root_link_pos_w[0].cpu().numpy())
        log["cmd"].append(command.cpu().numpy())
      mppi.shift(replan_every)

      error = float((box.data.root_link_pos_w[0, :2] - goals[goal_idx]).norm())
      if real.upright()[0] < 0.5:
        fell_at = t
        break
      if error < args.tolerance:
        reached_at.append(t)
        if goal_idx == len(goals) - 1:
          break
        goal_idx += 1

    # Stand still for a moment so the video shows where the box ended up.
    paths.clear()
    stop = torch.zeros(3, device=args.device)
    panel = readouts(t, stop)
    for _ in range(round(1.5 / real.step_dt)):
      real.step(stop)
      video.add_image(hud.draw(real.render(), panel))

  final = box.data.root_link_pos_w[0].cpu().numpy()
  real.close()
  planner.close()
  arrays = {name: np.stack(values) for name, values in log.items()}
  payload: dict[str, Any] = {"dt": plan_dt, "goals": goals_np, **arrays}
  np.savez(args.out.with_suffix(".npz"), **payload)

  final_error = float(np.linalg.norm(final[:2] - goals_np[goal_idx]))
  print(f"\n[push] Box {BOX_MASS} kg, start {box_xy}, goals {goals_np.tolist()}.")
  print(
    f"[push] Goals reached (within {args.tolerance} m): {len(reached_at)} of"
    f" {len(goals_np)}, at {np.round(reached_at, 1).tolist()} s."
  )
  if fell_at is not None:
    print(f"[push] The robot FELL at {fell_at:.1f} s.")
  elif len(reached_at) < len(goals_np):
    print(f"[push] Time limit of {args.duration:.0f} s reached.")
  print(
    f"[push] At rest the box is {final_error:.3f} m from goal {goal_idx + 1}."
    f" Box centre height {final[2]:.3f} m (upright is {BOX_HALF_SIZE[2]} m)."
  )
  print(
    f"[push] Planning: {len(plan_s)} plans, median {np.median(plan_s):.2f} s each"
    f" ({args.samples} worlds x {args.horizon:.1f} s), one plan per"
    f" {replan_every * plan_dt:.1f} s of motion."
  )
  print(f"[push] Video: {args.out}")


if __name__ == "__main__":
  main()
