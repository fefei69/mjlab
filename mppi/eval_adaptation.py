"""How fast does the box jump adapt? Move the change closer to the edge.

Gravity and foot friction change at a line a set distance before the edge, from
2.5 m (about 1 s before take-off) down to 0.5 m. For each distance, each
planner runs every robot with gravity and friction fixed at nominal, estimated
online, and true. Also reports how far the online gravity estimate was off at
the trigger, and how long after the change the trigger came.

  uv run python -m mppi.eval_adaptation --checkpoint logs/rsl_rl/.../model_2999.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mjlab.tasks.box_jump.course import (
  DEMO_COURSE,
  DEMO_FRICTION_RANGE,
  DEMO_GRAVITY_RANGE,
)
from mppi.jump_mppi import JumpMppi, JumpMppiCfg, JumpTrackingMppi
from mppi.jump_planner import JumpModelCfg
from mppi.run_box_jump import MODES, TASK_ID, load_policy, make_env, run_mode


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--task", default=TASK_ID)
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--distances", type=float, nargs="+", default=[2.5, 1.5, 1.0, 0.5])
  p.add_argument("--planners", nargs="+", default=["mppi", "tracking"])
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  args.gravity, args.friction = DEMO_GRAVITY_RANGE, DEMO_FRICTION_RANGE
  args.change_before, args.change_prob = args.distances[0], 1.0
  args.duration, args.video, args.vz_jitter = 5.0, False, 0.0
  args.replan_every, args.samples_shown = 5, 0

  model_path = args.checkpoint.parent / "jump_model.json"
  model = JumpModelCfg.load(model_path) if model_path.exists() else JumpModelCfg()
  env = make_env(args)
  vec, policy = load_policy(env, args)
  params = env.event_manager.get_term_cfg("physics_change").params
  edge = DEMO_COURSE.layout(0.0).edge

  print(
    f"\n[adapt] {args.num_envs} robots; gravity {args.gravity}, friction"
    f" {args.friction}; jump model {'calibrated' if model_path.exists() else 'default'}."
  )
  print(
    "[adapt] change before edge | planner  | fixed  online  true |"
    " g error at trigger | change to trigger"
  )
  for distance in args.distances:
    params["onset_x"] = (edge - distance, edge - distance)
    args.change_before = distance
    for name in args.planners:
      planner = (
        JumpTrackingMppi(args.num_envs, model, device=args.device)
        if name == "tracking"
        else JumpMppi(args.num_envs, model, JumpMppiCfg(), device=args.device)
      )
      rates, err, delay = [], np.nan, np.nan
      for mode in MODES:
        r = run_mode(mode, env, vec, policy, planner, args)
        rates.append(r["success"].mean())
        if mode == "online":
          err = np.nanmedian(np.abs(r["gravity_used"] - r["gravity"]))
          delay = np.nanmedian(r["trigger_t"] - r["change_t"])
      print(
        f"[adapt] {distance:17.1f} m | {name:8s} | {rates[0]:5.0%} {rates[1]:6.0%}"
        f" {rates[2]:5.0%} | {err:12.2f} m/s^2 | {delay:10.2f} s",
        flush=True,
      )
  env.close()


if __name__ == "__main__":
  main()
