"""Tune DIAL-MPC on the box-jump course: a few settings, several seeds each.

Runs each setting with the planner given the true gravity and friction by
default, so that the planner itself is tuned, not the estimators. One
environment and one set of planning worlds per planning step are reused.

  uv run python -m dial.tune
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
from mppi.run_box_jump import make_env

from dial.box_jump import TASK_ID, PlanningWorlds, run
from mjlab.tasks.box_jump.course import DEMO_CHANGE_BEFORE_EDGE

DEFAULTS = {
  "num_samples": 2048,
  "plan_dt": 0.01,
  "horizon_steps": 20,
  "nodes": 5,
  "rounds": 2,
  "rounds_init": 10,
  "jump_arc": 0.0,
  "speed": 1.5,
}
SETTINGS = {
  "baseline": {},
  "horizon 0.5 s": {"horizon_steps": 25},
  "jump arc 0.15 m": {"jump_arc": 0.15},
  "planning step 0.005 s": {"plan_dt": 0.005},
  "4 rounds per step": {"rounds": 4},
  # Longer horizons keep the nodes about 0.08 s apart.
  "horizon 0.6 s": {"horizon_steps": 30, "nodes": 7},
  "horizon 0.8 s": {"horizon_steps": 40, "nodes": 10},
}


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--settings", nargs="+", default=list(SETTINGS))
  p.add_argument("--modes", nargs="+", default=["oracle"])
  p.add_argument("--seeds", type=int, default=5)
  p.add_argument("--gravity", type=float, default=7.0)
  p.add_argument("--friction", type=float, default=0.5)
  p.add_argument("--duration", type=float, default=7.0)
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()

  env = make_env(
    argparse.Namespace(
      task=TASK_ID,
      num_envs=1,
      change_before=DEMO_CHANGE_BEFORE_EDGE,
      gravity=(args.gravity, args.gravity),
      friction=(args.friction, args.friction),
      change_prob=1.0,
      video=False,
      device=args.device,
    )
  )
  worlds: dict[tuple[int, float], PlanningWorlds] = {}
  print(
    f"\n[tune] DIAL-MPC, gravity {args.gravity} and friction {args.friction} from"
    f" {DEMO_CHANGE_BEFORE_EDGE} m before the edge; {args.seeds} seeds."
  )
  print("[tune] setting                | mode     | landed | fell | median end | plan")
  for name in args.settings:
    setting = {**DEFAULTS, **SETTINGS[name]}
    key = (setting["num_samples"], setting["plan_dt"])
    if key not in worlds:
      worlds[key] = PlanningWorlds(
        setting["num_samples"] + 1, setting["plan_dt"], env.step_dt, args.device
      )
    for mode in args.modes:
      runs = []
      for seed in range(args.seeds):
        run_args = copy.copy(args)
        for k, v in setting.items():
          setattr(run_args, k, v)
        run_args.seed, run_args.video = seed, False
        run_args.change_before, run_args.samples_shown = DEMO_CHANGE_BEFORE_EDGE, 0
        runs.append(run(mode, env, worlds[key], run_args))
      landed = sum(r["success"] for r in runs)
      fell = sum(r["fell"] for r in runs)
      end = np.median([r["final_x"] for r in runs])
      ms = np.median([r["plan_ms"] for r in runs])
      print(
        f"[tune] {name:22s} | {mode:8s} | {landed:.0f}/{args.seeds}    |"
        f" {fell:.0f}/{args.seeds}  | {end:6.2f} m   | {ms:4.0f} ms",
        flush=True,
      )
  env.close()


if __name__ == "__main__":
  main()
