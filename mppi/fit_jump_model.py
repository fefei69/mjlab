"""Calibrate the planner's jump model to a jump policy.

Runs many robots over the demo course under random gravity and friction, with
the planner given the true values but each robot's take-off command scaled by
a random factor, so that commands span the range. Then fits, by least squares
and medians:

- the vertical speed the policy takes off with, against the commanded one and
  gravity,
- its forward speed at lift-off, against the commanded one,
- the time from the trigger to lift-off,
- where the trunk touches down beyond the point-mass flight,
- how hard it brakes after landing, as a share of friction * gravity.

  uv run python -m mppi.fit_jump_model --checkpoint logs/rsl_rl/.../model_2999.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mjlab.tasks.box_jump.course import (
  DEMO_CHANGE_BEFORE_EDGE,
  DEMO_FRICTION_RANGE,
  DEMO_GRAVITY_RANGE,
)
from mppi.jump_planner import NOMINAL_GRAVITY, JumpModelCfg, JumpPlanner
from mppi.run_box_jump import TASK_ID, load_policy, make_env, run_mode


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--task", default=TASK_ID)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--vz-jitter", type=float, default=0.3)
  p.add_argument("--out", type=Path, default=None)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  args.gravity, args.friction = DEMO_GRAVITY_RANGE, DEMO_FRICTION_RANGE
  args.change_before, args.change_prob = DEMO_CHANGE_BEFORE_EDGE, 1.0
  args.duration, args.video = 5.0, False

  env = make_env(args)
  vec, policy = load_policy(env, args)
  r = run_mode("oracle", env, vec, policy, JumpPlanner(device=args.device), args)
  env.close()

  jumped = r["jumped"]
  print(f"\n[fit] {args.num_envs} robots, {jumped.mean():.0%} jumped.")
  j = {k: v[jumped] for k, v in r.items()}
  cfg = JumpModelCfg()

  # Vertical speed: offset + gain * command + gravity_gain * (g - 9.81).
  x = np.stack(
    [np.ones_like(j["vz_cmd"]), j["vz_cmd"], j["gravity"] - NOMINAL_GRAVITY], -1
  )
  coef, *_ = np.linalg.lstsq(x, j["vz"], rcond=None)
  cfg.vz_offset, cfg.vz_gain, cfg.vz_gravity_gain = (float(c) for c in coef)
  vz_fit = x @ coef
  r2 = 1 - np.var(j["vz"] - vz_fit) / np.var(j["vz"])

  cfg.vx_gain = float(np.median(j["vx"] / j["vx_cmd"]))
  cfg.push_time = float(np.median((j["takeoff_x"] - j["trigger_x"]) / j["vx_cmd"]))

  # Touch-down beyond the point-mass flight from where it actually lifted off.
  flight = 2 * np.clip(j["vz"], 0, None) / j["gravity"]
  ballistic = j["takeoff_x"] + j["vx"] * flight
  cfg.flight_offset = float(np.median(j["landed"] - ballistic))
  spread = float(np.std(j["landed"] - ballistic))

  # Braking, from robots that landed on the box and stopped there.
  ok = r["success"]
  stop = r["final_x"][ok] - r["landed"][ok]
  decel = r["landing_vx"][ok] ** 2 / (2 * np.clip(stop, 0.02, None))
  cfg.brake_efficiency = float(
    np.median(decel / (r["friction"][ok] * r["gravity"][ok]))
  )

  print(
    f"[fit] Take-off vz = {cfg.vz_offset:.2f} + {cfg.vz_gain:.2f} * command"
    f" + {cfg.vz_gravity_gain:.3f} * (g - 9.81), R2 {r2:.2f}."
  )
  print(f"[fit] Take-off vx = {cfg.vx_gain:.2f} * command.")
  print(f"[fit] Trigger to lift-off: {cfg.push_time:.3f} s.")
  print(
    f"[fit] Touch-down {cfg.flight_offset:+.3f} m beyond the point-mass flight"
    f" (spread {spread:.3f} m)."
  )
  print(
    f"[fit] Braking: {cfg.brake_efficiency:.2f} of friction * gravity"
    f" ({ok.sum()} robots that stopped on the box)."
  )
  out = args.out or args.checkpoint.parent / "jump_model.json"
  cfg.save(out, checkpoint=str(args.checkpoint), r2_vz=float(r2))
  print(f"[fit] Saved {out}")


if __name__ == "__main__":
  main()
