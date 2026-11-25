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
- how far it travels after landing, from a second run with the planner the
  demo uses (MPPI tracking a reference) and a late change, so that the braking
  fit also covers the faster landings of that planner.

  uv run python -m mppi.fit_jump_model --checkpoint logs/rsl_rl/.../model_2999.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab.tasks.box_jump.course import (
  DEMO_CHANGE_BEFORE_EDGE,
  DEMO_FRICTION_RANGE,
  DEMO_GRAVITY_RANGE,
)
from mppi.jump_mppi import JumpTrackingMppi
from mppi.jump_planner import NOMINAL_GRAVITY, JumpModelCfg, JumpPlanner, stop_distance
from mppi.run_box_jump import TASK_ID, load_policy, make_env, run_mode


def fit_braking(
  speed: np.ndarray,
  stop: np.ndarray,
  friction: np.ndarray,
  gravity: np.ndarray,
  quantile: float = 0.8,
  steps: int = 3000,
) -> tuple[float, float, float]:
  """Offset, impact speed and efficiency of the braking model, by quantile
  regression: the model's stop distance is exceeded by 1 - `quantile` of stops."""
  v, y = torch.as_tensor(speed).float(), torch.as_tensor(stop).float().clamp(min=0)
  mu_g = torch.as_tensor(friction * gravity).float()
  raw = torch.tensor([-2.0, 0.0, 0.0], requires_grad=True)  # Softplus-positive.
  opt = torch.optim.Adam([raw], lr=0.02)

  def params(r: torch.Tensor) -> tuple[torch.Tensor, ...]:
    offset, impact, eff = torch.nn.functional.softplus(r)
    return offset, impact, 0.3 + eff  # Efficiency at least 0.3 for sane extrapolation.

  for _ in range(steps):
    offset, impact, eff = params(raw)
    pred = offset + (v - impact).clamp(min=0).square() / (2 * eff * mu_g)
    err = y - pred
    loss = torch.maximum(quantile * err, (quantile - 1) * err).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
  offset, impact, eff = (float(x) for x in params(raw.detach()))
  return offset, impact, eff


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--task", default=TASK_ID)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--vz-jitter", type=float, default=0.3)
  p.add_argument("--out", type=Path, default=None)
  p.add_argument(
    "--save-raw", type=Path, default=None, help="Also save the runs' outcomes."
  )
  p.add_argument(
    "--braking-change-before",
    type=float,
    default=1.0,
    help="Change point of the second run (m before the edge): late, for fast landings.",
  )
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()
  args.gravity, args.friction = DEMO_GRAVITY_RANGE, DEMO_FRICTION_RANGE
  args.change_before, args.change_prob = DEMO_CHANGE_BEFORE_EDGE, 1.0
  args.duration, args.video = 5.0, False
  args.replan_every, args.samples_shown = 5, 0  # For the second run's MPPI.

  env = make_env(args)
  vec, policy = load_policy(env, args)
  r = run_mode("oracle", env, vec, policy, JumpPlanner(device=args.device), args)

  jumped = r["jumped"]
  print(f"\n[fit] {args.num_envs} robots, {jumped.mean():.0%} jumped.")
  j = {k: v[jumped] for k, v in r.items() if v.shape[:1] == jumped.shape}
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

  # Run-up. The gait makes the speed wobble within each stride, so use speed
  # averaged over 0.2 s. Steady speed against a command held for that long, and
  # the top acceleration over 0.2 s windows.
  trace = r["speed_trace"]  # (T, B, 4)
  command, speed, running = trace[..., 0], trace[..., 1], trace[..., 3]
  dt, k = 0.02, 10

  def window_mean(a: np.ndarray) -> np.ndarray:
    kernel = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, "valid"), 0, a)

  smooth, held = window_mean(speed), window_mean(command)
  wobble = window_mean(command**2) - held**2
  whole = window_mean(running) > 0.999
  steady = whole & (wobble < 1e-4) & (held > 0.5)
  x = np.stack([np.ones(steady.sum()), held[steady]], -1)
  (cfg.speed_offset, cfg.speed_gain), *_ = np.linalg.lstsq(
    x, smooth[steady], rcond=None
  )
  cfg.speed_offset, cfg.speed_gain = float(cfg.speed_offset), float(cfg.speed_gain)
  dv = smooth[k:] - smooth[:-k]
  both = whole[k:] & whole[:-k]
  cfg.run_accel = float(np.percentile(np.abs(dv[both]) / (k * dt), 95))

  # Braking, from robots that landed on the box and stopped there: the landing
  # impact takes away up to some speed, friction * gravity the rest. Fitted to
  # the 80th percentile of stopping distance, so the planner does not count on
  # its best stops. The grid planner's run-ups are slow, so a second run with
  # the demo's planner (given the fit so far) and a late change adds fast
  # landings; without them the fit extrapolates to them, and badly.
  def stopped(runs: tuple[dict[str, np.ndarray], ...], key: str) -> np.ndarray:
    """`key` of the robots that landed on the box and stopped there."""
    return np.concatenate([run[key][run["success"]] for run in runs])

  def fit_stops(runs: tuple[dict[str, np.ndarray], ...]) -> tuple[np.ndarray, ...]:
    data = [stopped(runs, k) for k in ("landing_vx", "friction", "gravity")]
    stops = stopped(runs, "final_x") - stopped(runs, "landed")
    braking = fit_braking(data[0], stops, data[1], data[2])
    cfg.brake_offset, cfg.brake_impact_speed, cfg.brake_efficiency = braking
    return (*data, stops)

  fit_stops((r,))
  args.vz_jitter, args.change_before = 0.0, args.braking_change_before
  tracking = JumpTrackingMppi(args.num_envs, cfg, device=args.device)
  r_fast = run_mode("oracle", env, vec, policy, tracking, args)
  env.close()
  if args.save_raw:
    runs = (("", r), ("fast_", r_fast))
    raw: dict[str, Any] = {f"{pre}{k}": v for pre, run in runs for k, v in run.items()}
    np.savez(args.save_raw, **raw)
  land_speed, friction, gravity, stops = fit_stops((r, r_fast))
  covered = float(
    np.mean(
      stops
      <= stop_distance(
        cfg,
        torch.as_tensor(land_speed),
        torch.as_tensor(friction),
        torch.as_tensor(gravity),
      ).numpy()
    )
  )

  print(
    f"[fit] Take-off vz = {cfg.vz_offset:.2f} + {cfg.vz_gain:.2f} * command"
    f" + {cfg.vz_gravity_gain:.3f} * (g - 9.81), R2 {r2:.2f}."
  )
  print(f"[fit] Take-off vx = {cfg.vx_gain:.2f} * command.")
  print(
    f"[fit] Run-up: steady speed = {cfg.speed_offset:.2f} + {cfg.speed_gain:.2f} *"
    f" command, top acceleration {cfg.run_accel:.1f} m/s^2."
  )
  print(f"[fit] Trigger to lift-off: {cfg.push_time:.3f} s.")
  print(
    f"[fit] Touch-down {cfg.flight_offset:+.3f} m beyond the point-mass flight"
    f" (spread {spread:.3f} m)."
  )
  print(
    f"[fit] Braking: the impact takes {cfg.brake_impact_speed:.2f} m/s over"
    f" {cfg.brake_offset:.2f} m, then {cfg.brake_efficiency:.2f} of friction *"
    f" gravity; covers {covered:.0%} of {len(stops)} stops (median stop"
    f" {np.median(stops):.2f} m; landing speeds {np.percentile(land_speed, 5):.1f}"
    f" to {np.percentile(land_speed, 95):.1f} m/s)."
  )
  out = args.out or args.checkpoint.parent / "jump_model.json"
  cfg.save(out, checkpoint=str(args.checkpoint), r2_vz=float(r2))
  print(f"[fit] Saved {out}")


if __name__ == "__main__":
  main()
