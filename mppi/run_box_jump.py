"""Run the jump policy on the demo course, with the planner choosing the jump.

Many robots run at once, each with its own gravity and foot friction after the
change on the run-up. The planner (`mppi/jump_planner.py`) picks each robot's
run-up speed, trigger point and take-off speed every step until the jump,
using one of three sources for gravity and friction:

- `nominal`: fixed at 9.81 and 1.0 (no adaptation),
- `online`: estimated from the feet's push and from foot slip,
- `oracle`: the simulator's true values.

Every mode sees the same robots and the same changes. `--video` records robot 0.

  uv run python -m mppi.run_box_jump --checkpoint logs/rsl_rl/.../model_2999.pt
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from pathlib import Path

import mediapy as media
import numpy as np
import torch

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor, ContactSensorCfg
from mjlab.tasks.box_jump.course import (
  DEMO_CHANGE_BEFORE_EDGE,
  DEMO_COURSE,
  DEMO_FRICTION_RANGE,
  DEMO_GRAVITY_RANGE,
  box_jump_terrain,
)
from mjlab.tasks.box_jump.mdp import JumpCommand, JumpCommandCfg, JumpPhase
from mjlab.tasks.box_jump.mdp import physics_change as PhysicsChange
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import ViewerConfig
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mppi.jump_mppi import (
  JumpMppi,
  JumpMppiCfg,
  JumpTrackingMppi,
  plan_path,
  speed_curve,
)
from mppi.jump_planner import (
  NOMINAL_GRAVITY,
  FrictionEstimator,
  GravityEstimator,
  JumpModelCfg,
  JumpPlanner,
  Reference,
  ReferencePlanner,
)
from mppi.viz import (
  PLAN_COLOR,
  TRAIL_COLOR,
  Hud,
  Rgba,
  cost_colors,
  draw_polyline,
  draw_speed_panel,
)

TASK_ID = "Mjlab-BoxJump-Unitree-Go1-History"
MODES = ("nominal", "online", "oracle")
FEET = ("FR", "FL", "RR", "RL")
LINE_COLOR: Rgba = (1.0, 0.85, 0.25, 0.9)
AIM_COLOR: Rgba = (0.30, 1.00, 0.85, 1.0)
REF_COLOR: Rgba = (0.75, 0.55, 1.00, 0.9)
# Chase camera: distance (m), elevation and azimuth (deg; 0 looks forward).
CAMERA = (4.5, -18.0, 55.0)
PHASE_NAMES = {int(p): p.name.lower() for p in JumpPhase}


def make_env(args: argparse.Namespace) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.scene.terrain = box_jump_terrain(DEMO_COURSE)
  jump = cfg.commands["jump"]
  assert isinstance(jump, JumpCommandCfg)
  jump.course = DEMO_COURSE
  jump.coach = False
  # Every physics step's foot forces, for the gravity estimate.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "feet_ground_contact":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.history_length = cfg.decimation
  # Fallen robots stay where they fell; outcomes are recorded here instead.
  cfg.terminations = {}
  onset = DEMO_COURSE.layout(0.0).edge - args.change_before
  cfg.events["physics_change"].params.update(
    start_gravity_range=(NOMINAL_GRAVITY, NOMINAL_GRAVITY),
    start_friction_range=(1.0, 1.0),
    gravity_range=tuple(args.gravity),
    friction_range=tuple(args.friction),
    onset_x=(onset, onset),
    prob=args.change_prob,
  )
  # A chase camera, behind and to the left of the robot, that sees the course
  # ahead and so the planned jump from the start.
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    entity_name="robot",
    distance=getattr(args, "cam_distance", CAMERA[0]),
    elevation=getattr(args, "cam_elevation", CAMERA[1]),
    azimuth=getattr(args, "cam_azimuth", CAMERA[2]),
    width=1280,
    height=720,
    # Every robot runs on the same course: draw robot 0 alone, not overlapping
    # neighbours.
    max_extra_envs=0,
  )
  return ManagerBasedRlEnv(
    cfg=cfg, device=args.device, render_mode="rgb_array" if args.video else None
  )


def load_policy(env: ManagerBasedRlEnv, args: argparse.Namespace):
  """The jump policy from `args.checkpoint`, and the wrapped env it acts in."""
  agent_cfg = load_rl_cfg(args.task)
  vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(vec, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  return vec, runner.get_inference_policy(device=args.device)


def _path(
  x_now: float,
  trigger_x: float,
  liftoff_x: float,
  touchdown_x: float,
  stop_x: float,
  vz: float,
  gravity: float,
  height: float,
) -> np.ndarray:
  """A predicted trunk path from the plan's key points, (P, 3) from the spawn."""
  flight = 2 * vz / gravity
  s = np.linspace(0.0, flight, 10)
  fx = liftoff_x + (touchdown_x - liftoff_x) * s / max(flight, 1e-6)
  fz = height + vz * s - 0.5 * gravity * s**2
  xs = np.concatenate([[x_now, trigger_x], fx, [stop_x]])
  zs = np.concatenate([[height, height], fz, [height]])
  return np.stack([xs, np.zeros_like(xs), zs], axis=-1)


def reference_curve(ref: Reference, x_now: float) -> tuple[np.ndarray, np.ndarray]:
  """Robot 0's reference speed from where it is to the take-off point."""
  end = float(ref.takeoff_x[0])
  xs = torch.linspace(x_now, max(end, x_now), 40, device=ref.speed.device)
  speeds = ref.speed_at(xs[None].expand(len(ref.speed), -1))[0]
  return xs.cpu().numpy(), speeds.cpu().numpy()


def run_mode(
  mode: str,
  env: ManagerBasedRlEnv,
  vec: RslRlVecEnvWrapper,
  policy,
  planner: JumpPlanner | JumpMppi,
  args: argparse.Namespace,
) -> dict[str, np.ndarray]:
  torch.manual_seed(args.seed)
  vec.reset()
  obs = vec.get_observations()
  n, device, dt = env.num_envs, env.device, env.step_dt
  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("jump")
  assert isinstance(cmd, JumpCommand)
  change = env.event_manager.get_term_cfg("physics_change").func
  assert isinstance(change, PhysicsChange)
  feet_sensor = env.scene["feet_ground_contact"]
  assert isinstance(feet_sensor, ContactSensor)
  foot_sites = robot.find_sites(FEET, preserve_order=True)[0]
  mass = float(env.sim.mj_model.body_mass[robot.indexing.body_ids.cpu()].sum())
  gravity_est = GravityEstimator(n, dt, mass, device=device)
  friction_est = FrictionEstimator(n, device=device)
  mppi = planner if isinstance(planner, JumpMppi) else None
  tracking = planner if isinstance(planner, JumpTrackingMppi) else None
  model_cfg = planner.model if isinstance(planner, JumpMppi) else planner.cfg
  references = ReferencePlanner(model_cfg, device=device) if tracking else None

  def estimates() -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "nominal":
      return (
        torch.full((n,), NOMINAL_GRAVITY, device=device),
        torch.ones(n, device=device),
      )
    if mode == "online":
      return gravity_est.gravity, friction_est.friction
    return change.gravity, change.friction

  reference = None
  if tracking is not None and references is not None:
    reference = references.plan(
      cmd.position()[:, 0], torch.zeros(n, device=device), cmd.course(), *estimates()
    )
    tracking.set_reference(reference)
  if mppi is not None:
    mppi.reset(cmd.position()[:, 0], cmd.course()[0])

  fell = torch.zeros(n, dtype=torch.bool, device=device)
  # Calibration runs scale each robot's take-off command by a fixed factor.
  jitter = 1 + args.vz_jitter * (2 * torch.rand(n, device=device) - 1)
  aimed = torch.full((n,), torch.nan, device=device)
  # The gravity and friction the plan was made with, frozen at the trigger.
  used_gravity = torch.full((n,), torch.nan, device=device)
  used_friction = torch.full((n,), torch.nan, device=device)
  planned_vz = torch.zeros(n, device=device)
  trigger_x = torch.full((n,), torch.inf, device=device)
  origin = env.scene.env_origins[0].cpu().numpy()
  layout = DEMO_COURSE.layout(0.0)
  onset = layout.edge - args.change_before
  height = float(cmd.position()[0, 2])  # Trunk above the box tops, standing.
  name = (
    "MPPI tracking a reference"
    if tracking is not None
    else "long-horizon MPPI"
    if mppi is not None
    else "grid planner"
  )
  hud = Hud(
    f"Box jump, {name}",
    f"Gravity and friction: {mode}",
    [
      *([("Sampled plans (light = low cost)", "ramp")] if mppi is not None else []),
      ("Chosen plan", PLAN_COLOR),
      *([("Reference take-off and landing", REF_COLOR)] if tracking else []),
      ("Path driven", TRAIL_COLOR),
      ("Gravity and friction change", LINE_COLOR),
    ],
  )
  # What robot 0's planner predicts, for drawing: the chosen path and samples.
  view: dict[str, object] = {"path": None, "samples": None, "costs": None}
  speeds: dict[str, object] = {"samples": [], "chosen": None}
  run_x: list[float] = []
  run_v: list[float] = []
  model = model_cfg
  grid_friction = [1.0]  # Robot 0's friction at its last grid plan.
  # Robot 0's reference, for drawing: take-off point, landing target, speeds.
  ref_view: dict[str, float] = {}
  trail: list[np.ndarray] = []

  def predicted(r, sample: int, x_now: float, g: float) -> np.ndarray:
    if mppi is not None:
      return plan_path(r, 0, sample, g, height, mppi.cfg.dt)
    # The grid planner's jump: lift-off just before the edge, at the planned
    # run-up speed, with the vertical speed the calibration predicts.
    m = model
    vx = m.vx_gain * float(cmd.run_speed[0])
    vz = m.vz_offset + m.vz_gain * float(planned_vz[0])
    vz += m.vz_gravity_gain * (g - NOMINAL_GRAVITY)
    stop = float(aimed[0]) + vx**2 / (2 * m.brake_efficiency * grid_friction[0] * g)
    edge = float(cmd.course()[0][0])
    return _path(
      x_now,
      float(trigger_x[0]),
      edge - m.takeoff_before_edge,
      float(aimed[0]),
      stop,
      max(vz, 0.0),
      g,
      height,
    )

  def draw(vis: DebugVisualizer) -> None:
    line = origin + np.array([onset, 0.0, 0.005])
    vis.add_box(
      line, np.array([0.03, layout.runup_half_width, 0.005]), np.eye(3), LINE_COLOR
    )
    samples, costs = view["samples"], view["costs"]
    if isinstance(samples, list) and isinstance(costs, np.ndarray):
      colors = cost_colors(costs)
      for i in np.argsort(costs)[::-1]:
        r, g, b, a = (float(c) for c in colors[i])
        draw_polyline(vis, origin + samples[i], 0.006, (r, g, b, a))
    path = view["path"]
    if isinstance(path, np.ndarray):
      draw_polyline(vis, origin + path, 0.014, PLAN_COLOR)
    if ref_view:
      # The reference's targets, like DIAL-MPC's: a take-off line, a landing spot.
      at = origin + np.array([ref_view["takeoff_x"], 0.0, 0.006])
      size = np.array([0.02, layout.runup_half_width, 0.006])
      vis.add_box(at, size, np.eye(3), REF_COLOR)
      land = origin + np.array([ref_view["landing_x"], 0.0, 0.02])
      vis.add_sphere(land, 0.06, REF_COLOR)
    if len(trail) > 1:
      draw_polyline(vis, np.array(trail), 0.01, TRAIL_COLOR)

  if args.video:
    env.manager_visualizers["jump_overlay"] = type(
      "Overlay", (), {"debug_vis": staticmethod(draw)}
    )()

  planned_at = 0.0
  # When each robot's gravity and friction changed, and when it was triggered.
  change_t = torch.full((n,), torch.nan, device=device)
  trigger_t = torch.full((n,), torch.nan, device=device)
  # Commanded and actual forward speed while running, to fit the speed lag.
  speed_trace: list[torch.Tensor] = []
  with contextlib.ExitStack() as stack:
    video = None
    if args.video:
      out = args.out.with_name(f"{args.out.stem}_{mode}{args.out.suffix}")
      out.parent.mkdir(parents=True, exist_ok=True)
      video = stack.enter_context(
        media.VideoWriter(out, shape=(720, 1280), fps=round(1.0 / dt))
      )
    for k in range(round(args.duration / dt)):
      t = k * dt
      # Estimates from this step's sensors.
      force, history = feet_sensor.data.force, feet_sensor.data.force_history
      assert force is not None and history is not None
      gravity_est.update(history, robot.data.root_link_lin_vel_w)
      friction_est.update(force, robot.data.site_lin_vel_w[:, foot_sites])
      gravity, friction = estimates()

      # Re-plan every robot that has not been triggered yet.
      running = cmd.phase == JumpPhase.RUN
      ids = running.nonzero().squeeze(-1)
      x_now = cmd.position()[:, 0]
      speed_now = robot.data.root_link_lin_vel_b[:, 0]
      if mppi is not None:
        if k % args.replan_every == 0:
          if tracking is not None and references is not None:
            # A global reference from the latest estimates, then track it.
            reference = references.plan(
              x_now, speed_now, cmd.course(), gravity, friction
            )
            tracking.set_reference(reference)
            if running[0]:
              ref_view.update(
                takeoff_x=float(reference.takeoff_x[0]),
                landing_x=float(reference.landing_x[0]),
              )
          mppi.shift(t - planned_at)
          planned_at = t
          best = mppi.plan(x_now, speed_now, cmd.course(), gravity, friction)
          jumps = best.in_horizon[:, 0]
          trigger_x = torch.where(jumps, best.trigger_x[:, 0], torch.inf)
          aimed = torch.where(running, best.touchdown_x[:, 0], aimed)
          planned_vz = torch.where(running, best.vz_cmd[:, 0], planned_vz)
          if reference is not None:
            aimed = torch.where(running & ~jumps, reference.landing_x, aimed)
            planned_vz = torch.where(running & ~jumps, reference.vz, planned_vz)
          if running[0] and mppi.samples is not None:
            g0 = float(gravity[0])
            pick = torch.randperm(mppi.cfg.num_samples)[: args.samples_shown]
            view["samples"] = [predicted(mppi.samples, int(i), 0.0, g0) for i in pick]
            view["costs"] = mppi.samples.cost[0, pick].cpu().numpy()
            view["path"] = predicted(best, 0, 0.0, g0)
            speeds["samples"] = [
              speed_curve(mppi.samples, 0, int(i), mppi.cfg.dt) for i in pick
            ]
            speeds["chosen"] = speed_curve(best, 0, 0, mppi.cfg.dt)
        run_speed = mppi.speed_at(t - planned_at)
        vz = (planned_vz * jitter).clamp(0.5, 2.0)
        if len(ids):
          cmd.set_plan(run_speed[ids], trigger_x[ids], vz[ids], env_ids=ids)
      elif len(ids):
        assert isinstance(planner, JumpPlanner)
        course = tuple(c[ids] for c in cmd.course())
        plan = planner.plan(
          x_now[ids],
          speed_now[ids],
          (course[0], course[1], course[2]),
          gravity[ids],
          friction[ids],
        )
        vz = (plan.vz * jitter[ids]).clamp(0.5, 2.0)
        cmd.set_plan(plan.run_speed, plan.trigger_x, vz, env_ids=ids)
        aimed[ids] = plan.landing_x
        planned_vz[ids] = plan.vz
        trigger_x[ids] = plan.trigger_x
        if running[0]:
          grid_friction[0] = float(friction[0])
          view["path"] = predicted(None, 0, float(x_now[0]), float(gravity[0]))
      if len(ids):
        used_gravity[ids] = gravity[ids]
        used_friction[ids] = friction[ids]
      if not running[0]:
        view["samples"] = view["costs"] = None  # Keep the last chosen plan only.

      acted_on = cmd.command[:, 0].clone()  # The command this step acts on.
      with torch.no_grad():
        obs, *_ = vec.step(policy(obs))
      fell |= cmd.position()[:, 2] < -0.1
      change_t = torch.where(change.changed & change_t.isnan(), t + dt, change_t)
      left = (cmd.phase != JumpPhase.RUN) & trigger_t.isnan()
      trigger_t = torch.where(left, t + dt, trigger_t)
      speed_trace.append(
        torch.stack(
          [
            acted_on,
            speed_now,
            robot.data.root_link_lin_vel_b[:, 0],
            (running & (cmd.phase == JumpPhase.RUN)).float(),
          ],
          dim=-1,
        )
      )
      trail.append(robot.data.root_link_pos_w[0].cpu().numpy())
      run_x.append(float(cmd.position()[0, 0]))
      run_v.append(float(robot.data.root_link_lin_vel_b[0, 0]))

      if video is not None:
        speed = robot.data.root_link_lin_vel_b[0, 0].item()
        readouts = [
          ("Time", f"{t:4.1f} s"),
          ("Phase", PHASE_NAMES[int(cmd.phase[0])]),
          ("Speed", f"{speed:4.2f} m/s"),
          ("Plan", f"run {cmd.run_speed[0]:.2f}, up {planned_vz[0]:.2f} m/s"),
          ("Gravity", f"{change.gravity[0]:5.2f} (est {gravity[0]:5.2f})"),
          ("Friction", f"{change.friction[0]:4.2f} (est {friction[0]:4.2f})"),
        ]
        frame = hud.draw(env.render(), readouts)  # type: ignore[arg-type]
        sampled = speeds["samples"] if running[0] else []
        chosen = speeds["chosen"]
        frame = draw_speed_panel(
          frame,
          sampled if isinstance(sampled, list) else [],
          view["costs"] if running[0] and mppi is not None else None,  # type: ignore[arg-type]
          chosen if isinstance(chosen, tuple) else None,
          (np.array(run_x), np.array(run_v)),
          layout.edge,
          (layout.pad_start, layout.pad_end),
          x_max=layout.pad_end + 0.5,
          reference=(
            reference_curve(reference, float(cmd.position()[0, 0]))
            if reference is not None and running[0]
            else None
          ),
        )
        video.add_image(frame)

  success = cmd.success & ~fell
  return {
    "success": success.cpu().numpy(),
    "fell": fell.cpu().numpy(),
    "jumped": (cmd.phase >= JumpPhase.FLIGHT).cpu().numpy(),
    "aimed": aimed.cpu().numpy(),
    "landed": cmd.landing_x.cpu().numpy(),
    "vz_cmd": cmd.takeoff_cmd[:, 1].cpu().numpy(),
    "vz": cmd.takeoff_vel[:, 2].cpu().numpy(),
    "vx_cmd": cmd.takeoff_cmd[:, 0].cpu().numpy(),
    "vx": cmd.takeoff_vel[:, 0].cpu().numpy(),
    "takeoff_x": cmd.takeoff_x.cpu().numpy(),
    "trigger_x": cmd.trigger_x.cpu().numpy(),
    "landing_vx": cmd.landing_vel[:, 0].cpu().numpy(),
    "final_x": cmd.position()[:, 0].cpu().numpy(),
    "edge": cmd.course()[0].cpu().numpy(),
    "gravity": change.gravity.cpu().numpy(),
    "friction": change.friction.cpu().numpy(),
    "gravity_used": used_gravity.cpu().numpy(),
    "change_t": change_t.cpu().numpy(),
    "trigger_t": trigger_t.cpu().numpy(),
    "friction_used": used_friction.cpu().numpy(),
    # (T, B, 4): command, speed before and after the step, still running.
    "speed_trace": torch.stack(speed_trace).cpu().numpy(),
  }


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--task", default=TASK_ID)
  p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--gravity", type=float, nargs=2, default=DEMO_GRAVITY_RANGE)
  p.add_argument("--friction", type=float, nargs=2, default=DEMO_FRICTION_RANGE)
  p.add_argument("--change-before", type=float, default=DEMO_CHANGE_BEFORE_EDGE)
  p.add_argument("--change-prob", type=float, default=1.0)
  p.add_argument("--duration", type=float, default=5.0)
  p.add_argument("--video", action="store_true", help="Record robot 0.")
  p.add_argument("--out", type=Path, default=Path("logs/mppi/box_jump.mp4"))
  p.add_argument("--save", type=Path, default=None, help="Save outcomes (.npz).")
  p.add_argument(
    "--jump-model",
    type=Path,
    default=None,
    help="Calibrated jump model. Default: jump_model.json next to the checkpoint,"
    " if there is one (mppi/fit_jump_model.py).",
  )
  p.add_argument("--vz-jitter", type=float, default=0.0, help=argparse.SUPPRESS)
  p.add_argument(
    "--planner",
    choices=("tracking", "mppi", "grid"),
    default="tracking",
    help="tracking: short-horizon MPPI tracking a global reference; mppi: one"
    " long-horizon MPPI over the whole run-up and jump; grid: a grid search.",
  )
  p.add_argument(
    "--replan-every", type=int, default=5, help="MPPI: policy steps per re-plan."
  )
  p.add_argument("--samples-shown", type=int, default=32, help="Drawn in the video.")
  p.add_argument("--cam-distance", type=float, default=CAMERA[0])
  p.add_argument("--cam-elevation", type=float, default=CAMERA[1])
  p.add_argument(
    "--cam-azimuth",
    type=float,
    default=CAMERA[2],
    help="0 looks forward, 90 from the left.",
  )
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  env = make_env(args)
  vec, policy = load_policy(env, args)
  model_path = args.jump_model or args.checkpoint.parent / "jump_model.json"
  model = JumpModelCfg.load(model_path) if model_path.exists() else JumpModelCfg()
  print(f"[jump] Jump model: {model_path if model_path.exists() else 'defaults'}")
  planner: JumpPlanner | JumpMppi
  if args.planner == "tracking":
    planner = JumpTrackingMppi(args.num_envs, model, device=args.device)
  elif args.planner == "mppi":
    planner = JumpMppi(args.num_envs, model, JumpMppiCfg(), device=args.device)
  else:
    planner = JumpPlanner(model, device=args.device)

  results = {}
  print(
    f"\n[jump] {args.planner} planner, {args.num_envs} robots; gravity {args.gravity}, friction"
    f" {args.friction} from {args.change_before} m before the edge"
    f" (probability {args.change_prob})."
  )
  print(
    f"[jump] {'mode':8s} success  fell  jumped  touch-down error  take-off vz error"
  )
  for mode in args.modes:
    r = run_mode(mode, env, vec, policy, planner, args)
    results[mode] = r
    jumped = r["jumped"]
    err = np.abs(r["landed"] - r["aimed"])[jumped]
    vz_err = np.abs(r["vz"] - r["vz_cmd"])[jumped]
    print(
      f"[jump] {mode:8s} {r['success'].mean():6.1%} {r['fell'].mean():6.1%}"
      f" {jumped.mean():6.1%}   {np.median(err) if err.size else np.nan:6.3f} m"
      f"          {np.median(vz_err) if vz_err.size else np.nan:6.3f} m/s"
    )
  if args.save:
    args.save.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
      args.save, **{f"{m}_{k}": v for m, r in results.items() for k, v in r.items()}
    )
    print(f"[jump] Saved {args.save}")
  env.close()


if __name__ == "__main__":
  main()
