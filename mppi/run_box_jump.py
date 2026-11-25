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
from mppi.jump_planner import (
  NOMINAL_GRAVITY,
  FrictionEstimator,
  GravityEstimator,
  JumpModelCfg,
  JumpPlanner,
)
from mppi.viz import Hud, Rgba

TASK_ID = "Mjlab-BoxJump-Unitree-Go1-History"
MODES = ("nominal", "online", "oracle")
FEET = ("FR", "FL", "RR", "RL")
LINE_COLOR: Rgba = (1.0, 0.85, 0.25, 0.9)
AIM_COLOR: Rgba = (0.30, 1.00, 0.85, 1.0)
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
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    entity_name="robot",
    distance=4.0,
    elevation=-12.0,
    azimuth=90.0,
    width=1280,
    height=720,
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


def run_mode(
  mode: str,
  env: ManagerBasedRlEnv,
  vec: RslRlVecEnvWrapper,
  policy,
  planner: JumpPlanner,
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

  fell = torch.zeros(n, dtype=torch.bool, device=device)
  # Calibration runs scale each robot's take-off command by a fixed factor.
  jitter = 1 + args.vz_jitter * (2 * torch.rand(n, device=device) - 1)
  aimed = torch.full((n,), torch.nan, device=device)
  # The gravity and friction the plan was made with, frozen at the trigger.
  used_gravity = torch.full((n,), torch.nan, device=device)
  used_friction = torch.full((n,), torch.nan, device=device)
  planned_vz = torch.zeros(n, device=device)
  origin = env.scene.env_origins[0].cpu().numpy()
  layout = DEMO_COURSE.layout(0.0)
  onset = layout.edge - args.change_before
  hud = Hud(
    "Box jump with the planner",
    f"Gravity and friction: {mode}",
    [("Gravity and friction change", LINE_COLOR), ("Planned touch-down", AIM_COLOR)],
  )

  def draw(vis: DebugVisualizer) -> None:
    line = origin + np.array([onset, 0.0, 0.005])
    vis.add_box(
      line, np.array([0.03, layout.runup_half_width, 0.005]), np.eye(3), LINE_COLOR
    )
    if torch.isfinite(aimed[0]):
      at = origin + np.array([aimed[0].item(), 0.0, 0.01])
      vis.add_sphere(at, 0.05, AIM_COLOR)

  if args.video:
    env.manager_visualizers["jump_overlay"] = type(
      "Overlay", (), {"debug_vis": staticmethod(draw)}
    )()

  with contextlib.ExitStack() as stack:
    video = None
    if args.video:
      out = args.out.with_name(f"{args.out.stem}_{mode}{args.out.suffix}")
      out.parent.mkdir(parents=True, exist_ok=True)
      video = stack.enter_context(
        media.VideoWriter(out, shape=(720, 1280), fps=round(1.0 / dt))
      )
    for k in range(round(args.duration / dt)):
      # Estimates from this step's sensors.
      force, history = feet_sensor.data.force, feet_sensor.data.force_history
      assert force is not None and history is not None
      gravity_est.update(history, robot.data.root_link_lin_vel_w)
      friction_est.update(force, robot.data.site_lin_vel_w[:, foot_sites])
      if mode == "nominal":
        gravity = torch.full((n,), NOMINAL_GRAVITY, device=device)
        friction = torch.ones(n, device=device)
      elif mode == "online":
        gravity, friction = gravity_est.gravity, friction_est.friction
      else:
        gravity, friction = change.gravity, change.friction

      # Re-plan every robot that has not been triggered yet.
      running = cmd.phase == JumpPhase.RUN
      if running.any():
        ids = running.nonzero().squeeze(-1)
        course = tuple(c[ids] for c in cmd.course())
        plan = planner.plan(
          cmd.position()[ids, 0],
          robot.data.root_link_lin_vel_b[ids, 0],
          (course[0], course[1], course[2]),
          gravity[ids],
          friction[ids],
        )
        vz = (plan.vz * jitter[ids]).clamp(0.5, 2.0)
        cmd.set_plan(plan.run_speed, plan.trigger_x, vz, env_ids=ids)
        aimed[ids] = plan.landing_x
        planned_vz[ids] = plan.vz
        used_gravity[ids] = gravity[ids]
        used_friction[ids] = friction[ids]

      with torch.no_grad():
        obs, *_ = vec.step(policy(obs))
      fell |= cmd.position()[:, 2] < -0.1

      if video is not None:
        speed = robot.data.root_link_lin_vel_b[0, 0].item()
        readouts = [
          ("Time", f"{k * dt:4.1f} s"),
          ("Phase", PHASE_NAMES[int(cmd.phase[0])]),
          ("Speed", f"{speed:4.2f} m/s"),
          ("Plan", f"run {cmd.run_speed[0]:.2f}, up {planned_vz[0]:.2f} m/s"),
          ("Gravity", f"{change.gravity[0]:5.2f} (est {gravity[0]:5.2f})"),
          ("Friction", f"{change.friction[0]:4.2f} (est {friction[0]:4.2f})"),
        ]
        video.add_image(hud.draw(env.render(), readouts))  # type: ignore[arg-type]

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
    "friction_used": used_friction.cpu().numpy(),
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
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  env = make_env(args)
  vec, policy = load_policy(env, args)
  model_path = args.jump_model or args.checkpoint.parent / "jump_model.json"
  model = JumpModelCfg.load(model_path) if model_path.exists() else JumpModelCfg()
  print(f"[jump] Jump model: {model_path if model_path.exists() else 'defaults'}")
  planner = JumpPlanner(model, device=args.device)

  results = {}
  print(
    f"\n[jump] {args.num_envs} robots; gravity {args.gravity}, friction"
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
