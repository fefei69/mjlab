"""DIAL-MPC on the box-jump course: joint targets planned with full-robot rollouts.

There is no learned policy. Every control step (0.02 s), DIAL-MPC (`core.py`)
rolls 2,048 noisy joint-target sequences forward over 0.4 s in a second,
batched copy of the course, and moves its plan to the reward-weighted mean.

The rewards follow the authors' examples: a body target that moves along the
course in time (here at a set running speed, ending over the landing box),
staying level and facing forward, a trot schedule for the feet while running,
and contact rewards for feet on the landing box once the target is over it. We
add a penalty for feet or trunk below the box tops, which is a fall into the gap.

The planning copies use the planner's belief about gravity and foot friction:
fixed at nominal, estimated online (`mppi/jump_planner.py`), or the true values,
so the comparison matches the MPPI + RL planner's. The robot itself runs in the
same environment as that planner's demos, with gravity and friction changing at
the yellow line.

  uv run python -m dial.box_jump --gravity 7 --friction 0.5
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from mppi.jump_planner import NOMINAL_GRAVITY, FrictionEstimator, GravityEstimator
from mppi.run_box_jump import FEET, LINE_COLOR, make_env
from mppi.viz import PLAN_COLOR, TRAIL_COLOR, Hud, cost_colors, draw_polyline

import mjlab.tasks  # noqa: F401  (populates the task registry)
from dial.core import Dial, DialCfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.scene import Scene
from mjlab.sensor import ContactSensor
from mjlab.sim.sim import Simulation
from mjlab.tasks.box_jump.course import (
  DEMO_CHANGE_BEFORE_EDGE,
  DEMO_COURSE,
  box_jump_terrain,
)
from mjlab.tasks.box_jump.mdp import physics_change as PhysicsChange
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply
from mjlab.viewer.debug_visualizer import DebugVisualizer

# The task only supplies the robot, the course and the sensors; no policy runs.
TASK_ID = "Mjlab-BoxJump-Unitree-Go1"
MODES = ("nominal", "online", "oracle")
FOOT_GEOMS = tuple(f"{leg}_foot_collision" for leg in FEET)
# Trot, as in DIAL-MPC's examples: diagonal pairs (FR + RL, FL + RR).
TROT_PHASE = (0.0, 0.5, 0.5, 0.0)
TARGET_COLOR = (1.0, 0.45, 0.85, 0.9)


@dataclass(kw_only=True)
class RewardCfg:
  speed: float = 1.5
  """Speed of the body target along the course, m/s."""
  accel: float = 1.5
  """How fast the body target speeds up from rest, m/s^2."""
  height: float = 0.28
  """Trunk height above the box tops."""
  w_pos: float = 1.0
  w_height: float = 1.0
  w_upright: float = 1.0
  w_yaw: float = 0.3
  w_vel: float = 0.5
  w_gait: float = 0.1
  w_support: float = 2.0
  """Per foot below the box tops (in the gap or on the ground)."""
  w_fall: float = 10.0
  """Trunk low enough to have fallen."""
  w_land: float = 0.1
  """Per foot on the landing box, once the target is over it."""
  jump_arc: float = 0.0
  """Extra height of the body target over the gap, m: a reference arc for the
  jump, as MuJoCo MPC's backflip follows a reference height. 0 for none."""
  gait_cadence: float = 2.0
  gait_duty: float = 0.45
  gait_amplitude: float = 0.08


class Course:
  """The demo course, in metres along +x from the spawn point."""

  def __init__(self) -> None:
    layout = DEMO_COURSE.layout(0.0)
    self.edge = layout.edge
    self.pad_start, self.pad_end = layout.pad_start, layout.pad_end
    self.pad_half_width = layout.pad_half_width
    self.goal = 0.5 * (self.pad_start + self.pad_end)

  def target(self, t: float, cfg: RewardCfg) -> tuple[float, float]:
    """The body target's position and speed along +x at time `t`."""
    ramp = cfg.speed / cfg.accel
    if t < ramp:
      x, v = 0.5 * cfg.accel * t**2, cfg.accel * t
    else:
      x, v = 0.5 * cfg.speed * ramp + cfg.speed * (t - ramp), cfg.speed
    return (self.goal, 0.0) if x >= self.goal else (x, v)

  def arc(self, x: float, cfg: RewardCfg) -> float:
    """Height of the body target's jump arc at target position `x`."""
    start, end = self.edge - 0.3, self.pad_start + 0.3
    if cfg.jump_arc <= 0.0 or not start < x < end:
      return 0.0
    return cfg.jump_arc * math.sin(math.pi * (x - start) / (end - start))


class PlanningWorlds:
  """Copies of the course and robot, stepped together at the planner's timestep,
  with the planner's belief about gravity and foot friction."""

  def __init__(
    self, num_worlds: int, plan_dt: float, ctrl_dt: float, device: str
  ) -> None:
    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs = num_worlds
    cfg.scene.terrain = box_jump_terrain(DEMO_COURSE)
    cfg.scene.sensors = ()
    self.scene = Scene(cfg.scene, device)
    model = self.scene.compile()
    sim_cfg = replace(cfg.sim, mujoco=replace(cfg.sim.mujoco, timestep=plan_dt))
    self.sim = Simulation(num_envs=num_worlds, cfg=sim_cfg, model=model, device=device)
    self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
    # Writable copies of gravity and friction (unexpanded fields are shared views).
    self.sim.expand_model_fields(("opt.gravity", "geom_friction"))
    self.dt = ctrl_dt
    self.substeps = max(round(ctrl_dt / plan_dt), 1)
    robot = self.scene["robot"]
    assert isinstance(robot, Entity)
    self.robot = robot
    ctrl = torch.full((robot.num_joints,), -1, dtype=torch.long)
    for act in robot.actuators:
      for local, glob in zip(
        act.target_ids.tolist(), act.global_ctrl_ids.tolist(), strict=True
      ):
        ctrl[local] = glob
    self.ctrl = ctrl.to(device)
    self.default = robot.data.default_joint_pos[0].clone()
    self.feet = robot.find_sites(FEET, preserve_order=True)[0]
    self.foot_geoms = robot.indexing.geom_ids[
      robot.find_geoms(FOOT_GEOMS, preserve_order=True)[0]
    ]
    self.free_q = robot.indexing.free_joint_q_adr
    self.free_v = robot.indexing.free_joint_v_adr
    self.origin = self.scene.env_origins[0].clone()
    self.last_paths: torch.Tensor | None = None

  def set_physics(self, gravity: float, friction: float) -> None:
    model = self.sim.model
    model.opt.gravity[:] = torch.tensor([0.0, 0.0, -gravity], device=self.origin.device)
    model.geom_friction[:, self.foot_geoms, 0] = friction

  def targets(self, u: torch.Tensor) -> torch.Tensor:
    """Controls in [-1, 1] to joint targets: +-0.8 rad around the standing pose."""
    return self.default + 0.8 * u

  def rollout(
    self,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    controls: torch.Tensor,
    t0: float,
    course: Course,
    cfg: RewardCfg,
  ) -> torch.Tensor:
    """Mean per-step reward of every world's control sequence, (K,)."""
    data = self.sim.data
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    self.sim.forward()
    targets = self.targets(controls)
    total = torch.zeros(controls.shape[0], device=controls.device)
    paths = []
    for h in range(controls.shape[1]):
      data.ctrl[:, self.ctrl] = targets[:, h]
      for _ in range(self.substeps):
        self.sim.step()
      reward, trunk = self.reward(t0 + (h + 1) * self.dt, course, cfg)
      total += reward
      paths.append(trunk)
    self.last_paths = torch.stack(paths, dim=1)
    return total / controls.shape[1]

  def reward(
    self, t: float, course: Course, cfg: RewardCfg
  ) -> tuple[torch.Tensor, torch.Tensor]:
    data = self.sim.data
    qpos = data.qpos[:, self.free_q]
    trunk = qpos[:, :3] - self.origin
    quat = qpos[:, 3:7]
    vel = data.qvel[:, self.free_v[:3]]
    feet = self.robot.data.site_pos_w[:, self.feet] - self.origin
    x_tar, v_tar = course.target(t, cfg)

    up = quat_apply(
      quat, torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand_as(trunk)
    )
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    r_pos = -((trunk[:, 0] - x_tar).square() + trunk[:, 1].square())
    r_height = -(trunk[:, 2] - cfg.height - course.arc(x_tar, cfg)).square()
    r_upright = -(up - torch.tensor([0.0, 0.0, 1.0], device=up.device)).square().sum(-1)
    r_yaw = -yaw.square()
    r_vel = -(vel[:, 0] - v_tar).square()

    # Trot while running on the long box, away from the gap.
    running = (trunk[:, 0] < course.edge - 0.6).float()
    phase = torch.tensor(TROT_PHASE, device=quat.device) * 2 * math.pi
    angle = torch.remainder(
      t * 2 * math.pi * cfg.gait_cadence + math.pi - phase, 2 * math.pi
    )
    angle = (angle - math.pi) * 0.5 / (1 - cfg.gait_duty)
    lift = torch.cos(angle.clamp(-math.pi / 2, math.pi / 2)) * cfg.gait_amplitude
    r_gait = -(((feet[..., 2] - 0.02 - lift) / 0.05).square()).sum(-1) * running

    below = (feet[..., 2] < -0.05).float().sum(-1)
    fallen = (trunk[:, 2] < 0.12).float()
    on_pad = (
      (feet[..., 0] > course.pad_start)
      & (feet[..., 0] < course.pad_end)
      & (feet[..., 1].abs() < course.pad_half_width)
      & (feet[..., 2] < 0.05)
    ).float().sum(-1) * float(x_tar >= course.pad_start)

    reward = (
      cfg.w_pos * r_pos
      + cfg.w_height * r_height
      + cfg.w_upright * r_upright
      + cfg.w_yaw * r_yaw
      + cfg.w_vel * r_vel
      + cfg.w_gait * r_gait
      - cfg.w_support * below
      - cfg.w_fall * fallen
      + cfg.w_land * on_pad
    )
    return reward, trunk


def run(
  mode: str,
  env: ManagerBasedRlEnv,
  worlds: PlanningWorlds,
  args: argparse.Namespace,
) -> dict[str, float]:
  torch.manual_seed(args.seed)
  env.reset()
  dt = env.step_dt
  device = env.device
  robot = env.scene["robot"]
  assert isinstance(robot, Entity)
  change = env.event_manager.get_term_cfg("physics_change").func
  assert isinstance(change, PhysicsChange)
  feet_sensor = env.scene["feet_ground_contact"]
  assert isinstance(feet_sensor, ContactSensor)
  foot_sites = robot.find_sites(FEET, preserve_order=True)[0]
  mass = float(env.sim.mj_model.body_mass[robot.indexing.body_ids.cpu()].sum())
  gravity_est = GravityEstimator(1, dt, mass, device=device)
  friction_est = FrictionEstimator(1, device=device)
  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, JointPositionAction)
  # Joint target = action * scale + offset; DIAL-MPC plans the targets directly.
  scale = torch.as_tensor(term._scale, device=device).reshape(-1)[-12:]  # noqa: SLF001
  offset = torch.as_tensor(term._offset, device=device).reshape(-1)[-12:]  # noqa: SLF001
  dial = Dial(
    DialCfg(
      num_samples=args.num_samples,
      horizon_steps=args.horizon_steps,
      num_nodes=args.nodes,
      diffuse_steps=args.rounds,
      diffuse_steps_init=args.rounds_init,
      ctrl_dt=dt,
    ),
    robot.num_joints,
    device,
  )
  course, cfg = Course(), RewardCfg(speed=args.speed, jump_arc=args.jump_arc)
  origin = env.scene.env_origins[0].cpu().numpy()
  onset = course.edge - args.change_before
  plan_ms: list[float] = []
  trail: list[np.ndarray] = []
  view: dict[str, object] = {}
  lowest = math.inf

  hud = Hud(
    "Box jump, DIAL-MPC",
    f"Gravity and friction: {mode}; no learned policy",
    [
      ("Sampled plans (light = high reward)", "ramp"),
      ("Current plan", PLAN_COLOR),
      ("Body target", TARGET_COLOR),
      ("Path driven", TRAIL_COLOR),
      ("Gravity and friction change", LINE_COLOR),
    ],
  )

  def draw(vis: DebugVisualizer) -> None:
    line = origin + np.array([onset, 0.0, 0.005])
    vis.add_box(line, np.array([0.03, 0.5, 0.005]), np.eye(3), LINE_COLOR)
    samples, rewards = view.get("samples"), view.get("rewards")
    if isinstance(samples, np.ndarray) and isinstance(rewards, np.ndarray):
      colors = cost_colors(-rewards)
      for i in np.argsort(-rewards)[::-1]:
        r, g, b, a = (float(c) for c in colors[i])
        draw_polyline(vis, origin + samples[i], 0.006, (r, g, b, a))
    plan = view.get("plan")
    if isinstance(plan, np.ndarray):
      draw_polyline(vis, origin + plan, 0.014, PLAN_COLOR)
    target = view.get("target")
    if isinstance(target, float):
      vis.add_sphere(origin + np.array([target, 0.0, cfg.height]), 0.05, TARGET_COLOR)
    if len(trail) > 1:
      draw_polyline(vis, np.array(trail), 0.01, TRAIL_COLOR)

  if args.video:
    env.manager_visualizers["dial_overlay"] = type(
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
      t = k * dt
      force, history = feet_sensor.data.force, feet_sensor.data.force_history
      assert force is not None and history is not None
      gravity_est.update(history, robot.data.root_link_lin_vel_w)
      friction_est.update(force, robot.data.site_lin_vel_w[:, foot_sites])
      if mode == "nominal":
        belief = (NOMINAL_GRAVITY, 1.0)
      elif mode == "online":
        belief = (float(gravity_est.gravity[0]), float(friction_est.friction[0]))
      else:
        belief = (float(change.gravity[0]), float(change.friction[0]))
      worlds.set_physics(*belief)

      qpos = env.sim.data.qpos[0].clone()
      qvel = env.sim.data.qvel[0].clone()
      torch.cuda.synchronize()
      start = time.perf_counter()
      dial.improve(
        lambda u, q=qpos, v=qvel, t0=t: worlds.rollout(q, v, u, t0, course, cfg)
      )
      torch.cuda.synchronize()
      plan_ms.append(1e3 * (time.perf_counter() - start))

      target = worlds.targets(dial.action())
      env.step(((target - offset) / scale)[None])
      dial.shift()

      rel = robot.data.root_link_pos_w[0].cpu().numpy() - origin
      lowest = min(lowest, float(rel[2]))
      trail.append(robot.data.root_link_pos_w[0].cpu().numpy())
      if video is not None:
        paths = worlds.last_paths
        rewards = dial.last_rewards
        assert paths is not None and rewards is not None
        pick = torch.randperm(paths.shape[0] - 1)[: args.samples_shown]
        view["samples"] = paths[pick].cpu().numpy()
        view["rewards"] = rewards[pick].cpu().numpy()
        view["plan"] = paths[-1].cpu().numpy()  # The plan being refined.
        view["target"] = course.target(t, cfg)[0]
        speed = robot.data.root_link_lin_vel_b[0, 0].item()
        readouts = [
          ("Time", f"{t:4.1f} s"),
          ("Speed", f"{speed:4.2f} m/s"),
          ("Plan time", f"{plan_ms[-1]:4.0f} ms per 20 ms step"),
          ("Gravity", f"{change.gravity[0]:5.2f} (belief {belief[0]:5.2f})"),
          ("Friction", f"{change.friction[0]:4.2f} (belief {belief[1]:4.2f})"),
        ]
        video.add_image(hud.draw(env.render(), readouts))  # type: ignore[arg-type]

  rel = robot.data.root_link_pos_w[0].cpu().numpy() - origin
  upright = float(-robot.data.projected_gravity_b[0, 2]) > 0.8
  on_pad = (
    course.pad_start < rel[0] < course.pad_end
    and abs(rel[1]) < course.pad_half_width
    and rel[2] > 0.1
  )
  return {
    "success": float(on_pad and upright and lowest > -0.1),
    "fell": float(lowest < -0.1),
    "final_x": float(rel[0]),
    "plan_ms": float(np.median(plan_ms[1:])) if len(plan_ms) > 1 else math.nan,
  }


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
  p.add_argument("--gravity", type=float, default=7.0, help="After the change.")
  p.add_argument("--friction", type=float, default=0.5, help="After the change.")
  p.add_argument("--change-before", type=float, default=DEMO_CHANGE_BEFORE_EDGE)
  p.add_argument("--speed", type=float, default=1.5, help="Body target speed, m/s.")
  p.add_argument("--num-samples", type=int, default=2048)
  p.add_argument(
    "--horizon-steps", type=int, default=20, help="Control steps of 0.02 s."
  )
  p.add_argument("--nodes", type=int, default=5, help="Plan nodes (Hnode).")
  p.add_argument("--rounds", type=int, default=2, help="Annealing rounds per step.")
  p.add_argument("--rounds-init", type=int, default=10)
  p.add_argument(
    "--jump-arc", type=float, default=0.0, help="Body target arc over the gap, m."
  )
  p.add_argument(
    "--plan-dt",
    type=float,
    default=0.01,
    help="Planner physics step, s. DIAL-MPC uses 0.02, but against a robot"
    " simulated at 0.005 that loses its footing within 2 s.",
  )
  p.add_argument("--duration", type=float, default=7.0)
  p.add_argument("--video", action="store_true")
  p.add_argument("--samples-shown", type=int, default=32)
  p.add_argument("--out", type=Path, default=Path("logs/dial/box_jump.mp4"))
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  env_args = argparse.Namespace(
    task=TASK_ID,
    num_envs=1,
    change_before=args.change_before,
    gravity=(args.gravity, args.gravity),
    friction=(args.friction, args.friction),
    change_prob=1.0,
    video=args.video,
    device=args.device,
  )
  env = make_env(env_args)
  worlds = PlanningWorlds(args.num_samples + 1, args.plan_dt, env.step_dt, args.device)
  assert worlds.sim.data.qpos.shape[1] == env.sim.data.qpos.shape[1]
  print(
    f"\n[dial] DIAL-MPC, {args.num_samples} samples,"
    f" {0.02 * args.horizon_steps:.1f} s horizon; gravity"
    f" {args.gravity} and friction {args.friction} from {args.change_before} m"
    " before the edge."
  )
  for mode in args.modes:
    r = run(mode, env, worlds, args)
    print(
      f"[dial] {mode:8s} {'landed on the box' if r['success'] else 'missed'}"
      f"{' (fell)' if r['fell'] else ''}, ends {r['final_x']:.2f} m along;"
      f" planning {r['plan_ms']:.0f} ms per 20 ms control step."
    )
  env.close()


if __name__ == "__main__":
  main()
