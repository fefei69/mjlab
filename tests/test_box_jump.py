"""Tests for the box-jump course, per-world gravity and the mid-run change."""

from __future__ import annotations

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.box_jump.course import BoxJumpCourseCfg, box_jump_terrain
from mjlab.tasks.box_jump.mdp import JumpCommand, JumpPhase, physics_change
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
from mjlab.terrains.terrain_generator import TerrainGenerator

FEET = tuple(f"{leg}_foot_collision" for leg in ("FR", "FL", "RR", "RL"))

_FREE_BOX = """
  <mujoco>
    <worldbody>
      <body name="base" pos="0 0 1">
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def test_course_matches_its_layout():
  """On every row of a curriculum the boxes stand where `layout` puts them."""
  course = BoxJumpCourseCfg(gap_range=(0.3, 0.7), pad_length_range=(1.4, 0.8))
  terrain = box_jump_terrain(course, num_rows=3)
  assert terrain.terrain_generator is not None
  generator = TerrainGenerator(terrain.terrain_generator)
  spec = mujoco.MjSpec()
  generator.compile(spec)
  model = spec.compile()
  # Boxes standing on the ground up to the course height: (x min, x max).
  on_ground = abs(model.geom_pos[:, 2] - course.height / 2) < 1e-6
  tall = abs(model.geom_size[:, 2] - course.height / 2) < 1e-6
  boxes = {
    (round(float(p[0] - s[0]), 4), round(float(p[0] + s[0]), 4))
    for p, s in zip(
      model.geom_pos[on_ground & tall], model.geom_size[on_ground & tall], strict=True
    )
  }
  assert len(boxes) == 6
  for row in range(3):
    origin = generator.terrain_origins[row, 0]
    assert origin[2] == pytest.approx(course.height)
    layout = course.layout(row / 2)
    x = origin[0]
    assert (round(x - course.start_offset, 4), round(x + layout.edge, 4)) in boxes
    assert (round(x + layout.pad_start, 4), round(x + layout.pad_end, 4)) in boxes
  assert course.layout(1.0).pad_start - course.layout(1.0).edge == pytest.approx(0.7)


def test_gravity_can_differ_per_world(device):
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(_FREE_BOX))
  scene = Scene(
    SceneCfg(num_envs=2, env_spacing=3.0, entities={"robot": entity_cfg}), device
  )
  sim = Simulation(
    num_envs=2, cfg=SimulationCfg(njmax=20), model=scene.compile(), device=device
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  sim.expand_model_fields(("opt.gravity",))
  assert sim.get_default_field("opt.gravity").tolist() == pytest.approx(
    [0.0, 0.0, -9.81]
  )
  sim.model.opt.gravity[1] = torch.tensor([0.0, 0.0, -5.0], device=device)
  steps = 10
  for _ in range(steps):
    sim.step()
  t = steps * sim.mj_model.opt.timestep
  assert sim.data.qvel[:, 2].tolist() == pytest.approx([-9.81 * t, -5.0 * t], rel=1e-3)


def test_change_comes_at_each_robots_line_and_reset_undoes_it(device):
  cfg = unitree_go1_flat_env_cfg(play=True)
  cfg.scene.num_envs = 2
  cfg.scene.terrain = box_jump_terrain(BoxJumpCourseCfg())
  cfg.events = {k: v for k, v in cfg.events.items() if v.mode == "reset"}
  cfg.events["reset_base"].params["pose_range"] = {}
  cfg.events["physics_change"] = EventTermCfg(
    func=physics_change,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FEET),
      "gravity_range": (7.0, 7.0),
      "friction_range": (0.4, 0.4),
      "onset_x": (1.0, 1.0),
    },
  )
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    env.reset()
    robot = env.scene["robot"]
    feet = robot.indexing.geom_ids[robot.find_geoms(FEET)[0]]
    # Robot 0 is already past its line; robot 1 stays at the spawn point.
    pose = robot.data.root_link_pose_w.clone()
    pose[0, 0] += 1.5
    robot.write_root_link_pose_to_sim(pose)
    env.step(torch.zeros(2, env.action_manager.total_action_dim, device=device))

    gravity = env.sim.model.opt.gravity[:, 2]
    friction = env.sim.model.geom_friction[:, feet, 0]
    assert gravity.tolist() == pytest.approx([-7.0, -9.81])
    assert friction[0].tolist() == pytest.approx([0.4] * 4)
    assert friction[1].tolist() == pytest.approx([1.0] * 4)
    term = env.event_manager.get_term_cfg("physics_change").func
    assert isinstance(term, physics_change)
    assert term.changed.tolist() == [True, False]

    env.reset(env_ids=torch.tensor([0], device=device))
    assert env.sim.model.opt.gravity[0, 2].item() == pytest.approx(-9.81)
    assert env.sim.model.geom_friction[0, feet, 0].tolist() == pytest.approx([1.0] * 4)
    assert not term.changed[0]
  finally:
    env.close()


def test_jump_phases_follow_the_robot(device):
  """Past the trigger: take-off; in the air: flight; on the box: landed."""
  cfg = load_env_cfg("Mjlab-BoxJump-Unitree-Go1")
  cfg.scene.num_envs = 2
  # Nominal gravity and friction throughout: a predictable command.
  cfg.events["physics_change"].params.update(
    start_gravity_range=(9.81, 9.81), start_friction_range=(1.0, 1.0), prob=0.0
  )
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    env.reset()
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("jump")
    assert isinstance(cmd, JumpCommand)
    zero = torch.zeros(2, env.action_manager.total_action_dim, device=device)
    edge, pad_start, _ = cmd.course()
    origin = env.scene.env_origins

    stand = origin[0, 2] + robot.data.default_root_state[0, 2]

    def place(x: float, lift: float = 0.0, vz: float = 0.0) -> None:
      pose = robot.data.root_link_pose_w.clone()
      pose[0, 0] = origin[0, 0] + x
      pose[0, 2] = stand + lift
      robot.write_root_link_pose_to_sim(pose)
      vel = torch.zeros(2, 6, device=device)
      vel[0, 2] = vz
      robot.write_root_link_velocity_to_sim(vel)

    place(edge[0].item() - 0.05)
    env.step(zero)
    assert cmd.phase.tolist() == [JumpPhase.TAKEOFF, JumpPhase.RUN]
    assert 0.5 <= cmd.command[0, 3].item() <= 2.0  # Take off at this speed.
    assert cmd.command[1, 3] == 0.0

    place(edge[0].item() + 0.05, lift=0.3, vz=1.5)
    lifted = False
    for _ in range(4):
      env.step(zero)
      lifted |= bool(cmd.liftoff_now[0])
    assert lifted and cmd.phase[0] == JumpPhase.FLIGHT

    place(pad_start[0].item() + 0.5)
    env.step(zero)
    assert cmd.phase[0] == JumpPhase.LANDED
    assert cmd.success.tolist() == [True, False]
    assert cmd.command[0, 0] == 0.0 and cmd.command[0, 3] == 0.0  # Stop.
  finally:
    env.close()
