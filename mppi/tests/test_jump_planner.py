"""Tests for the jump planner and its gravity and friction estimators."""

import torch

from mppi.jump_planner import (
  NOMINAL_GRAVITY,
  CentroidalGravityEstimator,
  FrictionEstimator,
  GravityEstimator,
  JumpModelCfg,
  JumpPlanner,
)

COURSE = (torch.tensor([5.5]), torch.tensor([6.0]), torch.tensor([7.0]))


def _plan(planner: JumpPlanner, gravity: float, friction: float = 1.0):
  return planner.plan(
    torch.tensor([0.0]),
    torch.tensor([0.0]),
    COURSE,
    torch.tensor([gravity]),
    torch.tensor([friction]),
  )


def test_plan_lands_on_the_box_and_matches_ballistics():
  planner = JumpPlanner()
  plan = _plan(planner, NOMINAL_GRAVITY)
  cfg = planner.cfg
  assert plan.margin.item() > 0
  takeoff = 5.5 - cfg.takeoff_before_edge
  flight = 2 * plan.vz / NOMINAL_GRAVITY
  expected = takeoff + plan.run_speed * flight
  assert torch.allclose(plan.landing_x, expected, atol=1e-5)
  assert 6.0 + cfg.front_margin < plan.landing_x.item() < 7.0


def test_lower_gravity_and_friction_make_a_gentler_jump():
  planner = JumpPlanner()
  nominal = _plan(planner, NOMINAL_GRAVITY)
  low_g = _plan(planner, 7.0)
  # The same take-off flies further under low gravity, so ask for less of it.
  assert (low_g.run_speed * low_g.vz).item() < (nominal.run_speed * nominal.vz).item()
  slippery = _plan(planner, NOMINAL_GRAVITY, friction=0.4)
  # Braking is weaker: land earlier or slower.
  assert slippery.run_speed.item() <= nominal.run_speed.item()


def test_calibration_round_trips(tmp_path):
  cfg = JumpModelCfg(vz_gain=0.8, flight_offset=-0.1)
  cfg.save(tmp_path / "m.json", note="x")
  assert JumpModelCfg.load(tmp_path / "m.json") == cfg


def test_gravity_estimate_converges_from_foot_forces():
  mass, dt, gravity = 12.0, 0.02, 7.5
  est = GravityEstimator(2, dt, mass)
  vz = torch.zeros(2, 3)
  for k in range(200):
    # Robot 0 stands still; robot 1 bobs up and down.
    accel = torch.tensor([0.0, 3.0 * torch.sin(torch.tensor(k * 0.6)).item()])
    vz[:, 2] += accel * dt
    push = mass * (gravity + accel)  # Split over 4 feet and 4 substeps.
    history = torch.zeros(2, 4, 4, 3)
    history[..., 2] = (push / 4)[:, None, None]
    est.update(history, vz.clone())
  assert torch.allclose(est.gravity, torch.full((2,), gravity), atol=0.05)


def test_friction_estimate_follows_slips_and_respects_grip():
  est = FrictionEstimator(2, prior=1.0)
  force = torch.zeros(2, 4, 3)
  force[..., 2] = 50.0
  force[..., 0] = 20.0  # Tangential to normal: 0.4.
  vel = torch.zeros(2, 4, 3)
  vel[0, :, 0] = 0.5  # Robot 0's feet slide; robot 1's grip.
  for _ in range(30):
    est.update(force, vel)
  assert abs(est.friction[0].item() - 0.4) < 0.01
  assert est.friction[1].item() == 1.0  # Gripping at 0.4 proves nothing new.
  force[1, :, 0] = 60.0  # Gripping at 1.2: friction is at least that.
  est.update(force, vel)
  assert abs(est.friction[1].item() - 1.2) < 1e-5


def test_centroidal_gravity_follows_a_change_within_a_few_steps():
  mass, dt = 12.0, 0.02
  est = CentroidalGravityEstimator(1, dt, mass)
  vel = torch.zeros(1, 3)
  gravity = 9.81
  for k in range(40):
    if k == 20:
      gravity = 7.0  # Gravity drops mid-run.
    accel = 3.0 * torch.sin(torch.tensor(k * 0.6)).item()  # The body bobs.
    vel[:, 2] += accel * dt
    history = torch.zeros(1, 4, 4, 3)
    history[..., 2] = mass * (gravity + accel) / 4  # Split over the four feet.
    est.update(history, vel.clone())
    if k == 23:  # Three steps after the change: the median has moved.
      assert abs(est.gravity.item() - 7.0) < 1e-3
  assert abs(est.gravity.item() - 7.0) < 1e-3
