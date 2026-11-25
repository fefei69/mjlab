"""Tests for the box-jump MPPI planner."""

import torch

from mppi.jump_mppi import JumpMppi, JumpMppiCfg, plan_path

COURSE = (torch.full((2,), 5.5), torch.full((2,), 6.0), torch.full((2,), 7.0))


def _plan(gravity: float, friction: float, calls: int = 15):
  torch.manual_seed(0)
  mppi = JumpMppi(2, cfg=JumpMppiCfg(num_samples=128))
  x0, v0 = torch.zeros(2), torch.zeros(2)
  mppi.reset(x0, COURSE[0])
  g, mu = torch.full((2,), gravity), torch.full((2,), friction)
  for _ in range(calls - 1):
    mppi.plan(x0, v0, COURSE, g, mu)
  return mppi, mppi.plan(x0, v0, COURSE, g, mu)


def test_plan_lands_on_the_box_after_lifting_off_before_the_edge():
  mppi, best = _plan(9.81, 1.0)
  m = mppi.model
  assert (best.margin > 0).all()
  assert (best.liftoff_x <= 5.5 - m.takeoff_before_edge + 1e-3).all()
  assert ((best.touchdown_x > 6.0) & (best.stop_x < 7.0)).all()
  path = plan_path(best, 0, 0, 9.81, 0.3, mppi.cfg.dt)
  assert path[0, 0] == 0.0 and abs(path[-1, 0] - best.stop_x[0, 0].item()) < 1e-5
  assert path[:, 2].max() > 0.3  # The flight rises above running height.


def test_low_friction_slows_the_approach():
  _, grippy = _plan(9.81, 1.0)
  _, slippery = _plan(9.81, 0.4)
  # Braking is weaker, so it must land slower.
  assert (slippery.liftoff_vx < grippy.liftoff_vx - 0.2).all()


def test_shift_carries_the_plan_forward():
  mppi, _ = _plan(9.81, 1.0, calls=3)
  trigger = mppi.mean[:, mppi.cfg.num_knots].clone()
  later = mppi.speed_at(0.9)
  mppi.shift(0.9)
  assert torch.allclose(mppi.mean[:, mppi.cfg.num_knots], (trigger - 0.9).clamp(min=0))
  assert torch.allclose(mppi.speed_at(0.0), later, atol=1e-5)
