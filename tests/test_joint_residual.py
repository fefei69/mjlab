"""Tests for the per-joint fault score, on made-up data with a known answer."""

import pytest
import torch

from mjlab.tasks.velocity.mdp.joint_residual import JointModel, JointResidual

NAMES = ["hip", "thigh", "calf"]
DT = 0.02


@pytest.fixture
def model() -> JointModel:
  """A nominal model that predicts no velocity change, with unit noise."""
  m = JointModel(NAMES)
  for layer in m.net:
    if isinstance(layer, torch.nn.Linear):
      torch.nn.init.zeros_(layer.weight)
      torch.nn.init.zeros_(layer.bias)
  m.in_std.fill_(1.0)
  m.out_std.fill_(1.0)
  m.noise.fill_(1.0)
  m.threshold.fill_(1.0)
  return m.eval()


def run(model: JointModel, shortfall: torch.Tensor, steps: int) -> torch.Tensor:
  """Score after `steps` steps where each joint falls short by `shortfall`."""
  gen = torch.Generator().manual_seed(0)
  residual = JointResidual(model, num_envs=64, dt=DT)
  for _ in range(steps):
    q = torch.randn(64, 3, generator=gen)
    target = q + torch.randn(64, 3, generator=gen)
    noise = torch.randn(64, 3, generator=gen)
    # Velocity change toward the target is less than predicted by `shortfall`.
    qd_next = -torch.sign(target - q) * shortfall + noise
    residual.update(q, torch.zeros_like(q), target, qd_next)
  return residual.score


def test_noise_alone_stays_below_the_threshold(model: JointModel) -> None:
  score = run(model, torch.zeros(3), steps=200)
  assert score.abs().max() < model.threshold


def test_a_motor_that_falls_short_is_flagged_quickly(model: JointModel) -> None:
  """A shortfall of two usual errors on the calf crosses the threshold in 0.5 s."""
  score = run(model, torch.tensor([0.0, 0.0, 2.0]), steps=round(0.5 / DT))
  assert (score[:, 2] > model.threshold).all()
  assert (score[:, :2] < model.threshold).all()
