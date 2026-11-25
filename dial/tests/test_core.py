"""Tests for the DIAL-MPC planner core."""

import torch

from dial.core import Dial, DialCfg


def test_controls_pass_through_the_nodes_and_shift_moves_the_plan():
  dial = Dial(DialCfg(num_samples=8, horizon_steps=20, num_nodes=5), action_dim=1)
  dial.nodes = torch.linspace(0.0, 1.0, 6)[:, None]  # A ramp over 0.4 s.
  controls = dial.controls(dial.nodes)
  assert controls.shape == (20, 1)
  assert torch.allclose(controls[::4, 0], dial.nodes[:5, 0])  # Nodes every 0.08 s.
  dial.shift()
  # One 0.02 s step later, every node moves a quarter of the way to the next.
  assert torch.allclose(dial.nodes[0, 0], torch.tensor(0.05))
  assert torch.allclose(dial.nodes[-1, 0], torch.tensor(1.0))  # Held at the end.


def test_improve_moves_the_plan_towards_higher_reward():
  torch.manual_seed(0)
  dial = Dial(DialCfg(num_samples=256), action_dim=2)

  def reward(controls: torch.Tensor) -> torch.Tensor:
    return -(controls - 0.5).square().mean(dim=(1, 2))  # Best: all 0.5.

  for _ in range(5):
    dial.improve(reward)
  # The first node is the control already applied and never moves; the rest do.
  assert torch.allclose(dial.nodes[0], torch.zeros(2))
  assert ((dial.nodes[1:] - 0.5).abs() < 0.15).all()
