"""Tests for the online residual model, on made-up data with a known answer."""

import pytest
import torch

from mppi.planar_model import PlanarModel
from mppi.residual_check import (
  Commands,
  gait_noise,
  make_run,
  race_commands,
  score,
  track,
  true_change,
  varied_commands,
)
from mppi.residual_model import (
  ResidualCfg,
  ResidualModel,
  command_features,
  gain_offset_features,
)


@pytest.fixture
def nominal() -> PlanarModel:
  """A hand-written stand-in for the fitted model, with a similar lag and gain."""
  return PlanarModel(
    a=torch.diag(torch.tensor([0.76, 0.76, 0.33])),
    b=torch.diag(torch.tensor([0.26, 0.23, 0.62])),
    c=torch.zeros(3),
    dt=0.1,
    cmd_min=(-1.5, -1.0, -1.5),
    cmd_max=(3.0, 1.0, 1.5),
  )


def test_fresh_model_is_the_nominal_model(nominal: PlanarModel) -> None:
  model = ResidualModel(nominal, command_features)
  commands = torch.randn(5, 8, 3, generator=torch.Generator().manual_seed(0))
  pose, vel = torch.zeros(3), torch.tensor([1.0, 0.0, 0.2])
  for ours, theirs in zip(
    model.rollout(pose, vel, commands),
    nominal.rollout(pose, vel, commands),
    strict=True,
  ):
    assert torch.equal(ours, theirs)


def test_recovers_a_known_change_exactly_without_noise(nominal: PlanarModel) -> None:
  """With clean data and varied commands, every coefficient is identified."""
  gen = torch.Generator().manual_seed(0)
  steps = 600  # 60 s: slow gains need longer than offsets to settle exactly.
  cmd = varied_commands(steps, gen)
  coef = (true_change(nominal, "weak") + true_change(nominal, "veer")).expand(steps, 12)
  run = make_run(nominal, cmd, coef, torch.zeros(steps, 3))
  model = ResidualModel(nominal, command_features, ResidualCfg(leak=0.0))
  track(model, run)

  # Compare predictions away from the data, at commands the run never held.
  probes = torch.tensor([[2.0, 0.3, -0.8], [-1.0, -0.5, 1.2], [0.5, 0.0, 0.0]])
  truth = command_features(probes, probes) @ coef[0]
  assert torch.allclose(model.residual(probes, probes), truth, atol=5e-3)


@pytest.mark.parametrize(
  ("kind", "commands", "first_s", "second_s", "later"),
  [
    ("weak", race_commands, 0.45, 0.7, 0.6),
    # A veer shows only as the forward command varies, and it is a gain, which
    # moves slowly by design. So it is identified later.
    ("veer", varied_commands, 0.3, 0.35, 0.5),
  ],
)
def test_follows_a_change_and_its_undoing(
  nominal: PlanarModel,
  kind: str,
  commands: Commands,
  first_s: float,
  second_s: float,
  later: float,
) -> None:
  """Checks the share of the new model error removed, against loose minimums."""
  s = score(nominal, command_features, ResidualCfg(), kind, commands, seeds=4)
  assert s["jitter"] < 0.4, "Follows noise while the robot is healthy."
  assert s["first_s"] > first_s, "Too slow in the first second after the change."
  assert s["second_s"] > second_s, "Too slow in the second second."
  assert s["later"] > later, "Does not settle on the change."
  assert s["after_undo"] < 0.4, "Does not return to zero when the change is undone."


def test_constant_commands_do_not_wind_up(nominal: PlanarModel) -> None:
  """A gain and an offset cannot be told apart when the command never varies."""
  cfg = ResidualCfg()
  std = torch.tensor(cfg.meas_std)
  gen = torch.Generator().manual_seed(0)
  steps = 1200
  cmd = torch.tensor([3.0, 0.1, 1.4]).expand(steps, 3)
  coef = true_change(nominal, "drift").expand(steps, 12)
  run = make_run(nominal, cmd, coef, gait_noise(steps, std, nominal.dt, gen))
  model = ResidualModel(nominal, gain_offset_features, cfg)
  pred, away = track(model, run)

  # The uncertainty stops at the level where the leak balances the drift.
  limit = cfg.drift_std / (2 * cfg.leak - cfg.leak**2 * nominal.dt) ** 0.5
  assert model.coef_std.max() <= 1.001 * limit
  # At the command being held the correction is right.
  late = slice(steps // 2, None)
  error = ((pred - run.residual)[late] / std).pow(2).mean().sqrt()
  assert error < 0.4
  # Away from it the correction stays bounded: no worse than the change itself.
  assert (away[late] / std).abs().max() < (run.residual / std).abs().max() + 1.0


def test_steady_push_goes_to_the_offsets(nominal: PlanarModel) -> None:
  """A push under a held command is not taken for a drift that grows with speed.

  With the command held, a gain on the forward command and an offset explain the
  push equally well. Credited to the gain, the push would seem to shrink at lower
  speed, and the planner would brake to escape it.
  """
  cfg = ResidualCfg()
  std = torch.tensor(cfg.meas_std)
  slower = torch.tensor([1.0, 0.1, 1.4])
  errors = []
  for seed in range(4):
    gen = torch.Generator().manual_seed(seed)
    steps = 30  # Three seconds after the push starts.
    cmd = torch.tensor([3.0, 0.1, 1.4]).expand(steps, 3)
    coef = true_change(nominal, "drift").expand(steps, 12)
    run = make_run(nominal, cmd, coef, gait_noise(steps, std, nominal.dt, gen))
    model = ResidualModel(nominal, command_features, cfg)
    track(model, run)
    truth = command_features(slower, slower) @ coef[0]
    errors.append(((model.residual(slower, slower) - truth) / std).abs().max())
  # Gains as fast as offsets give about 0.9 here, and 0.4 with the default.
  assert torch.stack(errors).mean() < 0.6


def test_many_robots_match_one_robot_each(nominal: PlanarModel) -> None:
  """The batched filter gives each robot what its own filter would."""
  gen = torch.Generator().manual_seed(0)
  robots, steps = 3, 40
  vel = torch.randn(steps + 1, robots, 3, generator=gen)
  cmd = torch.randn(steps, robots, 3, generator=gen)
  batched = ResidualModel(nominal, command_features, num_envs=robots)
  singles = [ResidualModel(nominal, command_features) for _ in range(robots)]
  for t in range(steps):
    batched.update(vel[t], cmd[t], vel[t + 1])
    for i, single in enumerate(singles):
      single.update(vel[t, i], cmd[t, i], vel[t + 1, i])
  for i, single in enumerate(singles):
    torch.testing.assert_close(batched.coef[i], single.coef)
    torch.testing.assert_close(
      batched.residual(vel[-1], cmd[-1])[i], single.residual(vel[-1, i], cmd[-1, i])
    )
  batched.reset(torch.tensor([1]))
  assert batched.coef[1].abs().max() == 0
  assert batched.coef[0].abs().max() > 0
