"""Tests for the mid-episode motor fault on the Go1 fault task."""

import torch
from conftest import get_test_device

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import motor_fault


def test_fault_starts_mid_episode_and_is_undone_on_reset() -> None:
  device = get_test_device()
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-Go1-Fault")
  cfg.scene.num_envs = 4
  cfg.events["motor_fault"].params.update(
    fault_prob=1.0, strength_range=(0.2, 0.2), onset_s=(0.05, 0.059)
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  term = env.event_manager.get_term_cfg("motor_fault").func
  assert isinstance(term, motor_fault)
  default = env.sim.get_default_field("actuator_forcerange")[:, 1]
  limit = env.sim.model.actuator_forcerange[..., 1]
  action = torch.zeros(4, 12, device=device)
  env.reset()

  env.step(action)  # Onset is the second step.
  assert not term.active.any()
  torch.testing.assert_close(limit, default.expand_as(limit))

  obs, *_ = env.step(action)
  assert term.active.all()
  # Exactly one joint per robot is at 20%, and the critic sees which.
  weak = limit < default - 1e-6
  assert (weak.sum(dim=1) == 1).all()
  torch.testing.assert_close(limit[weak], 0.2 * default.expand_as(limit)[weak])
  critic = obs["critic"]
  assert isinstance(critic, torch.Tensor)
  torch.testing.assert_close(critic[:, -12:], term.strength)
  assert not (term.strength[:, term.joint_names.index("FL_calf_joint")] < 1).any()

  env.reset()
  assert not term.active.any()
  torch.testing.assert_close(limit, default.expand_as(limit))
  env.close()
