"""A motor fault that starts at a random time in each episode.

Weakens one joint's motor mid-episode by scaling its torque limit, and undoes it
on reset. A torque scale of 0 is a dead motor: the joint swings freely.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


class motor_fault:
  """Weaken one joint's motor at a random time in each episode.

  Use with ``mode="step"``. At reset each environment draws whether it gets a
  fault this episode, which joint, how strong the motor stays and when the fault
  starts. ``strength`` holds every joint's current torque scale (1 is healthy),
  for the critic or for logging.

  Params:
    asset_name: The robot entity.
    joint_names: Regex for the joints that may fail.
    fault_prob: Fraction of episodes with a fault.
    strength_range: Range of the failed motor's torque scale, drawn uniformly.
    onset_s: Range of the fault's start time within the episode, in seconds.
  """

  # Read by the event manager, as `requires_model_fields` would set them on a
  # function: give every world its own torque limits.
  model_fields = ("actuator_forcerange",)
  recompute = RecomputeLevel.none

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._asset: Entity = env.scene[cfg.params.get("asset_name", "robot")]
    pattern = re.compile(cfg.params["joint_names"])
    names = self._asset.joint_names
    self._eligible = [i for i, n in enumerate(names) if pattern.fullmatch(n)]
    if not self._eligible:
      raise ValueError(f"No joint matches {cfg.params['joint_names']!r}.")
    self.joint_names = names
    n, device = env.num_envs, env.device
    self.strength = torch.ones(n, len(names), device=device)
    self.active = torch.zeros(n, dtype=torch.bool, device=device)
    self._joint = torch.zeros(n, dtype=torch.long, device=device)
    self._planned = torch.ones(n, device=device)
    self._onset = torch.full((n,), -1, dtype=torch.long, device=device)
    self._steps = torch.zeros(n, dtype=torch.long, device=device)
    self._ctrl: torch.Tensor | None = None
    self._params = cfg.params
    self._draw(torch.arange(n, device=device))

  def _ctrl_ids(self) -> torch.Tensor:
    """The control index of each joint's motor, in joint order."""
    if self._ctrl is None:
      ctrl = torch.full((len(self.joint_names),), -1, dtype=torch.long)
      for act in self._asset.actuators:
        for local, glob in zip(
          act.target_ids.tolist(), act.global_ctrl_ids.tolist(), strict=True
        ):
          ctrl[local] = glob
      self._ctrl = ctrl.to(self._env.device)
    return self._ctrl

  def _draw(self, env_ids: torch.Tensor) -> None:
    p = self._params
    n, device = len(env_ids), self._env.device
    eligible = torch.tensor(self._eligible, device=device)
    self._joint[env_ids] = eligible[torch.randint(len(eligible), (n,), device=device)]
    lo, hi = p["strength_range"]
    self._planned[env_ids] = lo + (hi - lo) * torch.rand(n, device=device)
    lo, hi = p["onset_s"]
    onset = lo + (hi - lo) * torch.rand(n, device=device)
    onset = (onset / self._env.step_dt).long()
    faulty = torch.rand(n, device=device) < p["fault_prob"]
    self._onset[env_ids] = torch.where(faulty, onset, torch.full_like(onset, -1))
    self._steps[env_ids] = 0

  def _write(self, env_ids: torch.Tensor) -> None:
    ctrl = self._ctrl_ids()
    default = self._env.sim.get_default_field("actuator_forcerange")[ctrl]
    scale = self.strength[env_ids][:, :, None]
    model = self._env.sim.model
    model.actuator_forcerange[env_ids[:, None], ctrl] = default * scale

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self._env.num_envs, device=self._env.device)
    was_faulty = env_ids[self.active[env_ids]]
    self.strength[env_ids] = 1.0
    self.active[env_ids] = False
    if len(was_faulty):
      self._write(was_faulty)
    self._draw(env_ids)

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **_):
    del env, env_ids  # Step events run on all environments.
    self._steps += 1
    start = (self._steps == self._onset).nonzero(as_tuple=False).squeeze(-1)
    if len(start) == 0:
      return
    self.strength[start, self._joint[start]] = self._planned[start]
    self.active[start] = True
    self._write(start)


def motor_strength(
  env: ManagerBasedRlEnv, event_name: str = "motor_fault"
) -> torch.Tensor:
  """Every joint's torque scale from the motor-fault event: privileged, critic only."""
  term = env.event_manager.get_term_cfg(event_name).func
  assert isinstance(term, motor_fault)
  return term.strength
