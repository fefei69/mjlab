"""Gravity and foot friction that change once per episode, mid-run.

The change happens when the robot has come a set distance along +x from where it
spawned, so on the box-jump course it lands at a known point before the jump.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

__all__ = ["physics_change"]


class physics_change:
  """Change gravity and foot friction once, when the robot passes a line.

  Use with ``mode="step"``. At reset each environment draws its starting gravity
  and foot friction (nominal unless a starting range is given), whether it gets a
  change this episode, the values after the change, and how far along +x from
  its spawn point the change happens. ``gravity`` and ``friction`` hold every
  environment's current values.

  The Go1's feet have contact priority, so their friction is the friction of
  every foot contact; the terrain's own friction does not matter.

  Params:
    asset_cfg: The robot, with ``geom_names`` selecting the feet.
    gravity_range: Gravity after the change, in m/s^2.
    friction_range: Foot sliding friction after the change.
    onset_x: Range of the distance from the spawn point, along +x, at which the
      change happens, in metres.
    prob: Fraction of episodes with a change.
    start_gravity_range: Gravity at the start of each episode. Default nominal.
    start_friction_range: Foot friction at the start of each episode. Default
      nominal. This replaces any other foot-friction randomization.
    hard_prob: Share of changes drawn from the hard ranges below instead.
    hard_gravity_range, hard_friction_range: Low gravity and low friction, the
      combination that makes landing on and stopping on the box hardest.
  """

  # Read by the event manager: give every world its own gravity and friction.
  model_fields = ("opt.gravity", "geom_friction")
  recompute = RecomputeLevel.none

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._env = env
    self._asset: Entity = env.scene[asset_cfg.name]
    self._feet = self._asset.indexing.geom_ids[asset_cfg.geom_ids]
    self._params = cfg.params
    n, device = env.num_envs, env.device
    self.gravity = torch.zeros(n, device=device)
    self.friction = torch.zeros(n, device=device)
    self.changed = torch.zeros(n, dtype=torch.bool, device=device)
    self._new_gravity = torch.zeros(n, device=device)
    self._new_friction = torch.zeros(n, device=device)
    self._onset = torch.full((n,), torch.inf, device=device)
    self._nominal: tuple[float, float] | None = None
    self._draw(torch.arange(n, device=device))

  def _nominal_values(self) -> tuple[float, float]:
    """Gravity and foot friction of the compiled model."""
    if self._nominal is None:
      sim = self._env.sim
      gravity = sim.get_default_field("opt.gravity").norm().item()
      friction = sim.get_default_field("geom_friction")[self._feet, 0].mean().item()
      self._nominal = (gravity, friction)
    return self._nominal

  def _draw(self, env_ids: torch.Tensor) -> None:
    p = self._params
    n, device = len(env_ids), self._env.device

    def uniform(lo_hi: tuple[float, float]) -> torch.Tensor:
      lo, hi = lo_hi
      return lo + (hi - lo) * torch.rand(n, device=device)

    gravity, friction = self._nominal_values()
    self.gravity[env_ids] = uniform(p.get("start_gravity_range") or (gravity, gravity))
    self.friction[env_ids] = uniform(
      p.get("start_friction_range") or (friction, friction)
    )
    self._new_gravity[env_ids] = uniform(p["gravity_range"])
    self._new_friction[env_ids] = uniform(p["friction_range"])
    hard_prob = p.get("hard_prob", 0.0)
    if hard_prob > 0.0:
      hard = torch.rand(n, device=device) < hard_prob
      hard_ids = env_ids[hard]
      self._new_gravity[hard_ids] = uniform(p["hard_gravity_range"])[hard]
      self._new_friction[hard_ids] = uniform(p["hard_friction_range"])[hard]
    onset = uniform(p["onset_x"])
    happens = torch.rand(n, device=device) < p.get("prob", 1.0)
    self._onset[env_ids] = torch.where(happens, onset, torch.inf)

  def _write(self, env_ids: torch.Tensor) -> None:
    model = self._env.sim.model
    down = torch.zeros(len(env_ids), 3, device=self._env.device)
    down[:, 2] = -self.gravity[env_ids]
    model.opt.gravity[env_ids] = down
    friction = self.friction[env_ids][:, None].expand(-1, len(self._feet))
    model.geom_friction[env_ids[:, None], self._feet, 0] = friction

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self._env.num_envs, device=self._env.device)
    self.changed[env_ids] = False
    self._draw(env_ids)
    self._write(env_ids)

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **_):
    del env_ids  # Step events run on all environments.
    x = self._asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    start = (~self.changed & (x >= self._onset)).nonzero(as_tuple=False).squeeze(-1)
    if len(start) == 0:
      return
    self.changed[start] = True
    self.gravity[start] = self._new_gravity[start]
    self.friction[start] = self._new_friction[start]
    self._write(start)
