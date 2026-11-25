"""The planner side of the box jump: when to jump, how fast, and how high.

The jump policy runs at a commanded speed and, when triggered, takes off with a
commanded vertical speed (`mjlab.tasks.box_jump`). This planner picks the run-up
speed and the take-off speed for every robot at once, by searching a grid of
both through a simple model of the jump:

- **Take-off:** the trunk lifts off a little before the edge, at the run-up
  speed it can still reach before then, with the vertical speed the policy
  actually produces for a commanded one (a calibration, `JumpModelCfg`).
- **Flight:** ballistic, under the estimated gravity.
- **Landing:** the trunk must touch down past the start of the landing box, and
  then brake to a stop before its far end, at a deceleration set by the
  estimated friction and gravity.

It keeps the candidate with the largest margin to both ends, preferring lower
jumps when margins tie. Gravity and friction come from the estimators below, or
from fixed values (no adaptation), or from the simulator (an oracle).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

NOMINAL_GRAVITY = 9.81


@dataclass
class JumpModelCfg:
  """How the jump policy turns commands into a jump. Calibrate per policy."""

  vz_gain: float = 1.0
  vz_offset: float = 0.0
  vz_gravity_gain: float = 0.0
  """Vertical speed at lift-off = offset + gain * commanded + gravity_gain *
  (gravity - 9.81), m/s."""
  vx_gain: float = 1.0
  """Forward speed at lift-off = gain * run-up speed."""
  push_time: float = 0.12
  """Time from the trigger to lift-off, s."""
  takeoff_before_edge: float = 0.05
  """Trunk position at lift-off, metres before the edge."""
  flight_offset: float = 0.0
  """Touch-down beyond the point-mass prediction, m (the trunk is not the
  centre of mass, and the legs reach)."""
  run_accel: float = 2.0
  """Fastest forward acceleration on a grippy surface, m/s^2."""
  speed_gain: float = 1.0
  speed_offset: float = 0.0
  """Steady running speed = offset + gain * commanded, m/s."""
  speed_tau: float = 0.15
  """Time constant of reaching that speed, s. Short: the policy follows the
  command's own ramp closely (measured, not fitted)."""
  brake_efficiency: float = 0.5
  """Braking deceleration as a share of friction * gravity."""
  front_margin: float = 0.05
  """The trunk must touch down this far past the start of the landing box."""
  back_margin: float = 0.3
  """...and stop this far before its end, so the front feet stay on it."""
  vz_cost: float = 0.05
  """Metres of margin worth giving up for 1 m/s less take-off speed."""

  def save(self, path: Path, **extra: object) -> None:
    path.write_text(json.dumps({**asdict(self), **extra}, indent=2) + "\n")

  @classmethod
  def load(cls, path: Path) -> JumpModelCfg:
    data = json.loads(path.read_text())
    return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


@dataclass
class JumpPlan:
  run_speed: torch.Tensor
  vz: torch.Tensor
  trigger_x: torch.Tensor
  landing_x: torch.Tensor
  """Predicted trunk touch-down, along +x from the spawn point."""
  margin: torch.Tensor
  """Distance to the nearer of the two limits; negative when none fits."""


class JumpPlanner:
  def __init__(
    self,
    cfg: JumpModelCfg | None = None,
    run_speeds: tuple[float, float, int] = (1.0, 3.0, 41),
    vzs: tuple[float, float, int] = (0.5, 2.0, 31),
    device: str = "cpu",
  ) -> None:
    self.cfg = cfg or JumpModelCfg()
    speeds = torch.linspace(*run_speeds, device=device)
    heights = torch.linspace(*vzs, device=device)
    grid = torch.cartesian_prod(speeds, heights)  # (K, 2)
    self._v, self._vz = grid[:, 0], grid[:, 1]

  def predict(
    self,
    x: torch.Tensor,
    speed: torch.Tensor,
    edge: torch.Tensor,
    gravity: torch.Tensor,
    friction: torch.Tensor,
    run_speed: torch.Tensor,
    vz_cmd: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Touch-down point, stopping distance and trigger point.

    Robot quantities are (B, 1) and candidates (1, K), or all (B,).
    """
    cfg = self.cfg
    takeoff_x = edge - cfg.takeoff_before_edge
    # The run-up speed the robot can still reach before taking off.
    accel = torch.minimum(
      torch.full_like(friction, cfg.run_accel),
      cfg.brake_efficiency * friction * gravity,
    )
    room = (takeoff_x - x).clamp(min=0.0)
    reachable = (speed.square() + 2 * accel * room).sqrt()
    steady = cfg.speed_offset + cfg.speed_gain * run_speed
    vx = cfg.vx_gain * torch.minimum(steady, reachable)
    vz = (
      cfg.vz_offset
      + cfg.vz_gain * vz_cmd
      + cfg.vz_gravity_gain * (gravity - NOMINAL_GRAVITY)
    )
    flight = 2 * vz.clamp(min=0.0) / gravity
    landing_x = takeoff_x + vx * flight + cfg.flight_offset
    stop = vx.square() / (2 * cfg.brake_efficiency * friction * gravity)
    trigger_x = takeoff_x - vx * cfg.push_time
    return landing_x, stop, trigger_x

  def plan(
    self,
    x: torch.Tensor,
    speed: torch.Tensor,
    course: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gravity: torch.Tensor,
    friction: torch.Tensor,
  ) -> JumpPlan:
    """The best (run-up speed, take-off speed) for each robot, shapes (B,)."""
    cfg = self.cfg
    edge, pad_start, pad_end = (c[:, None] for c in course)
    landing_x, stop, trigger_x = self.predict(
      x[:, None],
      speed[:, None],
      edge,
      gravity[:, None],
      friction[:, None],
      self._v[None],
      self._vz[None],
    )
    short = pad_start + cfg.front_margin - landing_x
    long = landing_x + stop - (pad_end - cfg.back_margin)
    margin = -torch.maximum(short, long)
    best = (margin - cfg.vz_cost * self._vz[None]).argmax(dim=1)
    rows = torch.arange(len(x), device=x.device)
    return JumpPlan(
      run_speed=self._v[best],
      vz=self._vz[best],
      trigger_x=trigger_x[rows, best],
      landing_x=landing_x[rows, best],
      margin=margin[rows, best],
    )


class GravityEstimator:
  """Gravity from the feet's push and the trunk's motion: momentum balance.

  Over one policy step, mass * (vertical acceleration) = upward foot force -
  mass * gravity, with the foot force averaged over every physics step (the
  contact sensor's force history). Averaged over about a stride. A low-rate
  accelerometer reading would miss the short foot impacts and read low.
  """

  def __init__(
    self, num_envs: int, dt: float, mass: float, tau: float = 0.3, device: str = "cpu"
  ):
    self.dt = dt
    self.mass = mass
    self.alpha = dt / (tau + dt)
    self.gravity = torch.full((num_envs,), NOMINAL_GRAVITY, device=device)
    self._prev_vz: torch.Tensor | None = None

  def update(self, force_history: torch.Tensor, vel_w: torch.Tensor) -> torch.Tensor:
    """force_history (B, feet, substeps, 3) world contact forces; vel_w (B, 3)."""
    push = force_history[..., 2].mean(dim=-1).sum(dim=-1).abs()
    vz = vel_w[:, 2]
    if self._prev_vz is not None:
      accel_up = (vz - self._prev_vz) / self.dt
      sample = push / self.mass - accel_up
      self.gravity += self.alpha * (sample - self.gravity)
    self._prev_vz = vz.clone()
    return self.gravity

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.gravity[ids] = NOMINAL_GRAVITY
    self._prev_vz = None


class FrictionEstimator:
  """Foot friction from how hard the feet push sideways, and when they slip.

  A foot that grips proves friction is at least its tangential-to-normal force
  ratio; a foot that slides shows it. The estimate starts at a prior, moves
  towards the ratio seen on slipping feet, and never drops below the largest
  ratio seen on gripping ones. Friction is only observable when the robot
  pushes near the limit, so a low friction can go unnoticed until it slips.
  """

  def __init__(
    self,
    num_envs: int,
    prior: float = 1.0,
    rate: float = 0.3,
    slip_speed: float = 0.15,
    min_normal: float = 20.0,
    device: str = "cpu",
  ):
    self.prior = prior
    self.rate = rate
    self.slip_speed = slip_speed
    self.min_normal = min_normal
    self.friction = torch.full((num_envs,), prior, device=device)

  def update(self, force_w: torch.Tensor, foot_vel_w: torch.Tensor) -> torch.Tensor:
    """force_w, foot_vel_w: (B, feet, 3) contact force and foot velocity."""
    normal = force_w[..., 2].abs()
    loaded = normal > self.min_normal
    ratio = force_w[..., :2].norm(dim=-1) / normal.clamp(min=1e-6)
    slipping = loaded & (foot_vel_w[..., :2].norm(dim=-1) > self.slip_speed)
    gripping = loaded & ~slipping
    slip_ratio = (ratio * slipping).sum(dim=-1) / slipping.sum(dim=-1).clamp(min=1)
    any_slip = slipping.any(dim=-1)
    self.friction = torch.where(
      any_slip, self.friction + self.rate * (slip_ratio - self.friction), self.friction
    )
    grip_bound = torch.where(gripping, ratio, 0.0).amax(dim=-1)
    self.friction = torch.maximum(self.friction, grip_bound)
    return self.friction

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.friction[ids] = self.prior


@dataclass
class Reference:
  """The jump to aim for, per robot (B,): where to lift off, at what speed, how
  hard, and the approach-speed curve that gets there."""

  takeoff_x: torch.Tensor
  speed: torch.Tensor
  """Forward speed at the trigger and lift-off."""
  vz: torch.Tensor
  """Commanded take-off vertical speed."""
  cruise: torch.Tensor
  """Speed cap before the approach curve takes over."""
  accel: torch.Tensor
  """Acceleration of the approach curve."""
  landing_x: torch.Tensor
  margin: torch.Tensor

  def speed_at(self, x: torch.Tensor) -> torch.Tensor:
    """Reference speed at positions `x`, (B, ...): hold the cruise speed, then
    change speed at a constant rate to reach the take-off speed at lift-off."""
    shape = (-1,) + (1,) * (x.dim() - 1)
    left = (self.takeoff_x.view(shape) - x).clamp(min=0.0)
    curve = (self.speed.view(shape).square() + 2 * self.accel.view(shape) * left).sqrt()
    return torch.minimum(self.cruise.view(shape), curve)


class ReferencePlanner:
  """A global search for the jump to aim for, from each robot's current state.

  Candidates are the forward speed at take-off and the take-off vertical speed.
  The trunk lifts off just before the edge. A speed counts only if the robot can
  still reach it by then, speeding up or slowing down no faster than its top
  acceleration and its traction allow. The rest is the grid planner's model: a
  ballistic flight under the estimated gravity, and braking under the estimated
  friction. Recomputed from scratch at every call, so a new gravity or friction
  estimate changes the reference at once.
  """

  def __init__(
    self,
    cfg: JumpModelCfg | None = None,
    speeds: tuple[float, float, int] = (0.6, 3.0, 49),
    vzs: tuple[float, float, int] = (0.5, 2.0, 31),
    device: str = "cpu",
  ) -> None:
    self.cfg = cfg or JumpModelCfg()
    grid = torch.cartesian_prod(
      torch.linspace(*speeds, device=device), torch.linspace(*vzs, device=device)
    )
    self._v, self._vz = grid[:, 0], grid[:, 1]

  def plan(
    self,
    x: torch.Tensor,
    speed: torch.Tensor,
    course: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gravity: torch.Tensor,
    friction: torch.Tensor,
  ) -> Reference:
    m = self.cfg
    edge, pad_start, pad_end = (c[:, None] for c in course)
    g, mu = gravity[:, None], friction[:, None]
    v_now = speed.clamp(min=0.0)[:, None]
    takeoff_x = edge - m.takeoff_before_edge
    accel = torch.minimum(torch.full_like(mu, m.run_accel), m.brake_efficiency * mu * g)
    room = (takeoff_x - v_now * m.push_time - x[:, None]).clamp(min=0.0)
    lowest = (v_now.square() - 2 * accel * room).clamp(min=0.0).sqrt()
    highest = (v_now.square() + 2 * accel * room).sqrt()
    v, vz_cmd = self._v[None], self._vz[None]
    vx = m.vx_gain * v
    vz = (
      m.vz_offset + m.vz_gain * vz_cmd + m.vz_gravity_gain * (g - NOMINAL_GRAVITY)
    ).clamp(min=0.0)
    landing_x = takeoff_x + vx * 2 * vz / g + m.flight_offset
    stop_x = landing_x + vx.square() / (2 * m.brake_efficiency * mu * g)
    margin = -torch.maximum(
      pad_start + m.front_margin - landing_x, stop_x - (pad_end - m.back_margin)
    )
    unreachable = torch.relu(lowest - v) + torch.relu(v - highest)
    score = margin - m.vz_cost * vz_cmd - 10.0 * unreachable
    best = score.argmax(dim=1)
    rows = torch.arange(len(x), device=x.device)
    take = self._v[best]
    return Reference(
      takeoff_x=takeoff_x[:, 0],
      speed=take,
      vz=self._vz[best],
      cruise=torch.maximum(take, speed.clamp(min=0.0)),
      accel=0.8 * accel[:, 0],
      landing_x=landing_x[rows, best],
      margin=margin[rows, best],
    )
