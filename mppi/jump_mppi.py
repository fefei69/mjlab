"""MPPI for the box jump: the run-up speed profile, when to jump, and how hard.

A candidate plan is a run-up speed profile over the horizon (a few knots,
interpolated linearly), a trigger time and a take-off vertical speed. A cheap
model rolls every candidate forward for every robot at once:

- **Run-up:** the policy's forward speed settles at a calibrated offset and gain
  of the command, reached quickly but no faster than its top acceleration or
  its traction (friction * gravity) allows.
- **Take-off:** at the trigger the trunk pushes off for a short calibrated time,
  then lifts off with the vertical speed the policy actually produces for the
  commanded one.
- **Flight:** ballistic, under the estimated gravity.
- **Landing:** the trunk must touch down past the start of the landing box, and
  brake to a stop before its end, at a deceleration set by the estimated
  friction and gravity.

The update is MPPI's (`mppi/core.py`), batched over robots: costs normalized by
their spread, softmax weights, a few iterations per call, and the plan carried
over to the next call shifted by the time elapsed. The calibration of the
policy's jump comes from `mppi/fit_jump_model.py` (`JumpModelCfg`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mppi.jump_planner import NOMINAL_GRAVITY, JumpModelCfg, Reference


@dataclass(kw_only=True)
class JumpMppiCfg:
  horizon: float = 4.5
  """Seconds ahead: enough to reach the edge from the start at 1.2 m/s."""
  dt: float = 0.05
  num_samples: int = 256
  num_knots: int = 6
  iterations: int = 3
  temperature: float = 0.3
  speed_range: tuple[float, float] = (0.0, 3.0)
  vz_range: tuple[float, float] = (0.5, 2.0)
  sigma_speed: float = 0.3
  sigma_trigger: float = 0.15
  sigma_vz: float = 0.15
  w_miss: float = 20.0
  """Cost per metre of touch-down outside the landing window, or of lift-off past
  the edge."""
  w_margin: float = 2.0
  """Reward per metre of margin to the nearer end of the landing window."""
  max_margin: float = 0.25
  """Margin beyond this earns nothing more."""
  w_vz: float = 0.05
  """Cost per m/s of commanded take-off speed: lower jumps when margins tie."""
  w_smooth: float = 0.2
  """Cost per (m/s)^2 of change between neighbouring speed knots."""


@dataclass
class Rollout:
  """Predictions for (B, K) candidate plans."""

  cost: torch.Tensor
  speed_cmd: torch.Tensor
  """(B, K, N) commanded speed at each model step."""
  x: torch.Tensor
  """(B, K, N + 1) trunk position along +x before the jump."""
  trigger_t: torch.Tensor
  trigger_x: torch.Tensor
  liftoff_x: torch.Tensor
  liftoff_vx: torch.Tensor
  liftoff_vz: torch.Tensor
  vz_cmd: torch.Tensor
  flight_time: torch.Tensor
  touchdown_x: torch.Tensor
  stop_x: torch.Tensor
  margin: torch.Tensor
  in_horizon: torch.Tensor
  """Whether the plan jumps within the horizon (always, for the long horizon)."""


class JumpMppi:
  can_defer = False
  """Whether a plan may put the trigger beyond the horizon: no jump yet."""

  def __init__(
    self,
    num_envs: int,
    model: JumpModelCfg | None = None,
    cfg: JumpMppiCfg | None = None,
    device: str = "cpu",
  ) -> None:
    self.model = model or JumpModelCfg()
    self.cfg = cfg or JumpMppiCfg()
    self.device = device
    self.steps = round(self.cfg.horizon / self.cfg.dt)
    nk = self.cfg.num_knots
    self._knot_t = torch.linspace(0.0, self.cfg.horizon, nk, device=device)
    self._lo = torch.tensor(
      [self.cfg.speed_range[0]] * nk + [0.0, self.cfg.vz_range[0]], device=device
    )
    trigger_hi = self.cfg.horizon + (0.5 if self.can_defer else 0.0)
    self._hi = torch.tensor(
      [self.cfg.speed_range[1]] * nk + [trigger_hi, self.cfg.vz_range[1]],
      device=device,
    )
    self._sigma = torch.tensor(
      [self.cfg.sigma_speed] * nk + [self.cfg.sigma_trigger, self.cfg.sigma_vz],
      device=device,
    )
    self.mean = torch.zeros(num_envs, nk + 2, device=device)
    self.best: Rollout | None = None
    """The current plan's own rollout, from the last call to `plan`."""
    self.samples: Rollout | None = None
    """Robot 0's candidates from the last iteration, for drawing."""

  # Plan parameters.

  def reset(self, x: torch.Tensor, edge: torch.Tensor, speed: float = 2.0) -> None:
    """Start every robot from a plain plan: run at `speed`, jump at the edge."""
    m, nk = self.model, self.cfg.num_knots
    self.mean[:, :nk] = speed
    reach = (edge - m.takeoff_before_edge - x) / speed - m.push_time
    self.mean[:, nk] = reach.clamp(0.0, self.cfg.horizon)
    self.mean[:, nk + 1] = 1.2

  def shift(self, seconds: float) -> None:
    """Carry the plan over to `seconds` later."""
    nk = self.cfg.num_knots
    speeds = self.speed_profile(self.mean[:, None])[:, 0]  # (B, N)
    t = (self._knot_t + seconds).clamp(max=self.cfg.horizon)
    index = (t / self.cfg.dt).round().long().clamp(max=self.steps - 1)
    self.mean[:, :nk] = speeds[:, index]
    self.mean[:, nk] = (self.mean[:, nk] - seconds).clamp(min=0.0)

  def speed_profile(self, u: torch.Tensor) -> torch.Tensor:
    """(B, K, D) plans to (B, K, N) commanded speeds at each model step."""
    b, k = u.shape[:2]
    knots = u[..., : self.cfg.num_knots].clamp(*self.cfg.speed_range)
    flat = knots.reshape(b * k, 1, -1)
    out = F.interpolate(flat, size=self.steps, mode="linear", align_corners=True)
    return out.reshape(b, k, self.steps)

  def speed_at(self, seconds: float) -> torch.Tensor:
    """The current plan's commanded speed `seconds` after the last plan, (B,)."""
    speeds = self.speed_profile(self.mean[:, None])[:, 0]
    index = min(round(seconds / self.cfg.dt), self.steps - 1)
    return speeds[:, index]

  # Model.

  def rollout(
    self,
    u: torch.Tensor,
    x0: torch.Tensor,
    v0: torch.Tensor,
    course: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gravity: torch.Tensor,
    friction: torch.Tensor,
  ) -> Rollout:
    """Predict (B, K) candidate plans `u` from each robot's position and speed."""
    cfg, m = self.cfg, self.model
    b, k = u.shape[:2]
    nk = cfg.num_knots
    edge, pad_start, pad_end = (c[:, None] for c in course)
    g, mu = gravity[:, None], friction[:, None]
    speed_cmd = self.speed_profile(u)
    in_horizon = u[..., nk] < cfg.horizon
    if not self.can_defer:
      in_horizon = torch.ones_like(in_horizon)
    trigger_t = u[..., nk].clamp(0.0, cfg.horizon)
    vz_cmd = u[..., nk + 1].clamp(*cfg.vz_range)

    # Run-up: a lagged speed, limited by traction.
    accel = (
      torch.minimum(torch.full_like(mu, m.run_accel), m.brake_efficiency * mu * g)
      * cfg.dt
    )
    x = x0[:, None].expand(b, k).clone()
    v = v0[:, None].expand(b, k).clone()
    xs, vs = [x], [v]
    target = m.speed_offset + m.speed_gain * speed_cmd
    for n in range(self.steps):
      dv = ((target[..., n] - v) * (cfg.dt / m.speed_tau)).clamp(-accel, accel)
      x = x + (v + 0.5 * dv) * cfg.dt
      v = v + dv
      xs.append(x)
      vs.append(v)
    x_run, v_run = torch.stack(xs, dim=-1), torch.stack(vs, dim=-1)

    # The trigger, between two model steps.
    f = trigger_t / cfg.dt
    i0 = f.floor().long().clamp(max=self.steps - 1)
    w = (f - i0).clamp(0.0, 1.0)

    def at_trigger(series: torch.Tensor) -> torch.Tensor:
      lo = series.gather(-1, i0[..., None])[..., 0]
      hi = series.gather(-1, (i0 + 1)[..., None])[..., 0]
      return lo + w * (hi - lo)

    trigger_x, trigger_v = at_trigger(x_run), at_trigger(v_run)

    # Push-off, flight and braking.
    liftoff_x = trigger_x + trigger_v * m.push_time
    liftoff_vx = m.vx_gain * trigger_v
    liftoff_vz = (
      m.vz_offset + m.vz_gain * vz_cmd + m.vz_gravity_gain * (g - NOMINAL_GRAVITY)
    ).clamp(min=0.0)
    flight_time = 2 * liftoff_vz / g
    touchdown_x = liftoff_x + liftoff_vx * flight_time + m.flight_offset
    stop_x = touchdown_x + liftoff_vx.square() / (2 * m.brake_efficiency * mu * g)

    short = pad_start + m.front_margin - touchdown_x
    long = stop_x - (pad_end - m.back_margin)
    margin = -torch.maximum(short, long)
    out = Rollout(
      cost=torch.zeros_like(margin),
      speed_cmd=speed_cmd,
      x=x_run,
      trigger_t=trigger_t,
      trigger_x=trigger_x,
      liftoff_x=liftoff_x,
      liftoff_vx=liftoff_vx,
      liftoff_vz=liftoff_vz,
      vz_cmd=vz_cmd,
      flight_time=flight_time,
      touchdown_x=touchdown_x,
      stop_x=stop_x,
      margin=margin,
      in_horizon=in_horizon,
    )
    out.cost = self.cost(u, out, x_run, v_run, edge)
    return out

  def cost(
    self,
    u: torch.Tensor,
    r: Rollout,
    x_run: torch.Tensor,
    v_run: torch.Tensor,
    edge: torch.Tensor,
  ) -> torch.Tensor:
    """Land inside the window with margin, lift off before the edge."""
    del x_run, v_run
    cfg, m = self.cfg, self.model
    past_edge = torch.relu(r.liftoff_x - (edge - m.takeoff_before_edge))
    knots = u[..., : cfg.num_knots].clamp(*cfg.speed_range)
    return (
      cfg.w_miss * (torch.relu(-r.margin) + past_edge)
      - cfg.w_margin * r.margin.clamp(max=cfg.max_margin)
      + cfg.w_vz * r.vz_cmd
      + cfg.w_smooth * knots.diff(dim=-1).square().sum(dim=-1)
    )

  def seeds(self, x0: torch.Tensor, v0: torch.Tensor) -> torch.Tensor | None:
    """Extra candidates to score each iteration, (B, E, D), or None."""
    del x0, v0
    return None

  # MPPI.

  def plan(
    self,
    x0: torch.Tensor,
    v0: torch.Tensor,
    course: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gravity: torch.Tensor,
    friction: torch.Tensor,
  ) -> Rollout:
    """Improve every robot's plan; returns the plans' own rollouts, (B, 1)."""
    cfg = self.cfg
    b, d = self.mean.shape
    for _ in range(cfg.iterations):
      noise = torch.randn(b, cfg.num_samples, d, device=self.device) * self._sigma
      raw = self.mean[:, None] + noise
      raw[:, 0] = self.mean  # Always score the current plan itself.
      extra = self.seeds(x0, v0)
      if extra is not None:
        raw[:, 1 : 1 + extra.shape[1]] = extra
      out = self.rollout(raw, x0, v0, course, gravity, friction)
      # A robot that has fallen can give nonsense; never let it poison the plan.
      cost = torch.nan_to_num(out.cost, nan=1e6, posinf=1e6, neginf=-1e6)
      z = (cost - cost.min(dim=1, keepdim=True).values) / (
        cost.std(dim=1, keepdim=True) + 1e-6
      )
      weights = torch.softmax(-z / cfg.temperature, dim=1)
      mean = (weights[..., None] * raw).sum(dim=1)
      # As in `mppi/core.py`: unclamped, but not far past the limits.
      self.mean = torch.clamp(mean, self._lo - self._sigma, self._hi + self._sigma)
      self.samples = Rollout(
        **{k: v[:1] for k, v in vars(out).items()}  # Robot 0 only.
      )
    self.best = self.rollout(self.mean[:, None], x0, v0, course, gravity, friction)
    return self.best


def plan_path(
  r: Rollout, robot: int, sample: int, gravity: float, height: float, dt: float
) -> np.ndarray:
  """The trunk path of one rollout, (P, 3) in metres from the spawn point.

  Runs along the box at `height` above its top until the trigger, pushes off,
  flies a parabola under `gravity`, then brakes to a stop on the landing box.
  """
  x = r.x[robot, sample].cpu().numpy()
  if not bool(r.in_horizon[robot, sample]):  # No jump yet: the run-up only.
    run = x[::2]
    return np.stack([run, np.zeros_like(run), np.full_like(run, height)], axis=-1)
  n_trig = int(float(r.trigger_t[robot, sample]) / dt) + 1
  run = x[: max(n_trig, 1) : 2]
  trigger = float(r.trigger_x[robot, sample])
  liftoff = float(r.liftoff_x[robot, sample])
  vz = float(r.liftoff_vz[robot, sample])
  flight = float(r.flight_time[robot, sample])
  land = float(r.touchdown_x[robot, sample])
  stop = float(r.stop_x[robot, sample])
  s = np.linspace(0.0, flight, 10)
  # The flight, from lift-off to the predicted touch-down.
  fx = liftoff + (land - liftoff) * s / max(flight, 1e-6)
  fz = height + vz * s - 0.5 * gravity * s**2
  xs = np.concatenate([run, [trigger], fx, [stop]])
  zs = np.concatenate([np.full(len(run) + 1, height), fz, [height]])
  return np.stack([xs, np.zeros_like(xs), zs], axis=-1)


def speed_curve(
  r: Rollout, robot: int, sample: int, dt: float
) -> tuple[np.ndarray, np.ndarray]:
  """Predicted forward speed against position along the course, for drawing.

  The run-up until the trigger, the push-off and flight at the lift-off speed,
  then braking to a stop on the landing box.
  """
  x = r.x[robot, sample].cpu().numpy()
  if not bool(r.in_horizon[robot, sample]):  # No jump yet: the run-up only.
    return x, np.gradient(x, dt)
  n_trig = int(float(r.trigger_t[robot, sample]) / dt) + 1
  run_x = x[: max(n_trig, 2)]
  run_v = np.gradient(run_x, dt)
  vx = float(r.liftoff_vx[robot, sample])
  land = float(r.touchdown_x[robot, sample])
  stop = float(r.stop_x[robot, sample])
  xs = np.concatenate([run_x, [float(r.liftoff_x[robot, sample]), land, stop]])
  vs = np.concatenate([run_v, [vx, vx, 0.0]])
  return xs, vs


@dataclass(kw_only=True)
class JumpTrackingMppiCfg(JumpMppiCfg):
  """A short horizon that tracks a global reference (`ReferencePlanner`)."""

  horizon: float = 1.0
  num_knots: int = 4
  w_track: float = 4.0
  """Cost per (m/s)^2 of speed off the reference, averaged over the horizon."""
  w_takeoff: float = 20.0
  """Cost per m^2 of lift-off away from the reference take-off point."""
  w_vz_ref: float = 2.0
  """Cost per (m/s)^2 of take-off speed command away from the reference."""
  w_no_jump: float = 20.0
  """Cost per metre run past the take-off point without jumping."""
  w_position: float = 2.0
  """Cost per m^2 of falling behind or running ahead of the reference in time."""
  w_wait: float = 30.0
  """Cost of not jumping within the horizon when the reference jumps in it."""
  w_timing: float = 10.0
  """Cost per s^2 of trigger time away from the reference's."""


class JumpTrackingMppi(JumpMppi):
  """MPPI over a short horizon, tracking a jump reference recomputed globally
  from the latest gravity and friction estimates (as sampling MPCs such as
  DIAL-MPC and MuJoCo MPC track targets or references over 0.25 to 0.5 s).

  The reference sets where to lift off, how fast and how hard, and the speed
  curve to get there. This planner tracks the speed curve, and once the jump
  falls inside its horizon, times the trigger and fine-tunes the take-off speed
  so that its own model lands the robot inside the window.
  """

  can_defer = True
  cfg: JumpTrackingMppiCfg

  def __init__(
    self,
    num_envs: int,
    model: JumpModelCfg | None = None,
    cfg: JumpTrackingMppiCfg | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(num_envs, model, cfg or JumpTrackingMppiCfg(), device)
    self.reference: Reference | None = None

  def command_for(self, speed: torch.Tensor) -> torch.Tensor:
    """The run-up command that settles at forward speed `speed`."""
    m = self.model
    return ((speed - m.speed_offset) / m.speed_gain).clamp(*self.cfg.speed_range)

  def reset(self, x: torch.Tensor, edge: torch.Tensor, speed: float = 2.0) -> None:
    """Start with the reference's speed, and no jump within the horizon."""
    nk = self.cfg.num_knots
    target = (
      self.reference.speed_at(x)
      if self.reference is not None
      else torch.full_like(x, speed)
    )
    self.mean[:, :nk] = self.command_for(target)[:, None]
    self.mean[:, nk] = self.cfg.horizon + 0.3
    self.mean[:, nk + 1] = self.reference.vz if self.reference is not None else 1.2
    del edge

  def seeds(self, x0: torch.Tensor, v0: torch.Tensor) -> torch.Tensor | None:
    """The reference itself: its speed curve as commands, the trigger where the
    current plan reaches the reference trigger point, and its take-off speed."""
    ref = self.reference
    if ref is None:
      return None
    m = self.model
    # Positions along the horizon if the robot followed the reference speeds.
    t = self._knot_t
    x = x0[:, None] + v0.clamp(min=0.0)[:, None] * t  # A first guess.
    knots = self.command_for(ref.speed_at(x))
    trigger_x = ref.takeoff_x - ref.speed * m.push_time
    gap = (trigger_x - x0).clamp(min=0.0)
    trigger_t = gap / v0.clamp(min=0.3)
    seed = torch.cat([knots, trigger_t[:, None], ref.vz[:, None]], dim=-1)[:, None]
    return seed

  def set_reference(self, ref: Reference) -> None:
    self.reference = ref

  def cost(
    self,
    u: torch.Tensor,
    r: Rollout,
    x_run: torch.Tensor,
    v_run: torch.Tensor,
    edge: torch.Tensor,
  ) -> torch.Tensor:
    cfg, m = self.cfg, self.model
    ref = self.reference
    assert ref is not None, "Call set_reference before plan."
    knots = u[..., : cfg.num_knots].clamp(*cfg.speed_range)
    # The reference in time: follow its speed curve from the robot's current
    # state, speeding up or slowing down no faster than the curve's rate.
    x_ref, t_ref = self.reference_in_time(x_run[:, 0, 0], v_run[:, 0, 0])
    # Track it along the predicted run-up, up to the trigger.
    steps = torch.arange(x_run.shape[-1], device=x_run.device) * cfg.dt
    before = steps <= torch.where(r.in_horizon, r.trigger_t, cfg.horizon)[..., None]
    count = before.sum(dim=-1).clamp(min=1)
    error = (v_run - ref.speed_at(x_run)).square()
    track = (error * before).sum(dim=-1) / count
    behind = ((x_run - x_ref[:, None]).square() * before).sum(dim=-1) / count
    jump = r.in_horizon.float()
    # When the reference jumps within the horizon, waiting costs, and the
    # trigger should come when the reference's does.
    due = (t_ref < cfg.horizon - cfg.dt).float()[:, None]
    timing = (r.trigger_t - t_ref.clamp(max=cfg.horizon)[:, None]).square()
    takeoff = (r.liftoff_x - ref.takeoff_x[:, None]).square()
    vz_off = (r.vz_cmd - ref.vz[:, None]).square()
    past_edge = torch.relu(r.liftoff_x - (edge - m.takeoff_before_edge))
    landing = cfg.w_miss * (
      torch.relu(-r.margin) + past_edge
    ) - cfg.w_margin * r.margin.clamp(max=cfg.max_margin)
    # Without a jump in the horizon, running past the take-off point is a fall.
    overrun = torch.relu(x_run[..., -1] - ref.takeoff_x[:, None])
    return (
      cfg.w_track * track
      + cfg.w_position * behind
      + jump * (cfg.w_takeoff * takeoff + cfg.w_vz_ref * vz_off + landing)
      + jump * due * cfg.w_timing * timing
      + (1 - jump) * (cfg.w_no_jump * overrun + due[..., 0:1] * cfg.w_wait)
      + cfg.w_smooth * knots.diff(dim=-1).square().sum(dim=-1)
    )

  def reference_in_time(
    self, x0: torch.Tensor, v0: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference positions over the horizon, (B, N + 1), and the time it
    reaches the trigger point, (B,) (infinite if not within the horizon)."""
    ref, cfg, m = self.reference, self.cfg, self.model
    assert ref is not None
    rate = ref.accel * cfg.dt
    x, v = x0.clone(), v0.clamp(min=0.0)
    xs = [x]
    for _ in range(self.steps):
      v = torch.minimum(torch.maximum(ref.speed_at(x), v - rate), v + rate)
      x = x + v * cfg.dt
      xs.append(x)
    x_ref = torch.stack(xs, dim=-1)
    trigger_x = ref.takeoff_x - ref.speed * m.push_time
    reached = x_ref >= trigger_x[:, None]
    first = reached.float().argmax(dim=-1).float() * cfg.dt
    t_ref = torch.where(reached.any(dim=-1), first, torch.inf)
    return x_ref, t_ref
