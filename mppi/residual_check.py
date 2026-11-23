"""Check the online residual model on made-up data.

The "robot" here is the fitted planar model plus a change we choose, plus noise
shaped like the gait's. Only on such data is the true answer known, so this is
where the fit is tuned before it meets the simulator. Each run is 40 s: healthy,
a change at 10 s, the change undone at 25 s.

  uv run python -m mppi.residual_check
  uv run python -m mppi.residual_check --drift-std 0.9   # Faster and noisier.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

from mppi.planar_model import PlanarModel
from mppi.residual_model import (
  FEATURE_SETS,
  Features,
  ResidualCfg,
  ResidualModel,
  command_features,
)
from mppi.tracker import DEFAULT_CHECKPOINT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Timeline of every run, in seconds.
DURATION, CHANGE_AT, UNDO_AT = 40.0, 10.0, 25.0

# The nominal model's error on the recorded healthy race is mostly periodic, from
# the gait: about 70% of its variance sits in a wave with a period near 1/3 s.
GAIT_PERIOD, GAIT_SHARE = 1.0 / 3.0, 0.7

Commands = Callable[[int, torch.Generator], torch.Tensor]


def gait_noise(
  steps: int, std: torch.Tensor, dt: float, gen: torch.Generator
) -> torch.Tensor:
  """(steps, 3) noise with standard deviation `std`: a gait wave plus white noise."""
  t = torch.arange(steps)[:, None] * dt
  phase = 2 * math.pi * torch.rand(3, generator=gen)
  wave = math.sqrt(2 * GAIT_SHARE) * torch.sin(2 * math.pi * t / GAIT_PERIOD + phase)
  white = math.sqrt(1 - GAIT_SHARE) * torch.randn(steps, 3, generator=gen)
  return std * (wave + white)


def varied_commands(steps: int, gen: torch.Generator) -> torch.Tensor:
  """Random commands, each held for 0.2 to 3 s, as when the model was fitted."""
  lo, hi = torch.tensor([-1.0, -0.6, -1.2]), torch.tensor([3.0, 0.6, 1.5])
  out: list[torch.Tensor] = []
  while len(out) < steps:
    cmd = lo + (hi - lo) * torch.rand(3, generator=gen)
    cmd = cmd * (torch.rand(3, generator=gen) > 0.3)
    out += [cmd] * int(torch.randint(2, 31, (1,), generator=gen))
  return torch.stack(out[:steps])


def race_commands(steps: int, gen: torch.Generator) -> torch.Tensor:
  """Commands like the circle race's: flat out forward, the rest barely moving."""
  mean, std = torch.tensor([3.0, 0.1, 1.43]), torch.tensor([0.0, 0.15, 0.14])
  cmd = mean + std * torch.randn(steps, 3, generator=gen)
  return cmd.clamp(torch.tensor([-1.5, -1.0, -1.5]), torch.tensor([3.0, 1.0, 1.5]))


COMMANDS: dict[str, Commands] = {"varied": varied_commands, "race": race_commands}

CHANGES = {
  "weak": "30% less forward speed and yaw rate",
  "drift": "drifts 0.2 m/s sideways and 0.3 rad/s in yaw",
  "veer": "yaws 0.15 rad/s for each m/s of forward command",
}


def true_change(nominal: PlanarModel, kind: str) -> torch.Tensor:
  """(12,) coefficients of a change, in the layout of `command_features`.

  These are in plain units, not the scaled ones the filter works in. A steady
  effect of size d on axis i needs a per-step residual of d (1 - a_ii), because
  the nominal model carries most of the velocity over from the step before.
  """
  coef = torch.zeros(3, 4)
  carry = 1.0 - nominal.a.diagonal().cpu()
  if kind == "weak":
    coef[0, 0] = -0.3 * nominal.b[0, 0]
    coef[2, 2] = -0.3 * nominal.b[2, 2]
  elif kind == "drift":
    coef[1, 3] = 0.2 * carry[1]
    coef[2, 3] = 0.3 * carry[2]
  elif kind == "veer":
    coef[2, 0] = 0.15 * carry[2]
  else:
    raise ValueError(f"Unknown change: {kind}")
  return coef.flatten()


@dataclass
class Run:
  vel: torch.Tensor
  """(steps + 1, 3) velocity over each plan step."""
  cmd: torch.Tensor
  """(steps, 3) command sent at each step."""
  residual: torch.Tensor
  """(steps, 3) the true correction to the nominal model at each step."""


def make_run(
  nominal: PlanarModel, cmd: torch.Tensor, coef: torch.Tensor, noise: torch.Tensor
) -> Run:
  """Simulate the made-up robot. `coef` is (steps, 12), one true change per step."""
  vel, residual = [torch.zeros(3)], []
  for k in range(cmd.shape[0]):
    residual.append(command_features(vel[-1], cmd[k]) @ coef[k])
    vel.append(nominal.next_velocity(vel[-1], cmd[k]) + residual[-1] + noise[k])
  return Run(torch.stack(vel), cmd, torch.stack(residual))


def scenario(
  nominal: PlanarModel,
  kind: str | None,
  commands: Commands,
  std: torch.Tensor,
  seed: int,
) -> Run:
  """A 40 s run with the change `kind` between 10 s and 25 s. None means no change."""
  gen = torch.Generator().manual_seed(seed)
  steps = round(DURATION / nominal.dt)
  coef = torch.zeros(steps, 12)
  if kind is not None:
    on, off = round(CHANGE_AT / nominal.dt), round(UNDO_AT / nominal.dt)
    coef[on:off] = true_change(nominal, kind)
  cmd = commands(steps, gen)
  return make_run(nominal, cmd, coef, gait_noise(steps, std, nominal.dt, gen))


def track(
  model: ResidualModel, run: Run, probe: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
  """Run the filter over `run` from a fresh start.

  Returns two (steps, 3) tensors, both taken before the step's own sample is
  seen, as the planner would use them: the residual predicted for the step, and
  the residual predicted at the command `probe` (standstill if None).
  """
  probe = torch.zeros(3) if probe is None else probe
  model.reset()
  pred, away = [], []
  for k in range(run.cmd.shape[0]):
    pred.append(model.residual(run.vel[k], run.cmd[k]))
    away.append(model.residual(probe, probe))
    model.update(run.vel[k], run.cmd[k], run.vel[k + 1])
  return torch.stack(pred), torch.stack(away)


def rms(x: torch.Tensor) -> float:
  return x.pow(2).mean().sqrt().item()


def score(
  nominal: PlanarModel,
  features: Features,
  cfg: ResidualCfg,
  kind: str,
  commands: Commands,
  seeds: int,
) -> dict[str, float]:
  """How much of the new model error the residual removes, and its jitter.

  Errors are in units of the nominal model's usual error (`cfg.meas_std`).
  """
  std = torch.tensor(cfg.meas_std)
  model = ResidualModel(nominal, features, cfg)
  err, true, pred = [], [], []
  for seed in range(seeds):
    run = scenario(nominal, kind, commands, std, seed)
    p, _ = track(model, run)
    err.append((p - run.residual) / std)
    true.append(run.residual / std)
    pred.append(p / std)
  e, t, p = torch.stack(err), torch.stack(true), torch.stack(pred)
  on, off = round(CHANGE_AT / nominal.dt), round(UNDO_AT / nominal.dt)
  per_s = round(1.0 / nominal.dt)

  def removed(window: slice) -> float:
    return 1.0 - rms(e[:, window]) / rms(t[:, window])

  return {
    "first_s": removed(slice(on + 1, on + 1 + per_s)),
    "second_s": removed(slice(on + 1 + per_s, on + 1 + 2 * per_s)),
    "later": removed(slice(on + 5 * per_s, off)),
    "jitter": rms(p[:, 3 * per_s : on]),
    "after_undo": rms(p[:, off + 2 * per_s :]),
  }


def wind_up(
  nominal: PlanarModel, features: Features, cfg: ResidualCfg, seeds: int
) -> tuple[float, float]:
  """Five healthy minutes of race commands, which hardly vary.

  Returns the largest coefficient uncertainty reached, and the size of the
  residual predicted at standstill over the last 100 s, in units of the usual
  error. The truth is zero, so both should stay small.
  """
  std = torch.tensor(cfg.meas_std)
  model = ResidualModel(nominal, features, cfg)
  steps = round(300.0 / nominal.dt)
  away, widest = [], 0.0
  for seed in range(seeds):
    gen = torch.Generator().manual_seed(1000 + seed)
    cmd = race_commands(steps, gen)
    noise = gait_noise(steps, std, nominal.dt, gen)
    run = make_run(nominal, cmd, torch.zeros(steps, 12), noise)
    away.append(track(model, run)[1][-round(100.0 / nominal.dt) :] / std)
    widest = max(widest, model.coef_std.max().item())
  return widest, rms(torch.stack(away))


def recorded_race(nominal: PlanarModel, cfg: ResidualCfg, log: Path) -> None:
  """The same filter on a real healthy run, where the noise is the real gait's."""
  data = np.load(log)
  vel, cmd = torch.tensor(data["vel"]), torch.tensor(data["cmd"])
  std = torch.tensor(cfg.meas_std)
  nominal_err = vel[1:] - nominal.next_velocity(vel[:-1], cmd[:-1])
  run = Run(vel, cmd[:-1], torch.zeros(len(vel) - 1, 3))
  skip = round(3.0 / nominal.dt)
  print(f"\nRecorded healthy race ({log}), after the first 3 s, in usual errors:")
  print(f"  {'':12s}  one-step error    mean residual (forward, sideways, yaw)")
  bias = (nominal_err[skip:] / std).mean(dim=0)
  print(
    f"  {'nominal':12s}  {rms(nominal_err[skip:] / std):14.2f}"
    f"    {bias[0]:+.2f} {bias[1]:+.2f} {bias[2]:+.2f}  (its own mean error)"
  )
  for name, features in FEATURE_SETS.items():
    pred, _ = track(ResidualModel(nominal, features, cfg), run)
    left = (nominal_err - pred)[skip:] / std
    mean = (pred[skip:] / std).mean(dim=0)
    print(
      f"  {name:12s}  {rms(left):14.2f}    {mean[0]:+.2f} {mean[1]:+.2f} {mean[2]:+.2f}"
    )


# Figure. Colours and line weights follow the project's chart conventions: three
# series colours checked for colour-blind separation, grey for the truth.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = {"offset": "#2a78d6", "gain+offset": "#eb6834", "command": "#1baf7a"}
LABELS = {
  "offset": "Offset only (3)",
  "gain+offset": "Gain and offset (6)",
  "command": "All commands (12)",
}
AXES = ("forward, m/s", "sideways, m/s", "yaw rate, rad/s")
# The axis each change is drawn on, so the figure covers all three.
SHOWN_AXIS = {"weak": 0, "drift": 1, "veer": 2}


def save_figure(
  nominal: PlanarModel, cfg: ResidualCfg, path: Path, seed: int = 0
) -> None:
  """One run per change and command style: the true correction and each estimate."""
  std = torch.tensor(cfg.meas_std)
  t = np.arange(round(DURATION / nominal.dt)) * nominal.dt
  plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
  fig, grid = plt.subplots(
    len(CHANGES), len(COMMANDS), figsize=(11, 7.6), sharex=True, facecolor=SURFACE
  )
  for row, (kind, text) in enumerate(CHANGES.items()):
    for col, (cmd_name, commands) in enumerate(COMMANDS.items()):
      ax = grid[row, col]
      run = scenario(nominal, kind, commands, std, seed)
      axis = SHOWN_AXIS[kind]
      ax.axvspan(CHANGE_AT, UNDO_AT, color=GRID, alpha=0.45, lw=0)
      for name, features in FEATURE_SETS.items():
        pred, _ = track(ResidualModel(nominal, features, cfg), run)
        ax.plot(t, pred[:, axis], color=SERIES[name], lw=1.6, label=LABELS[name])
      ax.plot(t, run.residual[:, axis], color=INK, lw=1.3, label="True correction")
      ax.set_facecolor(SURFACE)
      ax.grid(True, color=GRID, lw=0.6)
      ax.set_axisbelow(True)
      for side in ("top", "right"):
        ax.spines[side].set_visible(False)
      for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
      ax.tick_params(colors=MUTED, length=0)
      ax.set_title(
        f"{kind.capitalize()}: {text}" if col == 0 else "",
        loc="left",
        color=INK,
        fontsize=9.5,
        pad=6,
      )
      ax.set_ylabel(f"Correction, {AXES[axis]}", color=INK_2)
      if row == 0:
        ax.text(
          1.0,
          1.04,
          f"{cmd_name.capitalize()} commands",
          transform=ax.transAxes,
          ha="right",
          color=INK_2,
        )
      if row == len(CHANGES) - 1:
        ax.set_xlabel("Time, s", color=INK_2)
  handles, labels = grid[0, 0].get_legend_handles_labels()
  fig.legend(
    handles[::-1],
    labels[::-1],
    loc="upper left",
    bbox_to_anchor=(0.06, 0.945),
    ncols=4,
    frameon=False,
    labelcolor=INK_2,
  )
  fig.suptitle(
    "Online residual on made-up data: the change is applied at 10 s (shaded) "
    "and undone at 25 s",
    x=0.065,
    ha="left",
    color=INK,
    fontsize=11.5,
    fontweight="bold",
  )
  fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=160, facecolor=SURFACE)
  plt.close(fig)


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  default_model = DEFAULT_CHECKPOINT.with_name("planar_model.json")
  p.add_argument("--model-file", type=Path, default=default_model)
  p.add_argument("--drift-std", type=float, default=ResidualCfg.drift_std)
  p.add_argument("--leak", type=float, default=ResidualCfg.leak)
  p.add_argument("--gain-drift-ratio", type=float, default=ResidualCfg.gain_drift_ratio)
  p.add_argument("--seeds", type=int, default=8, help="Runs averaged per number.")
  p.add_argument("--log", type=Path, default=Path("logs/mppi/circle_race.npz"))
  p.add_argument("--out", type=Path, default=Path("logs/mppi/residual_check.png"))
  args = p.parse_args()
  torch.set_num_threads(1)  # The matrices are tiny, so threads only add overhead.

  nominal = PlanarModel.load(args.model_file)
  cfg = ResidualCfg(
    drift_std=args.drift_std, leak=args.leak, gain_drift_ratio=args.gain_drift_ratio
  )
  print(
    f"Settings: drift_std {cfg.drift_std}, leak {cfg.leak},"
    f" gain_drift_ratio {cfg.gain_drift_ratio:.2f}, {args.seeds} runs each"
  )

  print("\nShare of the new model error removed, and jitter in usual errors:")
  header = (
    f"  {'commands':8s} {'change':6s} {'features':12s}"
    "  first s  second s  later   jitter  after undo"
  )
  print(header)
  for cmd_name, commands in COMMANDS.items():
    for kind in CHANGES:
      for name, features in FEATURE_SETS.items():
        s = score(nominal, features, cfg, kind, commands, args.seeds)
        print(
          f"  {cmd_name:8s} {kind:6s} {name:12s}"
          f"  {s['first_s']:6.0%}  {s['second_s']:7.0%}  {s['later']:5.0%}"
          f"   {s['jitter']:5.2f}  {s['after_undo']:9.2f}"
        )

  print("\nSpeed against noise, 'command' features, averaged over the six cases:")
  print(f"  {'drift_std':9s}  first s  second s  later   jitter")
  for drift in (0.25, 0.4, 0.6, 0.9, 1.3):
    trial = replace(cfg, drift_std=drift)
    rows = [
      score(nominal, command_features, trial, kind, commands, args.seeds)
      for commands in COMMANDS.values()
      for kind in CHANGES
    ]
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    print(
      f"  {drift:9.2f}  {mean['first_s']:6.0%}  {mean['second_s']:7.0%}"
      f"  {mean['later']:5.0%}   {mean['jitter']:5.2f}"
    )

  print("\nFive healthy minutes of race commands, 'command' features:")
  print(f"  {'leak':5s}  widest coefficient std  residual predicted at standstill")
  for leak in (0.0, 0.05, 0.1, 0.2, 0.5):
    trial = replace(cfg, leak=leak)
    widest, away = wind_up(nominal, command_features, trial, min(args.seeds, 4))
    print(f"  {leak:5.2f}  {widest:22.2f}  {away:32.2f}")

  if args.log.exists():
    recorded_race(nominal, cfg, args.log)

  save_figure(nominal, cfg, args.out)
  print(f"\nFigure: {args.out}")


if __name__ == "__main__":
  main()
