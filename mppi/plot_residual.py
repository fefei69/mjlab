"""Plot the planner's residual over time, around a change, from race logs.

Takes the logs that `demo_circle_race` writes next to its video: one run with a
fixed model and one with the residual, under the same change. The top three
panels show, per axis, the nominal model's error on each plan step (what the
residual is fitted to) and the residual's correction for that step, predicted
before the step was seen. The bottom panel shows both runs' distance from the
centre line, against the track's edges.

  uv run python -m mppi.demo_circle_race --pull 0 -30 --no-video \\
    --out logs/mppi/residual_fixed.mp4
  uv run python -m mppi.demo_circle_race --pull 0 -30 --residual command \\
    --no-video --out logs/mppi/residual_command.mp4
  uv run python -m mppi.plot_residual logs/mppi/residual_fixed.npz \\
    logs/mppi/residual_command.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

# The project's chart colours, as in residual_check.py.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CORRECTED, FIXED = "#2a78d6", "#898781"
AXES = ("forward, m/s", "sideways, m/s", "yaw rate, rad/s")


def style(ax: Axes) -> None:
  ax.set_facecolor(SURFACE)
  ax.grid(True, color=GRID, lw=0.6)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
  ax.tick_params(colors=MUTED, length=0)


def main() -> None:
  p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  p.add_argument("fixed", type=Path, help="Log of the run with the fixed model.")
  p.add_argument("corrected", type=Path, help="Log of the run with the residual.")
  p.add_argument("--radius", type=float, default=2.5)
  p.add_argument("--half-width", type=float, default=0.4)
  p.add_argument("--out", type=Path, default=Path("logs/mppi/residual_plot.png"))
  args = p.parse_args()

  fixed, corrected = np.load(args.fixed), np.load(args.corrected)
  if "correction" not in corrected.files:
    raise SystemExit(f"{args.corrected} has no correction. Re-run the race demo.")
  dt = float(corrected["dt"])
  key = "changed" if "changed" in corrected.files else "pulling"
  changing = corrected[key].astype(bool)
  change_at = float(np.argmax(changing)) * dt if changing.any() else None

  plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
  fig, axes = plt.subplots(4, 1, figsize=(9, 8.2), sharex=True, facecolor=SURFACE)
  t = np.arange(len(corrected["pose"])) * dt
  for axis, ax in enumerate(axes[:3]):
    style(ax)
    ax.plot(
      t,
      corrected["nominal_error"][:, axis],
      color=MUTED,
      lw=0.8,
      label="Nominal model's error",
    )
    ax.plot(
      t,
      corrected["correction"][:, axis],
      color=CORRECTED,
      lw=2.0,
      label="Residual's correction",
    )
    ax.set_ylabel(AXES[axis], color=INK_2)
  axes[0].set_title(
    "Correction learned online, run with the residual",
    loc="left",
    color=INK,
    fontsize=10,
  )
  axes[0].legend(
    loc="lower right",
    bbox_to_anchor=(1.0, 1.0),
    frameon=False,
    labelcolor=INK_2,
    ncols=2,
  )

  ax = axes[3]
  style(ax)
  for edge in (-args.half_width, args.half_width):
    ax.axhline(edge, color=INK_2, lw=1.0, ls="--")
  ax.text(
    t[-1],
    -args.half_width,
    "track edge ",
    color=INK_2,
    ha="right",
    va="bottom",
    fontsize=8,
  )
  for log, color, name in (
    (fixed, FIXED, "Fixed model"),
    (corrected, CORRECTED, "Residual"),
  ):
    tt = np.arange(len(log["pose"])) * dt
    off = np.hypot(log["pose"][:, 0], log["pose"][:, 1]) - args.radius
    ax.plot(tt, off, color=color, lw=2.0, label=name)
    ax.annotate(
      name,
      (tt[-1], off[-1]),
      xytext=(4, 0),
      textcoords="offset points",
      color=INK_2,
      va="center",
      fontsize=8.5,
    )
  ax.set_title(
    "Distance from the centre line, outward positive",
    loc="left",
    color=INK,
    fontsize=10,
  )
  ax.set_ylabel("m", color=INK_2)
  ax.set_xlabel("Time, s", color=INK_2)
  ax.legend(
    loc="lower right",
    bbox_to_anchor=(1.0, 1.0),
    frameon=False,
    labelcolor=INK_2,
    ncols=2,
  )

  if change_at is not None:
    for ax in axes:
      ax.axvline(change_at, color=INK, lw=1.0)
    axes[0].annotate(
      "change starts",
      (change_at, 1.0),
      xycoords=("data", "axes fraction"),
      xytext=(4, -10),
      textcoords="offset points",
      color=INK,
      fontsize=8.5,
    )
  fig.tight_layout()
  args.out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.out, dpi=160, facecolor=SURFACE)
  print(f"Figure: {args.out}")


if __name__ == "__main__":
  main()
