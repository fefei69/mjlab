"""Drawing for the demo videos: MPPI rollouts in the scene, and a text panel.

Sampled rollouts are colored by cost on a single-hue ramp, light for low cost and
dark for high cost, so the good ones stand out against the dark floor. The chosen
plan has its own color and is drawn thicker. The panel names both in a legend.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.viewer.debug_visualizer import DebugVisualizer

Rgb = tuple[float, float, float]
Rgba = tuple[float, float, float, float]

# One hue, monotone in lightness: low cost (light) to high cost (dark).
COST_RAMP = np.array(
  [
    [1.00, 0.94, 0.80],
    [1.00, 0.78, 0.47],
    [0.92, 0.45, 0.20],
    [0.58, 0.23, 0.09],
    [0.30, 0.12, 0.06],
  ]
)
PLAN_COLOR: Rgba = (0.30, 1.00, 0.85, 1.0)
TRAIL_COLOR: Rgba = (0.62, 0.70, 0.82, 0.9)


def cost_colors(costs: np.ndarray) -> np.ndarray:
  """One RGBA color per rollout, from the rank of its cost.

  The lowest cost gets the lightest, most opaque color. Ranks are used instead
  of cost values so that a few very bad rollouts do not flatten the scale.
  """
  n = len(costs)
  rank = np.empty(n)
  rank[np.argsort(costs)] = np.linspace(0.0, 1.0, n)
  stops = np.linspace(0.0, 1.0, len(COST_RAMP))
  rgb = [np.interp(rank, stops, COST_RAMP[:, c]) for c in range(3)]
  alpha = 0.95 - 0.7 * rank
  return np.stack([*rgb, alpha], axis=1)


def draw_path(
  vis: DebugVisualizer,
  points: np.ndarray,
  z: float,
  radius: float,
  color: Rgba,
  markers: bool = False,
) -> None:
  """Draw a polyline through (N, 2) ground points at height `z`."""
  pts = np.concatenate([points[:, :2], np.full((len(points), 1), z)], axis=1)
  for a, b in zip(pts[:-1], pts[1:], strict=True):
    if np.linalg.norm(b - a) > 1e-4:
      vis.add_cylinder(a, b, radius, color)
  if markers:
    for point in pts[1:]:
      vis.add_sphere(point, 2.2 * radius, color)


def draw_polyline(
  vis: DebugVisualizer, points: np.ndarray, radius: float, color: Rgba
) -> None:
  """Draw a polyline through (N, 3) points."""
  for a, b in zip(points[:-1], points[1:], strict=True):
    if np.linalg.norm(b - a) > 1e-4:
      vis.add_cylinder(a, b, radius, color)


def draw_rollouts(
  vis: DebugVisualizer, paths: np.ndarray, costs: np.ndarray, z: float, radius: float
) -> None:
  """Draw sampled rollouts, (N, H, 2) paths with (N,) costs, colored by cost."""
  colors = cost_colors(costs)
  # Worst first, so that the best rollouts end up on top.
  for i in np.argsort(costs)[::-1]:
    r, g, b, a = colors[i]
    draw_path(vis, paths[i], z, radius, (float(r), float(g), float(b), float(a)))


def _font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  try:
    return ImageFont.truetype(name, size)
  except OSError:
    return ImageFont.load_default(size)


def _to_255(color: Sequence[float]) -> tuple[int, int, int]:
  return (round(255 * color[0]), round(255 * color[1]), round(255 * color[2]))


class Hud:
  """A panel in the top-left corner: title, live readouts and a color legend.

  Legend entries pair a label with either a color, shown as a line, or the
  string "ramp", shown as the cost ramp.
  """

  INK = (240, 240, 240)
  MUTED = (185, 190, 198)

  def __init__(
    self,
    title: str,
    subtitle: str,
    legend: Sequence[tuple[str, Rgba | str]],
    note: str | None = None,
  ) -> None:
    self._title = title
    self._subtitle = subtitle
    self._legend = legend
    self._note = note
    self._title_font = _font("DejaVuSans-Bold.ttf", 22)
    self._font = _font("DejaVuSans.ttf", 17)
    self._mono = _font("DejaVuSansMono.ttf", 17)

  def draw(self, frame: np.ndarray, readouts: Sequence[tuple[str, str]]) -> np.ndarray:
    """Return `frame` with the panel drawn on it. Readouts are (label, value)."""
    pad, line, swatch = 14, 25, 46
    width = 440
    rows = len(readouts) + len(self._legend) + (1 if self._note else 0)
    height = 2 * pad + 34 + 26 + line * rows + 22
    image = Image.fromarray(frame).convert("RGBA")
    panel = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    x0, y0 = 16, 16
    draw.rounded_rectangle(
      (x0, y0, x0 + width, y0 + height), radius=10, fill=(12, 16, 24, 185)
    )
    x, y = x0 + pad, y0 + pad
    draw.text((x, y), self._title, font=self._title_font, fill=self.INK)
    y += 34
    draw.text((x, y), self._subtitle, font=self._font, fill=self.MUTED)
    y += 26 + 6
    for label, value in readouts:
      draw.text((x, y), label, font=self._font, fill=self.MUTED)
      draw.text((x + 120, y), value, font=self._mono, fill=self.INK)
      y += line
    y += 10
    for label, color in self._legend:
      mid = y + 11
      if isinstance(color, str):
        for i in range(swatch):
          t = i / (swatch - 1)
          stops = np.linspace(0.0, 1.0, len(COST_RAMP))
          rgb = [float(np.interp(t, stops, COST_RAMP[:, c])) for c in range(3)]
          draw.line((x + i, mid - 5, x + i, mid + 5), fill=(*_to_255(rgb), 255))
      else:
        draw.line((x, mid, x + swatch, mid), fill=(*_to_255(color), 255), width=5)
      draw.text((x + swatch + 12, y), label, font=self._font, fill=self.INK)
      y += line
    if self._note:
      draw.text((x, y), self._note, font=self._font, fill=self.MUTED)
    return np.asarray(Image.alpha_composite(image, panel).convert("RGB"))


Curve = tuple[np.ndarray, np.ndarray]


def draw_speed_panel(
  frame: np.ndarray,
  samples: Sequence[Curve],
  costs: np.ndarray | None,
  chosen: Curve | None,
  actual: Curve,
  edge: float,
  box: tuple[float, float],
  x_max: float,
  v_max: float = 3.0,
  reference: Curve | None = None,
  reference_color: Rgba = (0.75, 0.55, 1.00, 0.9),
) -> np.ndarray:
  """A speed-against-position chart in the bottom-right corner of `frame`.

  Sampled plans are coloured by cost as in the scene, the chosen plan and the
  speed actually run are drawn on top. The landing box is shaded and the edge
  of the run-up box marked.
  """
  image = Image.fromarray(frame).convert("RGBA")
  panel = Image.new("RGBA", image.size, (0, 0, 0, 0))
  draw = ImageDraw.Draw(panel)
  w, h = 430, 210
  x0, y0 = image.size[0] - w - 16, image.size[1] - h - 16
  draw.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=10, fill=(12, 16, 24, 185))
  font = _font("DejaVuSans.ttf", 15)
  draw.text((x0 + 14, y0 + 10), "Speed along the course", font=font, fill=Hud.INK)
  left, right, top, bottom = x0 + 46, x0 + w - 16, y0 + 38, y0 + h - 30

  def px(x: np.ndarray | float, v: np.ndarray | float):
    u = left + (right - left) * np.clip(np.asarray(x) / x_max, 0, 1)
    z = bottom - (bottom - top) * np.clip(np.asarray(v) / v_max, 0, 1)
    return u, z

  bx0, _ = px(box[0], 0)
  bx1, _ = px(box[1], 0)
  draw.rectangle((float(bx0), top, float(bx1), bottom), fill=(237, 148, 51, 70))
  ex, _ = px(edge, 0)
  draw.line((float(ex), top, float(ex), bottom), fill=(*Hud.MUTED, 255), width=1)
  draw.line((left, bottom, right, bottom), fill=(*Hud.MUTED, 255), width=1)
  draw.line((left, top, left, bottom), fill=(*Hud.MUTED, 255), width=1)
  small = _font("DejaVuSans.ttf", 12)
  for v in (0, 1, 2, 3):
    _, z = px(0, v)
    draw.text((x0 + 14, float(z) - 7), f"{v}", font=small, fill=Hud.MUTED)
  draw.text((right - 60, bottom + 6), f"{x_max:.0f} m", font=small, fill=Hud.MUTED)
  draw.text((x0 + 14, top - 2), "m/s", font=small, fill=Hud.MUTED)

  def line(curve: Curve, color: tuple[int, int, int, int], width: int) -> None:
    u, z = px(*curve)
    if len(u) > 1:
      draw.line(list(zip(u.tolist(), z.tolist(), strict=True)), fill=color, width=width)

  if costs is not None and len(samples):
    colors = cost_colors(costs)
    for i in np.argsort(costs)[::-1]:
      r, g, b, a = colors[i]
      line(samples[i], (*_to_255((r, g, b)), round(255 * a)), 1)
  if reference is not None:
    line(reference, (*_to_255(reference_color), 255), 2)
  if chosen is not None:
    line(chosen, (*_to_255(PLAN_COLOR), 255), 3)
  line(actual, (*_to_255(TRAIL_COLOR), 255), 2)
  return np.asarray(Image.alpha_composite(image, panel).convert("RGB"))
