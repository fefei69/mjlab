"""The box-jump course: a long box to build up speed, a gap, a smaller box.

Both boxes stand on flat ground with their tops at the same height. The robot
spawns near the back of the long box, facing the gap along +x. Missing the small
box, short or long, drops the robot to the ground.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_plane
from mjlab.utils import spec_config as spec_cfg

_SKY = spec_cfg.TextureCfg(
  name="sky",
  type="skybox",
  builtin="gradient",
  rgb1=(0.55, 0.66, 0.80),
  rgb2=(0.10, 0.12, 0.16),
  width=512,
  height=3072,
)
_RUNUP_COLOR = (0.36, 0.48, 0.66, 1.0)
_PAD_COLOR = (0.93, 0.58, 0.20, 1.0)


@dataclass(frozen=True)
class CourseLayout:
  """Where the course's parts are, in metres along +x from the spawn point."""

  edge: float
  """Front edge of the long box, where the gap starts."""
  pad_start: float
  pad_end: float
  runup_half_width: float
  pad_half_width: float
  height: float
  """Height of both box tops above the ground."""


@dataclass(kw_only=True)
class BoxJumpCourseCfg(SubTerrainCfg):
  """One course per terrain tile. Ranges go from difficulty 0 to difficulty 1."""

  height: float = 0.4
  runup_length: float = 6.0
  runup_width: float = 1.0
  gap_range: tuple[float, float] = (0.5, 0.5)
  pad_length_range: tuple[float, float] = (1.0, 1.0)
  pad_width: float = 1.0
  start_offset: float = 0.5
  """Spawn point's distance from the back end of the long box."""
  margin: float = 1.0
  """Ground before the long box and after the small box."""

  def tile_size(self) -> tuple[float, float]:
    """The smallest tile that holds the course at every difficulty."""
    length = (
      2 * self.margin
      + self.runup_length
      + max(self.gap_range)
      + max(self.pad_length_range)
    )
    width = max(self.runup_width, self.pad_width) + 2 * self.margin
    return (length, width)

  def layout(self, difficulty: float) -> CourseLayout:
    gap = float(np.interp(difficulty, (0.0, 1.0), self.gap_range))
    pad_length = float(np.interp(difficulty, (0.0, 1.0), self.pad_length_range))
    edge = self.runup_length - self.start_offset
    return CourseLayout(
      edge=edge,
      pad_start=edge + gap,
      pad_end=edge + gap + pad_length,
      runup_half_width=self.runup_width / 2,
      pad_half_width=self.pad_width / 2,
      height=self.height,
    )

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng  # The layout is set by the difficulty alone.
    body = spec.body("terrain")
    layout = self.layout(difficulty)
    spawn = np.array([self.margin + self.start_offset, self.size[1] / 2, self.height])

    def box(start: float, end: float, half_width: float) -> mujoco.MjsGeom:
      """A box from the ground up to the course height, x from spawn."""
      return body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=((end - start) / 2, half_width, self.height / 2),
        pos=(spawn[0] + (start + end) / 2, spawn[1], self.height / 2),
      )

    ground = make_plane(body, self.size, 0.0, center_zero=False)[0]
    ground.material = _floor_material(spec)  # A checker, so speed is visible.
    runup = box(-self.start_offset, layout.edge, layout.runup_half_width)
    pad = box(layout.pad_start, layout.pad_end, layout.pad_half_width)
    geometries = [
      TerrainGeometry(geom=ground),
      TerrainGeometry(geom=runup, color=_RUNUP_COLOR),
      TerrainGeometry(geom=pad, color=_PAD_COLOR),
    ]
    return TerrainOutput(origin=spawn, geometries=geometries)


def _floor_material(spec: mujoco.MjSpec) -> str:
  """The checker floor's material, added to `spec` the first time."""
  name = "box_jump_floor"
  if spec.material(name) is None:
    spec.add_texture(
      name=name,
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
      mark=mujoco.mjtMark.mjMARK_EDGE,
      rgb1=(0.2, 0.3, 0.4),
      rgb2=(0.1, 0.2, 0.3),
      markrgb=(0.8, 0.8, 0.8),
      width=300,
      height=300,
    )
    material = spec.add_material(
      name=name, texuniform=True, texrepeat=(4.0, 4.0), reflectance=0.2
    )
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
  return name


def box_jump_terrain(course: BoxJumpCourseCfg, num_rows: int = 1) -> TerrainEntityCfg:
  """A terrain of `num_rows` courses; row r has difficulty r / (num_rows - 1)."""
  generator = TerrainGeneratorCfg(
    size=course.tile_size(),
    num_rows=num_rows,
    num_cols=1,
    curriculum=True,
    sub_terrains={"box_jump": course},
    border_width=10.0,
  )
  cfg = TerrainEntityCfg(terrain_type="generator", terrain_generator=generator)
  cfg.textures = (*cfg.textures, _SKY)
  return cfg


# The demo (cpw, 2026-11-24): about 5.5 m of run-up to reach 2 to 3 m/s, a gap
# near Go1's body length (0.65 m), and a landing box short enough that a 20 to 40%
# error in jump length misses it. Gravity and foot friction change 2.5 m before
# the edge, about 1 s before take-off at 2.5 m/s.
DEMO_COURSE = BoxJumpCourseCfg()
DEMO_GRAVITY_RANGE = (7.0, 12.0)
"""Gravity after the change, in m/s^2. Nominal is 9.81."""
DEMO_FRICTION_RANGE = (0.4, 1.0)
"""Foot friction after the change. Nominal is 1.0; training saw 0.3 to 1.5."""
DEMO_CHANGE_BEFORE_EDGE = 2.5
"""Distance before the edge at which gravity and friction change, in metres."""
