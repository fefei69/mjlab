"""MPPI velocity planning on top of the trained Go1 velocity policy.

MPPI plans velocity commands (forward, sideways, yaw rate) and the RL policy
tracks them. Run the scripts as modules from the repository root:

  uv run python -m mppi.fit_planar_model
  uv run python -m mppi.demo_circle_race
  uv run python -m mppi.demo_box_push
"""

# Importing mjlab selects the headless EGL renderer. It has to happen before any
# module in this package imports mujoco, so it lives here.
import mjlab  # noqa: F401
