"""MPPI velocity planning on top of the trained Go1 velocity policy.

MPPI plans velocity commands (forward, sideways, yaw rate) and the RL policy
tracks them. Run the scripts as modules from the repository root:

  uv run python -m mppi.fit_planar_model
  uv run python -m mppi.demo_circle_race
  uv run python -m mppi.demo_box_push
  uv run python -m mppi.residual_check    # The online residual, on made-up data.
  # The online residual in the race, against a 30 N pull to the robot's right:
  uv run python -m mppi.demo_circle_race --pull 0 -30 --residual offset
"""

# Importing mjlab selects the headless EGL renderer. It has to happen before any
# module in this package imports mujoco, so it lives here.
import mjlab  # noqa: F401
