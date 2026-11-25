"""DIAL-MPC on the box-jump course, for comparison with the MPPI + RL planner.

DIAL-MPC (Xue et al., 2024; github.com/LeCAR-Lab/dial-mpc) plans joint targets
directly with full-robot rollouts and no learned policy. `core.py` is its
planner, `box_jump.py` runs it on the course.
"""
