"""Unitree Go1 box-jump environment configurations.

Built on the Go1 rough-terrain velocity config for the robot, its contact
sensors and action scale, with the terrain, command, observations, rewards,
events, terminations and curriculum replaced.
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.box_jump import mdp as jump_mdp
from mjlab.tasks.box_jump.course import (
  DEMO_FRICTION_RANGE,
  DEMO_GRAVITY_RANGE,
  BoxJumpCourseCfg,
  box_jump_terrain,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_rough_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

FEET = tuple(f"{leg}_foot_collision" for leg in ("FR", "FL", "RR", "RL"))
# Training courses, easiest row first: the gap grows and the landing box
# shrinks. The demo course (gap 0.5 m, box 1.0 m) is two thirds of the way.
TRAIN_COURSE = BoxJumpCourseCfg(gap_range=(0.1, 0.7), pad_length_range=(2.0, 1.0))
TRAIN_ROWS = 10
# Actor history for the policy with memory: 25 policy steps, 0.5 s.
HISTORY_LENGTH = 25
# Gravity and foot friction: each episode starts somewhere in these ranges, and
# in half of them they change once on the run-up.
PHYSICS_PARAMS = {
  "asset_cfg": SceneEntityCfg("robot", geom_names=FEET),
  "start_gravity_range": DEMO_GRAVITY_RANGE,
  "start_friction_range": (DEMO_FRICTION_RANGE[0], 1.2),
  "gravity_range": DEMO_GRAVITY_RANGE,
  "friction_range": DEMO_FRICTION_RANGE,
  "onset_x": (0.5, 4.5),
  "prob": 0.5,
}


def unitree_go1_box_jump_env_cfg(
  play: bool = False, history_length: int = 0
) -> ManagerBasedRlEnvCfg:
  """Go1 runs along the long box and jumps onto the small one on command.

  With `history_length` > 0 the actor sees that many past steps of every
  observation except the command and the course ahead.
  """
  cfg = unitree_go1_rough_env_cfg()

  # Few contacts on two boxes: the flat-terrain solver budget is enough.
  cfg.sim.njmax = 300
  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.episode_length_s = 6.0

  cfg.scene.terrain = box_jump_terrain(TRAIN_COURSE, num_rows=TRAIN_ROWS)
  cfg.scene.terrain.max_init_terrain_level = 0
  # No ray casts: the course ahead comes from the command term.
  cfg.scene.sensors = tuple(
    s
    for s in (cfg.scene.sensors or ())
    if s.name not in ("terrain_scan", "foot_height_scan", "self_collision")
  )

  cfg.commands = {"jump": jump_mdp.JumpCommandCfg(course=TRAIN_COURSE)}

  # Observations.
  actor = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.3, n_max=0.3),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    # The accelerometer reads gravity plus the trunk's acceleration, in m/s^2.
    "accelerometer": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_acc"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
      clip=(-50.0, 50.0),
      scale=0.1,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "jump"}
    ),
    "course": ObservationTermCfg(
      func=jump_mdp.course_ahead, clip=(-2.0, 8.0), scale=0.25
    ),
  }
  critic = {
    **{k: replace(v, noise=None) for k, v in actor.items()},
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "physics": ObservationTermCfg(func=jump_mdp.physics_params),
    "base_height": ObservationTermCfg(func=jump_mdp.base_height),
    "phase": ObservationTermCfg(func=jump_mdp.jump_phase),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
    ),
  }
  if history_length > 0:
    for name, term in actor.items():
      if name not in ("command", "course"):
        actor[name] = replace(term, history_length=history_length)
  cfg.observations["actor"].terms = actor
  cfg.observations["critic"].terms = critic

  # Events: start on the long box facing the gap; gravity and friction change.
  cfg.events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": cfg.events["reset_robot_joints"],
    "encoder_bias": cfg.events["encoder_bias"],
    "base_com": cfg.events["base_com"],
    "physics_change": EventTermCfg(
      func=jump_mdp.physics_change, mode="step", params=dict(PHYSICS_PARAMS)
    ),
  }

  # Rewards are scaled by the 0.02 s step, so one-off terms (lift-off, falling)
  # carry large weights: 300 at lift-off is worth up to 6, like 3 s of tracking.
  cfg.rewards = {
    "track_velocity": RewardTermCfg(
      func=jump_mdp.track_jump_velocity, weight=2.0, params={"std": 0.5}
    ),
    "track_yaw_rate": RewardTermCfg(
      func=jump_mdp.track_jump_yaw_rate, weight=0.5, params={"std": 0.5}
    ),
    "takeoff_velocity": RewardTermCfg(
      func=jump_mdp.takeoff_velocity,
      weight=300.0,
      params={"std_vz": 0.5, "std_vx": 0.6},
    ),
    "takeoff_push": RewardTermCfg(func=jump_mdp.takeoff_push, weight=5.0),
    "landed_on_pad": RewardTermCfg(func=jump_mdp.landed_on_pad, weight=2.0),
    "flat_orientation": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-2.0),
    "torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-5),
    "thigh_collision": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "thigh_ground_touch"},
    ),
    "shank_collision": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "shank_ground_touch"},
    ),
    "terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_off": TerminationTermCfg(func=jump_mdp.fell_off_course),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation, params={"limit_angle": math.radians(80.0)}
    ),
    "trunk_contact": TerminationTermCfg(
      func=mdp.illegal_contact, params={"sensor_name": "trunk_ground_touch"}
    ),
  }

  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(func=jump_mdp.terrain_levels_jump),
  }

  # Watch from the side, following the robot.
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_ROOT,
    entity_name="robot",
    distance=4.0,
    elevation=-12.0,
    azimuth=90.0,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.terminations = {"fell_off": cfg.terminations["fell_off"]}
  return cfg
