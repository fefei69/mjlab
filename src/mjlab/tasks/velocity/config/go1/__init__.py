from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  FAULT_HISTORY_LENGTH,
  unitree_go1_flat_env_cfg,
  unitree_go1_flat_fault_env_cfg,
  unitree_go1_rough_env_cfg,
)
from .rl_cfg import unitree_go1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-Go1",
  env_cfg=unitree_go1_rough_env_cfg(),
  play_env_cfg=unitree_go1_rough_env_cfg(play=True),
  rl_cfg=unitree_go1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go1",
  env_cfg=unitree_go1_flat_env_cfg(),
  play_env_cfg=unitree_go1_flat_env_cfg(play=True),
  rl_cfg=unitree_go1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go1-Fault",
  env_cfg=unitree_go1_flat_fault_env_cfg(),
  play_env_cfg=unitree_go1_flat_fault_env_cfg(play=True),
  rl_cfg=unitree_go1_ppo_runner_cfg(experiment_name="go1_fault"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go1-Fault-History",
  env_cfg=unitree_go1_flat_fault_env_cfg(history_length=FAULT_HISTORY_LENGTH),
  play_env_cfg=unitree_go1_flat_fault_env_cfg(
    play=True, history_length=FAULT_HISTORY_LENGTH
  ),
  rl_cfg=unitree_go1_ppo_runner_cfg(experiment_name="go1_fault_history"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go1-Fault-Score",
  env_cfg=unitree_go1_flat_fault_env_cfg(fault_score=True),
  play_env_cfg=unitree_go1_flat_fault_env_cfg(play=True, fault_score=True),
  rl_cfg=unitree_go1_ppo_runner_cfg(experiment_name="go1_fault_score"),
  runner_cls=VelocityOnPolicyRunner,
)
