from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg

from .env_cfgs import HISTORY_LENGTH, unitree_go1_box_jump_env_cfg

register_mjlab_task(
  task_id="Mjlab-BoxJump-Unitree-Go1",
  env_cfg=unitree_go1_box_jump_env_cfg(),
  play_env_cfg=unitree_go1_box_jump_env_cfg(play=True),
  rl_cfg=unitree_go1_ppo_runner_cfg(experiment_name="go1_box_jump"),
)

register_mjlab_task(
  task_id="Mjlab-BoxJump-Unitree-Go1-History",
  env_cfg=unitree_go1_box_jump_env_cfg(history_length=HISTORY_LENGTH),
  play_env_cfg=unitree_go1_box_jump_env_cfg(play=True, history_length=HISTORY_LENGTH),
  rl_cfg=unitree_go1_ppo_runner_cfg(experiment_name="go1_box_jump_history"),
)

register_mjlab_task(
  task_id="Mjlab-BoxJump-Unitree-Go1-Robust",
  env_cfg=unitree_go1_box_jump_env_cfg(robust=True),
  play_env_cfg=unitree_go1_box_jump_env_cfg(play=True, robust=True),
  rl_cfg=unitree_go1_ppo_runner_cfg(experiment_name="go1_box_jump_robust"),
)

register_mjlab_task(
  task_id="Mjlab-BoxJump-Unitree-Go1-History-Robust",
  env_cfg=unitree_go1_box_jump_env_cfg(history_length=HISTORY_LENGTH, robust=True),
  play_env_cfg=unitree_go1_box_jump_env_cfg(
    play=True, history_length=HISTORY_LENGTH, robust=True
  ),
  rl_cfg=unitree_go1_ppo_runner_cfg(experiment_name="go1_box_jump_history_robust"),
)
