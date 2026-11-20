# Baseline policy: Go1, flat terrain

The policy every demo in `mppi/` uses by default, kept in git so the demos run straight after a clone.

| File | What it is |
|---|---|
| `model_9999.pt` | PPO velocity-tracking policy for the undamaged Go1. It sees no history. |
| `planar_model.json` | The planar motion model fitted to that policy, used by the circle race. |

- **Trained:** 2026-11-10, task `Mjlab-Velocity-Flat-Unitree-Go1`, 4096 environments, 10,000 iterations, on mjlab commit `936f7a1`. W&B run `27yeeuu4`. The training command and results are in [notes/progress.md](../../../notes/progress.md).
- **Planar model:** fitted on 2026-11-11 over commands up to 3.0 m/s forward and 1.5 rad/s yaw. Regenerate it with `uv run python -m mppi.fit_planar_model`, which overwrites the file here.
- **Adding more:** each policy the team needs to share gets its own folder next to this one. Do not commit every training run. A checkpoint is about 5 MB and stays in git history for good.
