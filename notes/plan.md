# RL & OC Project Plan: Fast Adaptation to Leg Damage

> Snapshot of the plan as pasted into a Claude Code session on 2026-11-11, with the wording unchanged. Two parts didn't survive the paste: the system architecture figure and the experiments table. If the plan also lives in a shared doc, that doc is the source of truth, and this file should be updated when it changes.

## In one sentence

A meta-RL locomotion policy adapts its gait within seconds after a quadruped loses part of a leg. An MPPI planner over its velocity commands, with a small motion model re-fit online, keeps it on its path.

RL is the tracking policy (meta-RL with PPO). OC is the planner (MPPI with online system identification).

## Demo goal

A simulated Unitree Go1 walks a path and loses part of a leg mid-run. Within seconds it walks again and returns to the path, with no retraining.

- **Task:** follow a path of waypoints at a commanded speed.
- **Damage:** at a random time, one lower leg (calf) is cut off or a knee motor dies. Some damages are held out of training.
- **Success:** short recovery time (seconds until velocity tracking is back within a set tolerance), small path deviation after the chop, no falls.

## Method

Two layers: an MPPI planner picks velocity commands, and a meta-RL policy tracks them. Both adapt online without weight updates.

*(Figure missing from this snapshot: system architecture, 2 layers, 5 parts.)*

- **Tracking controller (RL):** a PPO policy that sees a short history of observations and runs at 50 Hz. It trains across many leg damages that start at random times mid-episode, so it learns to change its gait within seconds. This is black-box meta-RL.
- **Planner (OC):** MPPI over velocity commands (forward, sideways, yaw rate), with a 1–2 s horizon, re-planning at 10–20 Hz. Its model is the robot's planar motion under the policy. A few gains and offsets are re-fit every step by least squares on the last second of motion.
- **Why this split:** a damaged robot limps, veers and slows down. The policy fixes how it walks; the planner fixes where it goes. Training is one PPO run plus a tiny model.
- **Considered and dropped:** a fully learned body model with MPPI (GrBAL-style) and a TD-MPC-style world model. Both are heavier than this project needs.

## Starting point and tools

We start from mjlab's Go1 velocity-tracking task: PyTorch on MuJoCo Warp, with an Isaac Lab-style manager API.

- **Where:** mjlab, pinned to one release. Its Go1 velocity tasks train with RSL-RL PPO.
- **Why mjlab:** an Isaac Lab-style API, PyTorch, and simulator state in GPU tensors with no copying. Most of the plan is built in.
- **Built in:**
  - Observation history for the policy, set per observation term or group.
  - Mid-episode events, and per-environment geom size, body mass and PD gain randomization.
  - Plain MuJoCo model files and data structures, so the demo video can run in CPU MuJoCo.
- **What we add:**
  - A chop event: at a random time, shrink the calf and foot geoms, lower the leg's mass and zero the knee gains. Restore them on reset.
  - A command term that takes velocity commands from the planner instead of random ones.
  - MPPI and the least-squares motion model, in PyTorch on the GPU.
- **Fallback:** MuJoCo Playground 0.2.0 (`pip install playground==0.2.0`), the last JAX/MJX release, which has a Go1 joystick environment.

## Experiments

Four controllers face the same damages, so each row shows what one layer adds. The main metric is recovery time after the chop.

*(Table missing from this snapshot. The rows we assume are listed in [ideas.md](ideas.md), entry 2.1. Paste the real table here.)*

- **Damages:** train on a range (which leg, how much of the calf, dead knee motor); test on combinations held out of training.
- **Metrics:** recovery time, velocity-tracking error vs. time since the chop, path deviation, falls.
- **Optional:** an oracle planner whose model is told the true damage, as an upper bound. The simple path follower can be pure pursuit.

## Milestones and work split

Five milestones, in order; each one produces a result we can show.

1. **Baseline:** train mjlab's Go1 velocity-tracking task and confirm it tracks velocity commands.
2. **Damage injection:** damage at a random time per episode (calf geometry, dead knee motor), plus held-out test damages.
3. **Meta-RL policy:** PPO with observation history across damages, compared against the no-history policy (rows 1–2).
4. **Planner:** MPPI over velocity commands with the least-squares motion model, on a path task (rows 3–4).
5. **Evaluation and demo:** the full table, recovery-time plots, and the chop video in CPU MuJoCo.

Suggested split for two people: one takes milestones 1–3 (environment, damage, policy). The other builds milestone 4 in parallel against the undamaged baseline. Both do 5.

- [ ] Set the deadline and checkpoint dates.

## Risks and fallbacks

- **The policy can't walk after a full calf cut.** Use a curriculum: dead knee motor and short cuts first, then longer cuts.
- **The planar model is too simple for the damaged gait.** Add speed-dependent terms or a small learned residual, still re-fit online.
- **mjlab is young and its API still changes.** Pin one release, and fall back to Playground 0.2.0 if it blocks us.
- **In MuJoCo Warp, collision on/off bits are shared across parallel worlds.** Shrink or move the geoms instead, and do the visual chop in CPU MuJoCo.

## References

- **LocoFormer** (Skild AI, CoRL 2025): the leg-chop adaptation demo that motivates this project; videos.
- **Learning to Adapt in Dynamic, Real-World Environments** (Nagabandi et al., ICLR 2019): meta-learned models with MPC, including a robot that loses a leg.
- **DATT** (CoRL 2023): an RL policy paired with an online adaptive estimator.
- **mjlab:** our starting point. Docs on events, domain randomization, observation history and per-world limits.
- **MuJoCo Playground:** the fallback; version 0.2.0 is the last JAX/MJX release.
