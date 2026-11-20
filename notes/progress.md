# Progress

## Milestones

| # | Milestone (from [plan.md](plan.md)) | Status | Notes |
|---|---|---|---|
| 1 | Baseline: train Go1 velocity tracking | Done | The flat-terrain run from 2026-11-10 tracks velocity commands. It is an undamaged policy without history, so it is a reference point, not row 1 of the experiments table. |
| 2 | Damage injection | Not started | Design notes are in [ideas.md](ideas.md), section 4. |
| 3 | Meta-RL policy | Not started | |
| 4 | Planner | Started | MPPI, the planar model and its least-squares fit exist and run on the undamaged policy in two demos (2026-11-11). Still missing: re-fitting the model online, and a path task with damage. |
| 5 | Evaluation and demo | Not started | |

Still to set: the deadline and checkpoint dates.

## Setup

- **Training PC:** one RTX 5080, reached over SSH. Rendering works headless, so videos can be recorded without a display.
- **Code:** the fork `fefei69/mjlab` is the remote `origin`, and `mujocolab/mjlab` is `upstream`. Project work goes on the branch `dev`, which is local only so far.
- **Version:** commit `936f7a1`, which is mjlab v1.6.0 plus 60 commits, with MuJoCo and MuJoCo Warp 3.16.
- **Training budget:** 10,000 PPO iterations with 4096 environments take 1 h 14 min, so several runs fit in a day.
- **Results:** everything under `logs/` is ignored by git and exists only on the training PC.
- **Shared baseline:** the final baseline checkpoint and its fitted planar model are also in the repository, in `mppi/checkpoints/go1_flat_baseline/`, and the demos load them from there by default.

## Log

### 2026-11-20

- Wrote [roles.md](roles.md): five roles, one per person. cpw takes the method role, which is the design of the planner's residual dynamics model. The other four are open.
- No work on the items since 2026-11-12. The task list had not been pushed.
- Put the baseline checkpoint (4.6 MB) and its planar model into the repository, in `mppi/checkpoints/go1_flat_baseline/`, so teammates can run the demos straight after a clone.
- Added item S4, the randomization the policies train on and the held-out set, to the Robot changes role.
- cpw's role now covers a residual model at two levels. The planner's residual comes first. A joint-level residual given to the walking policy as an input comes second, as an exploration ([ideas.md](ideas.md), entries 3.11 and 5.7). Whether this changes the four-way comparison is not decided.

### 2026-11-12

- Worked out how the policy and the planner would each adapt online: history conditioning as black-box meta-RL, and a residual on the planar model re-fitted by least squares ([ideas.md](ideas.md), entries 5.6 and 3.11). Nothing built.
- Drew the system figure, shown at the top of [ideas.md](ideas.md).
- The comparison for the final experiments is now a two-by-two of four controllers ([ideas.md](ideas.md), entry 2.1).
- Wrote [tasks.md](tasks.md): action items for the four teammates to pick from. Which change to test on is still open ([tasks.md](tasks.md), section 2).

### 2026-11-11

- Read the plan and checked its claims about mjlab against the code. Everything it lists as built in exists. The review's suggestions are in [ideas.md](ideas.md), sections 3 to 6.
- Read the LLA-MPPI paper, the closest prior work. Summary in [related-work.md](related-work.md); what we take from it is in ideas.md, entries 2.5, 2.6 and 3.3.
- Discussed how to frame the contribution (ideas.md, entries 1.1 and 1.2).
- Discussed which task to build the planner on first: racing or box pushing (ideas.md, entry 3.4). Not decided.
- Created the branch `dev` from `main` at `936f7a1`.
- Started this notes folder.
- Built two MPPI demos on the undamaged baseline policy, described below. The leg-damage parts of the plan (chop event, history policy) are not started.

**MPPI demos.** In both, MPPI plans velocity commands (forward, sideways, yaw rate) and the trained RL policy tracks them. The code is in `mppi/` on the branch `dev`, not committed yet.

| File | What it does |
|---|---|
| `mppi/core.py` | MPPI over a command sequence. It takes any function that scores command sequences. |
| `mppi/tracker.py` | Runs the trained policy under planned commands, and copies simulation state between a real and a planning environment. |
| `mppi/planar_model.py` | The planar motion model `v_next = A v + B u + G s(u) + c` and its least-squares fit. `s(u)` holds speed-dependent terms. |
| `mppi/fit_planar_model.py` | Collects data from the policy and fits the model. |
| `mppi/sweep_commands.py` | Measures what the policy does with commands outside its trained ranges. |
| `mppi/demo_circle_race.py` | Demo 1: race around a circle, planning with the planar model. |
| `mppi/demo_box_push.py` | Demo 2: push a box through goals, planning with simulator rollouts. |
| `mppi/viz.py` | Draws the sampled MPPI rollouts into the videos, colored by cost, plus a text panel with live readouts and a legend. |

*Planar model fit, first version.* This fit used only the command range the policy was trained on (forward −1.5 to 2.0 m/s, sideways ±1.0 m/s, yaw ±0.7 rad/s) and had no speed-dependent terms. 512 robots were driven for 60 s each with commands that change every 0.2 to 3 s. No robot fell. The model steps every 0.1 s.

- **What the fit says:** the robot delivers 99% of a commanded forward velocity and 94% of a commanded sideways velocity or yaw rate, with a time constant of about 0.3 s for linear velocity and 0.1 s for yaw rate. Offsets and cross-coupling are close to zero.
- **How well it predicts**, on robots held out of the fit, against the naive model that assumes the robot moves exactly as commanded:

  | Prediction | Fitted model | Naive model |
  |---|---|---|
  | Position, 1 s ahead (robot travels 0.52 m) | 0.058 m | 0.121 m |
  | Position, 2 s ahead (robot travels 0.95 m) | 0.102 m | 0.192 m |
  | Heading, 2 s ahead | 0.039 rad | 0.047 rad |

- **What it means for the plan:** the plan's risk that "the planar model is too simple" does not show up on the undamaged robot, provided the model includes the response lag.
- **Where it is saved:** `logs/rsl_rl/go1_velocity/2026-11-10_17-04-19/planar_model.json`. That file now holds the wider fit described under "Demo 1, faster version". Running the fit with `--cmd-max 2.0 1.0 0.7 --cmd-min -1.5 -1.0 -0.7 --linear` reproduces this first one.

*Demo 1: circle race, first version.* Track radius 2.5 m, half-width 0.4 m, 30 s. MPPI uses the planar model with a 1.5 s horizon and re-plans every 0.1 s. Its commands were limited to the trained range.

| Planner setting | Average speed | Lap times (s) | Mean distance from centre line | Heading off the track direction |
|---|---|---|---|---|
| Fitted model (the video) | 1.64 m/s | 9.6, 9.6, 9.6 | 0.04 m | 11 deg |
| Naive model | 1.60 m/s | 9.8, 9.8, 9.8 | 0.11 m | 12 deg |
| Fitted model, no heading cost | 1.61 m/s | 9.4, 10.0, 9.9 | 0.38 m | 43 deg |

- **Planning is real time:** 7 ms per plan (2048 samples, 4 rounds), against a 100 ms budget.
- **The lap time was set by the command limits given to MPPI.** An earlier version of these notes called that the policy's yaw-rate limit. That was wrong: 0.7 rad/s is only the range the policy was trained on, and it can turn much faster (see the faster version below).
- **Video:** replaced by the faster version's video.

*Demo 1, faster version.* This is the current default and the current video. A sweep of commands beyond the trained range showed that the policy can do much more than it was trained on, so MPPI's limits were widened.

- **Command sweep** (`mppi/sweep_commands.py`). One robot per command, each ramped in over 2 s and then held:

  | Command | Reached |
  |---|---|
  | Yaw 2.0 rad/s, standing | 1.92 rad/s |
  | Forward 3.0 m/s, straight | 2.81 m/s |
  | Forward 4.0 m/s, straight | 3.22 m/s |
  | Forward 3.0 m/s with yaw 1.4 rad/s | 2.48 m/s and 1.09 rad/s |

  Robots started to fall at forward commands of 3.75 m/s.
- **New command limits:** forward −1.5 to 3.0 m/s, sideways ±1.0 m/s, yaw ±1.5 rad/s. They come from the fit's stress test, which drives 512 robots for 60 s with abrupt random commands. Inside these limits 0 to 2 robots fell (three runs). With limits of 3.5 m/s and 2.0 rad/s, 19 fell, and with 4.0 m/s, 39 fell.
- **Model:** re-fitted over the new limits, with speed-dependent terms added and still by least squares. Predicting 2 s ahead it is off by 0.151 m, against 0.268 m for the naive model (the robots travel 1.24 m on average). The terms matter at the limit. For the command 3.0 m/s with 1.4 rad/s the robot does 2.48 m/s and 1.09 rad/s. The fit without the terms predicts 2.90 m/s and 1.29 rad/s, and the fit with them predicts 2.64 m/s and 1.20 rad/s.
- **Planner changes:** the robot may use the whole track width, with no cost for leaving the centre line, and the horizon is 1.0 s.

| Version | Lap time | Ground speed | Line taken |
|---|---|---|---|
| First version | 9.6 s | 1.64 m/s | Centre line, radius 2.5 m |
| Faster version (the video) | 5.5 to 5.6 s | 2.40 m/s | Inside edge, radius 2.14 m |

- **Where the gain comes from:** higher speed and a shorter line. With the new planner settings but the old command limits, a lap takes 9.0 s.
- **Reliability:** no fall in the video run or in three more runs of 120 s each. The robot stays on the track, at most 0.39 m from the centre line where the edge is at 0.40 m.
- **Fitted against naive model:** with the same limits, the naive model laps in 5.6 to 5.9 s and leaves the track (0.45 m from the centre line).
- **Why not faster still:** at 4.0 m/s the robot fell in 2 of 3 races. Limits of 3.5 m/s and 2.0 rad/s were not faster in a test with the earlier planner settings, and that range had 19 falls in the stress test. Going faster needs a policy trained on wider command ranges ([ideas.md](ideas.md), entry 5.5).
- **Video:** `logs/mppi/circle_race.mp4`. Log: `logs/mppi/circle_race.npz`.

*Demo 2: box push.* A 2 kg box, 0.5 × 0.5 × 0.4 m, is pushed through three goals in order. MPPI rolls out 384 command sequences in 384 parallel simulator worlds over a 2 s horizon, with no hand-written model of the box.

- **Result in the video:** all 3 goals reached, at 3.2, 8.0 and 11.6 s. The box came to rest 0.055 m from the last goal and stayed upright. The robot did not fall.
- **Repeatability:** eight other runs also reached every goal: one with a single goal, three seeds of a two-goal course, a different two-goal layout, and three more runs of the three-goal course. Times vary between runs: the second goal was reached after 6.4 to 11.0 s, and the third after 10.6 to 13.8 s.
- **Not real time:** each plan takes 0.77 s and there is one plan per 0.2 s of motion, so the demo runs about 4 times slower than real time. The real world is paused while planning.
- **Video:** `logs/mppi/box_push.mp4`. Log: `logs/mppi/box_push.npz`.

*What the videos show.* Both draw the MPPI rollouts on the floor. The sampled plans are colored by cost on one orange ramp, light for low cost and dark for high cost, and the chosen plan is aqua. A panel in the corner shows the time, the command and a legend. In the box push, thin lines are predicted robot paths and thick lines are predicted box paths. 64 of the samples are drawn; `--rollouts-shown` changes that.

Commands, run from the repository root:

```sh
uv run python -m mppi.fit_planar_model     # Once per checkpoint, about 30 s.
uv run python -m mppi.demo_circle_race     # About 30 s.
uv run python -m mppi.demo_box_push        # About 1 min.

uv run python -m mppi.sweep_commands        # Optional: what the policy can do, about 20 s.

# Variations.
uv run python -m mppi.demo_circle_race --model ideal --radius 3.0
uv run python -m mppi.demo_circle_race --facing-weight 0
uv run python -m mppi.demo_circle_race --no-video --duration 120
# The race inside the trained command range only (slower):
uv run python -m mppi.demo_circle_race --cmd-min -1.5 -1 -0.7 --cmd-max 2 1 0.7
uv run python -m mppi.demo_box_push --goals 3.5 -1.0 --seed 1
```

Each script prints many "solver iterations limit reached" warnings from MuJoCo Warp. They are harmless; add `2>&1 | grep "\[race\]"` (or `"\[push\]"`, `"\[fit\]"`) to see only the results.

### 2026-11-10

**Baseline training run (milestone 1).** Task `Mjlab-Velocity-Flat-Unitree-Go1`, PPO from rsl-rl, 4096 environments, 10,000 iterations, finished in 1 h 13 min with no errors.

| Iteration | Time | Reward | Episode length | Velocity error (xy / yaw) |
|---|---|---|---|---|
| 100 | 0:51 | 54.3 | 950 | 1.43 / 0.48 |
| 1000 | 7:43 | 92.2 | 1000 | 0.38 / 0.25 |
| 4950 | 36:43 | 94.4 | 1000 | 0.36 / 0.23 |
| 5050 | 37:27 | 83.5 | 992 | 0.49 / 0.27 |
| 9999 | 1:13:42 | 84.0 | 1000 | 0.44 / 0.28 |

- **It stops falling early.** Episodes last the full 1000 steps from about iteration 1000 on.
- **Most of the learning happens in the first 8 minutes.**
- **The dip at iteration 5000 is the curriculum, not a regression.** The commanded speeds widen there to −1.5…2.0 m/s forward and ±0.7 rad/s yaw.
- **Read the velocity errors as relative.** mjlab accumulates them per command period, so compare them between runs and don't read them as m/s.

Where the results are:

- **Run directory:** `logs/rsl_rl/go1_velocity/2026-11-10_17-04-19/`
- **Final checkpoint:** `model_9999.pt` in that directory, with one saved every 50 iterations.
- **Exported policy:** `2026-11-10_17-04-19.onnx`
- **Video of the final policy:** `videos/play/rl-video-step-0.mp4` (20 s, 1280×720)
- **Training videos:** 120 clips in `videos/train/`
- **W&B:** run `27yeeuu4` in the project `mjlab`

Commands, for reproducing it:

```sh
# Train. The grep drops MuJoCo Warp's solver-iteration warnings, which flood the log.
PYTHONUNBUFFERED=1 uv run train Mjlab-Velocity-Flat-Unitree-Go1 \
  --env.scene.num-envs 4096 --video True \
  --env.viewer.width 1280 --env.viewer.height 720 2>&1 \
  | grep --line-buffered -vE "^(solver|linesearch) iterations limit reached|OverflowType\.(LS_)?ITERATIONS" \
  | tee logs/go1_flat_train.log

# Record the final policy. It runs until interrupted, so stop it with Ctrl-C once
# the video is written.
uv run play Mjlab-Velocity-Flat-Unitree-Go1 \
  --checkpoint-file logs/rsl_rl/go1_velocity/2026-11-10_17-04-19/model_9999.pt \
  --num-envs 1 --video True --video-length 1000 \
  --video-width 1280 --video-height 720
```

Boolean flags need an explicit value in mjlab's command line: `--video True`, not a bare `--video`.
