# Progress

## Milestones

| # | Milestone (from [plan.md](plan.md)) | Status | Notes |
|---|---|---|---|
| 1 | Baseline: train Go1 velocity tracking | Done | The flat-terrain run from 2026-11-10 tracks velocity commands. It is an undamaged policy without history, so it is a reference point, not row 1 of the experiments table. |
| 2 | Damage injection | Started | Mid-episode motor faults for training (`mdp.motor_fault`, 2026-11-24). No calf cut yet. |
| 3 | Meta-RL policy | Started | Preliminary P1, P2 and a fault-score policy, 3,000 iterations each (2026-11-24). |
| 4 | Planner | Started | MPPI, the planar model and its least-squares fit exist and run on the undamaged policy in two demos (2026-11-11). The online residual model exists, is tested on made-up data, and runs in the race loop against a steady pull on the trunk (2026-11-20). Still missing: a safe default for its features, and a path task with damage. |
| 5 | Evaluation and demo | Started | The main demo is now the box jump ([ideas.md](ideas.md), entry 6.3). Its scene is built (2026-11-24); there is no jump skill yet. |

Still to set: the deadline and checkpoint dates.

## Setup

- **Training PC:** one RTX 5080, reached over SSH. Rendering works headless, so videos can be recorded without a display.
- **Code:** the fork `fefei69/mjlab` is the remote `origin`, and `mujocolab/mjlab` is `upstream`. Project work goes on the branch `dev`, which is local only so far.
- **Version:** commit `936f7a1`, which is mjlab v1.6.0 plus 60 commits, with MuJoCo and MuJoCo Warp 3.16.
- **Training budget:** 10,000 PPO iterations with 4096 environments take 1 h 14 min, so several runs fit in a day.
- **Results:** everything under `logs/` is ignored by git and exists only on the training PC.
- **Shared baseline:** the final baseline checkpoint and its fitted planar model are also in the repository, in `mppi/checkpoints/go1_flat_baseline/`, and the demos load them from there by default.
- **Running tests on the training PC:** its shell puts ROS on Python's path, and pytest then fails before it starts. Run tests as `env -u PYTHONPATH uv run pytest mppi/tests`. `make check` needs no such change.

## Log

### 2026-11-24

**Change of direction (cpw):** the main demo is now a run-up along a long box and a jump onto a smaller box, with friction and gravity changing while the robot runs, so it has to adapt fast enough to land on the box. Motor faults become a bonus task. Demo numbers (box sizes, gap, how far and when gravity and friction change) are not set yet.

**Can the current framework do it? No.** Checked against the code:

- **No jump skill.** Every policy only tracks a planar velocity command on flat ground (baseline up to 3 m/s forward, P1 and P2 up to 1 m/s).
- **The planner can't predict a jump.** The planar model has no height and no flight phase.
- **The flat-task policies see no terrain.** The rough Go1 task already has a 1.6 x 1.0 m height scan, and mjlab has box terrains.
- **Gravity can't vary per robot yet.** `Simulation.expand_model_fields` only handles top-level model fields, and gravity lives in `model.opt`. MuJoCo Warp itself allows per-world gravity (`opt.gravity` is a batched array), so this is a small patch.
- **The policy sees only gravity's direction.** Its gravity input is a unit vector, so g's size is visible only through how the body moves.
- **Friction is already randomized per robot** (foot friction 0.3 to 1.5, at startup). Changing it mid-run is a step event like `motor_fault`.

What carries over: the MPPI core, the batched online residual, history-based training (as in P2), step events for mid-run changes, and batched evaluation.

**Scene built** ([ideas.md](ideas.md), entry 6.3, with the numbers and why):

- The course is a terrain tile (`src/mjlab/tasks/box_jump/course.py`): a 6 m run-up box, a 0.5 m gap and a 1 x 1 m landing box, all 0.4 m high.
- **Per-robot gravity:** mjlab can now expand `opt.gravity`.
- **`physics_change` step event:** changes gravity and foot friction once, when the robot passes a line.
- **Preview** (`uv run python -m mppi.demo_box_jump`, video `logs/mppi/box_jump_scene.mp4`): the velocity policy runs at the gap at 2.5 m/s, then hits the face of the landing box and drops into the gap.
- **Tests:** `tests/test_box_jump.py`.

All action items for the planner side, plus preliminary versions of S2, S4, P1 and P2 so they could be tested. Details in [ideas.md](ideas.md), entry 3.11, "Sixth" to "Eighth step", and entry 5.7. Committed in `32ac2d2ba` and `2e95e0b3f`; the fault-trained checkpoints are not in git. Videos of three fault cases are in `logs/mppi/fault_r3.mp4`, `fault_thigh.mp4` and `fault_knee.mp4` (training PC only).

- **S2, mid-run motor fault** (`mdp.motor_fault`): one joint per episode, torque scale 0 to 0.5, 80% of episodes, start between 1 and 12 s, undone on reset. The critic sees every joint's torque scale. Preliminary S4 in `FAULT_TRAIN_PARAMS`; the front left knee is held out.
- **P1, P2 and a fault-score policy**, 3,000 iterations each, in `mppi/checkpoints/go1_fault*/`. Under random commands up to 1 m/s, after a dead knee: baseline 82% fall, P1 34%, P2 16%. They are trained only up to 1 m/s (the curriculum widens at 5,000 iterations).
- **R5, per-joint fault score** (`mdp.joint_fault_score`, `mppi/fit_joint_model.py`): a motor below 10% torque is flagged in a median 0.54 s, on the right joint 89% of the time, with 5% of healthy robots raising a false alarm in 20 s.
- **R3, slow down once a joint is flagged** (`demo_circle_race --fault-limits`): full speed when healthy (no false alarms), and after a weak motor as safe as always-slow (0 of 9 falls, against 6 of 9 always-fast). No help with a dead motor on the baseline policy.
- **R1:** the planar residual runs for many robots at once (`num_envs`), and features can be learned offline (`mppi/learn_features.py`); they are 5 to 10% better than the hand-chosen ones on familiar conditions and no better on the held-out joint.
- **The residual's test on fault-trained policies:** where the robot survives with a changed gait, the residual halves the model error (P2, held-out knee, 0.19 to 0.09 m/s), but with three races per cell no difference in falls or progress is reliable.
- **Bug fixed:** the tracker shared motor parameters across robots, so a fault set on one robot changed all of them. Earlier single-robot results are unaffected.

### 2026-11-23

- **Residual: gains now drift slower than offsets** (`ResidualCfg.gain_drift_ratio`, default 1/3, option `--gain-drift-ratio`). On the 30 N pull test the 12-coefficient residual went from 2 falls in 10 runs to 0, off the track 0% (0 to 2%) during the pull, progress 1.83 m/s. Cost: 11 to 24% of steps over the inside edge before the pull, against 0 to 8%. Details and the full table in [ideas.md](ideas.md), entry 3.11, "Third step".
- **Residual plot over time** (`mppi/plot_residual.py`, R2): the correction follows the nominal model's error on all three axes after the pull.
- **First motor faults** (`--fault`, ideas.md entry 3.11, "Fourth step"): faults that change the planar motion also break the gait of the current policy. The 12-coefficient residual cuts the model error by 40 to 60% but most runs still fall.
- **Same faults inside the trained command range** ("Fifth step"): with the knee at 20% no run falls, so speed caused most falls. Where the robot survives, the fault barely changes the planar motion, so the residual has little to fix. Its real test needs a policy trained on faults (P1, P2).

### 2026-11-21

- Wrote the course proposal, [proposal.tex](proposal.tex): repository link, description, tools, algorithms, planned work with experiments and expected results, and preliminary work. The limit is two pages.
- **Not compiled.** There is no LaTeX on the training PC. Its length is an estimate from font metrics: about 1.9 pages at 10 pt with 0.7 in margins. Compile it before submitting.
- **Checked:** every number against the raw run logs and the code, and all seven references against the papers.
- **Still to do before submitting:** put the team members' names in the title block, and commit and push the residual code and these notes, because the proposal links to the branch `dev` and describes work that is not on it yet.
- **Stated as the plan in the proposal, still open in these notes:** motor faults on one leg as the damage ([tasks.md](tasks.md), section 2), and a recovery time of about 2 s as the expected result for the policy with history, taken from "a second or two" in [ideas.md](ideas.md), entry 5.6.
- Drew the system as a LaTeX figure, [system_figure.tex](system_figure.tex): the control chain from task cost to robot, the planner's motion model with its residual and Kalman filter, the feedback paths, and the damage. Shaded boxes are the parts that adapt online. Unlike [system.png](system.png) it does not mark what is built and what is planned.
- **The figure is not compiled either,** for the same reason as the proposal. Its layout was checked with a preview drawn from the same coordinates (`logs/mppi/system_figure_preview.png`, on the training PC only), which shows no overlaps.
- **The figure is now inside the proposal,** inline, so the proposal stays one file to paste into Overleaf. To make room, the title went to one line and the text lost what the figure shows (the rates and command components, where the planner's command goes) plus a few details: the heading term of the cost, the planning-time expected result (still under preliminary work), and the description of the joint-level residual. Estimated length with the figure: 1.91 pages. Still not compiled here.
- **New fact from checking the proposal:** on the healthy robot, before the pull, the 3- and 6-coefficient residuals put the robot over the inside edge of the track in 20 to 30% of the plan steps, by up to 7.5 cm (ten runs each). The fixed model never does. The 12-coefficient residual does so in 0 to 7.5% of the steps.

### 2026-11-20

- Wrote [roles.md](roles.md): five roles, one per person. cpw takes the method role, which is the design of the planner's residual dynamics model. The other four are open.
- No work on the items since 2026-11-12. The task list had not been pushed.
- Put the baseline checkpoint (4.6 MB) and its planar model into the repository, in `mppi/checkpoints/go1_flat_baseline/`, so teammates can run the demos straight after a clone.
- Added item S4, the randomization the policies train on and the held-out set, to the Robot changes role.
- cpw's role now covers a residual model at two levels. The planner's residual comes first. A joint-level residual given to the walking policy as an input comes second, as an exploration ([ideas.md](ideas.md), entries 3.11 and 5.7). Whether this changes the four-way comparison is not decided.
- Sent the roles message to the team. No role picks recorded yet.
- Looked up where the residual idea stands in the literature ([related-work.md](related-work.md), "Online residual models"). cpw chose Neural-Fly as the design to follow.
- Built the first step of the planner's residual model (item R1), described below. Not committed yet.
- Put the residual into the race loop and tested it against a steady pull on the trunk (item R2, first version), described below. Not committed yet.

**Online residual model, first step.** The fitted planar model stays fixed and a residual on top of it is fitted during the run, by Neural-Fly's update: a Kalman filter on a few coefficients, with a leak toward zero. The features are a replaceable part, hand-chosen for now. This step checked it on made-up data only, where the true change is known.

| File | What it does |
|---|---|
| `mppi/residual_model.py` | The residual model, its online update, and three hand-chosen feature sets (3, 6 and 12 coefficients). |
| `mppi/residual_check.py` | Makes up runs with a known change, measures how fast and how noisily the fit follows it, and draws the figure. |
| `mppi/tests/test_residual_model.py` | Five tests, about 3 s. |
| `mppi/planar_model.py` | Changed: the rollout loop is now a shared function, so the corrected model can be rolled out like the fitted one. The race demo runs as before (lap 5.5 s). |

Results with the default settings, 8 runs per number, 12-coefficient features:

| Commands | Change | Model error removed in the first second | In the second | Later |
|---|---|---|---|---|
| Varied | 30% weaker | 51% | 77% | 75% |
| Varied | Steady drift | 58% | 80% | 85% |
| Varied | Veer growing with speed | 54% | 72% | 74% |
| Race-like | 30% weaker | 64% | 88% | 89% |
| Race-like | Steady drift | 63% | 85% | 85% |
| Race-like | Veer growing with speed | 64% | 89% | 90% |

- While the robot is healthy the residual wanders by about a quarter of the nominal model's usual error.
- The target of "about a second" is not met: it is closer to two. The full findings and what stays open are in [ideas.md](ideas.md), entry 3.11.
- Figure: `logs/mppi/residual_check.png`, on the training PC only.

```sh
uv run python -m mppi.residual_check
env -u PYTHONPATH uv run pytest mppi/tests
```

**Residual in the race loop, against a pull on the trunk.** The race demo can now pull on the trunk with a steady force (`--pull`) and plan with the corrected model (`--residual`). The test: healthy for 8 s, then 30 N to the robot's right for 22 s. Ten runs per row, because runs with the same seed differ.

| Planner's model | Falls | Time off the track during the pull | Progress along the track, 15 to 30 s |
|---|---|---|---|
| Fixed | 0 of 10 | 90% (87 to 92%) | 1.64 m/s |
| Corrected, offset per axis | 0 of 10 | 0% (0 to 4%) | 1.73 m/s |
| Corrected, gain and offset per axis | 0 of 10 | 0% (0 to 3%) | 1.73 m/s |
| Corrected, 12 coefficients | 2 of 10 | 0% (0 to 20%) in the runs that stayed up | 2.00 m/s in the runs that stayed up |

- The correction keeps the robot on the track. The fixed model lets it be pulled 0.14 m off the track and stay there.
- The 12 coefficients are not safe yet: when the pull starts the planner brakes hard and swings its commands, and the robot fell in 2 of 10 runs. The offset alone is smooth. The likely cause and a scratch test of a fix are in [ideas.md](ideas.md), entry 3.11, "Second step".
- On the healthy robot the correction makes the commands jumpier, and with the offset alone the robot is over the inside edge 28% of the time, by up to 6 cm.
- Changed files: `mppi/demo_circle_race.py` (the two options, three more readouts in the video panel, and a fix to a lap counted at the start), `mppi/tracker.py` (`pull_trunk`), `mppi/residual_model.py` (the list of feature sets moved here).
- Videos, on the training PC only: `logs/mppi/pull_compare.mp4` (fixed model beside the offset correction), `logs/mppi/pull_fixed.mp4`, `logs/mppi/pull_corrected.mp4`, and `logs/mppi/pull_command_fall.mp4` (a fall with the 12 coefficients).

```sh
uv run python -m mppi.demo_circle_race --pull 0 -30 --out logs/mppi/pull_fixed.mp4
uv run python -m mppi.demo_circle_race --pull 0 -30 --residual offset --out logs/mppi/pull_corrected.mp4
ffmpeg -i logs/mppi/pull_fixed.mp4 -i logs/mppi/pull_corrected.mp4 \
  -filter_complex "[0:v]pad=iw+8:ih:0:0:black[l];[l][1:v]hstack" logs/mppi/pull_compare.mp4
```

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
