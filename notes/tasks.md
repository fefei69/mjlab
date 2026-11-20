# Action items

Start with [roles.md](roles.md) and pick a role there. This file is the detail behind the roles: every item, what it involves and when it counts as done. If you want an item outside your role, put your name in its **Owner** column in section 4.

- **How to claim:** edit this file on the branch `dev`, write your name in the Owner column, and push. Do it early so two people don't start the same item.
- **Pairs are fine.** Two names on one item is allowed, and a good idea for the hard ones.
- **When you finish:** add a short entry with your result to [progress.md](progress.md) and mark the item done here.

The difficulty and size ratings are rough guesses, written before anyone has tried the tasks. Correct them when you know better.

## 1. Where the project stands

The goal is a simulated Go1 quadruped that keeps following its path when something about it changes mid-run, for example a motor dying. The system has two layers, and both are meant to adapt online. The picture is at the top of [ideas.md](ideas.md).

- **Planner:** MPPI plans velocity commands (forward, sideways, yaw rate) ten times a second. It predicts where the robot will go with a small planar motion model.
- **Tracker:** a PPO policy turns each velocity command into joint targets, fifty times a second.

What exists today, on the branch `dev`:

| Part | State |
|---|---|
| PPO velocity policy for the undamaged robot | Trained. It sees no history. |
| MPPI planner (`mppi/core.py`) | Built and working |
| Planar motion model, fitted offline by least squares (`mppi/planar_model.py`) | Built and working |
| Two demos: race around a circle, push a box through goals | Working, with videos |
| Anything that changes the robot mid-run | Not built |
| Policy with history | Not built |
| Planner that re-fits its model online | Not built, design in [ideas.md](ideas.md) entry 3.11 |

Results and the commands to reproduce them are in [progress.md](progress.md). The original plan is in [plan.md](plan.md).

### Before you start

- **Code:** clone the fork, check out `dev`, run `uv sync`. Always run things with `uv run`.
- **Trained policy:** the checkpoint is not in git, because training outputs are ignored. Ask cpw for it, or retrain it yourself in about 75 minutes with the command in [progress.md](progress.md).
- **GPU:** everything so far ran on one PC with an RTX 5080. Training needs a GPU like that. Ask cpw how to share that machine. The table below says which items need it.
- **Try it:** `uv run python -m mppi.demo_circle_race` runs the race and writes a video. If that works, your setup is fine.

## 2. Open decision: which change do we test on?

We have not decided what "something about the robot changes" means in the final experiments. This table lists the candidates. The column on planar motion matters most: the planner can only adapt to a change that alters how the robot moves over the ground.

| Change | What happens to the robot | Effect on planar motion (expected, not measured) | Work to build in mjlab |
|---|---|---|---|
| Weak or dead motor on one leg | It limps and veers to one side | Strong and one-sided | Low. Scale one joint's gains or torque limit (`dr.pd_gains`, `dr.effort_limits`). |
| Whole leg off (all three motors) | It walks on three legs | Strong and one-sided | Low. Same code as above. |
| Shortened lower leg (the "calf cut") | One leg no longer reaches the ground normally | Strong and one-sided | High. Geometry, contact, foot sensors and rewards all need changes ([ideas.md](ideas.md) entries 4.1 to 4.5). |
| Payload, or a steady pull on the trunk | It is heavier or dragged sideways | Moderate. Slower, and drifting if the load is off-centre. | Low as an external force (`apply_external_force_torque`). Real added mass is slower to simulate. |
| Low friction | The feet slip | Only at high speed or in turns | Low (`dr.geom_friction`). |
| Terrain: slopes or rough ground | It climbs or stumbles | Depends on the slope | High. Needs the rough-terrain policy, longer training and a planner cost that knows the terrain. |

**Proposal, for cpw to confirm:**

1. **Train and test on motor faults on one leg:** any leg, any joint, any severity from slightly weak to dead, including the whole leg off. Hold some combinations out of training.
2. **Also test on payload or pull and on low friction, without training on them.** They show whether the system copes with a kind of change it has never seen.
3. **Treat the calf cut as a stretch goal** for the final video.
4. **Leave terrain out.** It is a different project.

Reasons:

- **Both layers get work.** A fault on one leg changes the gait, which the policy must adapt to, and makes the robot veer, which the planner must correct.
- **Severity is a number.** That gives a clean split between changes seen in training and changes held out.
- **It is cheap to build first.** The low-effort changes share one piece of code.
- **It can be compared.** The closest prior work, LLA-MPPI, tests a leg with all motors off, a payload and friction ([related-work.md](related-work.md)).

Item S1 below measures the "expected" column, so the decision can be made on data.

## 3. The comparison we need at the end

Four controllers, in a two-by-two table. Each one switches one layer's adaptation on or off.

| | Planner with a fixed model | Planner that re-fits a residual online |
|---|---|---|
| **Policy without history** | 1. Baseline | 3. Residual only |
| **Policy with history** | 2. History only | 4. Ours |

What each comparison shows:

- **2 against 1:** what the policy's adaptation is worth.
- **3 against 1:** what the planner's adaptation is worth.
- **4 against 2 and 3:** whether the two add up, or whether one makes the other unnecessary.

Rules that keep it fair:

- **Same planner everywhere.** All four use the same MPPI, cost and command limits. Only the model differs.
- **Same training for both policies.** Both are trained on the same changes for the same number of steps. Only the history differs.
- **Same trials.** Every controller faces the same changes at the same times.

What to measure, over 50 trials per controller and change:

- **Recovery time:** seconds until velocity tracking is back within a set tolerance.
- **Path error** after the change.
- **Falls.**
- **Time to finish** the course.

Report changes seen in training and changes held out separately, with confidence intervals.

## 4. Action items

Difficulty is Easy, Medium or Hard. Size is S (a day or two), M (three to five days) or L (a week or more).

| ID | Item | Difficulty | Size | Skills you need | GPU | Needs first | Owner |
|---|---|---|---|---|---|---|---|
| S1 | Measure which changes matter | Easy to Medium | M | Python, PyTorch tensors, reading simulator code | Yes | Nothing | |
| S2 | Build the mid-run change for training | Medium | M | mjlab's event system, MuJoCo basics | Yes | Nothing for motor faults | |
| S3 | Build the calf cut (stretch) | Hard | L | MuJoCo geometry and contact, reward design | Yes | S2 | |
| T1 | Path task with turns in both directions | Easy to Medium | M | Python, plane geometry, cost design | Yes | Nothing | |
| P1 | Train the policy without history on the changes | Medium | M | Running PPO training, reading training curves | Yes | S2 | |
| P2 | Train the policy with history | Medium to Hard | M to L | PPO training, observation and network setup | Yes | S2 | |
| P3 | Show that the history policy really adapts | Medium | M | PyTorch, plotting, analysing a trained network | Yes | P2 | |
| P4 | Train on wider command ranges (optional) | Easy to Medium | S to M | PPO training setup | Yes | Nothing | |
| R1 | Online residual model | Medium | M | Linear algebra (least squares), PyTorch, pytest | No | Nothing | cpw |
| R2 | Put the residual into the planner loop | Medium | M | Python, MPPI basics, plotting | Yes | R1 | cpw |
| R3 | Command limits after a change | Hard, open question | M to L | Control intuition, designing experiments | Yes | R2 | |
| R4 | Oracle planner (optional) | Medium | M | Simulator state handling, GPU tensors | Yes | S2, T1 | |
| X1 | Evaluation script | Medium | M | Python, statistics, careful bookkeeping | Yes | Nothing to start | |
| X2 | Run the four-way comparison | Easy to Medium | M | Running experiments, plotting | Yes | P1, P2, R2, X1, T1 | |
| X3 | Final demo videos | Medium | M | Python, video tools, an eye for presentation | Yes | V2, and X2 for the final cut | |
| V1 | Figures and a shared visual style | Medium | M to L | Plotting in Python, visual design, reading logs | Little | Nothing to start | |
| V2 | Demo ideas and storyboard | Easy to Medium | S to M | Imagination, understanding the method, sketching | No | Nothing | |
| W1 | Related work and citations | Easy | S to M | Reading papers, writing | No | Nothing | |
| W2 | Tests for the planner code | Easy to Medium | S to M | pytest, PyTorch | Little | Nothing | |
| W3 | Report | Medium | L | Technical writing, building an argument | No | Nothing to start, X2 for the results | |
| W4 | Slides and talk | Medium | M | Slide design, presenting | No | V1, X3 | |

**Can start today, with nothing else finished:** S1, S2, T1, R1, X1, V1, V2, W1, W2, W3, P4.

**Left over if nobody picks them:** R3, R4, P4 and W2 are open to anyone.

**The chain that decides the schedule:** S2, then P1 and P2 (each a training run of over an hour, probably repeated), then X2. Start S2 early.

The items are grouped into five roles in [roles.md](roles.md):

- **Method, residual dynamics model (cpw):** R1, R2, and R3 if time allows.
- **Robot changes:** S1, S2, with S3 as a stretch.
- **Policy:** P1, P2, P3, with P4 as an option.
- **Evaluation:** T1, X1, X2, with W2 and R4 as options.
- **Story and demos:** V1, V2, X3, W1, W3, W4.

### The story and demos role

We want one person to own how the project is told and shown: the report, the slides, the figures and the demos. It is a full role that starts on day one, not a job for the last week.

- **It shapes the experiments.** The figures we want decide what the experiments must log, and the demo we want decides what the simulation must be able to show. Both need to be known early.
- **It needs the method, not the code.** You have to understand the system well enough to explain it, and be comfortable with Python for plots. You do not need to train policies.
- **It is hard to do well.** Rated Medium because the bar is a report and a talk that a stranger follows on first reading.

## 5. What each item involves

### S1. Measure which changes matter

Apply each candidate change from section 2 to the policy we already have, mid-run, and measure what happens. No training.

- **Do:** for each change and a few severities, run many robots in parallel, apply the change after a few seconds, and record falls, velocity tracking error, and how the fitted planar model shifts (its gains and offsets).
- **Done when:** there is a table with one row per change and severity, and a short recommendation for section 2.
- **Start from:** `mppi/sweep_commands.py` and `mppi/fit_planar_model.py`, which already run many robots under commands and fit the model. mjlab's `dr` functions change the simulated robot.
- **Why it's a good first item:** you learn the codebase and settle the open decision.

### S2. Build the mid-run change for training

A piece of the environment that applies a change at a random time in each episode and undoes it on reset.

- **Do:** write it as an mjlab event, which is a function the environment calls every step. Start with motor faults. Keep the change's parameters (which leg, which joint, how severe, whether it has happened yet) in a tensor that other code can read.
- **Done when:** a training run starts with it, a video shows the change happening mid-episode, and the parameters can be logged and fed to the critic.
- **Watch out for:** [ideas.md](ideas.md) entries 4.2, 4.6 and 4.7. In short, write it as a class with a `reset` hook, pick one change time per episode, and avoid changing body mass because that is slow.

### S3. Build the calf cut (stretch)

The original demo: one lower leg is cut short mid-run.

- **Do:** shrink and move the lower-leg and foot collision shapes, give the stump friction, move or mask the foot's sensors and reward terms, and show the cut in the video.
- **Done when:** the robot can be cut mid-run in simulation without the physics or the rewards breaking.
- **Why it's hard:** five separate things need care. They are listed in [ideas.md](ideas.md) entries 4.1 to 4.5.

### T1. Path task with turns in both directions

The race demo runs on a circle, which is a weak test: the commands are constant, so the planner's model barely matters ([ideas.md](ideas.md) entry 3.7).

- **Do:** generalise `mppi/demo_circle_race.py` from a circle to a path given as waypoints, such as a figure-eight or a slalom. Provide the cost for MPPI, the path error and progress measures, and the drawing for the video.
- **Done when:** the robot runs laps on a figure-eight with the existing policy and planner, and the script reports path error and lap time.

### P1. Train the policy without history on the changes

This is the policy for controllers 1 and 3.

- **Do:** train the existing policy setup with S2 switched on. Tell the critic the change's true parameters, but not the actor.
- **Done when:** there is a checkpoint, training curves, and a video of the policy coping with a change.
- **Watch out for:** rewards that punish the adapted gait. See [ideas.md](ideas.md) entries 4.4 and 4.5.

### P2. Train the policy with history

This is the policy for controllers 2 and 4, and half of what the project claims.

- **Do:** give the actor its recent observations and actions. Start with a stacked window of 0.5 to 1 s, which is one setting in mjlab (`history_length`). Keep everything else identical to P1.
- **Done when:** there is a checkpoint trained with the same budget as P1, and a first comparison against P1 on tracking after a change.
- **Optional:** a recurrent policy (GRU) as a second variant. mjlab's configuration already supports it.
- **Background:** [ideas.md](ideas.md) entry 5.6 explains why history is needed and what else is needed with it.

### P3. Show that the history policy really adapts

A policy can survive changes by being robust without adapting. We need evidence for adaptation.

- **Do:** plot tracking error against time since the change. Blank or shuffle the history at test time and measure the drop. Train a small probe that predicts the change's parameters from the policy's internal features.
- **Done when:** there are three figures, one per check, with a sentence each on what they show.

### P4. Train on wider command ranges (optional)

The current policy was trained up to 2.0 m/s forward and 0.7 rad/s yaw. It happens to work up to about 3.0 m/s and 1.5 rad/s, and falls beyond that ([ideas.md](ideas.md) entry 5.5).

- **Do:** train with wider command ranges so that fast commands are trained and not just lucky.
- **Done when:** `mppi/sweep_commands.py` shows the new policy reaching higher speeds without falling.

### R1. Online residual model

The planner's own adaptation. This item is pure math and code, with no simulator needed.

- **Do:** keep the offline model fixed and fit a small correction on top of it from recent motion, by least squares that forgets old samples and is pulled toward zero. Start with a gain and an offset per axis.
- **Done when:** tests on made-up data pass. A sudden change in gain or offset is recovered within about a second of samples, no change leaves the residual near zero, and constant commands do not make it blow up.
- **Start from:** `PlanarModel` in `mppi/planar_model.py` and the design in [ideas.md](ideas.md) entry 3.11.

### R2. Put the residual into the planner loop

- **Do:** re-fit the residual at every planning step during a run and let MPPI plan with it. Test it on the existing policy with simple changes, before P1 and P2 exist.
- **Done when:** a plot shows the residual jumping when a change is applied, and a comparison shows path error with the fixed model against the residual model.
- **Start from:** the loop in `mppi/demo_circle_race.py`. It already measures the two things the fit needs at every step: the command sent and the velocity that resulted.

### R3. Command limits after a change

An open question, more research than engineering.

- **The problem:** MPPI is allowed to command up to 3.0 m/s. A robot with a dead motor may fall at that speed. A lower gain in the residual tells MPPI the robot is slower, but not that a command has become unsafe ([ideas.md](ideas.md) entry 3.9).
- **Do:** find a rule that shrinks MPPI's command limits after a change, and test whether it prevents falls without costing much speed.
- **Done when:** there is a rule, and numbers for falls and lap time with and without it.

### R4. Oracle planner (optional)

An upper bound for the comparison: a planner that is told the true change.

- **Do:** plan by rolling candidate commands out in parallel copies of the simulator that have the true change applied, as the box-push demo does for the box.
- **Done when:** it runs on the path task and gives a fifth row for the results table.
- **Start from:** `mppi/demo_box_push.py` and `Tracker.rollout` in `mppi/tracker.py`.

### X1. Evaluation script

One script that produces the numbers for section 3.

- **Do:** run a given controller on a given change for many trials without rendering, and write recovery time, path error, falls and finish time to a file, with confidence intervals.
- **Done when:** it produces a results table for the existing policy and planner on at least one change.
- **Worth knowing:** the demos run one robot at a time. Fifty trials for every controller and change will take hours unless the script runs robots in parallel, which means running MPPI for many robots at once.

### X2. Run the four-way comparison

- **Do:** run X1 for the four controllers on the changes we decided on, and make the results table and the recovery plots.
- **Done when:** the table in section 3 is filled in, for changes seen in training and changes held out.

### X3. Final demo videos

Build the demos chosen in V2.

- **Do:** produce the headline clip, which shows the change happening and the robot recovering, and the supporting clips for the talk. Keep them short, and make each one understandable with the sound off and no narration.
- **Done when:** the clips are in the slides and the headline clip also works on its own.
- **Start from:** `mppi/viz.py`, which draws the planner's sampled plans and a text panel into the videos, and the two existing demo videos.

### V1. Figures and a shared visual style

Every figure in the report, the slides and the videos should look like it belongs to the same project.

- **Do first:** fix a style. Most important is one color per controller, used for the four controllers in every plot and every video. Also fonts, line widths and how uncertainty is drawn.
- **Then:** the result figures. We expect to need tracking error against time since the change with one line per controller, the four-way results with confidence intervals, the planner's residual over time around a change, and falls per controller.
- **And:** the method figures. The system figure exists (`system.svg`, shown in [ideas.md](ideas.md)). Still missing are a figure of the four-way comparison and one that shows a single trial on a timeline.
- **Done when:** there is a small plotting module that everyone uses, and each figure can be regenerated from saved results with one command.
- **Start now:** the existing results in [progress.md](progress.md) and the logs the demos write (`logs/mppi/*.npz`) are enough to design the style on real data. Tell X1 early what the figures need logged.

### V2. Demo ideas and storyboard

Decide what we show, before anyone builds it.

- **Do:** collect demo ideas, pick the ones that make the point fastest, and sketch each as a storyboard: what is on screen, what happens when, and what the viewer should conclude.
- **Done when:** the team has agreed on one headline demo and two or three supporting ones, each with a storyboard and a note on what must be built for it.

Ideas to start from, none of them built or tested:

- **The moment.** The robot runs a course, a motor dies, and the screen splits into the four controllers from that same instant, with a timer showing when each has recovered.
- **The planner's view.** The fan of sampled plans bends after the change as the model updates, next to a small gauge of the residual jumping.
- **Path traces from above.** One color per controller over the same course, showing where each leaves the path.
- **Escalation.** One continuous run with faults that get worse, from a weak motor to a whole leg off, until the system can no longer cope. It shows the limits honestly.
- **A change it has never seen.** A payload or a slippery floor that was not in training.
- **Repair.** The fault is undone mid-run and the system adapts back, which shows adaptation in both directions.
- **Box push with a damaged robot.** The existing box demo, with a fault applied on the way.
- **Break it live.** A button in the interactive viewer that kills a motor during the talk. This is the riskiest idea and needs a recorded backup.

### W1. Related work and citations

No coding.

- **Do:** read the papers listed in [related-work.md](related-work.md) under "To read" and "From the plan's references", and write for each what it does and how we differ. Check the citations for the formal framing in [ideas.md](ideas.md) entry 5.6, which were written from memory.
- **Done when:** [related-work.md](related-work.md) covers every paper we will cite.

### W2. Tests for the planner code

The code in `mppi/` has no automated tests yet. A good item for learning the code.

- **Do:** write pytest tests. Examples: MPPI finds the minimum of a simple cost and respects its limits, the model fit recovers a known model from made-up data, and copying simulator state between two environments gives identical motion.
- **Done when:** the tests run with `uv run pytest`.

### W3. Report

- **Do first:** an outline with the claim in one sentence, the figures each section needs, and what each experiment has to show. Share it with the team in the first week, so the experiments are built to fill it.
- **Then:** write the method and related-work sections while the experiments run, and the results once X2 is done. State the limits plainly, including what did not work.
- **Done when:** someone outside the team can read the report once and say what we did, what we found and how we know.
- **Start from:** the system figure and the framing in [ideas.md](ideas.md) (entries 1.1, 5.6 and 3.11) and the positioning in [related-work.md](related-work.md).
- **Deadline:** not set yet.

### W4. Slides and talk

- **Do:** build the talk around one claim and the headline demo. One idea per slide, the videos embedded and tested on the machine that will present, and a rehearsal with the team.
- **Done when:** the talk has been rehearsed once in full and fits its time slot.
