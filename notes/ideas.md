# Ideas and open questions

Each entry has a status: `open` (under discussion), `adopted` (agreed, and reflected in [plan.md](plan.md)) or `dropped` (with the reason). The date is when the entry was raised.

The entries so far came out of two reviews on 2026-11-11, the plan checked against mjlab's code and the LLA-MPPI paper (see [related-work.md](related-work.md)), and the discussions that followed. Entries 3.2, 3.4, 3.5, 3.9 and 3.10 were adopted that day, when the two MPPI demos were built (see [progress.md](progress.md)). Everything else is still open.

## System overview

![System overview: an MPPI planner sends velocity commands to an RL tracking policy, which drives a simulated Go1. Both layers adapt online.](system.png)

*The system has two layers, and each adapts online in its own way. The planner re-fits a residual in its motion model from how the robot actually moved (entry 3.11). The policy adapts its gait through its observation history (entry 5.6). Solid boxes exist today, dashed boxes are planned, and orange marks the parts that adapt. Drawn on 2026-11-12.*

The picture is `system.png`, rendered from `system.svg`. When a planned part is built, move its box from the dashed group to the solid group in `system.svg`, then render the PNG again from the repository root:

```sh
google-chrome --headless=new --no-sandbox --hide-scrollbars \
  --force-device-scale-factor=2 --window-size=980,670 \
  --screenshot="$PWD/notes/system.png" "file://$PWD/notes/system.svg"
```

## 1. What makes the project interesting

### 1.1 Frame it as adaptation at two levels, with no predefined damage models

**Status:** `open` · 2026-11-11

The first thought was that our MPPI planner also adapts online. On its own that doesn't set us apart. LLA-MPPI's planner adapts online too, by switching to whichever simulator in its bank fits best, and Nagabandi et al. adapt a learned model online for MPC. Three narrower differences do hold:

- **Fitting instead of selecting.** LLA-MPPI picks one of K damage models written down in advance. We re-fit gains and offsets continuously, so nothing limits us to damages someone listed. Our held-out damages test exactly this, and LLA-MPPI never runs that test.
- **Two layers adapt, not one.** Our policy adapts implicitly through its observation history, and our planner adapts explicitly through least squares. LLA-MPPI has a single adapting layer.
- **Cost.** We solve a least-squares problem with a few parameters each step. They run K full physics simulators in parallel.

Proposed one-line framing: "adaptation at two levels, with no predefined damage models, tested on damages unseen in training."

### 1.2 Risk: the two adapting layers compete

**Status:** `open` · 2026-11-11

If the history policy compensates well, velocity tracking stays accurate after the damage, the fitted motion model stays near identity, and the adaptive planner shows no gain over a fixed one. The planner's adaptation only matters for what the policy cannot fix, such as a lower top speed or a steady drift to one side on three legs.

This is reasoning, not a measured result. Entries 2.2 to 2.4 are there to make the planner's contribution visible if it exists.

## 2. Experiments

### 2.1 The four controllers to compare

**Status:** `adopted` · 2026-11-12

Decided by cpw on 2026-11-12: a two-by-two comparison, where each controller switches one layer's adaptation on or off.

| | Planner with a fixed model | Planner that re-fits a residual online |
|---|---|---|
| **Policy without history** | 1. Baseline | 3. Residual only |
| **Policy with history** | 2. History only | 4. Ours |

cpw named three of the four: ours, the policy with history alone, and the residual with a policy without history. The fourth, with neither, is filled in here as the plain baseline. The rules that keep the comparison fair and the measures are in [tasks.md](tasks.md), section 3.

This replaces the rows assumed on 2026-11-11, which used a simple path follower for the first two.

### 2.2 One pair of rows must isolate the planner's adaptation

**Status:** `open` · 2026-11-11

Use the same policy in both rows, with the planner on a frozen motion model in one and on the online fit in the other. The two-by-two comparison in 2.1 contains this pair twice, once for each policy.

### 2.3 Include damages severe enough to bias tracking

**Status:** `open` · 2026-11-11

The planner needs something left to correct after the policy has adapted (see 1.2). Damages that cap the top speed or cause a steady sideways drift are the likely candidates.

### 2.4 Plot the fitted gains and offsets over time around the chop

**Status:** `open` · 2026-11-11

This is direct evidence that the planner's model changed in response to the damage, separate from the task metrics.

### 2.5 Add a whole-leg-disabled damage

**Status:** `open` · 2026-11-11

Turn off all three motors of one leg. This matches LLA-MPPI's "leg lock" test, where they succeed in 46 of 50 trials, so at least one of our results can sit beside theirs.

### 2.6 Borrow LLA-MPPI's evaluation protocol

**Status:** `open` · 2026-11-11

- 50 randomized trials per condition.
- Success rate and fall rate with 95% confidence intervals.
- Time to goal.
- Controller update time. Their whole-body planner takes 14–18 ms per update. A policy plus MPPI over three velocity commands should be much cheaper, which would be a fair selling point, but we haven't measured ours.

### 2.7 Row 1 should also train across damages

**Status:** `open` · 2026-11-11

The baseline run from 2026-11-10 is an undamaged policy without history. Row 1 should be trained across the same damages as row 2, so that history is the only difference between them.

### 2.8 Which change to test on

**Status:** `open` · 2026-11-12

Not decided: motor faults, a shortened leg, payload, friction, terrain, or several of them. The candidates, a proposal and the reasons are in [tasks.md](tasks.md), section 2. The proposal is to train and test on motor faults on one leg, to test on payload and friction without training on them, to keep the calf cut as a stretch goal, and to leave terrain out. Item S1 in the task list measures how much each candidate changes the robot's planar motion, so the choice can be made on data.

## 3. Planner

### 3.1 A gain and an offset can't be separated under a constant command

**Status:** `open` · 2026-11-11

With a nearly constant command, least squares over the last second can't tell a gain from an offset, because both explain the same velocity error. Two options:

- Regularize the fit toward the previous estimate.
- Fit only the offset when the command barely changes.

This is reasoning, not something we have tested.

### 3.2 The planner command term needs the stock behavior turned off

**Status:** `adopted` · 2026-11-11

Built in `mppi/tracker.py`. It keeps mjlab's command term, switches off resampling, heading, standing and forward-only modes, and writes the command tensor directly. It also recomputes the observation after writing a command, so the policy acts on a new command in the same step, without the 20 ms delay described below.

mjlab's stock velocity command puts 30% of environments in heading mode and 10% in standing mode, and resamples every 3–8 s. For the planner, turn those off or subclass the term, then write the command tensor directly.

The policy acts on a new command one step (20 ms) later, because observations are computed at the end of each step. Checked against the code.

### 3.3 Optional: a simulator bank as a damage identifier

**Status:** `open` · 2026-11-11

LLA-MPPI's model selection would fit mjlab: a second simulation with K parallel worlds, one damage hypothesis each. Each step, write the measured state into every world, apply the same joint targets, step once, and compare the predictions with what the robot did.

- **What it gives:** an explicit damage estimate (which leg, how much) to compare with what our policy infers implicitly.
- **What it doesn't give:** the planar motion model our MPPI uses. To drive the planner we would still need a planar model per hypothesis, for example fitted offline from the policy's rollouts with that damage.
- **Starting values from the paper:** a window of 10 steps, a switching margin of 0.5, 3 wins in a row, errors measured on joint positions. Their ablation found the error is U-shaped in window length, best around 20–25 steps.

Use this only as a model-selection component. Their full controller is not a practical baseline: the code isn't released, they use a Go2, and reimplementing whole-body MPPI would be a project of its own.

### 3.4 Build the planner on a racing task first, not box pushing

**Status:** `adopted` · 2026-11-11, with a change: both were built

Decision: build two demos on the undamaged policy, a race around a circle and a box push, with MPPI planning velocity commands and the RL policy tracking them. Both exist and work; results are in [progress.md](progress.md). The reasoning below is kept as written before the demos. Two of its untested points are now measured:

- **Fast-changing commands are fine.** With commands changing every 0.2 to 3 s, none of 512 robots fell in 60 s.
- **The response lag is real.** The fitted model has a time constant of about 0.3 s for linear velocity and 0.1 s for yaw rate. Including it halves the prediction error.

The question: as a first step, run MPPI over velocity commands with the already-trained baseline policy as the tracker, on either box pushing or racing.

Claude's recommendation is to do it, and to pick racing, meaning a timed lap around a track of waypoints.

- **Racing reuses everything.** MPPI's model is the planar motion model the plan already needs. The command term, the least-squares fit and the evaluation all carry over to the damage experiments. The only new asset is a track.
- **Racing gives the planner's model something to do before any damage.** Near top speed and in tight turns the policy can't deliver what is commanded, so the fitted gains and offsets move away from identity. That eases the risk in 1.2. After damage the limits drop and the planner has to slow down for corners, which makes a clear demo.
- **Box pushing can't use the planar model.** MPPI has to predict the box's motion under contact. The practical way is to roll out in the simulator itself (entry 3.5). That makes box pushing feasible, but it skips the fitted model, which is the project's own contribution. The baseline policy was also never trained with sustained contact, only with brief velocity pushes. Simulator rollouts would at least let MPPI see when the policy stumbles.

Things to watch with the baseline policy:

- **Stay inside the trained command ranges:** forward −1.5 to 2.0 m/s, sideways ±1.0 m/s, yaw ±0.7 rad/s. Checked in the config.
- **Smooth the commands.** In training a command changed only every 3–8 s. MPPI re-plans at 10–20 Hz, so penalize command changes or sample smooth noise. How the policy reacts to fast-changing commands is untested.
- **The model probably needs a response lag.** The robot's velocity doesn't jump to the command. A gain and an offset alone ignore that, and racing is where it matters most. This is reasoning, not a measurement.
- **A cheap first experiment:** drive the baseline policy with step and sweep commands, log commanded against achieved velocity, and see how well a planar model predicts 1–2 s ahead. That tests the plan's "planar model is too simple" risk before any MPPI code is written.

### 3.5 Simulator rollouts as MPPI's model

**Status:** `adopted` · 2026-11-11, for the box-push demo

Instead of a planar model, MPPI can roll out in the simulator: each sample is one parallel world, with the policy running inside it. This needs no hand-written model, so it works for tasks with contact such as box pushing.

- **What it takes in mjlab:** a second environment used only for planning. Before each plan, copy the real world's state into every planning world: the robot's root and joint state, the box's state, and the policy's last action, which is one of its observations. Built in `mppi/tracker.py` and measured: after a copy, a planning world stays within 1e-5 of the real one for 2 s.
- **A trap in the copy:** mjlab caches the last observation. After copying state, the observation must be recomputed, or the policy acts on stale data and the planning worlds diverge within one step.
- **Speed (measured):** a plan with 384 worlds and a 2 s horizon, which is 100 policy steps, takes 0.77 s. The steps run one after another, so the horizon sets the cost more than the number of worlds. With one plan per 0.2 s of motion, the demo runs about 4 times slower than real time. That is fine in simulation, because the real world is frozen while planning.
- **The planner's adapting part drops out.** The plan's adaptation is the least-squares fit of the planar model. Simulator rollouts don't use the planar model, and the planning worlds have the same physics as the real one, so there is no mismatch to fit. After damage, the planning worlds must be told the damage, which makes this the plan's optional oracle planner. If the damage has to be identified instead, it becomes LLA-MPPI's approach (entry 3.3).
- **Two ways to put adaptation back here, if we want it.** Both are reasoning, untested.
  - Fit the simulator's own parameters online, such as the box's mass or the damage. This is not a linear least-squares problem, and it is the ground LLA-MPPI already covers.
  - Keep the planning worlds nominal and fit a small correction to their predicted base motion by least squares. This stays cheap, but the correction is applied after the rollout, so it is not consistent with contacts such as the box.
- **Shared code:** write MPPI once with a swappable rollout function. The sampling and weighting are the same for the simulator rollout and for the planar model, so neither is wasted work.

### 3.6 Test the planner's adaptation early with simple dynamics shifts

**Status:** `open` · 2026-11-11

The least-squares fit can be exercised with the baseline policy, before the chop event or any retraining exists. Use the planar model on a path or racing task, and apply a shift mid-run that mjlab can already make per world: add payload mass, lower the ground friction, or scale down one leg's PD gains. The policy's tracking should degrade and the fitted gains and offsets should move.

This shows early whether the planner's adaptation works at all, and how quickly it reacts. That mjlab can vary these three per world was checked in the code. The experiment itself is untested.

### 3.7 Race on a track with straights and direction changes, not a circle

**Status:** `open` · 2026-11-11

The circle demo showed two limits of a circle as a test (measured, see [progress.md](progress.md)):

- **The lap time was set by the command limits, not by the planner.** In the first version MPPI was limited to the trained range, 0.7 rad/s of yaw, which fixed the lap at about 9.5 s. This entry first called that the policy's limit. That was wrong: the policy turns at up to about 1.9 rad/s when asked (entry 3.9).
- **A circle mostly hides the value of the fitted model.** In steady state the commands are constant, so the response lag doesn't matter. Inside the trained range, the fitted model and the naive one reached the same speed, and the fitted one only stayed closer to the centre line (4 cm against 11 cm). At the wider limits the difference is larger: the naive model is slower and leaves the track.

A figure-eight or a slalom has direction changes, where the lag matters, and straights, where the speed limit matters. It would be a better test for the planner, and later for the online re-fit.

### 3.8 The planner runs sideways when allowed

**Status:** `open` · 2026-11-11

With no cost on heading, MPPI ran the circle at 43 degrees to the track, using the policy's sideways velocity. It was no faster on the circle (same lap time). The demo penalizes it for a nose-first racing line. Worth remembering for the damage experiments: after a chop, moving sideways or at an angle may be the better gait, so a heading cost could hide a useful adaptation.

### 3.9 Take MPPI's command limits from what the policy can do, not from its training ranges

**Status:** `adopted` · 2026-11-11, for the circle race

The policy tracks commands far outside the ranges it was trained on: yaw up to about 1.9 rad/s against 0.7 trained, and forward speed up to about 2.8 m/s against 2.0 trained. Measured with `mppi/sweep_commands.py`. Widening MPPI's limits to forward 3.0 m/s and yaw 1.5 rad/s cut the lap time from 9.6 s to 5.5 s with no falls. Above that the robot starts to fall. Numbers are in [progress.md](progress.md).

This matters for the damage experiments. A damaged robot can do less, so limits that were safe before the chop may make it fall afterwards. The planner then needs limits that shrink with the damage, either from a sweep per damage or from the online fit.

### 3.10 Speed-dependent terms in the planar model

**Status:** `adopted` · 2026-11-11

The plan lists "add speed-dependent terms" as the fallback if the planar model is too simple. They are now in: nine products of each command with the size of each command, fitted in the same least-squares solve.

- **On average they change little:** the error 2 s ahead drops from 0.158 m to 0.151 m.
- **At the limit they matter:** for a command of 3.0 m/s with 1.4 rad/s, the linear fit overstates the yaw rate by 0.20 rad/s and the fit with the terms by 0.11 rad/s.
- **In the race they make the planner consistent:** with the terms, all six planner settings tried lapped in 5.5 to 5.6 s. Without them, one of the six settled into a slower sideways gait at 6.1 s.

Even with the terms the model is still optimistic at the limit. Terms that depend on the current velocity, not only the command, are a possible next step.

### 3.11 How the planner would adapt online

**Status:** `adopted` · raised 2026-11-12, design agreed and first step built 2026-11-20

The residual model and its online fit exist and pass their tests on made-up data (see "First step" at the end of this entry). They are not in the planner's loop yet, so in the demos the planar model is still fitted once, offline, and stays fixed during a run. The paragraphs below are kept as written at the time, and later ones revise earlier ones.

The MPPI algorithm itself would not change. Only the model it plans with would: its parameters would be re-fitted during the run from the robot's own recent motion. Each plan step already produces one sample for that fit, namely the velocity over the last step, the command that was sent, and the velocity that resulted. The steps would be:

1. Keep the samples from the last second or so.
2. Re-solve the least-squares fit on them, every plan step. It is a tiny problem.
3. Hand the new parameters to MPPI's rollouts for the next plan.

Proposed form, raised on 2026-11-12: a residual model. The offline model stays fixed as the nominal model, and only a small correction on top of it is re-fitted online:

  v_next = nominal(v, u) + residual(v, u)

- **What is fitted:** the nominal model's recent prediction errors, so the target is "what the robot did minus what the nominal model said". With no damage the errors are near zero and so is the residual.
- **Size of the residual:** start with a gain and an offset per axis (6 numbers), as in the plan. A fuller linear residual in the velocity and command is the next step up.
- **How it is fitted:** least squares that forgets old samples gradually and is pulled toward zero. The pull toward zero means the residual only moves when the data shows something, which also covers the cases of too little data and constant commands below.
- **What it gives for free:** the size of the residual is a damage signal. It can be plotted around the chop (entry 2.4).
- **Its limit:** the residual is fitted on what the robot is doing now, so it is only reliable near the current commands. MPPI could plan toward commands where it is wrong. Keeping the residual small in parameter count limits this.

This is close in shape to DATT in the plan's references, which pairs an RL policy with an online adaptive estimator. A residual only on the planar motion should be enough here, because the policy absorbs the change in how the legs must move. LLA-MPPI's disturbance-observer baseline failed on leg lock (0 of 50), but there the residual had to explain whole-body dynamics with no adapting policy underneath.

Three problems to design around, none tested yet:

- **Too little data.** One second is 10 samples, and the full model has 16 parameters per velocity component. Fit only a few parameters online, such as a gain and an offset per axis on top of the offline model, which is what the plan says. Alternatively, pull the fit toward the offline model so that the data only changes what it actually shows.
- **Constant commands** (entry 3.1). With no variation in the command, a gain and an offset cannot be told apart. Pulling toward the offline model handles this too: what the data does not show stays at its old value.
- **Speed against noise.** The gait makes the measured velocity oscillate, so a very short window fits noise. A window of about a second, or a fit that forgets old samples gradually, is the trade-off.

What it should look like at a chop: the model's predictions suddenly miss, and within about a second the fitted gain drops and an offset appears, for example a steady drift to one side. MPPI's predicted paths then bend the way the damaged robot really moves, so it steers against the drift and slows for turns.

The command limits are a separate problem (entry 3.9). A lower gain tells MPPI the robot is slower, but not that a command has become unsafe.

Entry 3.6 describes how to test this before the chop event exists.

Development plan, agreed with cpw on 2026-11-20. cpw owns this part and develops it on a small set of changes of their own, without waiting for the final set.

Choose the changes by their effect on planar motion, so that each thing the residual must do is exercised:

| Effect on planar motion | Example change | What the residual should do |
|---|---|---|
| The robot delivers less of what is commanded | All motors weaker, or a backward drag on the trunk | Lower its gain |
| The robot drifts steadily | One leg's motor weak, or a sideways pull on the trunk | Grow an offset |
| Nothing visible in planar motion | A small friction change at low speed | Stay near zero |

The third row checks that the residual does not react to noise. Undoing a change mid-run checks that it adapts back.

Order of work:

1. **Made-up data.** Change a gain or an offset of the nominal model by a known amount, add noise, and check that the fit recovers it and how fast. Only here is the true answer known.
2. **A steady force on the trunk** in the simulator. mjlab can apply an external force to a body, so this needs no new environment code. Tried on 2026-11-20, see "Second step" below.
3. **Motor faults,** once item S2 in [tasks.md](tasks.md) provides them, so there is one implementation of the changes and not two.

Keep the development set apart from the test set. A few numbers get tuned on these changes: how fast the fit forgets, how strongly it is pulled toward zero, and which parameters it fits. Tuning them on the changes used in the final comparison would give this controller an unfair advantage over the other three, so the held-out set from item S4 stays unseen during development.

The residual is fitted during the run and not trained, so this is a development set of changes, not domain randomization in the training sense.

Design agreed with cpw on 2026-11-20: follow Neural-Fly ([related-work.md](related-work.md)), with hand-chosen features as the baseline row.

Neural-Fly writes the residual as learned features times a few coefficients, `residual = Φ(v, u) · a`. A small network Φ is trained offline and then frozen. Only the coefficients `a` are fitted during the run. In the paper Φ has 4 outputs shared by the three force axes, so 12 coefficients adapt online, the same count as our larger hand-chosen option.

What this would change for us:

- **The online fit stays the same kind of thing.** The paper's update is a Kalman filter on `a` with a leak toward zero. Per plan step, with `e` the nominal model's error, and without the paper's extra tracking-error term, which needs a tracking controller that we do not have at this level:

  ```
  a⁻ = (1 − λ dt) a                      # leak toward zero
  P⁻ = (1 − λ dt)² P + Q dt              # stay ready to change
  K  = P⁻ Φᵀ (Φ P⁻ Φᵀ + R dt)⁻¹
  a  = a⁻ − K (Φ a⁻ − e)                 # correct by the prediction error
  P  = (I − K Φ) P⁻ (I − K Φ)ᵀ + K R dt Kᵀ
  ```

  The three knobs are the leak `λ`, how fast `a` may move `Q`, and the measurement noise `R`. This replaces the running-sums form sketched earlier and does the same job. The paper proves convergence without requiring varied inputs, which is our constant-commands problem.
- **A new offline stage.** Φ is trained on motion data from several different changes, so that for each change a least-squares `a` explains that change's errors. The paper used six wind conditions and 12 minutes of flight. It also trains a second network against Φ, to keep information about the condition out of the features. We would start without that.
- **The sentence above no longer holds.** The features would be trained on changes, so the planner gets a training set of changes too, and its held-out set must stay unseen for the same reason as the policy's.
- **Data needs the robot to stay up.** The data comes from the robot walking under each change, so it needs a policy that does not fall under them, and it must be collected again for each policy, because the planar motion depends on the policy underneath.
- **It gives us our ablation.** The paper compares its learned features against a constant feature, which is a disturbance observer, and the learned ones win. Ours would be three rows: offset only, hand-chosen features (command and constant), learned features.

The first step is unchanged: build the online update with the features as a replaceable part, and test it on made-up data with the hand-chosen features. Learned features come after there is data from real changes.

How sure: the update equations were read from the paper's supplement. The network sizes and the details of the offline training come from a summary of its methods section and have not been checked against the text.

**First step, built on 2026-11-20.** The code is `mppi/residual_model.py`, the made-up-data check is `mppi/residual_check.py`, and the tests are in `mppi/tests/`. Everything below was measured on made-up data unless it says otherwise: the fitted planar model plus a change we chose, plus noise shaped like the gait's. The simulator results are under "Second step" below.

What was built:

- **The update above, with the features as a replaceable part.** Three hand-chosen feature sets exist: an offset per axis (3 coefficients, a disturbance observer), a gain and an offset per axis (6), and every axis responding to all three commands plus an offset (12).
- **Two scalings that were not in the earlier sketch.** The features see the velocity and command divided by the command range. The residual is expressed in units of the nominal model's usual error on the healthy robot, per axis (0.10 m/s, 0.06 m/s, 0.06 rad/s, from the fit's report). With both, one setting suits every axis, and the 12-coefficient set follows noise about half as much as without them.
- **Two knobs.** `drift_std` (default 0.6) sets the speed: higher reacts faster and follows noise more. `leak` (default 0.1 per second) returns coefficients to zero when the data says nothing about them.

What was found:

| Finding | Numbers |
|---|---|
| It reacts in about two seconds, not one | With the 12 coefficients it removes 51 to 64% of the new model error in the first second after a change and 72 to 89% in the second. Raising `drift_std` to 0.9 gains 6 points in the first second and costs a third more jitter. |
| It returns to zero when the change is undone | The residual after the undo is the same size as before the change: about 0.25 of the usual error. |
| The leak is needed when commands hardly vary | Five healthy minutes of race-like commands with no leak: the coefficient uncertainty keeps growing (7.4 and rising) and the residual predicted at standstill wanders to 0.52 usual errors. With the default leak: 1.1 and 0.20. |
| Which features are needed depends on the change | Under varied commands, the offset alone removes 41% of a "weaker" change in the long run, against 78% for gain and offset. Only the 12 coefficients follow a veer that grows with speed: 74%, against 53% and 48%. |
| Under race commands the feature sets are equal | All three remove 85 to 92%, because the commands hardly vary and an offset explains everything. Simpler is then slightly better. |
| The healthy residual is not zero on the real robot | On the recorded healthy race the nominal model is off by -0.5, -0.6 and -1.0 usual errors on average (forward, sideways, yaw), because the race sits at the edge of the range the model was fitted on. The residual learns exactly that. So the residual's size is a damage signal only against its healthy level. |
| Most of the remaining error is the gait | On that recorded race the residual lowers the one-step error only from 1.25 to 1.09 usual errors. The error is strongly periodic: its correlation with itself reaches plus or minus 0.7 at delays of 0.2 to 0.5 s. These features cannot predict that. The made-up noise copies it roughly, as 70% of the variance in one wave with a period of 1/3 s. |

Open after this step:

- **Is two seconds fast enough?** It was in the one case tried so far, the pull under "Second step" below, which is an easy one. If it turns out not to be, the gait wave is the thing to remove first, for example by averaging the velocity over one gait period before fitting. Not tried.
- **A held command hides the gain.** While a command is held, only the correction at that command is learned. The split between gain and offset arrives when the command next changes. This is the limit noted above, now measured: in the constant-command test the correction at standstill is 2.1 usual errors where the truth is 3.4.
- **The made-up noise is white plus one wave.** The real gait's error may depend on speed. The recorded race is the only real data used so far, and it is healthy and at one operating point.

**Second step, tried on 2026-11-20: a steady pull on the trunk, with the residual in the race loop.** The race demo gained two options: `--pull` applies a steady horizontal force to the trunk that turns with the robot, and `--residual` lets MPPI plan with the corrected model, updated at every plan step. The test is the circle race (radius 2.5 m, track 0.4 m to each side), healthy for 8 s, then a 30 N pull to the robot's right, which is outward. Ten runs of 30 s per row. Runs with the same seed differ, so the rows are counts over runs, not single results.

| Planner's model | Falls | Time off the track during the pull | Where it settles | Progress along the track, 15 to 30 s |
|---|---|---|---|---|
| Fixed (no correction) | 0 of 10 | 90% (87 to 92%) | 0.54 m outside the centre line, so 0.14 m off the track | 1.64 m/s |
| Offset per axis (3 coefficients) | 0 of 10 | 0% (0 to 4%) | 0.32 m outside, on the track | 1.73 m/s |
| Gain and offset per axis (6) | 0 of 10 | 0% (0 to 3%) | 0.31 m outside, on the track | 1.73 m/s |
| All commands and offset (12) | 2 of 10 | 0% (0 to 20%) in the runs that stayed up | Near the centre line | 2.00 m/s in the runs that stayed up |

What this shows:

- **The correction does its job at the task level.** With the fixed model the robot is pulled off the track and stays off, while its plans keep pointing back inside. With any of the three feature sets it settles inside the track.
- **The planner's model is less wrong, but not back to healthy.** One-step error in forward and sideways velocity: 0.17 m/s healthy, 0.38 m/s under the pull with the fixed model, 0.25 m/s with the correction.
- **The 12 coefficients make the planner jerk when the pull starts, and the robot can fall.** In the two seconds after the pull starts the planner brakes from 3 m/s to under 1 m/s and swings its commands across their range: the command changes by 0.92 per plan step, against 0.14 with the fixed model and 0.16 with the offset alone. Both falls came 1.6 to 1.8 s after the pull started. A third fall, in a run made for the video, came 7 s after the pull started: the robot never settled, moving at about 1 m/s and swinging between 0.2 and 0.8 m outside the centre line until it fell. So the trouble is not always over after two seconds.
- **The likely cause is the limit in entry 3.1.** While the commands are held, a steady pull and a drift that grows with forward speed look the same, so the 12 coefficients blame part of the pull on the forward command and the planner brakes. This is a reading of the logs, not proven. One scratch test supports it: with the gain features weighted at a third of the offset's, 0 of 10 runs fell, the robot was off the track 0% of the time (0 to 15%), and progress was 1.83 m/s. That variant is not in the code.
- **Two seconds was fast enough here, in an easy case.** The robot starts on the inside edge, so it has 0.75 m of room before it leaves the track, and it drifts outward for about three seconds before settling.
- **The correction costs something on the healthy robot.** With the offset alone the commands change three times as much per step before the pull (0.43 against 0.14) and the robot is over the inside edge 28% of the time, by up to 6 cm. The fixed model stays just inside. With the 12 coefficients the healthy robot runs nearer the centre line and laps in 5.8 to 5.9 s against 5.5 to 5.6 s (one run each, no pull).

Open after this step:

- **Which feature set is the method's default.** The offset alone is safe and is the disturbance-observer baseline. The 12 coefficients track better once settled and are unsafe in the transient. A slower prior on the gains, as in the scratch test, is the obvious thing to build and test next. Built and tested on 2026-11-23, see the third step below.
- **Only one pull was tested in depth.** Single runs with a 20 N pull to the right, a 30 N pull to the left and a 30 N drag backward went the same way, but one run each is not evidence.
- **The pull is not leg damage.** It is a steady offset, which is the case an offset feature fits exactly. Motor faults (step 3) are where richer features should matter.

**Third step, 2026-11-23: gains drift slower than offsets.** `ResidualCfg.gain_drift_ratio` (default 1/3) lets gain coefficients drift at a third of `drift_std`, offsets at the full rate. Gains are found automatically: the coefficients whose features are zero at rest. While the commands are held, a steady push can be explained by an offset or by a gain on the forward command; with equal rates the filter split it, and the planner braked to shrink a drift that does not depend on speed. Now such an error goes to the offset first. Same pull test, ten runs per row:

| Planner's model | Falls | Off the track during the pull | Off the track before the pull | Progress 15 to 30 s | Command change, first 2 s of the pull |
|---|---|---|---|---|---|
| Gain and offset (6), equal rates | 0 of 10 | 0% (0 to 3%) | 20 to 29% | 1.74 m/s | 0.58 |
| Gain and offset (6), slow gains | 0 of 10 | 0% (0 to 3%) | 26 to 31% | 1.75 m/s | 0.34 |
| All commands and offset (12), equal rates | 2 of 10 | 0% (0 to 20%), runs that stayed up | 0 to 8% | 2.01 m/s, runs that stayed up | 1.05 |
| All commands and offset (12), slow gains | 0 of 10 | 0% (0 to 2%) | 11 to 24% | 1.83 m/s | 0.52 |
| Offset only (3), for reference | 0 of 10 | 0% (0 to 4%) | 24 to 30% | 1.74 m/s | 0.17 |

"Command change" is the summed absolute change of the three commands per plan step, averaged over the first 2 s of the pull. The fixed model gives 0.14.

- The 12 coefficients no longer fall, and their transient is half as jerky. They now beat the offset alone on this test: same safety, faster, and less time over the inside edge before the pull.
- The cost: the healthy 12-coefficient robot is over the inside edge 11 to 24% of the time before the pull, against 0 to 8% with equal rates.
- On made-up data (`residual_check`), gain-type changes are learned more slowly with varied commands: the share of the error removed after two seconds drops from 75% to 62% for "weak" and from 74% to 64% for "veer". Two unit tests were loosened for this, and one was added (`test_steady_push_goes_to_the_offsets`).
- Proposed default: 12 coefficients with slow gains, to be confirmed on motor faults, where gain changes are expected.

**Fourth step, 2026-11-23: first motor faults.** `--fault JOINTS --fault-torque X --fault-gain Y` scales one or more motors' torque limit and PD gains from `--change-at` (8 s) on, through `Tracker.weaken_joints`. Fixed model, single runs: the front right knee at 30% torque, or the whole front right leg at 30%, changes almost nothing (model error 0.17 to 0.19 m/s). The knee at 10% falls within 1.2 s. Then three runs per row, front right leg:

| Fault | Planner's model | Falls | Progress after the fault | Off the track after | Model error after |
|---|---|---|---|---|---|
| Thigh motor dead (torque and gains 0) | Fixed | 2 of 3 (at 9.3, 9.4 s) | 1.28 m/s | 0% | 0.66 m/s |
| | Offset (3) | 2 of 3 (9.0, 9.7 s) | 0.40 m/s | 23% | 0.37 m/s |
| | 12 coefficients | 2 of 3 (14.8, 25.3 s) | 0.15 m/s | 2% | 0.28 m/s |
| Knee at 15% torque | Fixed | 1 of 3 (9.0 s) | 0.60 m/s | 1% | 0.51 m/s |
| | Offset (3) | 3 of 3 (8.9 to 10.3 s) | 1.53 m/s | 21% | 0.33 m/s |
| | 12 coefficients | 3 of 3 (9.9 to 19.2 s) | 0.65 m/s | 0% | 0.31 m/s |
| Knee at 20% torque | Fixed | 2 of 3 (15.0, 15.3 s) | 1.70 m/s | 3% | 0.33 m/s |
| | Offset (3) | 2 of 3 (11.0, 13.3 s) | 1.64 m/s | 26% | 0.28 m/s |
| | 12 coefficients | 3 of 3 (14.3 to 24.3 s) | 1.92 m/s | 5% | 0.20 m/s |

Progress is averaged up to the fall where there is one.

- **The residual makes the model less wrong, but does not stop falls.** Model error drops by 40 to 60% with the 12 coefficients, and falls sometimes come later, but most runs fall with every model. With this policy the faults that change the planar motion are the ones that break the gait, and no velocity plan repairs a gait.
- **The offset alone is worse than the fixed model here:** off the track 21 to 26% of the time after the fault. A fault is not a steady push.
- **A confound:** MPPI may command up to 3 m/s, above the 2 m/s the policy was trained on. Falls may come from fast commands on a weak leg rather than from the fault alone. Running inside the trained range (`--cmd-max 2 1 0.7`) would separate the two, and is the question of R3 (command limits after a change).
- Three runs per row is a first look, not evidence.

**Fifth step, 2026-11-23: the same faults inside the trained command range** (`--cmd-min -1.5 -1 -0.7 --cmd-max 2 1 0.7`, so at most 2 m/s and 0.7 rad/s; on this circle that caps the speed along the track at about 1.75 m/s). Three runs per row:

| Fault | Planner's model | Falls | Progress after the fault | Off the track after | Model error after |
|---|---|---|---|---|---|
| None (healthy) | Fixed | 0 of 3 | 1.74 m/s | 1% | 0.23 m/s |
| Thigh motor dead | Fixed | 3 of 3 | 0.80 m/s | 9% | 0.48 m/s |
| | Offset (3) | 3 of 3 | 1.08 m/s | 14% | 0.37 m/s |
| | 12 coefficients | 2 of 3 | 0.31 m/s | 54% | 0.26 m/s |
| Knee at 15% torque | Fixed | 1 of 3 | 0.76 m/s | 7% | 0.30 m/s |
| | Offset (3) | 1 of 3 | 0.95 m/s | 5% | 0.24 m/s |
| | 12 coefficients | 1 of 3 | 0.79 m/s | 1% | 0.22 m/s |
| Knee at 20% torque | Fixed | 0 of 3 | 1.16 m/s | 1% | 0.22 m/s |
| | Offset (3) | 0 of 3 | 1.04 m/s | 2% | 0.27 m/s |
| | 12 coefficients | 0 of 3 | 1.16 m/s | 0% | 0.16 m/s |

- **Speed caused most of the falls in the fourth step.** With the knee at 20%, every run now survives, against 7 of 9 falling at up to 3 m/s. So command limits after a change (R3) matter.
- **Where the robot survives, the fault hardly changes its planar motion.** With the knee at 20% the fixed model's error is 0.22 m/s, the same as on the healthy robot (0.23 m/s). The robot is slower because MPPI asks for less, not because the model is wrong, so the residual has little to fix: same progress, off the track 0% against 1%.
- **Where the planar motion changes a lot, the gait breaks.** A dead thigh motor falls with every model, and the 12-coefficient residual leaves the track 54% of the time.
- **So the residual's real test needs a policy trained on faults (P1, P2).** A policy that survives a dead motor will walk differently, slower or veering, and that is the change the planner's model has to follow. With the current policy there is no fault that both keeps the robot up and changes its planar motion much.

**Sixth step, 2026-11-24: lower the command limits once a joint is flagged (R3).** The per-joint fault score from entry 5.7 watches the real robot. When any joint crosses its threshold, MPPI's limits drop from the fast range (3 m/s, 1.5 rad/s) to the trained range (2 m/s, 0.7 rad/s) for the rest of the run (`demo_circle_race --fault-limits`). Compared with always fast and always slow, baseline policy, fixed planar model, fault at 8 s, three runs per row (five for the healthy fast and slow rows):

| Fault | Always fast | Always slow | Slow once flagged |
|---|---|---|---|
| None | 0 falls, 2.78 m/s | 0 falls, 1.74 m/s | 0 falls, 2.79 m/s, no alarm |
| Knee at 15% | 2 of 3 fall | 0 of 3, 1.12 m/s after | 0 of 3, 1.06 m/s after |
| Knee at 20% | 1 of 3 | 0 of 3, 1.11 m/s | 0 of 3, 1.22 m/s |
| Rear knee at 10% | 3 of 3, off the track 53% | 0 of 3, 1.46 m/s | 0 of 3, 1.47 m/s |
| Knee dead | 3 of 3 | 1 of 3 | 3 of 3 |
| Thigh dead | 1 of 3 | 3 of 3 | 2 of 3 |

- **Every fault was flagged 0.2 to 0.5 s after it started, always on the right joint, and no healthy run raised an alarm.**
- **For a weak motor the rule gets the best of both:** full speed while healthy, and as safe as the slow limits after the fault (no falls in 9 runs, against 6 of 9 when always fast).
- **For a dead motor no limit helps the baseline policy.** It falls at any speed; that needs a policy trained on faults.
- **Open:** the rule is one switch. A rule that scales the limits with the score, or that brakes harder right after the alarm, is untested.

**Seventh step, 2026-11-24: features learned offline** (`mppi/learn_features.py`, as in Neural-Fly). 1024 robots, each under one condition for 40 s (20% healthy, 50% one weak motor at 0 to 50% torque, 30% a steady pull of 10 to 40 N in a random direction), random commands in the trained range. A small network F(v, u) with 6 coefficients is trained so that coefficients fitted by ridge regression on 3 s of a robot's data predict its next 3 s. Then the online filter runs over held-out robots, and over robots with the held-out joint (front left knee), and its one-step error is compared, in m/s and rad/s, forward, sideways, yaw:

| Policy, test set | Nominal only | Hand, offset (3) | Hand, 12 coefficients | Learned (6) |
|---|---|---|---|---|
| Baseline, held-out robots | 0.114, 0.090, 0.093 | 0.112, 0.071, 0.086 | 0.113, 0.071, 0.086 | 0.102, 0.069, 0.085 |
| Baseline, held-out joint | 0.104, 0.063, 0.126 | 0.104, 0.064, 0.118 | 0.104, 0.065, 0.117 | 0.098, 0.063, 0.116 |
| P1, held-out robots | 0.084, 0.079, 0.080 | 0.079, 0.053, 0.071 | 0.080, 0.054, 0.071 | 0.074, 0.052, 0.066 |
| P1, held-out joint | 0.103, 0.079, 0.211 | 0.100, 0.075, 0.204 | 0.099, 0.075, 0.204 | 0.103, 0.076, 0.218 |
| P2, held-out robots | 0.082, 0.073, 0.079 | 0.078, 0.050, 0.068 | 0.079, 0.051, 0.068 | 0.072, 0.049, 0.063 |
| P2, held-out joint | 0.078, 0.059, 0.105 | 0.073, 0.049, 0.100 | 0.072, 0.048, 0.100 | 0.074, 0.051, 0.101 |

- **The learned features are a little better than the hand-chosen ones on conditions like those they were trained on,** forward and yaw by 5 to 10%, for every policy. Most of the remaining error is the gait's own wobble, which no residual removes.
- **On the held-out joint they are no better,** and for P1 slightly worse in yaw. Six coefficients learned from faults on other joints do not cover the front left knee; the hand-chosen offsets do as well or better. More conditions in training, or more coefficients, may fix that.
- The baseline rows use commands up to 2 m/s; P1 and P2 rows up to 1 m/s, their trained range (see "Eighth step").
- **The three hand-chosen sets are equal on this data.** With commands held for 0.2 to 3 s, the extra gains add nothing a one-step fit can use.

**Eighth step, 2026-11-24: first fault-trained policies (preliminary).** P1 (no history), P2 (0.5 s of history) and a third policy that sees the per-joint fault score (entry 5.7), each trained 3,000 iterations on mid-episode motor faults (`mdp.motor_fault`, front left knee held out). Stopped early for time: the command curriculum widens only at 5,000 iterations, so all three are trained up to 1 m/s and 0.5 rad/s.

*Do they cope with a dead motor?* 128 robots each, random commands up to 1 m/s held 3 s, the motor dies at 2 s; share that tip past 60° within the next 8 s:

| Motor | Baseline | P1 | P2 (history) | P1 setup + fault score |
|---|---|---|---|---|
| Front right knee dead | 82% | 34% | 16% | 31% |
| Front right thigh dead | 62% | 5% | 3% | 6% |
| Front right hip dead | 93% | 72% | 36% | 31% |
| Front right knee at 20% | 0% | 0% | 0% | 0% |

- **Training on faults works, and history helps on top.** P2 falls half as often as P1 after a dead knee or hip. Whether P2 adapts or is just more robust is for P3 to show.
- **The fault score helps less than history.** It halves P1's falls after a dead hip but does not help with a dead knee.

*The race.* Circle, fault at 8 s, three runs per cell, every policy with a fixed planar model ("off") or the 12-coefficient residual ("command"). At up to 1 m/s and 0.5 rad/s every policy falls after a dead knee or thigh in nearly every run: the race keeps the robot at the edge of its range, which is harder than training. At a gentler cap (0.6 m/s, 0.3 rad/s):

| Fault | Policy | Falls, fixed / residual | Progress after, fixed / residual | Off the track after | Model error after |
|---|---|---|---|---|---|
| Thigh dead | P1 | 0 / 0 of 3 | 0.48 / 0.58 m/s | 22% / 19% | 0.21 / 0.19 m/s |
| | P2 | 0 / 0 of 3 | 0.76 / 0.75 m/s | 0% / 14% | 0.09 / 0.09 m/s |
| Knee dead | P1 | 3 / 3 of 3 | | | |
| | P2 | 3 / 3 of 3 | | | |
| Held-out knee dead | P1 | 1 / 0 of 3 | 0.10 / -0.06 m/s | 32% / 65% | 0.31 / 0.22 m/s |
| | P2 | 2 / 1 of 3 | 0.39 / 0.20 m/s | 64% / 45% | 0.19 / 0.09 m/s |
| None | P1, P2 | 0 / 0 of 3 | 0.64 to 0.68 m/s | | |

- **Where a fault-trained policy survives with a changed gait, the residual cuts the planner's model error,** by half for P2 on the held-out knee (0.19 to 0.09 m/s). This is the case the earlier steps lacked.
- **It does not yet turn into better racing.** On the held-out knee the robot crawls (0.1 to 0.4 m/s) and leaves the track 30 to 65% of the time with either model. With three runs per cell, none of the differences in falls or progress is reliable.
- **Next:** train P1 and P2 the full 10,000 iterations, so they reach 2 m/s and cope better; then run the comparison with X1's batched evaluation over many trials instead of three races.

## 4. Damage injection

Background: MuJoCo Warp calls each parallel environment a "world". Some model fields can differ per world, including geom size and position, site position and PD gains. The collision settings `contype`, `conaffinity` and `condim` cannot.

Every entry in this section was checked against mjlab's code at commit `936f7a1`. None has been tried in a run yet.

### 4.1 Give the lower-leg capsules friction

**Status:** `open` · 2026-11-11

Go1's lower-leg collision capsules are frictionless (`condim 1`), and only the foot sphere has friction. After a chop the robot would stand on a frictionless stump and slide. `condim` can't vary per world, so set those capsules to `condim 3` in the model for all worlds. Intact legs rarely touch the ground with their lower legs, so normal walking shouldn't change.

### 4.2 Drop the mass change from the chop

**Status:** `open` · 2026-11-11

One Go1 lower leg, foot included, weighs 0.196 kg, which is 1.5% of the robot. Changing `body_mass` makes mjlab rerun its full physics recompute (`set_const`). In an event that runs every step, that recompute happens on every step for every world, even when nothing was chopped. Geometry, site and gain changes need no recompute.

### 4.3 Shrink and shift the geoms

**Status:** `open` · 2026-11-11

Capsules shrink around their centre, so shorten them with `geom_size` and move them up with `geom_pos`. Shrink the foot sphere to almost nothing.

### 4.4 Move or mask the foot site of the damaged leg

**Status:** `open` · 2026-11-11

A site is a named reference point on a body. Go1's foot sites (`FR`, `FL`, `RR`, `RL`) sit at the old foot centre and don't move when the geoms shrink. The reward terms `foot_clearance`, `foot_swing_height`, `foot_slip` and `soft_landing` depend on those sites or on the foot contact sensor, so they would track a foot that no longer exists.

Either move the damaged leg's site to the stump tip (`dr.site_pos`, per world) or mask those terms for that leg.

### 4.5 Relax the pose reward for the damaged leg

**Status:** `open` · 2026-11-11

The pose reward pulls every joint toward the normal stance, which fights a three-legged gait that holds the stump up.

### 4.6 Implement the chop as a class-based "step" event, not an "interval" event

**Status:** `done for motor faults` · 2026-11-24 (`mdp.motor_fault`)

mjlab events run at startup, on reset, at intervals, or on every step. Interval events re-arm after firing, so a 20 s episode could be chopped twice. A class-based term running every step can pick one chop time per episode in its `reset` hook and keep a "chopped" flag per environment.

### 4.7 Dead knee motor

**Status:** `open` · 2026-11-11

In Go1's model the knee joint is named `*_calf_joint`. A dead motor is `dr.pd_gains` with `operation="abs"` and gains (0, 0) on that joint. It needs no recompute.

## 5. Policy

### 5.1 History is one setting on the actor's observations

**Status:** `open` · 2026-11-11

The stock Go1 policy sees no history. Milestone 3 is mostly setting `history_length` on the actor observation group. Checked against the code.

### 5.2 A recurrent policy is a cheap ablation

**Status:** `open` · 2026-11-11

rsl-rl's recurrent policy (GRU or LSTM) is already available through mjlab's config, so comparing stacked history with a recurrent policy costs one extra training run.

### 5.3 Let the critic see the true damage

**Status:** `open` · 2026-11-11

Give the critic which leg is damaged, how much, and whether the motor is dead. The task already has a separate critic observation group, and privileged critic inputs usually speed up PPO.

### 5.4 Simulator rollouts will need the policy's history copied too

**Status:** `open` · 2026-11-11

The box-push demo copies simulation state into planning worlds (entry 3.5). That is complete for the current policy, which sees no history. Once the policy has observation history or is recurrent, the history buffers or hidden state must be copied as well, or the planning worlds start with the wrong memory.

### 5.5 Train the policy on wider command ranges

**Status:** `open` · 2026-11-11

The baseline run stopped at the curriculum stage with forward commands up to 2.0 m/s and yaw up to 0.7 rad/s. The next stage, up to 3.0 m/s, starts exactly where the run ended, so it was never trained. The policy already extrapolates to about 2.8 m/s and 1.9 rad/s, but it falls beyond that. Training longer, or with wider ranges for forward speed and yaw, is the remaining way to make the race faster. One training run takes a little over an hour.

### 5.6 What turns the PPO policy into meta-RL

**Status:** `open` · 2026-11-12

History alone is not enough. A policy with history trained only on the undamaged robot has nothing to adapt to and learns to ignore the history. Black-box meta-RL needs three things together:

- **Memory:** the actor sees its recent observations and actions, as a stacked window or through a recurrent network.
- **A distribution of tasks in training:** many damages, starting at random times mid-episode, with some episodes undamaged. The switch has to happen during an episode, or the policy never learns to notice it.
- **A hidden task:** the actor is never told the damage. It has to infer it from how the robot responds to its own actions. The critic may be told (entry 5.3).

Training is still ordinary PPO. The adaptation happens inside the forward pass, with no weight updates at test time. This is the same idea as in-context learning in language models: the history is the context, and the damage is the task inferred from it. With a transformer over a long history it is in-context learning in the literal sense, which is how LocoFormer is usually described.

Options for the memory, from least to most work:

| Option | What it is | Work in mjlab |
|---|---|---|
| Stacked history | The actor's MLP sees the last N observations | One setting, `history_length` |
| Recurrent policy | GRU or LSTM actor, the classic form of black-box meta-RL | A config change, already supported by rsl-rl |
| Teacher and student (RMA) | A teacher sees the true damage as a latent; a student learns to predict that latent from history | Two training phases and an extra network |
| Transformer over a long context | In-context learning in the literal sense | A new policy class and much more compute |

Suggested starting point, not yet tested: stacked history of 0.5 to 1 s, which is one to two gait cycles, with the true damage given to the critic. Use the recurrent policy as an ablation.

Black-box meta-RL is one family of meta-RL, not the whole field. All families train across a distribution of tasks and differ in where the adaptation to a new task happens:

| Family | Where adaptation happens | Example |
|---|---|---|
| Black-box, memory-based | In the policy's memory, with no weight updates | RL², LocoFormer |
| Gradient-based | In the weights, by a few gradient steps at test time | MAML |
| Task inference | In an explicit latent task variable inferred from recent data | PEARL, VariBAD; RMA is a supervised relative |
| Model-based | In a dynamics model that is updated online, then used for planning | Nagabandi et al. |

Black-box fits the policy because the robot has to adapt within seconds, mid-run, with no time for trial episodes or gradient steps. Our planner is a small hand-designed case of the model-based family: its least-squares re-fit is the adaptation, and MPPI is the planning.

Why history helps, in POMDP terms (raised 2026-11-12). The damage can be treated as a hidden part of the state: which leg, how much is cut, whether the motor is dead, and whether the chop has happened yet. With that variable included, the problem is an ordinary Markov problem again, but the policy cannot observe it. That makes it a POMDP, and in a POMDP the best action depends on the belief about the hidden state, which is computed from the history of observations and actions. A policy that sees only the current observation cannot hold a belief, so it can only be robust on average.

- **Why one observation is not enough:** the same joint angles and velocities occur with and without damage. The damage shows in how the robot responds to an action over the following steps.
- **What the memory does:** it plays the role of the belief. A stacked window is a truncated version of it, which is enough if one or two gait cycles identify the damage.
- **The baseline is already a POMDP.** Friction, the centre-of-mass offset and the encoder bias are randomized and hidden from the actor, and it does not see contacts or its height. Without history it is robust to them, not adapted to them.
- **The privileged critic** is the critic being given the hidden state, which is allowed because the critic is only used in training.
- **The planner does the same inference explicitly.** Its residual fit is a point estimate of how the hidden damage changes the planar motion. Pulling the fit toward zero acts as a prior, and forgetting old samples allows the hidden state to change.

Names for this setting that are worth citing, to be checked against the papers: hidden-parameter MDP (Doshi-Velez and Konidaris), hidden-mode MDP for a hidden state that switches (Choi, Yeung and Zhang), and Bayes-adaptive MDP for the meta-RL view (Duff).

Ways to check that the policy really adapts and is not just robust:

- **Against the no-history policy** trained on the same damages (rows 1 and 2 of the experiments table). The gap is the value of adaptation.
- **Recovery curve:** tracking error against time since the chop should fall over a second or two, not jump to a new steady level.
- **History ablation at test time:** blank or shuffle the history. Performance after damage should drop.
- **Probe:** the damage should be decodable from the policy's hidden features.

### 5.7 A residual dynamics model for the walking policy too

**Status:** `open` · 2026-11-20

Raised by cpw: use the residual idea at both levels, in the planner and in the walking policy. The policy would get an online estimate of how the robot's dynamics have changed, as an extra input.

- **The planner's residual cannot be reused as it is.** It describes the base's motion over the ground with a few numbers. It does not say which leg or joint is faulty, and that is what the policy needs in order to change its gait.
- **A joint-level residual would.** The recipe is the same and the signal is different: a nominal model of how each joint follows its target, fitted on the healthy robot, and an online fit of that model's recent errors. A dead motor shows up as a joint that does not follow its target. The result is one or two numbers per joint.
- **It has to be part of training.** The policy must be trained with the residual as an input, so the estimate has to be computed inside the training environment for thousands of robots at once. That is cheap to compute, but it makes the Policy role's training depend on this work.
- **How it relates to history.** The residual is a hand-designed summary of the recent past. A history policy has to learn its own summary. So the two can be compared directly: designed adaptation against learned adaptation.
- **Effect on the comparison in 2.1.** The two factors would become "history, yes or no" and "residual model, yes or no", with the residual feeding both layers. Telling the planner's residual apart from the policy's would need two more controllers. Not decided.

Order agreed with cpw on 2026-11-20: build the planner's residual first, because it needs no training and is already designed. Build the joint-level one second, and treat it as an exploration until the first one works. Agree the joint-level residual's interface with the Policy role early. None of this is tested.

**First version, 2026-11-24** (`src/mjlab/tasks/velocity/mdp/joint_residual.py`, fitted by `mppi/fit_joint_model.py`):

- **Nominal model: a small network, not a linear model per joint.** A linear model of each joint's velocity change from its tracking error and velocity explains only 20 to 50% of it (contacts and the other joints matter). A network over all joints' positions, velocities and tracking errors explains 93 to 98%, on held-out healthy robots.
- **Signal: a one-sided shortfall, not a fitted gain.** A Kalman filter on a gain and an offset per joint, like the planar residual, flagged healthy joints about 20% of the time: the effective gain swings with the gait. A weak motor differs only when it hits its torque limit, and then the joint moves toward its target less than predicted. The score is that shortfall, in units of the model's usual error, averaged over 0.5 s. Impacts push the error both ways and average out.
- **Threshold:** set so that 5% of healthy robots cross it within 20 s, under random commands up to 3 m/s.
- **Detection, baseline policy, 512 faulty robots:** torque below 10% of normal, 100% detected, median 0.54 s, right joint 89%; 10 to 25%, 56% detected, median 0.86 s; 25 to 50%, 31% detected, median 5.2 s. A motor that keeps a quarter of its torque rarely changes how the joint moves, and rarely matters.
- **As a policy input:** the observation term `joint_fault_score` computes it inside the training environment for every robot, divided by the threshold. The task `Mjlab-Velocity-Flat-Unitree-Go1-Fault-Score` trains a policy that sees it.
- **Known limit:** the nominal model is fitted to the baseline policy's gait. A policy that walks differently would need its own fit.

## 6. Demo and setup

### 6.1 Record the chop video in mjlab itself

**Status:** `open` · 2026-11-11

The plan records the chop in CPU MuJoCo. That may not be needed: mjlab's video renderer copies each world's geom size, position and color. The lower-leg mesh can be hidden and a shortened stump capsule shown in the same run. Checked against the code, not yet tried.

### 6.2 Pin commit `936f7a1`, not the v1.6.0 tag

**Status:** `open` · 2026-11-11

The fork is v1.6.0 plus 60 commits, including the upgrade from MuJoCo 3.11 to 3.16. The baseline ran on that commit, so pinning it keeps later runs comparable.

### 6.3 Box-jump demo: the main demo from 2026-11-24

**Status:** `scene built` · 2026-11-24

cpw changed the story: the main demo is now a run along a long box to build up speed and a jump onto a smaller box, while gravity and foot friction change; the robot has to adapt in time to land on the box. Motor faults become a bonus task.

**Numbers, picked as a starting point** (in `src/mjlab/tasks/box_jump/course.py`, `DEMO_*`; to retune once a jump policy exists):

| What | Value | Why |
|---|---|---|
| Long box | 6.0 x 1.0 m, top 0.4 m above the ground | 5.5 m of run-up from the spawn point reaches 2 to 3 m/s with room to spare |
| Gap | 0.5 m | Near Go1's body length (0.65 m); published robots of its size leap 0.6 m (Zhuang et al., Robot Parkour Learning, 2023) |
| Landing box | 1.0 x 1.0 m, same height | Short enough that a 20 to 40% error in jump length misses it |
| Gravity after the change | 7 to 12 m/s^2 (nominal 9.81) | Same take-off lands 40% farther at 7, 18% shorter at 12 (point mass) |
| Foot friction after the change | 0.4 to 1.0 (nominal 1.0; training saw 0.3 to 1.5) | At 0.4 the robot slides about 0.7 m when braking from 2 m/s, against 0.2 m at 1.0 (point mass) |
| When | 2.5 m before the edge | About 1 s before take-off at 2.5 m/s |

**Friction is the feet's.** Go1's foot geoms have contact priority, so their friction is the friction of every foot contact and the boxes' own friction does not matter. Changing "the ground's" friction means changing the feet's, as the existing training randomization does.

**Built:**

- `BoxJumpCourseCfg`, a terrain tile with the course. Its gap and landing-box length can grow with difficulty for a curriculum; `layout(difficulty)` gives where the parts are relative to the spawn point.
- Per-robot gravity: mjlab can now expand nested model fields such as `opt.gravity`. MuJoCo Warp already reads gravity per world.
- `physics_change`, a step event: each episode, gravity and foot friction start at nominal or at a random value, and change once when the robot passes a line a set distance from its spawn point.
- `mppi/demo_box_jump.py`: the velocity policy runs a fixed speed ramp at the gap. It reaches 2.57 m/s, the change comes at 2.3 s, and at the edge its front legs hit the face of the landing box and it drops into the gap. As expected: it has no jump skill.

**Not built yet:** a jump skill (the critical path), a jump model for the planner (run-up plus ballistic flight), and a task with rewards for landing on the box.

**MPPI today plans only (vx, vy, wz):** forward, sideways and turning speed, 10 steps of 0.1 s. Nothing vertical, no take-off timing.

### 6.4 Compute cost of a full-model sampling planner (DIAL-MPC style)

**Status:** `measured` · 2026-11-24

DIAL-MPC (LeCAR lab, arXiv 2409.15610) jumps with no learned policy: MPPI-style sampling over joint torques, rolling out the full robot model, with the jump given only by contact and body-position targets. It reports 50 Hz replanning on an RTX 4090 with 2,048 samples over 0.4 s (20 steps), and the 0.6 m crate climb took about 30 s to plan 2 s (not real time). It does not adapt online, and shows no running jumps.

**Our cost** on the RTX 5080, Go1 on the box course in mjlab (MuJoCo Warp). One planning iteration means copying the current state into every world and rolling out noisy joint targets, so the robots move, as in a real plan. Times in ms:

| Samples | Timestep | 0.4 s horizon | 1.0 s horizon |
|---|---|---|---|
| 256 | 5 ms | 61 | 157 |
| 1,024 | 5 ms | 82 | 211 |
| 2,048 | 5 ms | 109 | 286 |
| 4,096 | 5 ms | 169 | 445 |
| 8,192 | 5 ms | 288 | 755 |
| 2,048 | 10 ms | 67 | 173 |
| 1,024 | 20 ms | 29 | 73 |
| 2,048 | 20 ms | 41 | 106 |

- **Samples are cheap, steps are not.** Each physics step has a floor of roughly 0.8 to 1.5 ms however few worlds there are, so the horizon and timestep set the cost.
- **Moving robots cost two to three times more than standing ones** (110 against 47 ms for 2,048 samples over 0.4 s). The solver needs more iterations, and MuJoCo Warp floods "linesearch iterations limit reached" warnings (`ls_iterations` 20).
- **Real time at 50 Hz (20 ms) is out of reach**, even for one iteration at the cheapest setting.
- **A simulated demo can still run it** if the simulated robot waits for the planner. For example, 2,048 samples, 1 s horizon, 10 ms timestep, 2 iterations, replanning at 50 Hz is about 17 s of compute per simulated second.
- **Not measured:** cheaper contact settings for planning worlds (condim 3, fewer solver iterations). The 20 ms timestep stayed stable under random joint targets but was not tested on running or landing impacts.

### 6.5 The jump policy: take-off command split, with accelerometer

**Status:** `training` · 2026-11-24

cpw chose this split: the policy runs, takes off on command with a commanded vertical speed, lands and brakes; the planner decides the run-up speed and when to trigger, predicting the landing point from a ballistic model with gravity and friction estimated online.

**Tasks:** `Mjlab-BoxJump-Unitree-Go1` (no history) and `Mjlab-BoxJump-Unitree-Go1-History` (0.5 s of history, the main policy), in `src/mjlab/tasks/box_jump/`.

- **Output:** 12 joint position targets at 50 Hz, as for the velocity policy.
- **Actor inputs:**
  - the velocity policy's body inputs;
  - the IMU accelerometer, which reads gravity plus the trunk's acceleration;
  - the command `(vx, vy, wz, vz)`: `vz` is 0 while running; above 0 it means take off now at that vertical speed, held until touch-down. (A separate jump flag was dropped on 2026-11-24 at cpw's request: same information, no mode bit.);
  - the course ahead: distance to the edge and to the start and end of the landing box.
  - The history variant sees 25 past steps of everything except the command and the course: 1,207 inputs.
- **Critic adds:** true gravity and foot friction, trunk height, jump phase, and foot contacts.
- **The coach (the planner's stand-in in training):** steers along the centre line, ramps the run-up speed, triggers the jump so the trunk lifts off about 5 cm before the edge, and sets the vertical speed from a ballistic flight under the episode's true gravity, aiming the trunk 5 to 30 cm past the start of the landing box (±10% noise). After touch-down it commands a stop. The run-up speed is raised where needed so the take-off speed stays at most 2 m/s.
- **Phases:** run, take-off (triggered, feet still down), flight (all feet off for two steps), landed (first touch-down).
- **Rewards:**
  - planar velocity tracking, plus no vertical motion on the ground;
  - yaw rate;
  - at lift-off, once: closeness to the commanded vertical and forward speed (weight 300, worth up to 6);
  - upward push while taking off;
  - on the landing box after touch-down;
  - flat body, action rate, joint limits, torques, thigh and shank contacts;
  - falling: -200, worth -4.
- **Ends on:** trunk below the box tops, tilt over 80 degrees, trunk contact, or 6 s.
- **Curriculum:** 10 rows, gap 0.1 to 0.7 m, landing box 2.0 to 1.0 m. Landing on the box moves an environment up a row; anything else moves it down.
- **Gravity and friction:** each episode starts with gravity 7 to 12 m/s^2 and foot friction 0.4 to 1.2. In half of the episodes both change once on the run-up, to 7 to 12 and 0.4 to 1.0.

### 6.6 The planner side of the jump

**Status:** `built, waiting for the trained policies` · 2026-11-24

- **`mppi/jump_planner.py`, the planner:**
  - For every robot at once, it searches a grid of run-up speed (1 to 3 m/s) and take-off vertical speed (0.5 to 2 m/s) through a jump model, and replans every step until the trigger.
  - The model has four parts:
    - **Take-off:** the trunk lifts off just before the edge, at the speed the robot can still reach.
    - **Flight:** ballistic under the estimated gravity.
    - **Calibration:** the policy's actual take-off speed for a commanded one.
    - **Landing check:** the trunk must touch down past the start of the box and brake to a stop before its end, at a deceleration set by friction × gravity.
  - It keeps the plan with the largest margin to both ends.
- **Gravity estimate: momentum balance.** Gravity = average upward foot force ÷ mass − trunk vertical acceleration, using the contact sensor's force at every physics step.
  - Reading the accelerometer once per 20 ms policy step read 2 to 3 m/s^2 low while running, because it missed the short foot impacts. A real IMU integrates at a high rate, so this was a simulation artefact.
  - At the trigger the new estimate is within 0.12 m/s^2 (median).
- **Friction estimate.** Slipping feet show the friction; gripping feet bound it from below. It starts at a prior of 1.0. On the first, flawed checkpoint it read about 0.3 low at the trigger: conservative.
- **`mppi/run_box_jump.py`:** runs the policy with the planner on the demo course across many robots, with gravity and friction from one of three sources:
  - `nominal`: fixed at 9.81 and 1.0;
  - `online`: the estimates above;
  - `oracle`: the simulator's true values.
  - Every source sees the same changes. `--video` records robot 0.
- **`mppi/fit_jump_model.py`:** calibrates the jump model to a policy:
  - take-off vertical speed against the command and gravity;
  - forward speed;
  - trigger-to-lift-off time;
  - touch-down beyond the point-mass flight;
  - braking efficiency.
  - It saves `jump_model.json` next to the checkpoint.
- **The jump command** has a `coach=False` mode, in which a planner sets the plan through `set_plan`.

**First run, on a flawed early checkpoint** (iteration 250 of the first training, 64 robots, gravity 7 to 12 and friction 0.4 to 1.0 after the change):

| Source | Lands on box | Falls |
|---|---|---|
| Nominal | 61% | 39% |
| Online | 70% | 16% |
| Oracle | 92% | 6% |

A second run with 128 robots gave 66%, 76% and 74%, so these early numbers are noisy. The calibration on that checkpoint showed why: the policy ignored the take-off command (R^2 0.02).

**The flaw, fixed:** the first training counted the running gait's brief moment with all feet in the air as a lift-off, so the policy was never pushed to follow the commanded take-off speed (median error 1.3 m/s). A lift-off now needs three steps in the air with the trunk rising at 0.25 m/s or more, and the take-off reward is wider. In the retrained no-history policy the error is 0.30 m/s by iteration 200.

**After removing the jump flag** (no-history policy at iteration 300 of 3,000, calibrated planner, 2026-11-24):

- **Calibration:** the policy follows the take-off command better than the flawed one did, but not fully: actual vertical speed = 0.62 + 0.65 × commanded (R^2 0.30). From where it actually takes off, touch-down is within about 6 cm of the flight model plus a fixed -16 cm offset. The fitted braking, 1.11 × friction × gravity, is optimistic: landing impacts press the feet harder than the robot's weight.

| Case (256 or 64 robots) | Fixed values | Online estimates | True values |
|---|---|---|---|
| Full range: gravity 7 to 12, friction 0.4 to 1.0 (256) | 49% | 84% | 74% |
| Gravity 7.5, friction 0.8 (64) | 0% | 72% | 45% |
| Gravity 7, friction 0.5 (64) | 0% | 2% | 2% |

Share that land and stay on the box.

- **Fixed values fail as expected:** the planner believes gravity is 9.81, so the robot overshoots the box.
- **Online beats the true values,** because the friction estimate reads low and makes the planner gentler. Given the true friction, the optimistic braking model plans landings the robot cannot stop from. The braking model needs a conservative fit.
- **Gravity 7 with friction 0.5 is beyond this early policy:** it lands on target, then bounces and slides off.
- **Video bug, fixed:** the renderer drew two neighbouring robots on top of robot 0, because every robot shares one course. Videos recorded before the fix (`box_jump_800_*`) show other robots' falls; their statistics are unaffected. The runner now sets `max_extra_envs=0`. Comparison videos are stacked in rows (cpw's preference).

### 6.7 MPPI for the jump

**Status:** `built and working` · 2026-11-24

cpw asked for the jump to be planned by MPPI rather than the grid search.

- **What `mppi/jump_mppi.py` plans:** for every robot at once, a run-up speed profile over 4.5 s (6 knots), a trigger time and a take-off speed. The update is the same as in `mppi/core.py`, batched over robots: noise around the current plan, costs normalized by their spread, softmax weights, 3 iterations, and a warm start shifted by the time elapsed. It replans every 0.1 s and takes about 60 ms for 256 robots on the shared GPU.
- **Rollout model** (calibrated):
  - **Run-up:** the speed settles at an offset and gain of the command (0.83 + 0.65 × command), no faster than the top acceleration (2.2 m/s^2) or traction (friction × gravity).
  - **Take-off:** push-off for 0.12 s, then lift-off at the calibrated vertical speed (-0.12 + 1.12 × command, R^2 0.80).
  - **Flight:** ballistic under the estimated gravity, with touch-down 0.19 m short of the point-mass prediction.
  - **Braking:** 0.89 × friction × gravity, the 20th percentile, so the planner does not count on the robot's best stops.
- **Cost:** touch-down outside the landing window or lift-off past the edge (20 per metre), a reward for margin up to 0.25 m, a small cost on take-off speed, and smoothness of the speed profile.
- **Videos:** they always draw the sampled plans coloured by cost, the chosen plan (cyan) and the path driven (white). The chosen plan stays frozen after the trigger, to compare prediction with reality.
- **Calibration fixes:**
  - the run-up speed has no real lag; the earlier first-order lag fit was meaningless;
  - the speed trace now records the command the policy acted on;
  - braking uses a low percentile.

**Results** (no-history policy at iteration 900, 256 robots over gravity 7 to 12 and friction 0.4 to 1.0, changed 2.5 m before the edge):

| Planner | Fixed values | Online estimates | True values |
|---|---|---|---|
| MPPI | 82% (0.11 m) | 100% (0.09 m) | 100% (0.07 m) |
| Grid | 56% (0.18 m) | 97% (0.19 m) | 95% (0.20 m) |

Share that land and stay on the box (median touch-down error).

Hard case, gravity 7 and friction 0.5, 64 robots, MPPI: 0% with fixed values, 100% with online estimates, 100% with the true values. With the online or true values, MPPI runs fast early, slows to about 1 m/s before the edge, jumps high (2 m/s up), lands and stops. The grid search could not plan a slow-down and managed 2% in this case. Video: `logs/mppi/box_jump_mppi_hard_rows.mp4`.

**Final policies** (3,000 iterations each, calibrated, MPPI, 512 robots over gravity 7 to 12 and friction 0.4 to 1.0 changed 2.5 m before the edge; 2026-11-24):

| Policy | Fixed values | Online estimates | True values |
|---|---|---|---|
| No history | 73% (23% fall) | 99.0% | 99.2% |
| History (0.5 s) | 87% (1% fall) | 99.2% | 99.2% |

Share that land and stay on the box.

- **With the planner adapting,** both policies land on the box nearly every time.
- **Without it,** the policy with history copes much better (87% against 73%, and 1% falls against 23%). This is the four-way comparison: planner adaptation dominates, and policy history helps where the planner does not adapt.
- **Hard case, gravity 7 and friction 0.5 (64 robots):**
  - no history: 2%, 97% and 94% (fixed, online, true values);
  - history: 8%, 92% and 97%.
- **Calibration of the final policies:**
  - no history: take-off vertical speed -0.13 + 1.07 × command (R^2 0.85), braking 1.09 × friction × gravity at the 20th percentile;
  - history: 0.21 + 0.88 × command (R^2 0.70), braking 1.13.
- **Videos:** `logs/mppi/box_jump_final_hard_rows.mp4` and `box_jump_final_history_hard_rows.mp4`. They now use a chase camera (behind and to the left of the robot), so the planned jump at the end of the course is in view from the first frame. A speed-along-the-course chart shows the sampled speed profiles, the chosen one and the speed actually run. Before the change, the side camera followed the robot too closely to show the plan's jump until about 2 m from the edge, and the sampled plans' run-ups all overlapped on one line.

### 6.8 Reference + short-horizon MPPI, and how fast it adapts

**Status:** `built and measured` · 2026-11-24

cpw found the long-horizon visualization odd: the plan showed the jump from the first step. DIAL-MPC and MuJoCo MPC, checked in their source, both plan over 0.25 to 0.5 s with running costs only. A time-indexed target or reference, set from above, carries the long-term structure:
- DIAL-MPC moves the body and foot targets 0.4 m every 1 s;
- MuJoCo MPC's backflip follows a hand-computed height and orientation reference, with g fixed at 9.81.

cpw chose the same structure, since the project's focus is fast adaptation.

- **`ReferencePlanner` (`mppi/jump_planner.py`):** a global grid over take-off speed and take-off vertical speed, recomputed every 0.1 s from the latest gravity and friction estimates. It counts only take-off speeds the robot can still reach by speeding up or slowing down. It outputs the take-off point and speed, the vertical speed, the landing target, and an approach-speed curve.
- **`JumpTrackingMppi` (`mppi/jump_mppi.py`):** MPPI over 1 s that tracks the reference. It may put the jump beyond its horizon.
  - Its cost tracks the reference speed and the reference position over time. It also penalizes waiting when the reference jumps within the horizon, puts the trigger at the reference's time and the lift-off at its point, and keeps the take-off speed near the reference while landing inside the window.
  - A first version without the time-indexed terms never jumped: waiting cost nothing as long as the robot never reached the take-off point, so it slowed down and stopped. That is why the other projects index their targets by time.
- **Videos:** a short predicted path at first, the reference's take-off line and landing spot in purple, and the jump arcs only once the jump enters the horizon. `logs/mppi/box_jump_tracking_hard_rows.mp4`.

**Adaptation test** (`mppi/eval_adaptation.py`, `logs/mppi/eval_adaptation.txt`): no-history final policy, 256 robots, gravity 7 to 12 and friction 0.4 to 1.0, changed at a line closer and closer to the edge.

| Change before edge | Planner | Fixed | Online | True | Gravity error at trigger | Change to trigger |
|---|---|---|---|---|---|---|
| 2.5 m | long MPPI | 73% | 100% | 100% | 0.20 m/s^2 | 0.66 s |
| 2.5 m | tracking | 63% | 100% | 100% | 0.19 | 0.62 s |
| 1.5 m | long MPPI | 74% | 96% | 100% | 0.63 | 0.26 s |
| 1.5 m | tracking | 65% | 98% | 100% | 0.54 | 0.30 s |
| 1.0 m | long MPPI | 71% | 86% | 98% | 1.01 | 0.10 s |
| 1.0 m | tracking | 65% | 85% | 98% | 0.86 | 0.14 s |
| 0.5 m | long MPPI | 77% | 79% | 77% | 1.34 | -0.06 s |
| 0.5 m | tracking | 70% | 77% | 69% | 1.27 | -0.04 s |

- **The two planners adapt equally fast.** Both replan every 0.1 s.
- **The estimator is the bottleneck.** With the true values both stay at 98 to 100% down to 1.0 m before the edge, 0.1 s before the trigger. With the online estimates success falls to 85%, because the gravity estimate (0.3 s time constant) is still about 1 m/s^2 off at the trigger.
- **At 0.5 m the change comes after the trigger,** so no planner can help: about 75% whatever the source.
- **Next:** a faster gravity estimate, for example detecting the step change and resetting the filter.

### 6.9 DIAL-MPC on the box-jump course

**Status:** `built, first results` · 2026-11-24

cpw asked to see DIAL-MPC itself, kept out of `mppi/`: `dial/core.py` and `dial/box_jump.py`.

- **Planner (`dial/core.py`):** the authors' algorithm from their `dial_core.py`.
  - Sampling: 2,048 samples over a 0.4 s horizon (20 control steps of 0.02 s) with 6 nodes; noise grows 0.9x per node towards the end of the horizon and shrinks 0.5x per round.
  - Rounds: 10 on the first step, then 2 per step.
  - Update: softmax weights at temperature 0.05 on rewards relative to the current plan, divided by their spread; the first node is fixed and the plan shifts one step after each control step.
  - Our only change: nodes joined linearly, not by a quadratic spline.
- **Task (`dial/box_jump.py`):** joint targets planned directly (±0.8 rad around standing), with no learned policy.
  - The rollouts run in a second, batched copy of the course with the planner's belief about gravity and foot friction (fixed, online estimates, or true).
  - Rewards in DIAL's style, as means over the horizon: a body target moving along the course at 1.5 m/s and stopping over the landing box, height, upright, yaw, forward speed, a trot foot-height schedule before the gap, feet on the landing box, and penalties for feet or trunk below the box tops.
- **Planning physics step matters.** With DIAL's 0.02 s step against our robot simulated at 0.005 s, the robot lost its footing within 2 s, before the gap, whatever the target speed. With 0.01 s (two physics steps per control step) it runs the course. The authors plan and simulate with the same 0.02 s model.
- **First result,** gravity 7 and friction 0.5 from 2.5 m before the edge, one robot per source:
  - fixed values: jumps short and tumbles off the front edge of the box;
  - online estimates and true values: both land on the box and stop.
  - Planning takes 118 ms per 20 ms control step (2,049 worlds), about 6x slower than real time.
  - Video: `logs/dial/box_jump_rows.mp4`. More seeds in `logs/dial/seeds.txt`.

**With 5 seeds** (seeds 0 to 4, same hard case): fixed 0/5, online 1/5, true values 3/5 land and stay on the box. Most misses end 5.8 to 5.95 m along: short of the box or on its front edge (the gap is 5.5 to 6.0 m), and 3 of the online misses fell. Seed 0, the video, was a lucky run for the online case. In the same case the MPPI + RL planner lands 100% of 64 robots with online estimates.

The online estimators were built around the RL policy's gait and may read DIAL's gait worse; this has not been checked.

Still to try for DIAL: a 0.005 s planning step (matching the robot exactly, about 2x slower), more rounds or samples per step, and a stage-based jump reward like the authors' sequential jump.

**Horizon, 10 seeds each** (gravity 7, friction 0.5 from 2.5 m before the edge; `logs/dial/tune_*_10seeds.txt`):

| Horizon | Fixed | Online | True | Planning per 20 ms step |
|---|---|---|---|---|
| 0.4 s (DIAL's default) | 0/10 (1 fell) | 4/10 (4 fell) | 10/10 | about 120 ms |
| 0.8 s (10 nodes) | 1/10 (4 fell) | 10/10 | 10/10 | about 250 ms |

- **0.4 s is too short for this task** (cpw's suspicion). At 1.3 m/s it sees about 0.5 m ahead. At lift-off the touch-down sits at the very end of the horizon and the braking beyond it, so the plan cannot prepare for landing and stopping on a 1 m box.
- **With 0.8 s, online estimates are as good as the true values.** Fixed values still fail.
- **Cost:** about 12x slower than real time, against MPPI + RL planning 256 robots in about 60 ms per 0.1 s.
- **The authors' 0.4 s horizon** was for standing hops of 0.4 m every 1 s, with no run-up or braking.
- **Video:** `logs/dial/box_jump_h08_rows.mp4`. In its seed the fixed-values robot also landed, which is the 1-in-10 exception.

### 6.10 A faster gravity estimate: the whole robot's momentum

**Status:** `built and measured` · 2026-11-25

The first gravity estimate used the trunk's velocity in the momentum balance. The legs are about a quarter of the mass and swing fast, so their momentum changes read as gravity, and it needed a 0.3 s average.

`CentroidalGravityEstimator` (`mppi/jump_planner.py`) uses the centre-of-mass velocity of all bodies. Over one policy step, the feet's upward impulse minus the change of total momentum is then almost exactly mass × gravity × step. The estimate is the median of the last 5 steps (0.1 s). Samples outside 2 to 25 m/s^2 are ignored: a robot lying on the course carries its weight on its body, not its feet, and reads near zero. Online estimates passed to the planner are clamped to plausible values, and the MPPI replaces NaN costs.

**Measured** (128 robots, full range, online):
- trunk estimator: error while running median 0.16 and 90th percentile 0.81 m/s^2; time to within 0.3 m/s^2 after a change: median 0.80 s, 90th percentile 1.06 s;
- whole-robot estimator: 0.04 and 0.14 m/s^2; median 0.06 s, 90th percentile 0.08 s, about 13 times faster.

**Adaptation test again** (tracking planner, 256 robots; `logs/mppi/eval_adaptation_centroidal.txt`):

| Change before edge | Online, trunk estimator | Online, whole-robot estimator | True | Gravity error at trigger (trunk, whole-robot) |
|---|---|---|---|---|
| 2.5 m | 100% | 100% | 100% | 0.19, 0.04 |
| 1.5 m | 98% | 100% | 100% | 0.54, 0.04 |
| 1.0 m | 85% | 89% | 97% | 0.86, 0.05 |
| 0.5 m | 77% | 76% | 69% | about 1.3 (the change comes after the trigger) |

**What is left at 1.0 m** (a rerun gave 86%): both estimates are accurate at the trigger.
- Gravity error: median 0.05 m/s^2.
- Friction minus true: median -0.07, the same for misses and successes.
- So the gap to the true values is reaction time. The change comes only 0.14 s before the trigger; the median needs 3 steps (0.06 s) to switch, and the planner replans every 0.1 s. The plan's run-up was set with the old values.
- Next: replan every policy step near the edge, and a shorter median window.

**Faster reaction** (2026-11-25): the planner now replans every control step (0.02 s) once a robot is within 1.5 m of the edge, and the gravity median uses 3 steps (0.06 s). Adaptation test again (tracking planner, 256 robots; `logs/mppi/eval_adaptation_fast.txt`):

| Change before edge | Fixed | Online | True | Gravity error at trigger | Change to trigger |
|---|---|---|---|---|---|
| 2.5 m | 73% | 100% | 100% | 0.06 m/s^2 | 0.60 s |
| 1.5 m | 71% | 100% | 100% | 0.06 | 0.28 s |
| 1.0 m | 70% | 99% | 100% | 0.06 | 0.12 s |
| 0.5 m | 75% | 86% | 75% | 1.32 | -0.06 s |

- **At 1.0 m, online now matches the true values:** 99% against 100%. The change comes only 0.12 s before the trigger.
- **At 0.5 m the change comes after the trigger,** so the planner cannot use it. Online's 86% against 75% for the true values most likely comes from the friction estimate reading low, which gives more braking room. Not verified.

**Video:** `logs/mppi/box_jump_late_change_rows.mp4`, with gravity 7 and friction 0.5 changing 1.0 m before the edge. Over 64 robots: fixed 0%, online 100%, true 98%. With fixed values the robot jumps high (vz 1.95 m/s) and overshoots the box. With online or true values it switches to a low, fast jump (about 0.85 m/s up at about 2.9 m/s), lands and stops.
