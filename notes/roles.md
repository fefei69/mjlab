# Pick a role

The project has five roles, one per person. One is taken. Read the table, pick one of the other four, and reply in the team thread with the role you want. The **Owner** column is filled in from the thread.

When you have your role, mark your first item "in progress" in [tasks.md](tasks.md) and push to the branch `dev`. That also checks that your setup and your push access work.

A role is what you are responsible for, not a wall. Ask each other for help, and pair up on the hard parts. The detail for every item (S1, P2 and so on) is in [tasks.md](tasks.md), and the picture of the whole system is at the top of [ideas.md](ideas.md).

| Role | What you own | Difficulty | Skills you need | GPU | Owner |
|---|---|---|---|---|---|
| **Method: residual dynamics model** | How the system adapts online with a residual model: in the planner, and possibly in the policy | Hard | Linear algebra, control, PyTorch | Some | cpw (taken) |
| **Robot changes** | What goes wrong with the robot mid-run | Medium | mjlab's event system, MuJoCo basics, Python | Yes | |
| **Policy** | The walking policy, with and without history | Medium to Hard | Running PPO training, network and observation setup | Yes, a lot | |
| **Evaluation** | The course, the measurements and the final comparison | Medium | Python, statistics, careful bookkeeping | Yes | |
| **Story and demos** | The report, the slides, the figures and the videos | Medium | Writing, visual design, plotting in Python, presenting | Little | |

## Method: residual dynamics model (taken)

The planner predicts the robot's motion with a small model fitted before the run. This role designs the correction that is re-fitted during the run, so the planner notices when the robot has changed.

- **You deliver:** the residual model, its online fit, and evidence that it reacts to a change within about a second.
- **Also exploring:** the same idea one level down, a joint-level residual given to the walking policy as an input (R5). If it works, it adds one more policy for the Policy role to train.
- **Items:** R1, R2, R5 as an exploration, and R3 if time allows.
- **Works closely with:** Robot changes, for something to adapt to, Policy, for R5, and Evaluation, for the numbers.

## Robot changes

You build the thing the whole project reacts to: a change to the robot in the middle of a run, such as a motor dying on one leg.

- **You deliver:** a mid-run change that can be switched on in training and in the demos, a measurement of which kinds of change matter, and the randomization: which changes the policies train on and which are held out for testing.
- **First week:** S1. Apply candidate changes to the policy we already have and measure what each does. This settles an open decision for the whole team.
- **Items:** S1, S2, S4, and S3 as a stretch goal.
- **Others wait for you:** Policy cannot start its real training runs until S2 works and S4 is agreed.

## Policy

You train the walking policy twice on the same changes: once seeing only the current moment, once seeing its recent history. The difference between the two is half of what the project claims.

- **You deliver:** the two trained policies, and evidence that the one with history really adapts.
- **First week:** get the existing training running on the GPU machine, and train a policy with history on the undamaged robot as a dry run.
- **Items:** P1, P2, P3, and P4 as an option.
- **You wait for:** S2 and S4 from Robot changes, for the real runs. Agree the training ranges in S4 with them.

## Evaluation

You make the results trustworthy: one course, one set of measurements, and the same trials for every controller.

- **You deliver:** a course with turns in both directions, a script that measures any controller on it, and the final four-way comparison.
- **First week:** T1 and the start of X1, both on the policy and planner that already exist.
- **Items:** T1, X1, X2, and W2 and R4 as options.
- **You wait for:** everyone, for the final comparison. Nothing, to start. The held-out set you test on comes from S4.

## Story and demos

You own how the project is told and shown. This starts in the first week, because the figures decide what the experiments must record and the demos decide what the simulation must be able to show.

- **You deliver:** the report, the slides and talk, a consistent set of figures, and the demo videos.
- **First week:** V2, a storyboard of the demos we want, a one-page outline of the report (W3), and a first visual style (V1).
- **Items:** V1, V2, X3, W1, W3, W4.
- **You need:** to understand the method well enough to explain it. You do not train policies.
