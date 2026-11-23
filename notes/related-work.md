# Related work

Papers we've read, what they do, and how our project differs. Papers we haven't read yet are listed at the end.

## LLA-MPPI (the closest prior work)

*LLA-MPPI: Rapidly Adaptive Whole-body Control of Legged Robots with GPU-Accelerated Parallel Simulations.* Jung, AL-Sunni, Alvarez-Padilla, Manchester, Liu and Dolan (CMU and MIT). [arXiv 2610.10465](https://arxiv.org/abs/2610.10465), October 2026. Read on 2026-11-11.

It tackles the same failure as our plan, a leg disabled mid-walk on a Unitree quadruped, with the same simulator family (MuJoCo Warp) and the same planner family (MPPI). It adapts without any RL.

### What it does

- **Model selection ("look-back", on the GPU).** It keeps a bank of K MuJoCo Warp simulators, each with different physics such as payload, friction or a disabled leg. Every control step, all K simulators predict the robot's last transition from the measured state. The one with the lowest error over a sliding window becomes the robot's model. A challenger must beat the current model by a margin, and optionally win several times in a row, which stops noisy switching.
- **Planning ("look-ahead", on the CPU).** Whole-body MPPI plans joint-position targets through the selected model. The bank runs alongside it and takes about 2 ms on a laptop GPU for banks of up to a few hundred models.
- **No offline training at all.**

### Results

- **Leg lock.** One leg's three motors are disabled. The bank holds 5 models: the nominal robot plus one per leg.

  | Controller | Successes out of 50 |
  |---|---|
  | LLA-MPPI | 46 |
  | Oracle (told the true model) | 47 |
  | Particle-filter baseline | 31 |
  | Disturbance-observer baseline | 0 |
  | Nominal MPPI | 0 |

- **Across its four tasks.** 97.5% success, against 98.5% for the oracle, 74% for the particle filter, 42.5% for the disturbance observer and 25.5% for nominal MPPI. No falls, and 17.6 ms per controller update.
- **Hardware.** A Go2 has its rear-left leg disabled about 8 s into a walk and still reaches the goal. Nominal MPPI falls immediately.
- **RL comparison.** PPO is compared only on the payload task, not on leg failure. PPO trained with payload randomization also reached the goal, after 49–55 min of training. At run time PPO needs under 1 ms per step, against 14.4 ms for LLA-MPPI.

### How it compares with our plan

| | LLA-MPPI | Our plan |
|---|---|---|
| Adaptation | Explicit: picks the best-matching simulator from a bank | Implicit in the policy's observation history, plus a least-squares planar model in the planner |
| Planner | Whole-body MPPI over joint targets, about 17 ms per update | MPPI over 3 velocity commands |
| Training | None | One PPO run |
| Leg damage | One leg's three motors off | Lower leg cut (continuous) or dead knee, with held-out combinations |
| Robot | Go2, in simulation and on hardware | Go1, in simulation |

### The two gaps it leaves

1. **RL is never tested on leg failure.** The only PPO comparison is on the payload task.
2. **No unseen damages.** The paper says "All realized conditions lie within the corresponding model-bank ranges", and the authors list the finite, fixed bank as a limitation. Our held-out damages test exactly that.

### What we can't do with it

Run it as a baseline. The code isn't released (the project page said only "coming soon" on 2026-11-11), it uses a Go2 where mjlab ships a Go1, and reimplementing whole-body MPPI would be a project of its own. Its model selection alone is a possible add-on: see [ideas.md](ideas.md), entry 3.3.

## Online residual models (background for the method)

Added on 2026-11-20. The planner's residual model ([ideas.md](ideas.md), entry 3.11) was not taken from one paper. It combines two standard parts: a nominal model with a correction learned online, and an online update of its coefficients. The built version (2026-11-20) uses a Kalman filter with a leak toward zero, as Neural-Fly does, rather than recursive least squares with forgetting. The papers below are the closest uses of the first part. Each was checked by web search against its abstract only. Neural-Fly's method was then read further on 2026-11-20: its supplement directly, its methods section through a summary. None of the others has been read in full.

| Paper | Nominal model | What is fitted online | Planner |
|---|---|---|---|
| Fu, Levine, Abbeel, IROS 2016, [arXiv 1509.06841](https://arxiv.org/abs/1509.06841) | Neural network trained on earlier tasks | A local linear model, with the network as its prior | MPC |
| Kabzan et al., RA-L 2019, [arXiv 1711.06586](https://arxiv.org/abs/1711.06586) for the earlier version | Simple vehicle model | A Gaussian process on the model's error | MPC, on a real race car |
| Nagabandi et al., ICLR 2019, [arXiv 1803.11347](https://arxiv.org/abs/1803.11347) | Meta-trained neural network | The whole network, by a few gradient steps on recent data | MPPI, on a legged robot that loses a leg |
| Neural-Fly, O'Connell et al., Science Robotics 2022, [arXiv 2205.06908](https://arxiv.org/abs/2205.06908) | Drone model | Linear coefficients on features learned offline | None. A tracking controller |
| RTE, Chatzilygeroudis et al., 2018, [arXiv 1610.04213](https://arxiv.org/abs/1610.04213) | Outcomes of pre-computed behaviors, simulated on the intact robot | Gaussian processes on the gap between simulated and observed outcomes | Tree search, on a six-legged robot with a missing leg |

How ours relates to them:

- **Closest in the math: Neural-Fly.** Its online part also updates a few linear coefficients with a regularized recursive rule. Its features are learned offline from data. Ours are chosen by hand (the command and a constant). Learned features are the natural step up if the hand-chosen ones prove too weak.
- **Closest in the structure: RTE.** A planner chooses among high-level behaviors of a legged robot, and an online model corrects what each behavior is predicted to do after damage. Our velocity commands play the role of its behaviors.
- **Closest in the task: Nagabandi et al.** Same planner and same kind of failure, but the model predicts the low-level dynamics and there is no walking policy underneath.

A different way to use an online estimate: keep the planner's model fixed and cancel the disturbance below the planner. L1-adaptive MPPI (Pravitra et al., 2020, [arXiv 2004.00152](https://arxiv.org/abs/2004.00152), simulation only) does this for drones. A disturbance observer is equivalent to fitting only the offset in our residual. LLA-MPPI's disturbance-observer baseline scored 0 of 50 on leg lock (table above). That was in whole-body control with no adapting policy underneath, so it does not settle our case, but it is the result a reader will hold against us.

The fitting rule itself, recursive least squares with a forgetting factor, is textbook adaptive control (Ljung, *System Identification*; Åström and Wittenmark, *Adaptive Control*).

Not found: a paper that fits a small linear residual on a planar velocity model above an RL walking policy. The search was eight queries for papers already known to us, so this is not evidence that none exists.

## From the plan's references

Descriptions are the plan's own. These haven't been reviewed in the notes yet.

- **LocoFormer** (Skild AI, CoRL 2025): the leg-chop adaptation demo that motivates this project.
- **Learning to Adapt in Dynamic, Real-World Environments** (Nagabandi et al., ICLR 2019): meta-learned models with MPC, including a robot that loses a leg. Its model also adapts online, which matters for how we state our contribution (ideas.md, entry 1.1).
- **DATT** (Huang et al., CoRL 2023, [arXiv 2310.09053](https://arxiv.org/abs/2310.09053)): an RL policy paired with an online adaptive estimator. From its abstract, checked on 2026-11-20: in training the policy is given the true disturbance force, and in flight it is given an online estimate of it. That is the pattern for the joint-level residual fed to our policy (ideas.md, entry 5.7).

## To read

Both come from LLA-MPPI's bibliography.

- **RMA** (Kumar et al., RSS 2021): the standard history-based adaptation method. Our "policy with history" row resembles it.
- **Turrisi et al. 2026**, "Mixture-of-experts RL for fault-tolerant legged locomotion" ([arXiv 2606.25965](https://arxiv.org/abs/2606.25965)): RL aimed specifically at leg faults, so probably our closest RL competitor.
