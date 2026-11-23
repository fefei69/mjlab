# Project notes: fast adaptation to leg damage

A shared notebook for the RL & OC project: a simulated Unitree Go1 that keeps walking and returns to its path after losing part of a leg. It has two jobs: a place to brainstorm, and a record of what has been done.

## What's here

| File | What it holds | Update it when |
|---|---|---|
| [plan.md](plan.md) | The project plan, as a snapshot from 2026-11-11 | An idea is adopted and changes the plan |
| [ideas.md](ideas.md) | Ideas and open questions, each with a status, and the system figure | Something new comes up, or an idea is settled |
| [system.png](system.png) | The system figure shown in ideas.md: what is built and what is planned. [system.svg](system.svg) is its editable source. | A planned part is built |
| [roles.md](roles.md) | The five roles in the project, for each person to pick one | Someone picks a role |
| [tasks.md](tasks.md) | Action items for the team to pick from, with difficulty and skills, the open decision on what to test, and the four-way comparison | Someone claims or finishes an item, or a decision is made |
| [progress.md](progress.md) | Milestone status, setup, and a dated log of work and results | After each work session |
| [related-work.md](related-work.md) | Papers we've read: what they do and how we differ | We read a paper |
| [proposal.tex](proposal.tex) | The two-page course proposal, in LaTeX | The proposal is revised |
| [references.bib](references.bib) | BibTeX entries for the proposal | A reference is added |
| [system_figure.tex](system_figure.tex) | The system figure drawn in LaTeX (TikZ), for the proposal, the report and slides. [system_figure_standalone.tex](system_figure_standalone.tex) compiles it to a PDF of its own. | The system changes |

## How we use it

- **Ideas have a status:** `open` (still under discussion), `adopted` (agreed, and reflected in the plan) or `dropped` (with the reason). Dropped ideas stay in the file so we don't revisit them by accident.
- **The log is newest first.** Each entry has a date and says where the results are: run directory, W&B run ID, video path.
- **Notes point to results, they don't hold them.** Checkpoints, videos and training logs stay under `logs/`, which git ignores, so they exist only on the training PC.
- **Say how sure a claim is.** Mark whether something was checked against the code, measured in a run, or is only reasoning.

Claude Code keeps these files up to date during work sessions. Edit them by hand any time.
