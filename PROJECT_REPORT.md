# Smart Study Planner: Project Report

## Abstract

Smart Study Planner applies tabular Q-learning and SARSA to select study
activities using subject difficulty, confidence, exam urgency, energy, and
recent effort. A Flask application exposes training, history, analytics, and
an isolated OpenEnv-compatible evaluation environment.

## System and objectives

The system turns a student's subject configuration into an energy-aware daily
plan. It prioritizes difficult and low-confidence subjects without making
mock tests or breaks universally optimal, and stores plans per authenticated
user.

## Reinforcement-learning design

The state is `(time_slot, energy_level, previous_action_category,
progress_signature, break_bucket)`. Each subject's effort is represented by
three buckets (`0`, `1`, `2+`), while break history is capped at `2+`. The
action space contains regular, deep, revision, and mock actions per subject,
plus Meditation, Snack, and Water_Break. Deep study consumes two integer slots;
all other actions consume one.

Q-learning uses `Q <- Q + alpha (r + gamma max Q' - Q)` and SARSA uses the
observed next action. Terminal transitions do not bootstrap. Both agents use
seeded epsilon-greedy exploration and independent tables.

The reward combines activity value, difficulty, confidence gap, urgency,
remaining-time context, diminishing effort returns, and energy/break costs.
The implementation keeps these terms bounded and makes repeated mock tests
less attractive than need-based study.

## Application architecture

`study_planner.environment` and `agent` implement learning; `trainer` runs
episodes and creates schedules; `persistence` stores safe numeric archives;
`db` stores users, plans, progress, and notifications; `web` serves Flask
pages and APIs. `server/app.py` is the standalone OpenEnv service.

## Security and testing

Passwords use Werkzeug hashing, SQL queries are parameterized, ownership is
checked in plan APIs, model uploads are size- and schema-validated without
pickle loading, and OpenEnv rejects invalid actions. Focused smoke tests cover
environment transitions, both update loops, persistence validation, Flask
workflow, and invalid OpenEnv requests.

## Limitations and future work

The compact progress buckets trade exact history for a tractable table.
Statistical algorithm comparison should use many independent seeds in a
dedicated experiment runner. CSRF protection and a production session store
should be added before internet deployment.
