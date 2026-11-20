# Technical Architecture

## Components

Browser templates and `static/app.js` call Flask routes in
`study_planner/web.py`. The web layer builds a `StudyPlannerConfig`, trains
independent Q-learning and SARSA agents, serializes plans, and delegates
persistence to `study_planner/db.py`. The OpenEnv service uses
`OpenEnvStudyPlanner` separately from the study-planning MDP.

## Data flow

Subject form -> validated config -> environment reset -> epsilon-greedy
episodes -> greedy daily schedule -> weekly action-preserving projection ->
SQLite plan and progress records. User queries always filter by `user_id`.

## Model format

Models are compressed NumPy archives containing numeric `q_table`, string
state/action labels, a format version, and JSON metadata. Loading uses
`allow_pickle=False`, validates finite numeric values, dimensions, and action
mapping, and rejects incompatible archives.

## State and transitions

The environment tracks integer time slots, bounded energy, previous action
category, per-subject effort buckets, and bounded break history. Study lowers
energy and advances time; breaks restore energy and advance one slot; deep
study advances two slots. Terminal states do not return a bootstrapping target.

## Database

SQLite tables include users, plans, completions, progress events, notification
preferences, and notifications. Foreign keys are enabled on every connection,
and indexes cover user plan and activity lookups.
