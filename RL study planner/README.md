---
title: Smart Study Planner
emoji: "📘"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Reinforcement Learning Based Smart Study Planner

This project is a **browser-based website** that uses **Q-Learning** to build a daily study schedule for a student. The agent learns which subject to study in each time slot so that it maximizes useful study time and avoids fatigue from poor planning.

## What the project includes

- Time slots: Morning, Afternoon, Evening
- Actions: Study OS, Study Java, Study DSA, Take Break
- State: current time slot + last studied subject
- Reward design with difficulty, confidence, repeated-subject penalty, break penalty, and fatigue penalty
- Epsilon-greedy Q-learning training
- Reward graph saved with matplotlib
- RL schedule vs random schedule comparison
- Save and load Q-table support
- Simple Flask website with a modern responsive UI
- User accounts (register/login/logout)
- Per-user saved plan history (SQLite)
- Downloadable PDF timetable from history
- English/Hindi language toggle
- Weekly planner calendar view

## How Q-Learning works here

The agent keeps a Q-table that stores the value of taking each action in each state.

During training:

1. The environment starts with the first time slot.
2. The agent chooses an action using epsilon-greedy exploration.
3. The environment returns a reward based on the schedule quality.
4. The Q-table is updated so that good choices get higher values.
5. Over many episodes, the agent learns a better planning policy.

### State, action, reward

- **State**: `(current time slot, last studied subject)`
- **Action**: one of the study subjects or `Break`
- **Reward**: positive for studying useful subjects, negative for repeating the same subject too often or taking too many breaks

### Exploration vs exploitation

- **Exploration** means trying random actions to discover better plans.
- **Exploitation** means using the best-known action from the Q-table.
- Epsilon-greedy policy balances both.

## Adaptive reward idea

The reward is higher for weaker subjects and lower for subjects the student already knows well. If an exam date is given, the reward also increases as the exam gets closer.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the website:

```bash
python main.py
```

This starts the Flask server locally and opens the website in your browser. You can add subjects, set difficulty and confidence, upload a saved Q-table, train the agent, and view the schedule, plot, and Q-table right on the page.

## Deploy on Hugging Face Spaces

1. Create a new Space on Hugging Face and choose the `Python` SDK.
2. Upload or connect this repository.
3. Keep `requirements.txt` in the root; Spaces will install `Flask`, `numpy`, `matplotlib`, and `reportlab` automatically.
4. Make sure the Space starts from the root-level `app.py` file.
5. Leave the port handling to Spaces; the app listens on `0.0.0.0` and reads `PORT` automatically.

If you want to test the same startup pattern locally, run:

```bash
python app.py
```

## New user features

- Create an account from `/register` and login from `/login`.
- Generated plans are saved automatically to your account.
- Open `/history` to view past plans and download them as PDF.
- Toggle language using the EN/HI switch in the top bar.
- View a 7-day weekly plan in calendar table format after training.

## Output files

- `reward_vs_episodes.png` - reward curve
- `saved_q_table.npz` - learned Q-table

## Sample run output

```text
Reinforcement Learning Based Smart Study Planner
Training completed.
Final epsilon: 0.050

Optimized RL schedule
Morning    -> DSA
Afternoon  -> Java
Evening    -> Break
Total reward: 24.50

Random schedule
Morning    -> Break
Afternoon  -> OS
Evening    -> Java
Total reward: 8.00

Comparison result: RL-based schedule performed better.
```

## Notes for viva

- Reinforcement learning is a method where an agent learns by interacting with an environment and receiving rewards.
- Q-learning is a value-based RL algorithm that learns the best action for each state.
- The model in this project learns a better daily study plan by rewarding useful study decisions and penalizing poor scheduling.
