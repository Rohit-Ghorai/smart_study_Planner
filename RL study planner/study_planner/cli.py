from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .agent import QLearningAgent
from .config import StudyPlannerConfig
from .environment import StudyPlannerEnvironment
from .persistence import load_q_table, save_q_table
from .trainer import format_q_table, generate_greedy_schedule, generate_random_schedule, train_agent
from .visualization import plot_rewards


def _read_int(prompt: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter a whole number.")
                continue

        value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        return value


def _read_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid decimal number.")


def _collect_subject_data() -> tuple[list[str], list[int], list[int]]:
    subject_count = _read_int("How many subjects do you want to include?", 3, minimum=1)
    subjects: List[str] = []
    difficulties: List[int] = []
    strengths: List[int] = []

    print("Enter each subject name, difficulty, and current confidence level (1-5).")
    print("If you are unsure about confidence, use the same value as difficulty or 3.")

    for index in range(subject_count):
        subject_name = input(f"Subject {index + 1} name: ").strip() or f"Subject_{index + 1}"
        difficulty = _read_int(f"Difficulty for {subject_name}", 3, minimum=1, maximum=5)
        confidence = _read_int(f"Confidence for {subject_name}", 3, minimum=1, maximum=5)
        subjects.append(subject_name)
        difficulties.append(difficulty)
        strengths.append(confidence)

    return subjects, difficulties, strengths


def _build_config(load_existing_model: bool) -> StudyPlannerConfig:
    subjects, difficulties, strengths = _collect_subject_data()
    exam_date_text = input("Exam date (YYYY-MM-DD, optional): ").strip()

    if load_existing_model:
        return StudyPlannerConfig.from_input(
            subjects=subjects,
            difficulties=difficulties,
            strengths=strengths,
            exam_date_text=exam_date_text,
        )

    episodes = _read_int("Training episodes", 1500, minimum=100)
    alpha = _read_float("Learning rate alpha", 0.1)
    gamma = _read_float("Discount factor gamma", 0.9)
    epsilon = _read_float("Initial exploration rate epsilon", 1.0)

    return StudyPlannerConfig.from_input(
        subjects=subjects,
        difficulties=difficulties,
        strengths=strengths,
        exam_date_text=exam_date_text,
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )


def _print_schedule(title: str, schedule: list[str], reward: float, environment: StudyPlannerEnvironment) -> None:
    print(f"\n{title}")
    for slot, action in zip(environment.time_slots, schedule):
        print(f"{slot:<10} -> {action}")
    print(f"Total reward: {reward:.2f}")


def run_cli() -> None:
    print("Reinforcement Learning Based Smart Study Planner")
    print("-" * 52)

    load_existing_model = input("Load an existing Q-table before training? (y/N): ").strip().lower() in {"y", "yes"}
    config = _build_config(load_existing_model)
    environment = StudyPlannerEnvironment(config)
    agent = QLearningAgent(
        num_states=environment.num_states,
        num_actions=environment.num_actions,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_decay=config.epsilon_decay,
        min_epsilon=config.min_epsilon,
        seed=config.seed,
    )

    skip_training = False
    if load_existing_model:
        model_path = input("Path to Q-table file [saved_q_table.npz]: ").strip() or "saved_q_table.npz"
        if Path(model_path).exists():
            data = load_q_table(model_path)
            loaded_q_table = np.asarray(data["q_table"], dtype=float)

            if loaded_q_table.shape == agent.q_table.shape:
                agent.q_table = loaded_q_table
                skip_training = True
                print("Loaded Q-table successfully. Training will be skipped.")
            else:
                print("Loaded Q-table shape does not match the current environment. Training from scratch instead.")
        else:
            print("Saved Q-table not found. Training from scratch instead.")

    if skip_training:
        episode_rewards = []
    else:
        print("\nTraining the Q-learning agent...")
        episode_rewards = train_agent(environment, agent, config.episodes)

    print("\nTraining completed.")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print("\nQ-table:")
    print(format_q_table(environment, agent))

    if episode_rewards:
        plot_path = plot_rewards(episode_rewards, show=False)
        print(f"\nReward graph saved to: {plot_path}")
    else:
        print("\nReward graph was not generated because the model was loaded instead of trained.")

    rl_schedule, rl_reward = generate_greedy_schedule(environment, agent)
    random_schedule, random_reward = generate_random_schedule(environment)

    _print_schedule("Optimized RL schedule", rl_schedule, rl_reward, environment)
    _print_schedule("Random schedule", random_schedule, random_reward, environment)

    winner = "RL-based schedule" if rl_reward >= random_reward else "Random schedule"
    print(f"\nComparison result: {winner} performed better.")

    model_path = save_q_table(
        "saved_q_table.npz",
        agent.q_table,
        [
            f"{environment.time_slots[slot_index]} / Energy {energy_level} / {last_action_type}"
            for slot_index, energy_level, last_action_type in environment.state_space
        ],
        environment.actions,
    )
    print(f"Q-table saved to: {model_path}")

    print("\nKey idea: the agent learns which subject to place in each time slot by maximizing reward over many episodes.")
