from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np


from .agent import QLearningAgent, SARSAAgent
from .environment import StudyPlannerEnvironment


def train_agent(
    environment: StudyPlannerEnvironment,
    agent: QLearningAgent,
    episodes: int,
    log_interval: int = 100,
) -> List[float]:
    episode_rewards: List[float] = []

    for episode in range(episodes):
        state = environment.reset()
        state_index = environment.state_to_index[state]
        total_reward = 0.0
        done = False

        while not done:
            action_index = agent.choose_action(state_index)
            next_state, reward, done = environment.step(action_index)
            next_state_index = None if next_state is None else environment.state_to_index[next_state]
            agent.learn(state_index, action_index, reward, next_state_index, done)

            total_reward += reward
            if not done and next_state_index is not None:
                state_index = next_state_index

        agent.decay_exploration()
        episode_rewards.append(total_reward)

        if log_interval and (episode + 1) % log_interval == 0:
            recent_window = episode_rewards[-log_interval:]
            average_reward = float(np.mean(recent_window))
            print(
                f"Episode {episode + 1:4d}/{episodes} | "
                f"Average reward: {average_reward:7.2f} | Epsilon: {agent.epsilon:.3f}"
            )

    return episode_rewards


def train_sarsa_agent(
    environment: StudyPlannerEnvironment,
    agent: SARSAAgent,
    episodes: int,
    log_interval: int = 100,
) -> List[float]:
    """Train SARSA agent - uses actual next action for bootstrapping."""
    episode_rewards: List[float] = []

    for episode in range(episodes):
        state = environment.reset()
        state_index = environment.state_to_index[state]
        action_index = agent.choose_action(state_index)
        
        total_reward = 0.0
        done = False

        while not done:
            next_state, reward, done = environment.step(action_index)
            next_state_index = None if next_state is None else environment.state_to_index[next_state]
            
            if done:
                agent.learn(state_index, action_index, reward, None, None, done)
            else:
                next_action_index = agent.choose_action(next_state_index)
                agent.learn(state_index, action_index, reward, next_state_index, next_action_index, done)
                state_index = next_state_index
                action_index = next_action_index

            total_reward += reward

        agent.decay_exploration()
        episode_rewards.append(total_reward)

        if log_interval and (episode + 1) % log_interval == 0:
            recent_window = episode_rewards[-log_interval:]
            average_reward = float(np.mean(recent_window))
            print(
                f"Episode {episode + 1:4d}/{episodes} | "
                f"Average reward: {average_reward:7.2f} | Epsilon: {agent.epsilon:.3f}"
            )

    return episode_rewards


def generate_greedy_schedule(
    environment: StudyPlannerEnvironment,
    agent: QLearningAgent,
) -> Tuple[List[str], float]:
    state = environment.reset()
    state_index = environment.state_to_index[state]
    total_reward = 0.0
    schedule: List[str] = []

    done = False
    while not done:
        action_index = agent.choose_action(state_index, greedy=True)
        action_name = environment.actions[action_index]
        next_state, reward, done = environment.step(action_index)
        total_reward += reward
        schedule.append(action_name)

        if not done and next_state is not None:
            state_index = environment.state_to_index[next_state]

    return schedule, total_reward


def generate_greedy_schedule_with_explanations(
    environment: StudyPlannerEnvironment,
    agent: QLearningAgent,
    config,
) -> Tuple[List[str], float, List[Dict[str, str]]]:
    """Generate greedy schedule with AI explanations for each decision."""
    from datetime import date
    
    state = environment.reset()
    state_index = environment.state_to_index[state]
    total_reward = 0.0
    schedule: List[str] = []
    explanations: List[Dict[str, str]] = []

    done = False
    while not done:
        action_index = agent.choose_action(state_index, greedy=True)
        action_name = environment.actions[action_index]
        
        # Generate explanation
        slot_index, energy_level, last_action_type = environment.state_space[state_index]
        time_slot = environment.time_slots[slot_index]
        
        reasons = []
        
        if action_name in environment.break_actions:
            reasons.append("Rest needed for mental recovery")
            if environment.break_count >= 1:
                reasons.append("But already took breaks")
            
            if action_name == "Meditation":
                reasons.append("Meditation reduces stress")
            elif action_name == "Snack":
                reasons.append("Quick nutrition boost")
        else:
            # Parse subject from action
            subject = environment._parse_action(action_name)[0]
            difficulty = config.difficulties.get(subject, 3)
            strength = config.strengths.get(subject, 3)
            
            if difficulty >= 4:
                reasons.append(f"{subject} high difficulty ({difficulty}/5)")
            if strength < difficulty:
                reasons.append(f"{subject} low confidence ({strength}/5)")
                
            days_left = None
            if config.exam_date:
                days_left = (config.exam_date - date.today()).days
            
            if days_left and days_left <= 3:
                reasons.append(f"Exam urgent ({days_left} days left)")
            elif days_left and days_left <= 7:
                reasons.append(f"Exam soon ({days_left} days left)")
            
            if "_Deep" in action_name:
                reasons.append("Deep focused study session")
            elif "_Mock" in action_name:
                reasons.append("Mock exam to test knowledge")
            elif "_Rev" in action_name:
                reasons.append("Light revision reinforces concepts")
            
            if energy_level <= 2:
                reasons.append(f"Energy low ({energy_level}/5) - lighter session")
        
        explanations.append({
            "time_slot": time_slot,
            "subject": action_name,
            "reason": " • ".join(reasons) if reasons else "Optimal choice"
        })
        
        next_state, reward, done = environment.step(action_index)
        total_reward += reward
        schedule.append(action_name)

        if not done and next_state is not None:
            state_index = environment.state_to_index[next_state]

    return schedule, total_reward, explanations


def generate_random_schedule(environment: StudyPlannerEnvironment) -> Tuple[List[str], float]:
    environment.reset()
    total_reward = 0.0
    schedule: List[str] = []
    rng = random.Random(environment.config.seed)

    done = False
    while not done:
        action_index = rng.randrange(environment.num_actions)
        action_name = environment.actions[action_index]
        next_state, reward, done = environment.step(action_index)
        total_reward += reward
        schedule.append(action_name)

        if not done and next_state is not None:
            _ = next_state

    return schedule, total_reward


def generate_sarsa_greedy_schedule(
    environment: StudyPlannerEnvironment,
    agent: SARSAAgent,
) -> Tuple[List[str], float]:
    """Generate greedy schedule using SARSA-trained agent."""
    state = environment.reset()
    state_index = environment.state_to_index[state]
    total_reward = 0.0
    schedule: List[str] = []

    done = False
    while not done:
        action_index = agent.choose_action(state_index, greedy=True)
        action_name = environment.actions[action_index]
        next_state, reward, done = environment.step(action_index)
        total_reward += reward
        schedule.append(action_name)

        if not done and next_state is not None:
            state_index = environment.state_to_index[next_state]

    return schedule, total_reward


def format_q_table(environment: StudyPlannerEnvironment, agent: QLearningAgent) -> str:
    header = ["State"] + environment.actions
    lines = [" | ".join(f"{title:^14}" for title in header)]
    lines.append("-" * len(lines[0]))

    for state_index, state in enumerate(environment.state_space):
        slot_index, energy_level, last_action_type = state
        slot_name = environment.time_slots[slot_index]
        label = f"{slot_name} / E{energy_level} / {last_action_type}"
        q_values = [f"{value:>14.2f}" for value in agent.q_table[state_index]]
        lines.append(" | ".join([f"{label:<14}", *q_values]))

    return "\n".join(lines)
