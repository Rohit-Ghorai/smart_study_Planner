from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QLearningAgent:
    num_states: int
    num_actions: int
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)
        self.rng = np.random.default_rng(self.seed)

    def choose_action(self, state_index: int, greedy: bool = False) -> int:
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))

        row = self.q_table[state_index]
        best_value = np.max(row)
        best_actions = np.flatnonzero(row == best_value)
        return int(self.rng.choice(best_actions))

    def learn(self, state_index: int, action_index: int, reward: float, next_state_index: int | None, done: bool) -> None:
        current_q = self.q_table[state_index, action_index]
        next_best = 0.0 if done or next_state_index is None else float(np.max(self.q_table[next_state_index]))
        target = reward if done else reward + self.gamma * next_best
        self.q_table[state_index, action_index] = current_q + self.alpha * (target - current_q)

    def decay_exploration(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


@dataclass
class SARSAAgent:
    """State-Action-Reward-State-Action (SARSA) - on-policy learning algorithm."""
    num_states: int
    num_actions: int
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)
        self.rng = np.random.default_rng(self.seed)

    def choose_action(self, state_index: int, greedy: bool = False) -> int:
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))

        row = self.q_table[state_index]
        best_value = np.max(row)
        best_actions = np.flatnonzero(row == best_value)
        return int(self.rng.choice(best_actions))

    def learn(self, state_index: int, action_index: int, reward: float, next_state_index: int | None, next_action_index: int | None, done: bool) -> None:
        """SARSA update: uses the actual next action taken, not the greedy action."""
        current_q = self.q_table[state_index, action_index]
        next_q = 0.0
        
        if not done and next_state_index is not None and next_action_index is not None:
            next_q = float(self.q_table[next_state_index, next_action_index])
        
        target = reward if done else reward + self.gamma * next_q
        self.q_table[state_index, action_index] = current_q + self.alpha * (target - current_q)

    def decay_exploration(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

