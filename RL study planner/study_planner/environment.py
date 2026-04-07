from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import StudyPlannerConfig


# Enhanced state: (time_slot, energy_level, action_type)
# energy_level: 1-5 (1=exhausted, 5=fresh)
# action_type: "Idle", "Study", "Review", "Mock", "Break"
State = Tuple[int, int, str]


@dataclass
class EpisodeSnapshot:
    total_reward: float
    actions: List[str]


@dataclass
class SubjectProgress:
    """Track per-subject study progress during an episode."""
    deep_study_count: int = 0
    revision_count: int = 0
    mock_test_count: int = 0
    total_effort: float = 0.0


class StudyPlannerEnvironment:
    """Enhanced environment with rich state space and diverse action types."""
    
    def __init__(self, config: StudyPlannerConfig):
        self.config = config
        self.time_slots = list(config.time_slots)
        self.subjects = list(config.subjects)
        
        # Actions: Deep Study, Revision, Mock Test per subject + special breaks
        self.base_actions = self.subjects  # Basic study actions (1 slot)
        self.deep_study_actions = [f"{s}_Deep" for s in self.subjects]  # Deep study (1.5 slots)
        self.revision_actions = [f"{s}_Rev" for s in self.subjects]  # Light revision (0.5 slots)
        self.mock_test_actions = [f"{s}_Mock" for s in self.subjects]  # Full mock exam (1 slot, high value)
        self.break_actions = ["Meditation", "Snack", "Water_Break"]  # Different break types
        
        self.actions = self.base_actions + self.deep_study_actions + self.revision_actions + self.mock_test_actions + self.break_actions
        
        # Enhanced state space: (time_slot, energy_level, last_action_type)
        energy_levels = range(1, 6)  # 1 to 5
        last_action_types = ["Idle", "Study", "Review", "Mock", "Break"]
        
        self.state_space: List[State] = [
            (slot_index, energy, action_type)
            for slot_index in range(len(self.time_slots))
            for energy in energy_levels
            for action_type in last_action_types
        ]
        self.state_to_index: Dict[State, int] = {state: index for index, state in enumerate(self.state_space)}
        self.action_to_index: Dict[str, int] = {action: index for index, action in enumerate(self.actions)}

        self.reset()

    @property
    def num_states(self) -> int:
        return len(self.state_space)

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    def reset(self) -> State:
        """Reset environment for new episode."""
        self.current_slot_index = 0
        self.energy_level = 5  # Start fresh
        self.last_action_type = "Idle"
        self.break_count = 0
        self.total_study_effort = 0.0
        self.subject_progress: Dict[str, SubjectProgress] = {
            subject: SubjectProgress() for subject in self.subjects
        }
        return self._current_state()

    def _current_state(self) -> State:
        return (self.current_slot_index, self.energy_level, self.last_action_type)

    def _parse_action(self, action: str) -> Tuple[str, str]:
        """Parse action into (subject, action_type).
        
        Returns: (subject, type) where type in ["Study", "Deep", "Review", "Mock", "Break"]
        """
        if action in self.break_actions:
            return ("None", "Break")
        if "_Mock" in action:
            return (action.replace("_Mock", ""), "Mock")
        if "_Deep" in action:
            return (action.replace("_Deep", ""), "Deep")
        if "_Rev" in action:
            return (action.replace("_Rev", ""), "Review")
        return (action, "Study")

    def _subject_reward(self, subject: str, action_type: str) -> float:
        """Calculate reward for studying a subject with specific action type."""
        difficulty = self.config.difficulties.get(subject, 3)
        strength = self.config.strengths.get(subject, 3)
        progress = self.subject_progress.get(subject, SubjectProgress())

        # Base rewards by action type
        if action_type == "Deep":
            base_reward = 8.0  # Highest reward for deep study
        elif action_type == "Mock":
            base_reward = 10.0  # Highest value: tests knowledge
        elif action_type == "Review":
            base_reward = 3.0  # Lower reward for light review
        else:  # Regular study
            base_reward = 5.0

        # Difficulty bonus
        difficulty_bonus = float(difficulty)

        # Adaptive bonus: high bonus if low confidence
        adaptive_bonus = float(np.clip(difficulty - strength, -3, 3))

        # Time pressure bonus (more slots remaining = more bonus)
        time_bonus = float(max(0, len(self.time_slots) - self.current_slot_index)) * 0.3

        # Exam urgency bonus
        exam_bonus = self.config.exam_urgency_bonus()

        # Diminishing returns if studied same subject too much
        if action_type == "Deep":
            diminishing = -(progress.deep_study_count * 0.5)
        elif action_type == "Mock":
            diminishing = -(progress.mock_test_count * 1.0)
        elif action_type == "Review":
            diminishing = -(progress.revision_count * 0.2)
        else:
            diminishing = 0.0

        # Energy penalty if too tired
        energy_penalty = 0.0 if self.energy_level >= 3 else (3 - self.energy_level) * -1.5

        total_reward = base_reward + difficulty_bonus + adaptive_bonus + time_bonus + exam_bonus + diminishing + energy_penalty

        # Update progress
        if action_type == "Deep":
            progress.deep_study_count += 1
            progress.total_effort += 1.5
        elif action_type == "Mock":
            progress.mock_test_count += 1
            progress.total_effort += 1.0
        elif action_type == "Review":
            progress.revision_count += 1
            progress.total_effort += 0.5
        else:
            progress.total_effort += 1.0

        return max(0.0, total_reward)

    def _break_reward(self, break_type: str) -> float:
        """Calculate reward for taking a break."""
        reward = 1.0 if self.last_action_type == "Study" else 0.0

        # Penalty if too many breaks
        if self.break_count >= 2:
            reward -= 3.0

        # Penalty if no study done yet
        if self.total_study_effort == 0:
            reward -= 2.0

        # Bonus for meditation (stress relief)
        if break_type == "Meditation":
            reward += 1.5

        # Snack break is okay but not great
        if break_type == "Snack":
            reward += 0.5

        return reward

    def step(self, action_index: int) -> Tuple[Optional[State], float, bool]:
        """Execute one action and return (next_state, reward, done)."""
        if self.current_slot_index >= len(self.time_slots):
            return None, 0.0, True

        action = self.actions[action_index]
        subject, action_type = self._parse_action(action)

        # Calculate reward
        if action_type == "Break":
            reward = self._break_reward(subject)
            self.break_count += 1
            self.energy_level = min(5, self.energy_level + 2)  # Restore energy
            self.last_action_type = "Break"
        else:
            reward = self._subject_reward(subject, action_type)
            self.total_study_effort += 1.0
            self.energy_level = max(1, self.energy_level - 1)  # Consume energy
            # Normalize all study types to "Study" in state space
            self.last_action_type = "Study"

        # Advance time based on action intensity
        if action_type == "Deep":
            time_step = 2  # Deep study takes 1.5 slots (rounds to 2 for discrete)
        elif action_type == "Mock":
            time_step = 1  # Mock test takes 1 slot
        elif action_type == "Review":
            time_step = 1  # Review takes ~1 slot
        else:
            time_step = 1

        self.current_slot_index += time_step

        done = self.current_slot_index >= len(self.time_slots)
        next_state = None if done else self._current_state()
        return next_state, reward, done

    def simulate_schedule(self, action_names: Sequence[str]) -> EpisodeSnapshot:
        """Simulate a fixed schedule and return results."""
        self.reset()
        total_reward = 0.0
        executed_actions: List[str] = []

        for action_name in action_names:
            if action_name not in self.action_to_index:
                continue
            action_index = self.action_to_index[action_name]
            _, reward, done = self.step(action_index)
            total_reward += reward
            executed_actions.append(action_name)
            if done:
                break

        return EpisodeSnapshot(total_reward=total_reward, actions=executed_actions)
