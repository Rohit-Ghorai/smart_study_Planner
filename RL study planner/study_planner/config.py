from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_TIME_SLOTS: Tuple[str, ...] = ("Morning", "Afternoon", "Evening")


@dataclass
class StudyPlannerConfig:
    subjects: List[str]
    difficulties: Dict[str, int]
    strengths: Dict[str, int] = field(default_factory=dict)
    exam_date: Optional[date] = None
    time_slots: Tuple[str, ...] = DEFAULT_TIME_SLOTS
    episodes: int = 1500
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    seed: int = 42

    @classmethod
    def from_input(
        cls,
        subjects: Sequence[str],
        difficulties: Sequence[int],
        strengths: Sequence[int] | None = None,
        exam_date_text: str | None = None,
        episodes: int = 1500,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        seed: int = 42,
    ) -> "StudyPlannerConfig":
        parsed_subjects = [subject.strip() for subject in subjects if subject.strip()]
        difficulty_map = {
            subject: max(1, min(5, int(level))) for subject, level in zip(parsed_subjects, difficulties)
        }
        strength_values = strengths or []
        strength_map = {
            subject: max(1, min(5, int(level))) for subject, level in zip(parsed_subjects, strength_values)
        }

        exam_date = None
        if exam_date_text:
            try:
                exam_date = datetime.strptime(exam_date_text.strip(), "%Y-%m-%d").date()
            except ValueError:
                exam_date = None

        return cls(
            subjects=parsed_subjects,
            difficulties=difficulty_map,
            strengths=strength_map,
            exam_date=exam_date,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            seed=seed,
        )

    def exam_urgency_bonus(self) -> float:
        if self.exam_date is None:
            return 0.0

        days_left = (self.exam_date - date.today()).days
        if days_left <= 0:
            return 3.0
        if days_left <= 3:
            return 3.0
        if days_left <= 7:
            return 2.0
        if days_left <= 14:
            return 1.0
        return 0.0
