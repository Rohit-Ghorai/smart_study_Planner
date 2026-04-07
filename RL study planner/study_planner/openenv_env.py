from __future__ import annotations

from dataclasses import asdict, dataclass
from random import Random
from typing import Dict, List, Optional


@dataclass(frozen=True)
class OpenEnvTask:
    task_id: str
    title: str
    difficulty: str
    scenario: str
    max_steps: int
    pass_score: float
    action_rewards: Dict[str, float]


@dataclass
class OpenEnvStateModel:
    task_id: str
    title: str
    difficulty: str
    scenario: str
    step_count: int
    max_steps: int
    score: float
    done: bool
    action_space: List[str]


@dataclass
class OpenEnvStepResultModel:
    observation: OpenEnvStateModel
    reward: float
    score: float
    done: bool
    info: Dict[str, object]


class OpenEnvStudyPlanner:
    """Small real-world study planning environment used by OpenEnv checks."""

    ACTION_SPACE = [
        "diagnose_gaps",
        "prioritize_weak_subject",
        "schedule_deep_work",
        "add_revision_block",
        "add_mock_test",
        "ignore_signal",
    ]

    def __init__(self, seed: int = 42) -> None:
        self._rng = Random(seed)
        self._tasks: List[OpenEnvTask] = [
            OpenEnvTask(
                task_id="easy-01",
                title="Recover from one missed slot",
                difficulty="easy",
                scenario="Student missed one afternoon slot and exam is 10 days away.",
                max_steps=4,
                pass_score=0.65,
                action_rewards={
                    "diagnose_gaps": 0.24,
                    "prioritize_weak_subject": 0.28,
                    "add_revision_block": 0.20,
                    "schedule_deep_work": 0.14,
                    "add_mock_test": 0.10,
                    "ignore_signal": -0.22,
                },
            ),
            OpenEnvTask(
                task_id="medium-01",
                title="Balance fatigue and consistency",
                difficulty="medium",
                scenario="Student has low confidence in two subjects with repeated fatigue in evening sessions.",
                max_steps=5,
                pass_score=0.72,
                action_rewards={
                    "diagnose_gaps": 0.18,
                    "prioritize_weak_subject": 0.24,
                    "schedule_deep_work": 0.26,
                    "add_revision_block": 0.18,
                    "add_mock_test": 0.16,
                    "ignore_signal": -0.25,
                },
            ),
            OpenEnvTask(
                task_id="hard-01",
                title="Final-week exam rescue",
                difficulty="hard",
                scenario="Student is 3 days from exam, has inconsistent history, and needs high-yield recovery plan.",
                max_steps=6,
                pass_score=0.8,
                action_rewards={
                    "diagnose_gaps": 0.16,
                    "prioritize_weak_subject": 0.22,
                    "schedule_deep_work": 0.20,
                    "add_revision_block": 0.17,
                    "add_mock_test": 0.28,
                    "ignore_signal": -0.3,
                },
            ),
        ]

        self._active_task: Optional[OpenEnvTask] = None
        self._step_count = 0
        self._score = 0.0
        self._done = False

    def reset(self, task_id: str | None = None) -> OpenEnvStateModel:
        if task_id:
            selected = next((task for task in self._tasks if task.task_id == task_id), None)
            self._active_task = selected or self._tasks[0]
        else:
            self._active_task = self._rng.choice(self._tasks)

        self._step_count = 0
        self._score = 0.0
        self._done = False
        return self.state()

    def state(self) -> OpenEnvStateModel:
        if self._active_task is None:
            self.reset(self._tasks[0].task_id)

        assert self._active_task is not None
        return OpenEnvStateModel(
            task_id=self._active_task.task_id,
            title=self._active_task.title,
            difficulty=self._active_task.difficulty,
            scenario=self._active_task.scenario,
            step_count=self._step_count,
            max_steps=self._active_task.max_steps,
            score=round(self._score, 4),
            done=self._done,
            action_space=list(self.ACTION_SPACE),
        )

    def step(self, action: str) -> OpenEnvStepResultModel:
        if self._active_task is None:
            self.reset(self._tasks[0].task_id)

        assert self._active_task is not None

        if self._done:
            current_state = self.state()
            return OpenEnvStepResultModel(
                observation=current_state,
                reward=0.0,
                score=current_state.score,
                done=True,
                info={"reason": "episode_already_done"},
            )

        base_reward = self._active_task.action_rewards.get(action, -0.1)
        self._step_count += 1
        self._score = max(0.0, min(1.0, self._score + base_reward))

        if self._score >= 1.0 or self._step_count >= self._active_task.max_steps:
            self._done = True

        state = self.state()
        return OpenEnvStepResultModel(
            observation=state,
            reward=round(base_reward, 4),
            score=state.score,
            done=state.done,
            info={
                "pass_score": self._active_task.pass_score,
                "passed": state.score >= self._active_task.pass_score,
                "remaining_steps": max(0, self._active_task.max_steps - self._step_count),
            },
        )

    def tasks(self) -> List[Dict[str, object]]:
        return [
            {
                "task_id": task.task_id,
                "title": task.title,
                "difficulty": task.difficulty,
                "scenario": task.scenario,
                "max_steps": task.max_steps,
                "pass_score": task.pass_score,
                "agent_grader": "heuristic_reward_grader",
            }
            for task in self._tasks
        ]


def state_to_dict(state: OpenEnvStateModel) -> Dict[str, object]:
    return asdict(state)


def step_to_dict(result: OpenEnvStepResultModel) -> Dict[str, object]:
    payload = asdict(result)
    payload["observation"] = asdict(result.observation)
    return payload
