from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .agent import QLearningAgent
from .config import StudyPlannerConfig
from .environment import StudyPlannerEnvironment
from .trainer import generate_greedy_schedule, train_agent
from .visualization import plot_rewards


class StudyPlannerGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Reinforcement Learning Based Smart Study Planner")
        self.root.geometry("920x680")
        self.root.minsize(860, 620)

        self._build_interface()

    def _build_interface(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        title = ttk.Label(container, text="Reinforcement Learning Based Smart Study Planner", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w", pady=(0, 12))

        form = ttk.Frame(container)
        form.pack(fill="x")

        self.subjects_var = tk.StringVar(value="OS, Java, DSA")
        self.difficulties_var = tk.StringVar(value="5, 4, 4")
        self.strengths_var = tk.StringVar(value="3, 3, 3")
        self.exam_date_var = tk.StringVar(value="")
        self.episodes_var = tk.StringVar(value="1500")

        fields = [
            ("Subjects", self.subjects_var),
            ("Difficulty levels", self.difficulties_var),
            ("Confidence levels", self.strengths_var),
            ("Exam date (YYYY-MM-DD)", self.exam_date_var),
            ("Training episodes", self.episodes_var),
        ]

        for row, (label_text, variable) in enumerate(fields):
            ttk.Label(form, text=label_text).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=6)
            ttk.Entry(form, textvariable=variable, width=52).grid(row=row, column=1, sticky="ew", pady=6)

        form.columnconfigure(1, weight=1)

        button_row = ttk.Frame(container)
        button_row.pack(fill="x", pady=(14, 8))

        ttk.Button(button_row, text="Train Planner", command=self._train).pack(side="left")
        ttk.Button(button_row, text="Quit", command=self.root.destroy).pack(side="left", padx=8)

        self.output = tk.Text(container, height=22, wrap="word")
        self.output.pack(fill="both", expand=True, pady=(10, 0))

    def _parse_csv(self, value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    def _train(self) -> None:
        try:
            subjects = self._parse_csv(self.subjects_var.get())
            difficulties = [int(item) for item in self._parse_csv(self.difficulties_var.get())]
            strengths = [int(item) for item in self._parse_csv(self.strengths_var.get())]
            episodes = int(self.episodes_var.get().strip() or "1500")

            config = StudyPlannerConfig.from_input(
                subjects=subjects,
                difficulties=difficulties,
                strengths=strengths,
                exam_date_text=self.exam_date_var.get().strip(),
                episodes=episodes,
            )

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

            rewards = train_agent(environment, agent, config.episodes, log_interval=0)
            schedule, reward = generate_greedy_schedule(environment, agent)
            plot_path = plot_rewards(rewards, show=False)

            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, "Optimized schedule:\n")
            for slot, action in zip(environment.time_slots, schedule):
                self.output.insert(tk.END, f"{slot:<10} -> {action}\n")
            self.output.insert(tk.END, f"\nTotal reward: {reward:.2f}\n")
            self.output.insert(tk.END, f"Reward graph saved to: {plot_path}\n")
        except Exception as exc:  # pragma: no cover - UI feedback only
            messagebox.showerror("Training failed", str(exc))

    def run(self) -> None:
        self.root.mainloop()


def run_gui() -> None:
    StudyPlannerGUI().run()
