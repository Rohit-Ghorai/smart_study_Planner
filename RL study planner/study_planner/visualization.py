from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(rewards: Sequence[float], save_path: str = "reward_vs_episodes.png", show: bool = False) -> str:
    reward_array = np.asarray(rewards, dtype=float)
    figure, axis = plt.subplots(figsize=(10, 5))

    axis.plot(reward_array, label="Episode reward", alpha=0.5, linewidth=1.2)

    if len(reward_array) >= 10:
        window = max(5, len(reward_array) // 20)
        moving_average = np.convolve(reward_array, np.ones(window) / window, mode="valid")
        axis.plot(
            range(window - 1, window - 1 + len(moving_average)),
            moving_average,
            label=f"Moving average ({window})",
            linewidth=2.2,
        )

    axis.set_title("Reward vs Episodes")
    axis.set_xlabel("Episodes")
    axis.set_ylabel("Total reward")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()

    output_path = Path(save_path)
    figure.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
    return str(output_path.resolve())


def plot_rewards_as_base64(rewards: Sequence[float]) -> str:
    reward_array = np.asarray(rewards, dtype=float)
    figure, axis = plt.subplots(figsize=(10, 5))

    axis.plot(reward_array, label="Episode reward", alpha=0.55, linewidth=1.2)

    if len(reward_array) >= 10:
        window = max(5, len(reward_array) // 20)
        moving_average = np.convolve(reward_array, np.ones(window) / window, mode="valid")
        axis.plot(
            range(window - 1, window - 1 + len(moving_average)),
            moving_average,
            label=f"Moving average ({window})",
            linewidth=2.2,
        )

    axis.set_title("Reward vs Episodes")
    axis.set_xlabel("Episodes")
    axis.set_ylabel("Total reward")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")
