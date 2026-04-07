from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


def save_q_table(file_path: str, q_table: np.ndarray, state_labels: List[str], action_labels: List[str]) -> str:
    output_path = Path(file_path)
    np.savez_compressed(
        output_path,
        q_table=q_table,
        state_labels=np.asarray(state_labels, dtype=object),
        action_labels=np.asarray(action_labels, dtype=object),
    )
    return str(output_path.resolve())


def load_q_table(file_path: str) -> Dict[str, np.ndarray]:
    data = np.load(file_path, allow_pickle=True)
    return {
        "q_table": data["q_table"],
        "state_labels": data["state_labels"],
        "action_labels": data["action_labels"],
    }
