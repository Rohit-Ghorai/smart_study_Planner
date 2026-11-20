from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


class ModelValidationError(ValueError):
    """Raised when a serialized Q-table cannot be trusted or used."""


def save_q_table(
    file_path: str,
    q_table: np.ndarray,
    state_labels: List[str],
    action_labels: List[str],
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    table = np.asarray(q_table)
    if table.ndim != 2 or not np.issubdtype(table.dtype, np.number):
        raise ModelValidationError("Q-table must be a two-dimensional numeric array.")
    if not np.isfinite(table).all():
        raise ModelValidationError("Q-table contains NaN or infinite values.")
    if len(state_labels) != table.shape[0] or len(action_labels) != table.shape[1]:
        raise ModelValidationError("Model labels do not match Q-table dimensions.")

    output_path = Path(file_path)
    np.savez_compressed(
        output_path,
        format_version=np.asarray([2], dtype=np.int64),
        q_table=table.astype(np.float64, copy=False),
        state_labels=np.asarray(state_labels, dtype=np.str_),
        action_labels=np.asarray(action_labels, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(dict(metadata or {}), sort_keys=True), dtype=np.str_),
    )
    return str(output_path.resolve())


def load_q_table(
    file_path: str,
    expected_shape: Optional[tuple[int, int]] = None,
    expected_actions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    try:
        with np.load(file_path, allow_pickle=False) as data:
            required = {"q_table", "state_labels", "action_labels"}
            missing = required.difference(data.files)
            if missing:
                raise ModelValidationError(f"Model is missing fields: {', '.join(sorted(missing))}.")
            q_table = np.asarray(data["q_table"])
            state_labels = np.asarray(data["state_labels"])
            action_labels = np.asarray(data["action_labels"])
            metadata: Dict[str, Any] = {}
            if "metadata_json" in data.files:
                try:
                    metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ModelValidationError("Model metadata is invalid.") from exc
    except (OSError, ValueError) as exc:
        if isinstance(exc, ModelValidationError):
            raise
        raise ModelValidationError("The uploaded model is not a valid safe NumPy archive.") from exc

    if q_table.ndim != 2 or not np.issubdtype(q_table.dtype, np.number):
        raise ModelValidationError("Q-table must be a two-dimensional numeric array.")
    if not np.isfinite(q_table).all():
        raise ModelValidationError("Q-table contains NaN or infinite values.")
    if state_labels.ndim != 1 or action_labels.ndim != 1:
        raise ModelValidationError("Model labels must be one-dimensional.")
    if q_table.shape != (len(state_labels), len(action_labels)):
        raise ModelValidationError("Model labels do not match Q-table dimensions.")
    if expected_shape is not None and q_table.shape != expected_shape:
        raise ModelValidationError(
            f"Model shape {q_table.shape} is incompatible with expected shape {expected_shape}."
        )
    labels = [str(label) for label in action_labels.tolist()]
    if expected_actions is not None and labels != expected_actions:
        raise ModelValidationError("Model action mapping is incompatible with this planner.")

    return {
        "q_table": q_table.astype(np.float64, copy=False),
        "state_labels": np.asarray([str(label) for label in state_labels.tolist()], dtype=np.str_),
        "action_labels": np.asarray(labels, dtype=np.str_),
        "metadata": metadata,
    }
