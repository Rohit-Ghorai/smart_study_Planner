from __future__ import annotations

from typing import Any, Dict

from flask import Flask, request, session
from secrets import token_urlsafe

from study_planner.openenv_env import OpenEnvStudyPlanner, state_to_dict, step_to_dict


def create_server_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = token_urlsafe(32)
    environments: Dict[str, OpenEnvStudyPlanner] = {}

    def current_environment() -> OpenEnvStudyPlanner:
        environment_id = session.get("openenv_id")
        if not isinstance(environment_id, str):
            environment_id = token_urlsafe(16)
            session["openenv_id"] = environment_id
        if environment_id not in environments:
            environments[environment_id] = OpenEnvStudyPlanner(seed=42)
            environments[environment_id].reset("easy-01")
        return environments[environment_id]

    @app.post("/reset")
    def reset() -> tuple[Dict[str, Any], int]:
        payload = request.get_json(silent=True) or {}
        task_id = payload.get("task_id") if isinstance(payload, dict) else None
        state = current_environment().reset(task_id=task_id)
        return {"ok": True, "observation": state_to_dict(state)}, 200

    @app.get("/state")
    def state() -> tuple[Dict[str, Any], int]:
        return {"ok": True, "observation": state_to_dict(current_environment().state())}, 200

    @app.post("/step")
    def step() -> tuple[Dict[str, Any], int]:
        payload = request.get_json(silent=True) or {}
        action = ""
        if isinstance(payload, dict):
            action = str(payload.get("action", "")).strip()
        if not action:
            return {"ok": False, "error": "Missing action"}, 400
        try:
            result = current_environment().step(action)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}, 400
        return {"ok": True, **step_to_dict(result)}, 200

    @app.get("/validate")
    def validate() -> tuple[Dict[str, Any], int]:
        return {
            "ok": True,
            "service": "smart-study-planner-openenv",
            "status": "ready",
            "tasks": env.tasks(),
            "reward_range": [0.0, 1.0],
        }, 200

    return app


app = create_server_app()


def main() -> None:
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
