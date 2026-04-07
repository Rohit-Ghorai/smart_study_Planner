import argparse
import importlib
import json
import os
from typing import Any, Dict

from study_planner.openenv_env import OpenEnvStudyPlanner


# Required environment variables for submission.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if using from_docker_image().
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def _log(stage: str, message: str, **extra: Any) -> None:
    payload: Dict[str, Any] = {"message": message}
    if extra:
        payload["data"] = extra
    print(f"{stage} {json.dumps(payload, ensure_ascii=True)}", flush=True)


def _build_client() -> Any:
    openai_module = importlib.import_module("openai")
    OpenAI = getattr(openai_module, "OpenAI")

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required in environment.")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def run_inference(user_prompt: str, temperature: float = 0.2) -> str:
    _log("START", "inference_started", model=MODEL_NAME)
    _log("STEP", "client_initialization")
    client = _build_client()

    _log("STEP", "sending_chat_completion", prompt_chars=len(user_prompt))
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise and practical assistant. "
                    "Return clean, plain-text output only."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    text = (response.choices[0].message.content or "").strip()
    _log("END", "inference_completed", output_chars=len(text))
    return text


def run_baseline() -> Dict[str, Any]:
    """Deterministic baseline policy for reproducible scoring."""
    _log("START", "baseline_started")
    env = OpenEnvStudyPlanner(seed=42)
    task_order = ["easy-01", "medium-01", "hard-01"]
    policy = [
        "diagnose_gaps",
        "prioritize_weak_subject",
        "schedule_deep_work",
        "add_revision_block",
        "add_mock_test",
    ]

    results = []
    for task_id in task_order:
        _log("STEP", "task_started", task_id=task_id)
        env.reset(task_id=task_id)
        done = False
        step_index = 0
        final_score = 0.0
        while not done:
            action = policy[min(step_index, len(policy) - 1)]
            result = env.step(action)
            done = result.done
            final_score = result.score
            step_index += 1
        results.append({"task_id": task_id, "score": round(final_score, 4)})

    avg_score = round(sum(item["score"] for item in results) / len(results), 4)
    payload = {"scores": results, "average_score": avg_score}
    _log("END", "baseline_completed", average_score=avg_score)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI-compatible inference using HF router.")
    parser.add_argument("prompt", nargs="?", default="Give a short study strategy for exam week.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--mode",
        choices=["baseline", "llm"],
        default="baseline",
        help="baseline: deterministic scoring; llm: OpenAI-compatible completion",
    )
    args = parser.parse_args()

    if args.mode == "llm":
        result = run_inference(args.prompt, temperature=args.temperature)
        print(result)
        return

    baseline = run_baseline()
    print(json.dumps(baseline, ensure_ascii=True))


if __name__ == "__main__":
    main()
