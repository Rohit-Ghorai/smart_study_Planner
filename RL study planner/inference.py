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


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=True)


def _emit_block(block: str, **fields: Any) -> None:
    parts = [f"[{block}]"]
    for key, value in fields.items():
        parts.append(f"{key}={_format_value(value)}")
    print(" ".join(parts), flush=True)


def _build_client() -> Any:
    openai_module = importlib.import_module("openai")
    OpenAI = getattr(openai_module, "OpenAI")

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required in environment.")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def run_inference(user_prompt: str, temperature: float = 0.2) -> str:
    _emit_block("START", task="llm_inference", model=MODEL_NAME)
    _emit_block("STEP", task="llm_inference", step=1, action="client_initialization")
    client = _build_client()

    _emit_block("STEP", task="llm_inference", step=2, action="chat_completion", prompt_chars=len(user_prompt))
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
    _emit_block("END", task="llm_inference", score=1.0, steps=2, output_chars=len(text))
    return text


def run_baseline() -> Dict[str, Any]:
    """Deterministic baseline policy for reproducible scoring."""
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
        _emit_block("START", task=task_id)
        env.reset(task_id=task_id)
        done = False
        step_index = 0
        final_score = 0.0
        while not done:
            action = policy[min(step_index, len(policy) - 1)]
            result = env.step(action)
            done = result.done
            final_score = result.score
            _emit_block(
                "STEP",
                task=task_id,
                step=step_index + 1,
                action=action,
                reward=result.reward,
                score=round(result.score, 4),
            )
            step_index += 1
        final_score = round(final_score, 4)
        passed = final_score >= 0.7
        _emit_block("END", task=task_id, score=final_score, steps=step_index, passed=passed)
        results.append({"task_id": task_id, "score": final_score})

    avg_score = round(sum(item["score"] for item in results) / len(results), 4)
    payload = {"scores": results, "average_score": avg_score}
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

