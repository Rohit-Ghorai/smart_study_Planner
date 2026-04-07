import argparse
import json
import os
from typing import Any, Dict

from openai import OpenAI


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


def _build_client() -> OpenAI:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI-compatible inference using HF router.")
    parser.add_argument("prompt", help="User prompt text")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    result = run_inference(args.prompt, temperature=args.temperature)
    print(result)


if __name__ == "__main__":
    main()
