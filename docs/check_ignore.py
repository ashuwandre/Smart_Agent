import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "replace_with_your_openai_api_key":
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add a real key to the project-root .env file."
        )

    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        print("Credentials accepted.")
        print(f"First available model: {models.data[0].id}")

        model = os.getenv("PLANNER_MODEL", "gpt-4o-mini")
        response = client.responses.create(
            model=model,
            input="Reply with OK only.",
            max_output_tokens=16,
        )
        print(f"Live inference is working with {model}: {response.output_text}")
    except Exception as exc:
        print(type(exc).__name__)
        status_code = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None) or {}
        error = body.get("error", body) if isinstance(body, dict) else {}
        error_code = error.get("code") if isinstance(error, dict) else None
        error_message = (
            error.get("message")
            if isinstance(error, dict)
            else str(exc)
        )
        print(f"Status: {status_code}")
        print(f"Code: {error_code}")
        print(error_message or str(exc))

        if error_code == "insufficient_quota":
            print(
                "The key is valid, but its OpenAI project has no usable quota. "
                "Add billing/credits or use a key from a funded project."
            )


if __name__ == "__main__":
    main()