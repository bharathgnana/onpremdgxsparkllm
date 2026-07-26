#!/usr/bin/env python3
"""Measure local Qwen tool-call JSON validity over 50 deterministic requests."""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path(".bench/edge/tool-quality.json"))
    args = parser.parse_args()
    url = os.environ.get("QWEN_OPENAI_URL", "http://127.0.0.1:8000/v1/chat/completions")
    model = os.environ.get("QWEN_MODEL_ID", "qwen3-coder-30b")
    key = os.environ.get("QWEN_API_KEY", "local-qwen")
    valid = 0
    failures = []
    started = time.monotonic()
    for index in range(args.count):
        expected = f"src/module_{index}.py"
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": f"Inspect {expected}. You must call inspect_repository exactly once.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "inspect_repository",
                        "description": "Read one repository path.",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "inspect_repository"}},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.load(response)
            calls = body["choices"][0]["message"]["tool_calls"]
            arguments = json.loads(calls[0]["function"]["arguments"])
            if (
                len(calls) == 1
                and calls[0]["function"]["name"] == "inspect_repository"
                and arguments.get("path") == expected
            ):
                valid += 1
            else:
                failures.append({"index": index, "reason": "wrong call", "message": body})
        except Exception as exc:
            failures.append({"index": index, "reason": f"{type(exc).__name__}: {exc}"})
    rate = valid / args.count if args.count else 0
    result = {
        "requests": args.count,
        "valid": valid,
        "validity": rate,
        "required_validity": 0.98,
        "passed": rate >= 0.98,
        "wall_seconds": round(time.monotonic() - started, 3),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
