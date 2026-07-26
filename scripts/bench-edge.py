#!/usr/bin/env python3
"""Measure Anthropic endpoints against the hybrid edge latency gates."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ((128, 32), (2000, 32), (20000, 256))


def meminfo() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value, *_ = line.replace(":", "").split()
        if key in {"MemAvailable", "SwapFree", "SwapTotal"}:
            values[key] = round(int(value) / 1024 / 1024, 3)
    return values


def filler(tokens: int) -> str:
    phrase = "Reviewing code requires correctness security tests and compatibility. "
    return (phrase * ((tokens * 4 // len(phrase)) + 1))[: tokens * 4]


def probe(name: str, base: str, model: str, key: str, input_tokens: int, output_tokens: int) -> dict:
    body = {
        "model": model,
        "stream": True,
        "temperature": 0,
        "max_tokens": output_tokens,
        "messages": [
            {
                "role": "user",
                "content": filler(input_tokens)
                + "\nReturn a concise numbered code-review checklist.",
            }
        ],
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/messages",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": key},
        method="POST",
    )
    started = time.monotonic()
    first = None
    usage: dict = {}
    text = []
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                chunk = delta.get("text", "")
                if chunk:
                    first = first or time.monotonic()
                    text.append(chunk)
            if event.get("type") == "message_start":
                usage.update(event.get("message", {}).get("usage", {}))
            if event.get("type") == "message_delta":
                usage.update(event.get("usage", {}))
    ended = time.monotonic()
    out = int(usage.get("output_tokens") or max(1, len("".join(text)) // 4))
    decode_seconds = max(ended - (first or ended), 0.001)
    return {
        "lane": name,
        "requested_input_tokens": input_tokens,
        "requested_output_tokens": output_tokens,
        "reported_usage": usage,
        "ttft_seconds": round((first or ended) - started, 3),
        "wall_seconds": round(ended - started, 3),
        "decode_tok_s": round(out / decode_seconds, 3),
        "response_chars": len("".join(text)),
        "memory_after": meminfo(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane", choices=("qwen", "glm", "both"), default="both")
    p.add_argument("--quick", action="store_true", help="Run only the 128-token case.")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    lanes = {
        "qwen": ("http://127.0.0.1:8000", os.environ.get("QWEN_MODEL_ID", "qwen3-coder-30b"),
                 os.environ.get("QWEN_API_KEY", "local-qwen")),
        "glm": ("http://127.0.0.1:8001", os.environ.get("COLI_MODEL_ID", "glm-5.2-colibri"),
                os.environ.get("COLI_API_KEY", "")),
    }
    selected = lanes if args.lane == "both" else {args.lane: lanes[args.lane]}
    cases = DEFAULT_CASES[:1] if args.quick else DEFAULT_CASES
    results = []
    for lane, (base, model, key) in selected.items():
        lane_cases = cases if lane == "qwen" else tuple(c for c in cases if c[0] <= 2000)
        for ins, outs in lane_cases:
            print(f"{lane}: input≈{ins}, output≤{outs}", flush=True)
            result = probe(lane, base, model, key, ins, outs)
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "gates": {
            "qwen_20k": {"ttft_seconds_max": 30, "decode_tok_s_min": 15},
            "glm_2k": {"ttft_seconds_max": 600, "decode_tok_s_min": 3},
            "mem_available_gib_min": 10,
        },
    }
    target = args.output or ROOT / ".bench/edge" / f"{int(time.time())}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
