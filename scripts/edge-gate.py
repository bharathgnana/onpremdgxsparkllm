#!/usr/bin/env python3
"""Evaluate a bench-edge JSON artifact against the approved latency gates."""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text())
    failures = []
    saw_qwen_20k = False
    saw_glm_2k = False
    for row in payload.get("results", []):
        lane = row["lane"]
        requested = row["requested_input_tokens"]
        if lane == "qwen" and requested >= 20000:
            saw_qwen_20k = True
            if row["ttft_seconds"] > 30:
                failures.append(f"Qwen 20k TTFT {row['ttft_seconds']}s > 30s")
            if row["decode_tok_s"] < 15:
                failures.append(f"Qwen decode {row['decode_tok_s']} < 15 tok/s")
        if lane == "glm" and requested >= 2000:
            saw_glm_2k = True
            if row["ttft_seconds"] > 600:
                failures.append(f"GLM 2k TTFT {row['ttft_seconds']}s > 600s")
            if row["decode_tok_s"] < 3:
                failures.append(f"GLM decode {row['decode_tok_s']} < 3 tok/s")
        available = row.get("memory_after", {}).get("MemAvailable", 0)
        if available < 10:
            failures.append(f"{lane} memory reserve {available} GiB < 10 GiB")
    if not saw_qwen_20k:
        failures.append("missing required Qwen 20k benchmark row")
    if not saw_glm_2k:
        failures.append("missing required GLM 2k benchmark row")
    if failures:
        print("EDGE GATE: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("EDGE GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
