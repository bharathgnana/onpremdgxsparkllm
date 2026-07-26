#!/usr/bin/env python3
"""Claude UserPromptSubmit hook: inject the newest unread GLM review."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("glm_review", ROOT / "scripts/glm_review.py")
review = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(review)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        repo = Path(payload.get("cwd", ".")).resolve()
        top = review.run_git(repo, "rev-parse", "--show-toplevel").strip()
    except Exception:
        return 0
    for path in reversed(review.job_paths()):
        job = review.load_job(path)
        if (
            job.get("repo") != top
            or job.get("status") not in {"completed", "stale"}
            or job.get("notified_at")
            or not job.get("result")
        ):
            continue
        result = Path(job["result"])
        if not result.exists():
            continue
        content = result.read_text(errors="replace")[:6000]
        job["notified_at"] = review.utcnow()
        job["updated_at"] = review.utcnow()
        review.atomic_json(path, job)
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": (
                            "An asynchronous local GLM-5.2 review completed. "
                            "Treat it as advisory and verify its claims:\n\n" + content
                        ),
                    }
                }
            )
        )
        break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
