#!/usr/bin/env python3
"""Asynchronous local Qwen -> GLM code-review queue (stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get("GLM_REVIEW_STATE", Path.home() / ".local/state/onprem-glm-review"))
JOBS = STATE / "jobs"
QWEN_URL = os.environ.get("QWEN_OPENAI_URL", "http://127.0.0.1:8000/v1/chat/completions")
GLM_URL = os.environ.get("GLM_ANTHROPIC_URL", "http://127.0.0.1:8001/v1/messages")
QWEN_MODEL = os.environ.get("QWEN_MODEL_ID", "qwen3-coder-30b")
GLM_MODEL = os.environ.get("COLI_MODEL_ID", "glm-5.2-colibri")
MIN_AVAILABLE_GIB = float(os.environ.get("GLM_REVIEW_MIN_AVAILABLE_GIB", "10"))
MAX_DIFF_CHARS = int(os.environ.get("GLM_REVIEW_MAX_DIFF_CHARS", "24000"))
MAX_PACKET_CHARS = int(os.environ.get("GLM_REVIEW_MAX_PACKET_CHARS", "7600"))
MAX_REVIEW_TOKENS = int(os.environ.get("GLM_REVIEW_MAX_TOKENS", "192"))

SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|auth[_-]?token|password|passwd|secret|hf_token)"
    r"(\s*[=:]\s*)([^\s\"']+)"
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def diff_text(repo: Path) -> str:
    # Never materialize an unbounded repository-wide patch in a Claude hook.
    summary = run_git(repo, "diff", "--stat", "--summary", "HEAD")
    changed = run_git(
        repo, "diff", "--name-only", "--diff-filter=ACMRT", "HEAD"
    ).splitlines()
    patches: list[str] = []
    remaining = MAX_DIFF_CHARS
    for name in changed[:40]:
        if remaining <= 0:
            break
        patch = run_git(repo, "diff", "--no-ext-diff", "HEAD", "--", name)
        fragment = patch[:remaining]
        patches.append(fragment)
        remaining -= len(fragment)
    names = run_git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    extras: list[str] = []
    for name in names[:30]:
        path = repo / name
        if not path.is_file() or path.stat().st_size > 128_000:
            continue
        try:
            text = path.read_text(errors="strict")
        except (OSError, UnicodeError):
            continue
        if remaining <= 0:
            break
        fragment = f"\n--- /dev/null\n+++ b/{name}\n{text}"[:remaining]
        extras.append(fragment)
        remaining -= len(fragment)
    return summary + "\n" + "".join(patches) + "".join(extras)


def redact(text: str) -> str:
    text = SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}<redacted>", text)
    return "\n".join(
        "<redacted sensitive line>"
        if re.search(r"(?i)BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY", line)
        else line
        for line in text.splitlines()
    )


def snapshot(repo: Path) -> tuple[str, str]:
    text = redact(diff_text(repo))
    return hashlib.sha256(text.encode()).hexdigest(), text


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def qwen_is_busy() -> bool:
    metrics = QWEN_URL.split("/v1/", 1)[0] + "/metrics"
    try:
        with urllib.request.urlopen(metrics, timeout=2) as response:
            text = response.read().decode(errors="replace")
        match = re.search(r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.]+)", text, re.M)
        return bool(match and float(match.group(1)) > 0)
    except Exception:
        return False


def post_json(url: str, payload: dict, headers: dict[str, str], timeout: int) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def compact_with_qwen(job: dict, diff: str) -> str:
    clipped = diff[:MAX_DIFF_CHARS]
    prompt = f"""Create a compact packet for a senior code reviewer.
Do not review the code. State intent inferred from the diff, changed behavior,
risks, tests visible in the diff, and include only the most important diff
fragments. Never reproduce credentials. Keep the entire response under 1200
tokens.

Repository: {job['repo']}
Requested task: {job.get('task') or 'Review the current worktree changes.'}
Diff:
{clipped}
"""
    try:
        body = post_json(
            QWEN_URL,
            {
                "model": QWEN_MODEL,
                "temperature": 0,
                "max_tokens": 1400,
                "messages": [
                    {"role": "system", "content": "You compact code changes accurately."},
                    {"role": "user", "content": prompt},
                ],
            },
            {"authorization": f"Bearer {os.environ.get('QWEN_API_KEY', 'local-qwen')}"},
            180,
        )
        packet = body["choices"][0]["message"]["content"]
    except Exception as exc:
        packet = f"Compaction unavailable ({type(exc).__name__}).\n\n{clipped}"
    return redact(packet)[:MAX_PACKET_CHARS]


def ask_glm(packet: str) -> tuple[str, dict]:
    key = os.environ.get("COLI_API_KEY", "")
    started = time.monotonic()
    body = post_json(
        GLM_URL,
        {
            "model": GLM_MODEL,
            "temperature": 0,
            "max_tokens": MAX_REVIEW_TOKENS,
            "system": (
                "You are a senior coding verifier. Find correctness, security, "
                "compatibility and test gaps. Be concise. Return markdown with "
                "Verdict, Blocking findings, Advisory findings, and Tests."
            ),
            "messages": [{"role": "user", "content": packet}],
        },
        {"x-api-key": key},
        1200,
    )
    elapsed = time.monotonic() - started
    text = "".join(x.get("text", "") for x in body.get("content", []) if x.get("type") == "text")
    usage = body.get("usage", {})
    usage["wall_seconds"] = round(elapsed, 3)
    usage["output_tok_s"] = round(usage.get("output_tokens", 0) / elapsed, 3) if elapsed else 0
    return text, usage


def load_job(path: Path) -> dict:
    return json.loads(path.read_text())


def job_paths() -> list[Path]:
    return sorted(JOBS.glob("*.json"), key=lambda p: p.stat().st_mtime)


def enqueue(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        print(f"not a git worktree: {repo}", file=sys.stderr)
        return 2
    digest, diff = snapshot(repo)
    if not diff.strip():
        if not args.quiet:
            print("no changes to review")
        return 0
    JOBS.mkdir(parents=True, exist_ok=True)
    for path in job_paths():
        old = load_job(path)
        if old.get("repo") == str(repo) and old.get("diff_sha256") == digest:
            if not args.quiet:
                print(old["id"])
            return 0
    ident = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    job = {
        "id": ident,
        "repo": str(repo),
        "task": args.task or "",
        "diff_sha256": digest,
        "status": "pending",
        "created_at": utcnow(),
        "updated_at": utcnow(),
    }
    atomic_json(JOBS / f"{ident}.json", job)
    if not args.quiet:
        print(ident)
    return 0


def process_one(path: Path) -> None:
    job = load_job(path)
    if job["status"] != "pending":
        return
    if available_gib() < MIN_AVAILABLE_GIB or qwen_is_busy():
        return
    job.update(status="running", updated_at=utcnow())
    atomic_json(path, job)
    try:
        repo = Path(job["repo"])
        digest, diff = snapshot(repo)
        if digest != job["diff_sha256"]:
            job.update(status="stale", updated_at=utcnow())
            atomic_json(path, job)
            return
        packet = compact_with_qwen(job, diff)
        review, usage = ask_glm(packet)
        current_digest, _ = snapshot(repo)
        stale = current_digest != digest
        out_dir = repo / ".claude/glm-reviews"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{job['id']}.md"
        out.write_text(
            f"# GLM review {job['id']}\n\n"
            f"- Created: {job['created_at']}\n"
            f"- Diff SHA-256: `{digest}`\n"
            f"- Stale at completion: `{str(stale).lower()}`\n"
            f"- Usage: `{json.dumps(usage, sort_keys=True)}`\n\n{review.strip()}\n"
        )
        job.update(
            status="stale" if stale else "completed",
            result=str(out),
            usage=usage,
            packet_chars=len(packet),
            updated_at=utcnow(),
        )
    except Exception as exc:
        job.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated_at=utcnow())
    atomic_json(path, job)


def recover_interrupted_jobs() -> None:
    for path in job_paths():
        job = load_job(path)
        if job.get("status") == "running":
            job.update(status="pending", recovered_at=utcnow(), updated_at=utcnow())
            atomic_json(path, job)


def worker(args: argparse.Namespace) -> int:
    JOBS.mkdir(parents=True, exist_ok=True)
    recover_interrupted_jobs()
    while True:
        for path in job_paths():
            process_one(path)
        if args.once:
            return 0
        time.sleep(args.poll)


def status(_: argparse.Namespace) -> int:
    rows = [load_job(path) for path in job_paths()]
    if not rows:
        print("no review jobs")
        return 0
    for job in rows[-20:]:
        print(f"{job['id']} {job['status']:<9} {job['repo']} {job.get('result', '')}")
    return 0


def find_job(ident: str) -> tuple[Path, dict]:
    matches = list(JOBS.glob(f"{ident}*.json"))
    if len(matches) != 1:
        raise SystemExit(f"expected one job matching {ident!r}, found {len(matches)}")
    return matches[0], load_job(matches[0])


def show(args: argparse.Namespace) -> int:
    _, job = find_job(args.id)
    if job.get("result") and Path(job["result"]).exists():
        print(Path(job["result"]).read_text(), end="")
    else:
        print(json.dumps(job, indent=2, sort_keys=True))
    return 0


def wait_job(args: argparse.Namespace) -> int:
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        _, job = find_job(args.id)
        if job["status"] in {"completed", "failed", "stale"}:
            return show(args)
        time.sleep(2)
    print("timeout", file=sys.stderr)
    return 124


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("enqueue")
    e.add_argument("--repo", default=os.getcwd())
    e.add_argument("--task")
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=enqueue)
    w = sub.add_parser("worker")
    w.add_argument("--once", action="store_true")
    w.add_argument("--poll", type=int, default=10)
    w.set_defaults(func=worker)
    s = sub.add_parser("status")
    s.set_defaults(func=status)
    sh = sub.add_parser("show")
    sh.add_argument("id")
    sh.set_defaults(func=show)
    wa = sub.add_parser("wait")
    wa.add_argument("id")
    wa.add_argument("--timeout", type=int, default=1200)
    wa.set_defaults(func=wait_job)
    return p


if __name__ == "__main__":
    try:
        cli = parser()
        arguments = cli.parse_args()
        raise SystemExit(arguments.func(arguments))
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:500]}", file=sys.stderr)
        raise SystemExit(1)
