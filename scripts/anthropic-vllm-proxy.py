#!/usr/bin/env python3
"""Normalize Claude Code Anthropic requests before forwarding them to vLLM."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def normalize(raw: bytes) -> bytes:
    payload = json.loads(raw)
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return raw
    system = [text_content(payload.get("system", ""))]
    retained = []
    changed = False
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            system.append(text_content(message.get("content", "")))
            changed = True
        else:
            retained.append(message)
    if not changed:
        return raw
    payload["messages"] = retained
    payload["system"] = "\n\n".join(part for part in system if part)
    return json.dumps(payload, separators=(",", ":")).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream_port = 8000

    def log_message(self, fmt: str, *args: object) -> None:
        if os.environ.get("VLLM_PROXY_LOG"):
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        self.forward()

    def do_POST(self) -> None:
        self.forward()

    def forward(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length) if length else b""
        if self.command == "POST" and body:
            try:
                body = normalize(body)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_HEADERS
        }
        headers["Host"] = f"127.0.0.1:{self.upstream_port}"
        headers["Content-Length"] = str(len(body))
        upstream = http.client.HTTPConnection(
            "127.0.0.1", self.upstream_port, timeout=3600
        )
        try:
            upstream.request(self.command, self.path, body=body, headers=headers)
            response = upstream.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := response.read(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            upstream.close()
            self.close_connection = True


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=1
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--upstream-port", type=int, default=8000)
    parser.add_argument("--ensure", action="store_true")
    args = parser.parse_args()
    if args.ensure:
        if healthy(args.port):
            return 0
        subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--port",
                str(args.port),
                "--upstream-port",
                str(args.upstream_port),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(30):
            if healthy(args.port):
                return 0
            time.sleep(0.1)
        print(
            f"ERROR: compatibility proxy did not start on :{args.port}",
            file=sys.stderr,
        )
        return 1
    Handler.upstream_port = args.upstream_port
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
