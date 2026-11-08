#!/usr/bin/env python3
"""Shared helpers for tests/test_trailblazer_client.py and tests/test_trailblazer_scout.py.

A fake Trailblazer HTTP service on a random loopback port, a fake inotifywait
waiter, and a base TestCase that points TRAILBLAZER_STATE_DIR at a throwaway
directory. Not a test module: unittest and pytest do not collect it.
"""

import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import trailblazer  # noqa: E402

HEAD_BOB = {
    "id": 7,
    "kind": "function",
    "qualified": "headBob",
    "path": "assets/headBob.ts",
    "start_line": 10,
    "end_line": 24,
    "signature": "headBob(t: number)",
    "stale": False,
}
STEP_BOB = {
    "id": 8,
    "kind": "function",
    "qualified": "stepBob",
    "path": "assets/stepBob.ts",
    "start_line": 5,
    "end_line": 30,
    "signature": "stepBob(dt: number)",
    "stale": False,
}
BOB_OFFSET = {
    "id": 9,
    "kind": "const",
    "qualified": "bobOffset",
    "path": "assets/headBob.ts",
    "start_line": 3,
    "end_line": 3,
    "signature": "bobOffset = 0.12",
    "stale": False,
}

SEARCH_RESULT = [
    {"node": HEAD_BOB, "score": 0.83, "via": "vector"},
    {"node": STEP_BOB, "score": 0.79, "via": "vector"},
    {"node": BOB_OFFSET, "score": 1.0, "via": "exact"},
]

USES_RESULT = [
    {"node": STEP_BOB, "target": HEAD_BOB, "kind": "calls", "line": 42, "uncertain": True},
    {"node": BOB_OFFSET, "target": HEAD_BOB, "kind": "references", "line": 18, "uncertain": False},
]

STATUS_RESULT = {
    "files": 160,
    "nodes": 912,
    "edges": 2401,
    "pending_vectors": 0,
    "stale_files": 0,
    "following": "events",
    "embedding": "running",
}


class Handler(BaseHTTPRequestHandler):
    """Answers from the tables on the server object, and records what it was asked."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        op = self.path.strip("/")
        fake = cast("FakeServer", self.server)
        fake.calls.append((op, body))
        status, payload = fake.answer(op, body)
        out = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, format, *args):
        pass


class FakeServer(ThreadingHTTPServer):
    """A Trailblazer that answers from a table and remembers what it was asked."""

    def __init__(self, address, answer):
        super().__init__(address, Handler)
        self.calls = []
        self.answer = answer


def start_fake(answer):
    """Run a fake Trailblazer on a free loopback port; returns (url, server)."""
    server = FakeServer(("127.0.0.1", 0), answer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


def watched_answer(op, body):
    if op == "status":
        return 200, STATUS_RESULT
    if op == "search":
        return 200, SEARCH_RESULT
    if op == "uses":
        # One entry per edge into the definition(s) the name resolves to.
        if "id" in body:
            return 200, [e for e in USES_RESULT if e["target"]["id"] == body["id"]]
        return 200, [e for e in USES_RESULT if matches(e["target"]["qualified"], body.get("name"))]
    if op == "where":
        # Trailblazer answers where with bare node summaries, matched on the
        # qualified name or on its last segment.
        return 200, [n for n in (HEAD_BOB, STEP_BOB, BOB_OFFSET) if matches(n["qualified"], body.get("name"))]
    return 404, {"error": "unknown operation"}


def matches(qualified, asked):
    """Trailblazer falls back from a qualified name to its last segment."""
    if not asked:
        return False
    asked = asked.split("/")[0]
    return qualified == asked or qualified.split(".")[-1] == asked


class FakeWaiter:
    """Stands in for the inotifywait process: stderr lines, then an exit code.

    The waiter is what start_service() watches for "Watches established", so a
    fake yields that line (or fails before it) and wait() returns at once: no
    real inotify, no real backstop sleep, whatever the host's limits allow."""

    def __init__(self, lines, wait_code=0):
        self.stderr = iter(lines)
        self.wait_code = wait_code
        self.killed = False

    def wait(self):
        return self.wait_code

    def kill(self):
        self.killed = True




class TrailblazerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        self.repo = Path(self.tmp.name) / "repo"
        (self.repo / "assets").mkdir(parents=True)
        self.lines = [f"// line {n}" for n in range(1, 41)]
        (self.repo / "assets" / "headBob.ts").write_text("\n".join(self.lines) + "\n")
        (self.repo / "assets" / "stepBob.ts").write_text("\n".join(self.lines) + "\n")
        self.env = mock.patch.dict(
            "os.environ", {"TRAILBLAZER_STATE_DIR": str(self.state)}
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def write_endpoint(self, url, token="t0ken"):
        (self.state / "endpoint.json").write_text(json.dumps({"url": url, "token": token}))

    def serve(self, answer=watched_answer):
        url, server = start_fake(answer)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return url, server

    def no_systemctl(self):
        """Fails the test if anything reaches the real systemctl."""
        return mock.patch.object(trailblazer.subprocess, "run",
                                 side_effect=AssertionError("systemctl must not be called"))

