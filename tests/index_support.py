"""Shared helpers for tests/test_index.py and tests/test_index_sync.py.

A throwaway repository per test, a fake Qdrant-plus-embedder HTTP service that
records every request, and the MANIFEST_DIR patch that keeps real manifests out
of the home directory. Not a test module: unittest and pytest do not collect it.
"""

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import index  # noqa: E402


class TempRepoTestCase(unittest.TestCase):
    """A throwaway repository under test, with the process cwd elsewhere."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.process_cwd = self.root / "elsewhere"
        self.process_cwd.mkdir()
        previous_cwd = os.getcwd()
        os.chdir(self.process_cwd)
        self.addCleanup(os.chdir, previous_cwd)
        # The real manifests live under the home directory: never touch them.
        patcher = mock.patch.object(index, "MANIFEST_DIR", self.root / "manifests")
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, rel, text):
        path = self.work / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def lines_file(self, rel, count, prefix):
        return self.write(rel, "\n".join(f"{prefix} {n}" for n in range(1, count + 1)) + "\n")


class FakeResponse:
    """Just enough of an httpx response for index.py."""

    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = {} if payload is None else payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeIndexService:
    """A fake Qdrant plus a fake embedder behind one recording client.

    Every request is recorded as (method, url, body); embed calls are also
    kept separately so a test can read what was sent to the embedder or make
    it fail after N successful calls.
    """

    def __init__(self):
        self.requests = []            # (method, url, body)
        self.collections = set()
        self.embedded = []            # one entry per embed call: its input list
        self.fail_embed_after = None  # raise on the embed call after this many successes
        self.points = []              # every point ever upserted, in order
        self.query_points = []        # canned points for a query

    def client_factory(self, **kwargs):
        return FakeClient(self)

    def answer(self, method, url, body):
        self.requests.append((method, url, body))
        if url.startswith(index.EMBED_URL):
            if self.fail_embed_after is not None and len(self.embedded) >= self.fail_embed_after:
                raise OSError("embedder went away")
            self.embedded.append(body["input"])
            return FakeResponse(200, {"data": [{"embedding": [0.25] * index.EMBED_DIM}
                                               for _ in body["input"]]})
        prefix = f"{index.QDRANT_URL}/collections/"
        if url.startswith(prefix):
            rest = url[len(prefix):]
            name, _, tail = rest.partition("/")
            if method == "GET" and not tail:
                return FakeResponse(200 if name in self.collections else 404, {"result": {}})
            if method == "PUT" and not tail:
                self.collections.add(name)
                return FakeResponse(200, {"result": True})
            if method == "PUT" and tail.startswith("points"):
                self.points.extend(body["points"])
                return FakeResponse(200, {"result": True})
            if method == "POST" and tail.startswith("points/delete"):
                return FakeResponse(200, {"result": True})
            if method == "POST" and tail.startswith("points/query"):
                return FakeResponse(200, {"result": {"points": self.query_points}})
        raise AssertionError(f"unexpected request: {method} {url}")

    def embed_titles(self):
        """The file title of every input sent to the embedder, in order."""
        titles = []
        for inputs in self.embedded:
            for text in inputs:
                if text.startswith("title: "):
                    titles.append(text[len("title: "):].split(" | text: ", 1)[0])
        return titles

    def requests_named(self, method, fragment):
        return [r for r in self.requests if r[0] == method and fragment in r[1]]


class FakeClient:
    """Stands in for httpx.Client: every call goes to the recording service."""

    def __init__(self, service, **kwargs):
        self.service = service

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url):
        return self.service.answer("GET", url, None)

    def put(self, url, json=None):
        return self.service.answer("PUT", url, json)

    def post(self, url, json=None):
        return self.service.answer("POST", url, json)


class FakeServiceTestCase(TempRepoTestCase):
    def use_service(self, service=None):
        service = service or FakeIndexService()
        patcher = mock.patch.object(index, "httpx", SimpleNamespace(Client=service.client_factory))
        patcher.start()
        self.addCleanup(patcher.stop)
        return service

    def sync_quietly(self):
        with contextlib.redirect_stderr(io.StringIO()):
            return index.sync(str(self.work))
