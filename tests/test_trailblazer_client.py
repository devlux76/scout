#!/usr/bin/env python3
"""Tests for the trailblazer.py client: call(), watched(), search_hits(), graph text,
and ensure_watched()/start_service(), against a fake Trailblazer service.

Run with:  python3 -m unittest discover -s tests

Moved from ranger/tests/test_trailblazer.py (bug:6eed722); the fake service and
the waiter fake are in trailblazer_support.py.
"""

import contextlib
import io
import json
import subprocess
import time
import unittest
from pathlib import Path
from unittest import mock

from trailblazer_support import (
    HEAD_BOB, STATUS_RESULT, STEP_BOB, USES_RESULT, FakeWaiter, TrailblazerTestCase,
    watched_answer,
)

import trailblazer  # noqa: E402


class TrailblazerClientTests(TrailblazerTestCase):
    # --- call() and watched(): every failure is None, not an exception ---

    def test_no_endpoint_file_means_none(self):
        self.assertIsNone(trailblazer.call("status", {"repo": str(self.repo)}))
        self.assertIsNone(trailblazer.watched(self.repo))

    def test_unreadable_endpoint_file_means_none(self):
        (self.state / "endpoint.json").write_text("{not json")
        self.assertIsNone(trailblazer.watched(self.repo))

    def test_connection_refused_means_none(self):
        self.write_endpoint("http://127.0.0.1:9")  # nothing listens here
        self.assertIsNone(trailblazer.call("status", {"repo": str(self.repo)}))
        self.assertIsNone(trailblazer.watched(self.repo))

    def test_not_watched_is_its_own_answer(self):
        url, _server = self.serve(lambda op, body: (409, {"error": "not watched"}))
        self.write_endpoint(url)
        self.assertIs(trailblazer.call("status", {"repo": str(self.repo)}), trailblazer.NOT_WATCHED)
        self.assertIsNone(trailblazer.watched(self.repo))

    def test_error_status_means_none(self):
        url, _server = self.serve(lambda op, body: (401, {"error": "unauthorized"}))
        self.write_endpoint(url, token="wrong")
        self.assertIsNone(trailblazer.watched(self.repo))

    def test_watched_returns_the_status_dict(self):
        url, server = self.serve()
        self.write_endpoint(url)
        status = trailblazer.watched(self.repo) or {}
        self.assertEqual(status.get("nodes"), STATUS_RESULT["nodes"])
        op, body = server.calls[0]
        self.assertEqual(op, "status")
        self.assertEqual(body["repo"], str(self.repo))

    # --- search_hits(): index.py's hit shape, text read from disk ---

    def test_search_hits_convert_nodes_to_hits(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        hits = trailblazer.search_hits(self.repo, "head bob")
        self.assertEqual([h["path"] for h in hits], ["assets/headBob.ts", "assets/stepBob.ts", "assets/headBob.ts"])
        first = hits[0]
        self.assertEqual((first["start"], first["end"]), (10, 24))
        self.assertAlmostEqual(first["score"], 0.83)
        self.assertEqual(first["text"], "\n".join(self.lines[9:24]))
        self.assertEqual(hits[2]["start"], 3)
        self.assertEqual(hits[2]["text"], self.lines[2])

    def test_search_hits_of_a_missing_file_keep_their_range(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        (self.repo / "assets" / "stepBob.ts").unlink()
        hits = trailblazer.search_hits(self.repo, "head bob")
        self.assertEqual(hits[1]["text"], "")

    def test_search_hits_are_empty_when_the_service_is_down(self):
        self.write_endpoint("http://127.0.0.1:9")
        self.assertEqual(trailblazer.search_hits(self.repo, "head bob"), [])

    # --- structure_text(): users of the best hits, then named symbols ---

    def test_structure_text_names_the_best_hits_and_their_users(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        hits = trailblazer.search_hits(self.repo, "head bob")
        text = trailblazer.structure_text(self.repo, hits, "head bob")
        self.assertIn("headBob", text)
        self.assertIn("stepBob", text)
        self.assertIn("assets/stepBob.ts:42", text)
        self.assertIn("uncertain", text)

    def test_an_edge_keeps_its_uncertainty_and_provenance(self):
        edge = {"node": STEP_BOB, "kind": "calls", "line": 42,
                "uncertain": True, "provenance": "default_module"}
        self.assertEqual(trailblazer.edge_line(edge),
                         "assets/stepBob.ts:42 stepBob (calls, uncertain, default module)")

    def test_an_edge_without_provenance_renders_as_before(self):
        self.assertEqual(trailblazer.edge_line(USES_RESULT[0]),
                         "assets/stepBob.ts:42 stepBob (calls, uncertain)")
        self.assertEqual(trailblazer.edge_line(USES_RESULT[1]),
                         "assets/headBob.ts:18 bobOffset (references)")

    def test_structure_text_preserves_provenance_from_the_graph(self):
        uses = [dict(USES_RESULT[0], provenance="default_module")]
        url, _server = self.serve(lambda op, body: (200, uses) if op == "uses"
                                  else watched_answer(op, body))
        self.write_endpoint(url)
        hits = trailblazer.search_hits(self.repo, "head bob")
        text = trailblazer.structure_text(self.repo, hits, "head bob")
        self.assertIn("(calls, uncertain, default module)", text)

    def test_structure_text_reports_named_symbols(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        hits = trailblazer.search_hits(self.repo, "head bob")
        text = trailblazer.structure_text(self.repo, hits, "where is `stepBob` used?")
        self.assertIn("stepBob", text)
        self.assertIn("declared", text)

    def test_structure_text_is_empty_when_the_service_is_down(self):
        self.write_endpoint("http://127.0.0.1:9")
        self.assertEqual(trailblazer.structure_text(self.repo, [], "head bob"), "")

    # --- a graph that cannot answer is not a graph that says "nothing" ---

    def test_where_and_uses_are_none_when_the_graph_fails(self):
        url, _server = self.serve(lambda op, body: (503, {"error": "operation failed"})
                                  if op in ("where", "uses") else watched_answer(op, body))
        self.write_endpoint(url)
        self.assertIsNone(trailblazer.where_lines(self.repo, "headBob"))
        self.assertIsNone(trailblazer.uses_edges(self.repo, 7))

    def test_where_and_uses_are_empty_lists_for_a_real_empty_answer(self):
        url, _server = self.serve(lambda op, body: (200, []) if op in ("where", "uses")
                                  else watched_answer(op, body))
        self.write_endpoint(url)
        self.assertEqual(trailblazer.where_lines(self.repo, "headBob"), [])
        self.assertEqual(trailblazer.uses_edges(self.repo, 7), [])

    def test_structure_text_says_unknown_when_the_graph_fails(self):
        url, _server = self.serve(lambda op, body: (503, {"error": "operation failed"})
                                  if op in ("where", "uses") else watched_answer(op, body))
        self.write_endpoint(url)
        hits = trailblazer.search_hits(self.repo, "head bob")
        text = trailblazer.structure_text(self.repo, hits, "where is `stepBob` used?")
        self.assertIn("users unknown: graph unavailable", text)
        self.assertNotIn("no other definition uses it", text)
        self.assertNotIn("used by: nothing", text)

    def test_structure_text_reports_nothing_only_when_it_knows(self):
        url, _server = self.serve(lambda op, body: (200, []) if op in ("where", "uses")
                                  else watched_answer(op, body))
        self.write_endpoint(url)
        hits = trailblazer.search_hits(self.repo, "head bob")
        text = trailblazer.structure_text(self.repo, hits, "where is `stepBob` used?")
        self.assertIn("no other definition uses it", text)
        self.assertNotIn("unavailable", text)

    def test_a_graph_that_goes_away_mid_run_is_not_an_empty_graph(self):
        url, server = self.serve()
        self.write_endpoint(url)
        self.assertIsNotNone(trailblazer.watched(self.repo))  # status succeeds
        server.shutdown()  # ... and then the service stops
        server.server_close()
        self.assertIsNone(trailblazer.where_lines(self.repo, "headBob"))
        self.assertIsNone(trailblazer.uses_edges(self.repo, 7))

    # --- ensure_watched(): waking the service and asking for a watch ---
    def test_ensure_watched_asks_an_unwatched_repo_to_be_watched(self):
        asked = []

        def unwatched_until_asked(op, body):
            if op == "watch":
                asked.append(body["repo"])
                return 200, {"watched": True}
            if op == "status" and not asked:
                return 409, {"error": "not watched"}
            return watched_answer(op, body)

        url, server = self.serve(unwatched_until_asked)
        self.write_endpoint(url)
        with self.no_systemctl(), contextlib.redirect_stderr(io.StringIO()):
            status = trailblazer.ensure_watched(str(self.repo))
        self.assertEqual(status, STATUS_RESULT)
        self.assertEqual([op for op, _ in server.calls], ["status", "watch", "status"])
        self.assertEqual(asked, [str(self.repo)])

    def test_ensure_watched_needs_nothing_for_a_watched_repo(self):
        url, server = self.serve()
        self.write_endpoint(url)
        with self.no_systemctl():
            self.assertEqual(trailblazer.ensure_watched(str(self.repo)), STATUS_RESULT)
        self.assertEqual([op for op, _ in server.calls], ["status"])

    def test_ensure_watched_leaves_a_running_but_silent_service_alone(self):
        # Busy or wedged: a restart would throw away its indexing.
        self.write_endpoint("http://127.0.0.1:9")  # nothing is listening
        runs = []

        def systemctl(argv, **kw):
            runs.append(argv)
            return subprocess.CompletedProcess(argv, 0)  # is-active: yes

        with mock.patch.object(trailblazer.subprocess, "run", systemctl), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertIsNone(trailblazer.ensure_watched(str(self.repo)))
        self.assertEqual([argv[2] for argv in runs], ["is-active"])
        self.assertIn("leaving it alone", err.getvalue())

    def test_ensure_watched_starts_a_stopped_service_on_its_endpoint_event(self):
        # The service publishes endpoint.json by rename once its API listens; the
        # wait is on that event, so the status asked right after it is answered.
        # The waiter itself is faked: no real inotifywait, so the test passes
        # whatever the host's inotify instance limit allows.
        url, server = self.serve()
        runs = []

        def systemctl(argv, **kw):
            runs.append(argv)
            if argv[2] == "is-active":
                return subprocess.CompletedProcess(argv, 3)  # inactive
            temp = self.state / "endpoint.json.tmp"
            temp.write_text(json.dumps({"url": url, "token": "t0ken"}))
            temp.rename(self.state / "endpoint.json")
            return subprocess.CompletedProcess(argv, 0)

        waiter = FakeWaiter(["Setting up watches.\n", "Watches established.\n"], wait_code=0)
        with mock.patch.object(trailblazer.subprocess, "run", systemctl), \
             mock.patch.object(trailblazer.subprocess, "Popen", lambda *a, **k: waiter), \
             contextlib.redirect_stderr(io.StringIO()):
            status = trailblazer.ensure_watched(str(self.repo))
        self.assertEqual([argv[2] for argv in runs], ["is-active", "start"])
        self.assertEqual(status, STATUS_RESULT)
        self.assertEqual([op for op, _ in server.calls], ["status"])

    def test_a_start_that_publishes_nothing_fires_the_backstop(self):
        # The backstop is faked too: the waiter's wait() returns nonzero at once
        # instead of blocking for the real START_BACKSTOP_S.
        def systemctl(argv, **kw):
            return subprocess.CompletedProcess(argv, 3 if argv[2] == "is-active" else 0)

        waiter = FakeWaiter(["Setting up watches.\n", "Watches established.\n"], wait_code=1)
        with mock.patch.object(trailblazer.subprocess, "run", systemctl), \
             mock.patch.object(trailblazer.subprocess, "Popen", lambda *a, **k: waiter), \
             mock.patch.object(trailblazer, "START_BACKSTOP_S", 1), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertIsNone(trailblazer.ensure_watched(str(self.repo)))
        self.assertIn("anomaly", err.getvalue())

    def test_a_waiter_that_cannot_set_up_says_why_and_does_not_start(self):
        # inotifywait gives up when its instances are exhausted: its own stderr
        # must reach the operator, with the limit to raise, and the service
        # stays stopped — the operator decides, not the script.
        runs = []

        def systemctl(argv, **kw):
            runs.append(argv)
            return subprocess.CompletedProcess(argv, 3 if argv[2] == "is-active" else 0)

        waiter = FakeWaiter(["Setting up watches.\n",
                             "Couldn't initialize inotify: Too many open files\n"],
                            wait_code=1)
        with mock.patch.object(trailblazer.subprocess, "run", systemctl), \
             mock.patch.object(trailblazer.subprocess, "Popen", lambda *a, **k: waiter), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertFalse(trailblazer.start_service())
            self.assertIsNone(trailblazer.ensure_watched(str(self.repo)))
        message = err.getvalue()
        self.assertIn("Couldn't initialize inotify: Too many open files", message)
        self.assertIn("fs.inotify.max_user_instances", message)
        self.assertEqual([argv[2] for argv in runs], ["is-active", "is-active"])  # never started

    def test_a_search_slowed_by_indexing_still_answers(self):
        # Trailblazer's own embedding backlog delays a search past the plain
        # per-operation timeout; that must not read as "no hits".
        def slow_search(op, body):
            if op == "search":
                time.sleep(0.3)
            return watched_answer(op, body)

        url, _server = self.serve(slow_search)
        self.write_endpoint(url)
        with mock.patch.object(trailblazer.call, "__defaults__", (0.1,)):
            self.assertIsNone(trailblazer.call("search", {"repo": str(self.repo), "text": "x"}))
            hits = trailblazer.search_hits(str(self.repo), "head bob")
        self.assertIn("assets/headBob.ts", [h["path"] for h in hits])


if __name__ == "__main__":
    unittest.main()
