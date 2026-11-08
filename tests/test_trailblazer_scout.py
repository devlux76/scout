#!/usr/bin/env python3
"""Tests for how scout chooses its candidate source (trailblazer or its own index),
scout's command-line exit codes, and index.py's command line, against a fake
Trailblazer service.

Run with:  python3 -m unittest discover -s tests

Moved from ranger/tests/test_trailblazer.py (bug:6eed722); the fake service is in
trailblazer_support.py.
"""

import contextlib
import io
import subprocess
import sys
import unittest
from unittest import mock

from trailblazer_support import TrailblazerTestCase, exploding_sync, watched_answer

import index  # noqa: E402
import trailblazer  # noqa: E402
import scout  # noqa: E402


class CandidateSourceTests(TrailblazerTestCase):
    # --- scout.index_candidates(): which source it takes ---

    def test_index_candidates_uses_trailblazer_and_never_syncs(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        with mock.patch.object(index, "sync", exploding_sync):
            text, appendix = scout.index_candidates(str(self.repo), "head bob", False)
        self.assertIn("assets/headBob.ts:10-24", text)
        self.assertIn("headBob", appendix)
        # The block AFTER the label: a graph note would be printed there, so
        # assert on the text that follows the range label, not the text before it.
        block = text.split("assets/headBob.ts:10-24", 1)[1].split("\n\n", 1)[0]
        for label in ("defines:", "imports:", "imported by:", "uses:"):
            self.assertNotIn(label, block)  # no graph of its own

    def test_index_candidates_falls_back_to_its_own_index(self):
        calls = []

        def fake_sync(work_dir, verbose=False):
            calls.append(work_dir)
            return "ranger_test"

        hits = [{"path": "assets/headBob.ts", "start": 10, "end": 24, "score": 0.7,
                 "text": "\n".join(self.lines[9:24])}]
        with mock.patch.object(index, "sync", fake_sync), \
             mock.patch.object(index, "search", lambda *a, **k: hits):
            text, appendix = scout.index_candidates(str(self.repo), "head bob", False)
        self.assertEqual(calls, [str(self.repo)])
        self.assertIn("assets/headBob.ts:10-24", text)
        self.assertIsInstance(appendix, str)

    def test_index_candidates_survives_a_broken_index(self):
        with mock.patch.object(index, "sync", side_effect=RuntimeError("qdrant down")):
            self.assertEqual(scout.index_candidates(str(self.repo), "head bob", False), ("", ""))

    def test_index_candidates_uses_its_own_index_when_trailblazer_has_no_hits(self):
        # Trailblazer watches the repo but has nothing to say: vectors still being
        # built, or its search failed. That is not a reason to hand the model nothing.
        def no_hits(op, body):
            if op == "search":
                return 200, []
            return watched_answer(op, body)

        url, _server = self.serve(no_hits)
        self.write_endpoint(url)
        calls = []
        hits = [{"path": "assets/stepBob.ts", "start": 5, "end": 30, "score": 0.4, "text": ""}]
        with mock.patch.object(index, "sync", lambda *a, **k: calls.append(a) or "coll"), \
             mock.patch.object(index, "search", lambda *a, **k: hits):
            text, appendix = scout.index_candidates(str(self.repo), "head bob", False)
        self.assertEqual(len(calls), 1)
        self.assertIn("assets/stepBob.ts:5-30", text)

    def test_empty_trailblazer_hits_say_which_source_was_used(self):
        def no_hits(op, body):
            return (200, []) if op == "search" else watched_answer(op, body)

        url, _server = self.serve(no_hits)
        self.write_endpoint(url)
        with mock.patch.object(index, "sync", return_value="coll"), \
             mock.patch.object(index, "search", return_value=[]), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            scout.index_candidates(str(self.repo), "head bob", False)
        self.assertIn("trailblazer watches the repo but has no hits", err.getvalue())
        self.assertIn("using ranger's own index", err.getvalue())

    def test_no_notice_when_trailblazer_is_not_watched(self):
        self.write_endpoint("http://127.0.0.1:9")  # nothing is listening
        with mock.patch.object(index, "sync", return_value="coll"), \
             mock.patch.object(index, "search", return_value=[]), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            scout.index_candidates(str(self.repo), "head bob", False)
        self.assertEqual(err.getvalue(), "")

    # --- scout.py's command line: retrieval only, no model ---
    def test_scout_exits_1_when_no_index_source_can_answer(self):
        # No endpoint file, so Trailblazer cannot watch; ranger's own index
        # raises, so it cannot answer either: nothing could answer, and the
        # message must say that rather than claiming two false facts.
        err = io.StringIO()
        inactive = subprocess.CompletedProcess(["systemctl", "--user"], 3)
        with contextlib.redirect_stderr(err), \
             mock.patch.object(trailblazer.subprocess, "run", return_value=inactive), \
             mock.patch.object(trailblazer.subprocess, "Popen",
                               side_effect=OSError("inotifywait missing")), \
             mock.patch.object(index, "sync", side_effect=RuntimeError("qdrant down")), \
             mock.patch.object(sys, "argv", ["scout.py", "-C", str(self.repo), "-q", "head bob"]):
            with self.assertRaises(SystemExit) as caught:
                scout.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("no index source could answer", err.getvalue())
        self.assertIn("Trailblazer", err.getvalue())
        self.assertIn("own index", err.getvalue())

    def test_scout_exits_2_when_the_index_answered_with_no_hits(self):
        # The index answered (sync succeeded, search returned nothing) and the
        # question names no code-like symbol, so there are no candidates: a
        # different situation from having no source at all, so a different
        # message and exit status.
        def no_hits(op, body):
            return (200, []) if op == "search" else watched_answer(op, body)

        url, _server = self.serve(no_hits)
        self.write_endpoint(url)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), \
             mock.patch.object(index, "sync", return_value="coll"), \
             mock.patch.object(index, "search", return_value=[]), \
             mock.patch.object(sys, "argv",
                               ["scout.py", "-C", str(self.repo), "-q", "how does this work"]):
            with self.assertRaises(SystemExit) as caught:
                scout.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("the index answered but found no candidate locations", err.getvalue())

    def test_candidates_with_source_names_which_source_answered(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        text, _appendix, source = scout.candidates_with_source(str(self.repo), "head bob", False)
        self.assertEqual(source, "trailblazer")
        self.assertIn("assets/headBob.ts:10-24", text)

        with mock.patch.object(index, "sync", return_value="coll"), \
             mock.patch.object(index, "search", return_value=[]):
            text, _appendix, source = scout.candidates_with_source(
                str(self.repo), "how does this work", False, watched=None)
        self.assertEqual(source, "index")
        self.assertEqual(text, "")

        with mock.patch.object(index, "sync", side_effect=RuntimeError("qdrant down")):
            _text, _appendix, source = scout.candidates_with_source(
                str(self.repo), "head bob", False, watched=None)
        self.assertIsNone(source)

    # --- index.py's command line ---

    def run_index_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(sys, "argv", ["index.py", *argv]):
            index.main()
        return out.getvalue()

    def test_cli_query_uses_trailblazer_and_does_not_sync(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        out = self.run_index_cli("-C", str(self.repo), "-q", "head bob")
        self.assertIn("0.830  assets/headBob.ts:10-24", out)
        self.assertIn("assets/stepBob.ts:5-30", out)
        self.assertIn("assets/headBob.ts:3-3", out)

    def test_cli_own_index_flag_syncs_and_searches(self):
        self.write_endpoint("http://127.0.0.1:9")  # Trailblazer is not there
        calls = []
        hits = [{"path": "assets/headBob.ts", "start": 10, "end": 24, "score": 0.5, "text": ""}]
        with mock.patch.object(index, "sync", lambda *a, **k: calls.append(a)), \
             mock.patch.object(index, "search", lambda *a, **k: hits):
            out = self.run_index_cli("-C", str(self.repo), "-q", "head bob")
        self.assertEqual(len(calls), 1)
        self.assertIn("0.500  assets/headBob.ts:10-24", out)

    def test_cli_query_syncs_when_trailblazer_has_no_hits(self):
        def no_hits(op, body):
            return (200, []) if op == "search" else watched_answer(op, body)

        url, _server = self.serve(no_hits)
        self.write_endpoint(url)
        calls = []
        hits = [{"path": "assets/stepBob.ts", "start": 5, "end": 30, "score": 0.4, "text": ""}]
        with mock.patch.object(index, "sync", lambda *a, **k: calls.append(a)), \
             mock.patch.object(index, "search", lambda *a, **k: hits):
            out = self.run_index_cli("-C", str(self.repo), "-q", "head bob")
        self.assertEqual(len(calls), 1)
        self.assertIn("0.400  assets/stepBob.ts:5-30", out)


if __name__ == "__main__":
    unittest.main()
