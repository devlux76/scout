#!/usr/bin/env python3
"""Tests for how scout takes its candidates from Trailblazer, and scout's
command-line exit codes, against a fake Trailblazer service.

Run with:  python3 -m unittest discover -s tests

Scout has no index of its own (bug:3df441f): a repository Trailblazer does not
watch yields no candidates and exit 1, and a watched repository whose vectors
are still being built says so. Moved from ranger/tests/test_trailblazer.py
(bug:6eed722); the fake service is in trailblazer_support.py.
"""

import contextlib
import io
import subprocess
import sys
import unittest
from unittest import mock

from trailblazer_support import STATUS_RESULT, TrailblazerTestCase, watched_answer

import index  # noqa: E402
import trailblazer  # noqa: E402
import scout  # noqa: E402


def no_hits(op, body):
    return (200, []) if op == "search" else watched_answer(op, body)


def still_building(op, body):
    if op == "search":
        return 200, []
    if op == "status":
        return 200, {**STATUS_RESULT, "pending_vectors": 12}
    return watched_answer(op, body)


class CandidateSourceTests(TrailblazerTestCase):
    # --- scout.index_candidates(): Trailblazer's hits are the candidates ---

    def test_index_candidates_uses_trailblazer_and_has_no_index_of_its_own(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        self.assertFalse(hasattr(index, "sync"))
        self.assertFalse(hasattr(index, "search"))
        text, appendix = scout.index_candidates(str(self.repo), "head bob", False)
        self.assertIn("assets/headBob.ts:10-24", text)
        self.assertIn("headBob", appendix)
        # The block AFTER the label: a graph note would be printed there, so
        # assert on the text that follows the range label, not the text before it.
        block = text.split("assets/headBob.ts:10-24", 1)[1].split("\n\n", 1)[0]
        for label in ("defines:", "imports:", "imported by:", "uses:"):
            self.assertNotIn(label, block)  # no graph of its own

    def test_unwatched_repo_gives_no_candidates_and_says_how_to_watch_it(self):
        self.write_endpoint("http://127.0.0.1:9")  # nothing is listening
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            text, appendix, source = scout.candidates_with_source(
                str(self.repo), "head bob", False)
        self.assertIsNone(source)
        self.assertEqual((text, appendix), ("", ""))
        self.assertIn("trailblazer watch", err.getvalue())
        self.assertIn(str(self.repo), err.getvalue())

    def test_still_building_vectors_is_reported_not_papered_over(self):
        url, _server = self.serve(still_building)
        self.write_endpoint(url)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            text, _appendix, source = scout.candidates_with_source(
                str(self.repo), "head bob", False)
        self.assertEqual(source, "trailblazer")
        self.assertEqual(text, "")
        self.assertIn("still building vectors", err.getvalue())
        self.assertIn("pending_vectors=12", err.getvalue())

    def test_no_building_notice_when_vectors_are_done(self):
        url, _server = self.serve(no_hits)
        self.write_endpoint(url)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            _text, _appendix, source = scout.candidates_with_source(
                str(self.repo), "head bob", False)
        self.assertEqual(source, "trailblazer")
        self.assertNotIn("still building", err.getvalue())

    # --- scout.py's command line: retrieval only, no model ---

    def test_scout_exits_1_when_trailblazer_is_not_watching(self):
        # No endpoint file and no service to start: Trailblazer cannot answer,
        # so scout has no candidates and says how to fix that.
        err = io.StringIO()
        inactive = subprocess.CompletedProcess(["systemctl", "--user"], 3)
        with contextlib.redirect_stderr(err), \
             mock.patch.object(trailblazer.subprocess, "run", return_value=inactive), \
             mock.patch.object(trailblazer.subprocess, "Popen",
                               side_effect=OSError("inotifywait missing")), \
             mock.patch.object(sys, "argv", ["scout.py", "-C", str(self.repo), "-q", "head bob"]):
            with self.assertRaises(SystemExit) as caught:
                scout.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("trailblazer watch", err.getvalue())
        self.assertIn("scout: no candidates", err.getvalue())

    def test_scout_exits_2_when_trailblazer_answered_with_no_candidates(self):
        # Trailblazer answers, with no hits and no code-like name in the question:
        # a different situation from having no source at all, so a different
        # message and exit status.
        url, _server = self.serve(no_hits)
        self.write_endpoint(url)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), \
             mock.patch.object(sys, "argv",
                               ["scout.py", "-C", str(self.repo), "-q", "how does this work"]):
            with self.assertRaises(SystemExit) as caught:
                scout.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("Trailblazer answered but found no candidate locations for this question",
                      err.getvalue())

    def test_candidates_with_source_names_trailblazer(self):
        url, _server = self.serve()
        self.write_endpoint(url)
        text, _appendix, source = scout.candidates_with_source(str(self.repo), "head bob", False)
        self.assertEqual(source, "trailblazer")
        self.assertIn("assets/headBob.ts:10-24", text)

    # --- index.py renders hits; it does not own a command line any more ---

    def test_index_module_has_no_command_line(self):
        self.assertFalse(hasattr(index, "main"))


if __name__ == "__main__":
    unittest.main()
