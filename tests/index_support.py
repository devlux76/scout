"""Shared helpers for tests/test_index.py.

A throwaway repository per test, with the process cwd elsewhere. Not a test
module: unittest and pytest do not collect it.
"""

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


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

    def write(self, rel, text):
        path = self.work / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path
