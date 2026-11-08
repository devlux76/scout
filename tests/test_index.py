#!/usr/bin/env python3
"""Tests for index.py: the candidate text and exact-name matches scout prints.

Run with:  python3 -m unittest discover -s tests

bug:d4b7df1 - index.py had no unit tests. Every test builds a throwaway
repository under tempfile. Since bug:3df441f index.py only renders Trailblazer's
hits, so nothing here reaches Trailblazer, Qdrant or an embedder.
Shared helpers are in index_support.py.
"""

import unittest

from index_support import TempRepoTestCase

import index


class CandidatesTextTests(TempRepoTestCase):
    # bug:d4b7df1 - merging, rank order, previews, missing files.

    def build(self):
        self.write("a.py", "from b import b_helper\n\n"
                           "def a_main():\n" +
                           "\n".join(f"a filler {n}" for n in range(1, 18)) + "\n")
        self.write("b.py", "def b_helper():\n    pass\n" +
                           "\n".join(f"b line {n}" for n in range(3, 16)) + "\n")
        self.hits = [
            {"path": "a.py", "start": 1, "end": 10, "score": 0.91, "text": ""},
            {"path": "a.py", "start": 8, "end": 20, "score": 0.88, "text": ""},  # overlaps
            {"path": "b.py", "start": 4, "end": 9, "score": 0.75, "text": ""},
            {"path": "gone.py", "start": 1, "end": 3, "score": 0.60, "text": ""},
        ]

    def candidates(self):
        return index.candidates_text(str(self.work), self.hits, full=1, preview_lines=3)

    def test_overlapping_chunks_of_one_file_become_one_range(self):
        self.build()
        text = self.candidates()
        blocks = text.split("\n\n")
        self.assertEqual(len(blocks), 2)  # the missing file's hit is skipped
        self.assertTrue(blocks[0].startswith(f"{self.work / 'a.py'}:1-20 (similarity 0.91)"))
        self.assertNotIn("a.py:1-10", text)
        self.assertNotIn("8-20", text)
        self.assertNotIn("gone.py", text)

    def test_best_hit_comes_first_and_only_the_first_block_is_full(self):
        self.build()
        blocks = self.candidates().split("\n\n")
        # Rank order: the 0.91 hit's file first, then the 0.75 one.
        self.assertTrue(blocks[1].startswith(f"{self.work / 'b.py'}:4-9 (similarity 0.75) (preview)"))
        # First block: the whole merged range, numbered, no preview marker.
        self.assertNotIn("(preview)", blocks[0])
        self.assertIn("1: from b import b_helper", blocks[0])
        self.assertIn("20: a filler 17", blocks[0])
        # Later block: cut to preview_lines (4, 5, 6), labelled as a preview.
        self.assertIn("4: b line 4", blocks[1])
        self.assertIn("5: b line 5", blocks[1])
        self.assertIn("6: b line 6", blocks[1])
        self.assertNotIn("7: b line 7", blocks[1])


class SymbolMatchesTests(TempRepoTestCase):
    # bug:d4b7df1 - only code-like tokens are searched; found and missing names.

    def build(self):
        self.write("util.py", "def my_func():\n    pass\n"
                              "CAPS_CONST = 3\n"
                              "camelCaseThing = 1\n")
        self.write("other.py", "from util import my_func\nresult = my_func()\n")
        self.write("dup.py", "".join(f"dup_name = {n}\n" for n in range(13)))

    def matches(self, query):
        return index.symbol_matches(str(self.work), query)

    def test_found_symbols_print_their_matches(self):
        self.build()
        text = self.matches("where is my_func used?")
        self.assertIn("my_func (3 matches):", text)
        self.assertIn("util.py:1:def my_func():", text)
        self.assertIn("other.py:2:result = my_func()", text)

    def test_only_code_like_tokens_are_searched(self):
        self.build()
        text = self.matches("how does my_func find CAPS_CONST and camelCaseThing "
                            "but not plaintext words")
        self.assertIn("my_func (3 matches):", text)
        self.assertIn("CAPS_CONST (1 matches):", text)
        self.assertIn("camelCaseThing (1 matches):", text)
        for plain in ("how", "does", "find", "but", "not", "plaintext", "words"):
            self.assertNotIn(f"{plain}:", text)

    def test_backticks_name_a_symbol_and_a_missing_one_says_so(self):
        self.build()
        text = self.matches("what does `ghost_sym` say about `my_func`?")
        self.assertIn("ghost_sym: no exact match in the repository", text)
        self.assertIn("my_func (3 matches):", text)

    def test_many_matches_are_capped_with_a_count(self):
        self.build()
        text = self.matches("dup_name")
        self.assertIn("dup_name (13 matches):", text)
        self.assertIn("... 1 more matches", text)

if __name__ == "__main__":
    unittest.main()
