#!/usr/bin/env python3
"""Tests for index.py: the file walk, chunking, graph, formatters and candidate text.

Run with:  python3 -m pytest tests -q

bug:d4b7df1 - index.py had no unit tests. Every test builds a throwaway
repository under tempfile, so nothing here reaches a real Qdrant or embedder.
The sync and search tests (fake HTTP service) are in test_index_sync.py; shared
helpers are in index_support.py.
"""

import re
import unittest

from index_support import TempRepoTestCase

import index


class ListFilesTests(TempRepoTestCase):
    # bug:d4b7df1 - only text suffixes, size limits, hidden dirs and .gitignore.

    def build_tree(self):
        self.write("main.py", "print('hi')\n")
        self.write("notes.md", "# notes\n")
        self.write("data.bin", "binary-ish\n")            # suffix not in TEXT_SUFFIXES
        self.write("empty.py", "")                        # empty files are skipped
        self.write("big.ts", "x" * (index.MAX_FILE_BYTES + 1))     # above MAX_FILE_BYTES
        self.write("huge.json", "y" * (index.MAX_DATA_BYTES + 1))  # data file above MAX_DATA_BYTES
        self.write("small.json", '{"ok": true}\n')        # small data file is kept
        self.write(".hidden/sneaky.py", "print('no')\n")  # hidden dirs: skipped by rg
        self.write("ignored.py", "print('no')\n")
        (self.work / ".gitignore").write_text("ignored.py\n")

    def test_only_text_files_with_the_right_sizes_come_back(self):
        self.build_tree()
        files = index.list_files(str(self.work))
        self.assertEqual(set(files), {"main.py", "notes.md", "small.json"})
        stat = (self.work / "main.py").stat()
        self.assertEqual(files["main.py"], [int(stat.st_mtime), stat.st_size])

    def test_exactly_max_file_bytes_is_kept_but_a_byte_more_is_skipped(self):
        self.write("edge.ts", "z" * index.MAX_FILE_BYTES)
        self.write("over.ts", "z" * (index.MAX_FILE_BYTES + 1))
        files = index.list_files(str(self.work))
        self.assertIn("edge.ts", files)
        self.assertNotIn("over.ts", files)

    def test_an_empty_repo_yields_no_files(self):
        self.write("readme.txt", "")  # still empty
        self.assertEqual(index.list_files(str(self.work)), {})


class ChunkFileTests(TempRepoTestCase):
    # bug:d4b7df1 - CHUNK_LINES/CHUNK_OVERLAP windows and MAX_CHUNK_CHARS.

    def test_fifty_lines_yield_overlapping_windows(self):
        self.lines_file("fifty.py", 50, "line")
        chunks = index.chunk_file(str(self.work), "fifty.py")
        self.assertEqual([(s, e) for s, e, _ in chunks], [(1, 40), (33, 50)])
        first, second = chunks[0][2], chunks[1][2]
        # The overlap: lines 33..40 are inside both texts.
        for n in range(33, 41):
            self.assertIn(f"line {n}", first)
            self.assertIn(f"line {n}", second)
        self.assertNotIn("line 41", first)
        self.assertNotIn("line 32", second)

    def test_a_short_file_is_one_chunk(self):
        self.write("short.py", "one\ntwo\nthree\n")
        chunks = index.chunk_file(str(self.work), "short.py")
        self.assertEqual([(s, e) for s, e, _ in chunks], [(1, 3)])
        self.assertEqual(chunks[0][2], "one\ntwo\nthree")

    def test_a_blank_only_file_has_no_chunks(self):
        self.write("blank.py", "\n\n   \n\t\n")
        self.assertEqual(index.chunk_file(str(self.work), "blank.py"), [])

    def test_a_chunk_text_is_truncated_to_max_chunk_chars(self):
        self.write("long.py", "a" * 1000 + "\n" + "b" * 1000 + "\n")
        chunks = index.chunk_file(str(self.work), "long.py")
        self.assertEqual([(s, e) for s, e, _ in chunks], [(1, 2)])
        text = chunks[0][2]
        self.assertEqual(len(text), index.MAX_CHUNK_CHARS)
        self.assertTrue(text.startswith("a" * 1000 + "\n"))
        self.assertNotIn("b" * 1000, text)


class BuildGraphTests(TempRepoTestCase):
    # bug:d4b7df1 - import resolution, packages, imported_by and symbols.

    def build(self):
        self.write("src/main.ts",
                   "import { widget } from './widgets';\n"
                   "import { helper } from './util.js';\n"
                   "import React from 'react';\n"
                   "import deep from '@scope/pkg/deep/path';\n"
                   "export const MAIN_COUNT = 1;\n")
        self.write("src/page.tsx", "import { widget } from './widgets';\n")
        self.write("src/util.ts",
                   "export function helper() {}\n"
                   "export class Thing {}\n"
                   "const LOCAL = 3;\n")
        self.write("src/widgets/index.tsx", "export function widget() {}\n")
        self.write("app.py",
                   "from pkg.mod import x\n"
                   "import helper\n"
                   "MAX_SIZE = 10\n")
        self.write("pkg/mod.py", "def thing():\n    pass\n")
        self.write("helper.py", "THING = 5\n")
        self.write("sub/main2.py", "import localmod\n")
        self.write("sub/localmod.py", "def local_fn():\n    pass\n")
        self.write("notes.md", "just words\n")
        return index.build_graph(str(self.work), index.list_files(str(self.work)))

    def test_js_relative_extensionless_index_and_js_specifier_resolve(self):
        graph = self.build()
        self.assertEqual(graph["src/main.ts"]["imports"],
                         ["src/widgets/index.tsx", "src/util.ts"])
        self.assertEqual(graph["src/page.tsx"]["imports"], ["src/widgets/index.tsx"])

    def test_bare_specifiers_become_packages_not_imports(self):
        graph = self.build()
        self.assertEqual(graph["src/main.ts"]["packages"], ["react", "@scope/pkg"])
        self.assertEqual(graph["src/main.ts"]["imports"],
                         ["src/widgets/index.tsx", "src/util.ts"])

    def test_python_from_and_local_module_imports_resolve(self):
        graph = self.build()
        self.assertEqual(graph["app.py"]["imports"], ["pkg/mod.py", "helper.py"])
        # A sibling module in the importer's own directory resolves relative
        # to that directory, not to the repository root.
        self.assertEqual(graph["sub/main2.py"]["imports"], ["sub/localmod.py"])
        self.assertIn(("local_fn", 1), graph["sub/localmod.py"]["symbols"])

    def test_imported_by_is_filled_the_other_way(self):
        graph = self.build()
        self.assertEqual(set(graph["src/widgets/index.tsx"]["imported_by"]),
                         {"src/main.ts", "src/page.tsx"})
        self.assertEqual(set(graph["src/util.ts"]["imported_by"]), {"src/main.ts"})
        self.assertEqual(set(graph["helper.py"]["imported_by"]), {"app.py"})

    def test_symbols_carry_their_line_numbers(self):
        graph = self.build()
        self.assertIn(("MAIN_COUNT", 5), graph["src/main.ts"]["symbols"])
        self.assertIn(("widget", 1), graph["src/widgets/index.tsx"]["symbols"])
        self.assertIn(("helper", 1), graph["src/util.ts"]["symbols"])
        self.assertIn(("Thing", 2), graph["src/util.ts"]["symbols"])
        self.assertIn(("LOCAL", 3), graph["src/util.ts"]["symbols"])
        self.assertIn(("thing", 1), graph["pkg/mod.py"]["symbols"])
        self.assertIn(("MAX_SIZE", 3), graph["app.py"]["symbols"])
        self.assertIn(("THING", 1), graph["helper.py"]["symbols"])

    def test_non_code_files_get_empty_entries(self):
        graph = self.build()
        self.assertEqual(graph["notes.md"],
                         {"imports": [], "packages": [], "imported_by": [], "symbols": []})


class SymbolEdgesTests(TempRepoTestCase):
    # bug:d4b7df1 - defined/used through imports, strangers, and the limit.

    def build(self):
        self.write("util.ts", "export function widget() {}\n")
        self.write("main.ts", "import { widget } from './util';\nwidget();\n")
        self.write("stranger.ts", "const alias = widget;\n")  # same name, no import
        self.write("local.ts", "function lonely() {}\n")
        self.write("big.ts", "export function shared() {}\n")
        for n in range(10):
            self.write(f"u{n}.ts", "import { shared } from './big';\nshared();\n")
        return index.build_graph(str(self.work), index.list_files(str(self.work)))

    def test_defined_and_used_through_the_import(self):
        graph = self.build()
        text = index.symbol_edges(graph, "widget")
        self.assertEqual(text,
                         "defined in util.ts:1; used by main.ts\n"
                         "same name, no import link to a definition: stranger.ts")

    def test_a_local_symbol_says_it_has_no_other_user(self):
        graph = self.build()
        self.assertEqual(index.symbol_edges(graph, "lonely"),
                         "defined in local.ts:1; used by no other file (local to its file)")

    def test_a_name_nothing_defines_has_no_edges(self):
        graph = self.build()
        self.assertEqual(index.symbol_edges(graph, "nothing_at_all"), "")

    def test_the_user_list_is_truncated_with_a_count(self):
        graph = self.build()
        line = index.symbol_edges(graph, "shared").splitlines()[0]
        self.assertTrue(line.startswith("defined in big.ts:1; used by "), line)
        tail = line.split("used by ", 1)[1].split(", ")
        self.assertEqual(len(tail), 9)  # 8 users + the "+N more" marker
        self.assertEqual(tail[-1], "+2 more")
        self.assertTrue(all(re.fullmatch(r"u\d\.ts", name) for name in tail[:-1]), tail)
        # A smaller limit truncates harder.
        line = index.symbol_edges(graph, "shared", limit=3).splitlines()[0]
        self.assertEqual(line.split("used by ", 1)[1].split(", ")[-1], "+7 more")


class FileUsesTests(TempRepoTestCase):
    # bug:d4b7df1 - 'name (file)' for symbols taken from imported repo files.

    def build(self):
        names = [f"alpha{n:02d}" for n in range(15)]
        self.write("lib.py", "".join(f"def {name}():\n    pass\n" for name in names))
        self.write("app.py", "from lib import " + ", ".join(names) + "\n" + "\n".join(names) + "\n")
        return index.build_graph(str(self.work), index.list_files(str(self.work)))

    def test_uses_lists_the_symbols_taken_from_imports(self):
        graph = self.build()
        uses = index.file_uses(graph, "app.py")
        self.assertEqual(uses[0], "alpha00 (lib.py)")
        self.assertEqual(uses[1], "alpha01 (lib.py)")
        self.assertEqual(len(uses), 13)  # 12 shown + the truncation marker
        self.assertEqual(uses[-1], "+3 more")
        self.assertNotIn("alpha14 (lib.py)", uses[:-1])

    def test_a_smaller_limit_truncates_harder(self):
        graph = self.build()
        uses = index.file_uses(graph, "app.py", limit=3)
        self.assertEqual(uses, ["alpha00 (lib.py)", "alpha01 (lib.py)",
                               "alpha02 (lib.py)", "+12 more"])

    def test_a_file_that_imports_nothing_has_no_uses(self):
        graph = self.build()
        self.assertEqual(index.file_uses(graph, "lib.py"), [])


class GraphNoteTests(unittest.TestCase):
    # bug:d4b7df1 - defines/imports/imported by, restricted to a line range.

    def note(self, start=None, end=None, **kw):
        entry = {
            "imports": ["b.py"],
            "packages": ["react"],
            "imported_by": ["c.py"],
            "symbols": [("one", 2), ("two", 10)],
        }
        entry.update(kw)
        return index.graph_note(entry, start, end)

    def test_full_note_lists_defines_imports_and_imported_by(self):
        self.assertEqual(self.note(),
                         "defines: one, two\nimports: b.py, react\nimported by: c.py")

    def test_a_line_range_keeps_only_the_symbols_inside_it(self):
        self.assertEqual(self.note(1, 5),
                         "defines: one\nimports: b.py, react\nimported by: c.py")

    def test_a_range_with_no_symbol_inside_drops_the_defines_line(self):
        self.assertEqual(self.note(3, 9),
                         "imports: b.py, react\nimported by: c.py")

    def test_the_symbol_list_is_truncated_with_a_count(self):
        entry = {"imports": [], "packages": [], "imported_by": [],
                 "symbols": [(f"s{n}", n) for n in range(10)]}
        self.assertEqual(index.graph_note(entry, limit=3),
                         "defines: s0, s1, s2, +7 more")


class CandidatesTextTests(TempRepoTestCase):
    # bug:d4b7df1 - merging, rank order, previews, missing files, graph notes.

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

    def candidates(self, graph=None):
        return index.candidates_text(str(self.work), self.hits, full=1, preview_lines=3,
                                     graph=graph)

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

    def test_graph_notes_and_uses_appear_only_with_a_graph(self):
        self.build()
        without = self.candidates()
        self.assertNotIn("defines:", without)
        self.assertNotIn("uses:", without)
        graph = index.build_graph(str(self.work), index.list_files(str(self.work)))
        with_graph = self.candidates(graph)
        self.assertIn("defines: a_main", with_graph)
        self.assertIn("imports: b.py", with_graph)
        self.assertIn("uses: b_helper (b.py)", with_graph)
        self.assertIn("defines: b_helper", with_graph)
        self.assertIn("imported by: a.py", with_graph)


class SymbolMatchesTests(TempRepoTestCase):
    # bug:d4b7df1 - only code-like tokens are searched; found and missing names.

    def build(self):
        self.write("util.py", "def my_func():\n    pass\n"
                              "CAPS_CONST = 3\n"
                              "camelCaseThing = 1\n")
        self.write("other.py", "from util import my_func\nresult = my_func()\n")
        self.write("dup.py", "".join(f"dup_name = {n}\n" for n in range(13)))

    def matches(self, query, graph=None):
        return index.symbol_matches(str(self.work), query, graph=graph)

    def test_found_symbols_print_their_matches_and_edges(self):
        self.build()
        graph = index.build_graph(str(self.work), index.list_files(str(self.work)))
        text = self.matches("where is my_func used?", graph)
        self.assertIn("my_func (3 matches):", text)
        self.assertIn("util.py:1:def my_func():", text)
        self.assertIn("other.py:2:result = my_func()", text)
        self.assertIn("defined in util.py:1; used by other.py", text)

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


class StructureTextTests(TempRepoTestCase):
    # bug:d4b7df1 - best hits' notes and uses, then edges for named symbols.

    def build(self):
        self.write("util.py", "def helper():\n    pass\n")
        self.write("app.py", "from util import helper\nhelper()\n")
        self.graph = index.build_graph(str(self.work), index.list_files(str(self.work)))
        self.hits = [{"path": "app.py", "start": 1, "end": 2, "score": 0.5, "text": ""}]

    def test_best_hit_files_and_named_symbols_come_out_as_edges(self):
        self.build()
        text = index.structure_text(str(self.work), self.hits, "where is helper used",
                                    self.graph)
        lines = text.splitlines()
        self.assertEqual(lines, [
            "app.py",
            "  imports: util.py",
            "  uses: helper (util.py)",
            "helper",
            "  defined in util.py:1; used by app.py",
        ])

    def test_a_graph_that_knows_nothing_yields_no_text(self):
        self.build()
        self.assertEqual(index.structure_text(str(self.work), [], "how does this work",
                                              self.graph), "")
        self.assertEqual(index.structure_text(str(self.work), self.hits, "no names here",
                                              {}), "")


if __name__ == "__main__":
    unittest.main()
