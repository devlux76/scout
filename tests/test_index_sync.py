#!/usr/bin/env python3
"""Tests for index.py's sync and search against a fake Qdrant and embedder.

Run with:  python3 -m pytest tests -q

bug:d4b7df1 - first run, incremental skip, changed files, resume, payload, and
search. Helpers (FakeIndexService and friends) are in index_support.py.
"""

import json
import unittest
from unittest import mock

from index_support import FakeServiceTestCase

import index


class SyncTests(FakeServiceTestCase):
    # bug:d4b7df1 - first run, incremental skip, changed files, resume, payload.

    def build_repo(self):
        self.write("util.py", "def helper():\n    pass\n")
        self.write("app.py", "from util import helper\nhelper()\n")

    def test_first_run_creates_the_collection_indexes_and_writes_the_manifest(self):
        self.build_repo()
        service = self.use_service()
        name = self.sync_quietly()
        self.assertEqual(name, index.collection_name(str(self.work)))
        # 404 on the first GET caused the collection to be created.
        self.assertIn(("PUT", f"{index.QDRANT_URL}/collections/{name}",
                       {"vectors": {"size": index.EMBED_DIM, "distance": "Cosine"}}),
                      service.requests)
        # Every chunk went to the embedder, titled with its file.
        self.assertEqual(sorted(service.embed_titles()), ["app.py", "util.py"])
        self.assertTrue(all(text.startswith("title: ") for inputs in service.embedded
                            for text in inputs))
        # And every chunk landed as a point carrying the whole payload contract.
        self.assertEqual(len(service.points), 2)
        by_path = {p["payload"]["path"]: p for p in service.points}
        util = by_path["util.py"]["payload"]
        self.assertEqual(set(util), {"path", "start", "end", "text", "symbols",
                                     "imports", "packages"})
        self.assertEqual((util["start"], util["end"]), (1, 2))
        self.assertEqual(util["text"], "def helper():\n    pass")
        self.assertEqual(util["symbols"], ["helper"])
        self.assertEqual(util["imports"], [])
        self.assertEqual(util["packages"], [])
        app = by_path["app.py"]["payload"]
        self.assertEqual(app["imports"], ["util.py"])
        self.assertEqual(app["symbols"], [])
        # The manifest records every file with its mtime and size.
        manifest = json.loads((self.root / "manifests" / f"{name}.json").read_text())
        self.assertEqual(manifest, index.list_files(str(self.work)))

    def test_an_unchanged_repo_makes_no_embed_delete_or_upsert_request(self):
        self.build_repo()
        service = self.use_service()
        self.sync_quietly()
        service.requests.clear()
        service.embedded.clear()
        service.points.clear()
        name = self.sync_quietly()
        self.assertEqual([r[1] for r in service.requests],
                         [f"{index.QDRANT_URL}/collections/{name}"])
        self.assertEqual(service.embedded, [])
        self.assertEqual(service.points, [])

    def test_a_changed_file_is_deleted_then_reembedded_alone(self):
        self.build_repo()
        service = self.use_service()
        self.sync_quietly()
        self.write("util.py", "def helper():\n    pass\n    return 1\n")  # size changes
        service.requests.clear()
        service.embedded.clear()
        service.points.clear()
        self.sync_quietly()
        deletes = service.requests_named("POST", "points/delete")
        self.assertEqual(len(deletes), 1)
        self.assertEqual(deletes[0][2],
                         {"filter": {"must": [{"key": "path",
                                               "match": {"any": ["util.py"]}}]}})
        self.assertEqual(service.embed_titles(), ["util.py"])  # app.py untouched
        self.assertEqual({p["payload"]["path"] for p in service.points}, {"util.py"})

    def test_a_disappearing_file_is_dropped_without_any_embedding(self):
        self.build_repo()
        service = self.use_service()
        self.sync_quietly()
        (self.work / "app.py").unlink()
        service.requests.clear()
        service.embedded.clear()
        service.points.clear()
        name = self.sync_quietly()
        deletes = service.requests_named("POST", "points/delete")
        self.assertEqual(len(deletes), 1)
        self.assertEqual(deletes[0][2],
                         {"filter": {"must": [{"key": "path",
                                               "match": {"any": ["app.py"]}}]}})
        self.assertEqual(service.embedded, [])
        self.assertEqual(service.points, [])
        manifest = json.loads((self.root / "manifests" / f"{name}.json").read_text())
        self.assertEqual(set(manifest), {"util.py"})

    def test_an_interrupted_run_resumes_with_only_the_unfinished_files(self):
        # big.py spans three embed batches; the embedder dies during the run.
        # The manifest must hold exactly the file whose last chunk landed, and
        # the rerun must embed only the rest - no duplicated work.
        self.lines_file("big.py", 85, "line")   # 3 chunks of CHUNK_LINES windows
        self.write("one.py", "ONE = 1\n")
        self.write("two.py", "TWO = 2\n")
        current = index.list_files(str(self.work))
        ordered = {rel: current[rel] for rel in ("big.py", "one.py", "two.py")}
        service = self.use_service()
        service.fail_embed_after = 3
        name = index.collection_name(str(self.work))
        with mock.patch.object(index, "EMBED_BATCH", 1), \
             mock.patch.object(index, "list_files", lambda work_dir: dict(ordered)):
            with self.assertRaises(OSError):
                self.sync_quietly()
        # big.py finished (3 batches), then the embedder failed on one.py.
        saved = json.loads((self.root / "manifests" / f"{name}.json").read_text())
        self.assertEqual(saved, {"big.py": current["big.py"]})
        self.assertEqual(len(service.embedded), 3)  # exactly big.py's three chunks
        self.assertEqual(service.embed_titles(), ["big.py"] * 3)

        # Rerun with a working embedder: only one.py and two.py are embedded.
        service2 = self.use_service()
        service2.collections.add(name)
        with mock.patch.object(index, "list_files", lambda work_dir: dict(ordered)):
            self.assertEqual(self.sync_quietly(), name)
        self.assertEqual(sorted(service2.embed_titles()), ["one.py", "two.py"])
        self.assertEqual({p["payload"]["path"] for p in service2.points},
                         {"one.py", "two.py"})
        final = json.loads((self.root / "manifests" / f"{name}.json").read_text())
        self.assertEqual(final, current)


class SearchTests(FakeServiceTestCase):
    # bug:d4b7df1 - search() embeds the question and returns payload plus score.

    def test_search_returns_the_payload_with_the_score(self):
        self.write("util.py", "def helper():\n    pass\n")
        service = self.use_service()
        service.query_points = [
            {"payload": {"path": "util.py", "start": 1, "end": 2,
                         "text": "def helper():\n    pass"}, "score": 0.87},
        ]
        hits = index.search(str(self.work), "where is helper")
        self.assertEqual(hits, [{"path": "util.py", "start": 1, "end": 2,
                                 "text": "def helper():\n    pass", "score": 0.87}])
        embeds = service.requests_named("POST", index.EMBED_URL)
        self.assertEqual(embeds[0][2]["input"],
                         ["task: code retrieval | query: where is helper"])
        queries = service.requests_named("POST", "points/query")
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0][2],
                         {"query": [0.25] * index.EMBED_DIM, "limit": 8,
                          "with_payload": True})

    def test_search_passes_its_limit_through(self):
        self.write("util.py", "def helper():\n    pass\n")
        service = self.use_service()
        index.search(str(self.work), "helper", limit=3)
        queries = service.requests_named("POST", "points/query")
        self.assertEqual(queries[0][2]["limit"], 3)


if __name__ == "__main__":
    unittest.main()
