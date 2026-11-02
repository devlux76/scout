#!/usr/bin/env python3
"""Semantic index for ranger: repo files -> EmbeddingGemma 2 embeddings -> Qdrant.

sync() embeds new or changed files (incremental, keyed on mtime+size) and
search() returns the chunks closest to a question, so the ranger model starts
from candidate locations instead of exploring blind.

Standalone use (build or refresh the index, then optionally query it):
    python3 index.py -C <repo>
    python3 index.py -C <repo> -q "<question>"
    python3 index.py -C <repo> -q "<question>" --own-index

A query is answered by Trailblazer instead whenever it watches the repository:
its graph and vectors are already there, so nothing is embedded. --own-index
forces the index below to be synced and searched as it always was.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

# The skill that is running (scout or ranger share this module); names its messages.
PROG = Path(sys.argv[0]).stem if Path(sys.argv[0]).stem in ("scout", "ranger") else "ranger"

import httpx

try:
    import trailblazer
except ImportError:  # scout ships index.py without the client
    trailblazer = None

EMBED_URL = "http://127.0.0.1:8091/v1/embeddings"
EMBED_MODEL = "embeddinggemma-2"
EMBED_DIM = 768
QDRANT_URL = "http://127.0.0.1:6333"
MANIFEST_DIR = Path.home() / ".local/share/ranger/manifests"

CHUNK_LINES = 40
CHUNK_OVERLAP = 8
MAX_CHUNK_CHARS = 1800      # keeps a chunk well inside the embedder's context
MAX_FILE_BYTES = 400_000    # skip generated or data files
MAX_DATA_BYTES = 30_000     # data formats: index configs, not generated datasets
DATA_SUFFIXES = {".json", ".yaml", ".yml", ".txt"}
EMBED_BATCH = 16
TEXT_SUFFIXES = {
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".go", ".rs", ".c", ".h",
    ".cc", ".cpp", ".hpp", ".java", ".kt", ".rb", ".ex", ".exs", ".sh", ".sql",
    ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".html", ".css", ".glsl",
    ".wgsl", ".fx", ".cmake", ".lua", ".swift", ".cs", ".php",
}


def collection_name(work_dir):
    # The embedding model is part of the name: vectors from two models are never
    # comparable, so a change of model starts a fresh collection and re-embeds.
    return "ranger_" + hashlib.sha1(f"{EMBED_MODEL}\n{work_dir}".encode()).hexdigest()[:12]


def list_files(work_dir):
    """Text files, tracked or not. ripgrep skips hidden directories (so never .git)
    and honours .gitignore; --no-require-git applies it outside git repos too."""
    out = subprocess.run(["rg", "--files", "--no-require-git"], cwd=work_dir, capture_output=True, text=True)
    files = {}
    for rel in out.stdout.splitlines():
        p = Path(work_dir, rel)
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_size == 0 or st.st_size > MAX_FILE_BYTES:
            continue
        if p.suffix.lower() in DATA_SUFFIXES and st.st_size > MAX_DATA_BYTES:
            continue
        files[rel] = [int(st.st_mtime), st.st_size]
    return files


def chunk_file(work_dir, rel):
    """Overlapping line windows as (start_line, end_line, text)."""
    lines = Path(work_dir, rel).read_text(errors="replace").splitlines()
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + CHUNK_LINES, len(lines))
        text = "\n".join(lines[start:end])[:MAX_CHUNK_CHARS]
        if text.strip():
            chunks.append((start + 1, end, text))
        if end == len(lines):
            break
        start = end - CHUNK_OVERLAP
    return chunks


JS_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
JS_IMPORT = re.compile(r"""(?:^|[;\s])(?:import|export)\s[^'"`;]*?from\s*['"]([^'"]+)['"]|(?:^|[;\s])import\s*['"]([^'"]+)['"]|(?:require|import)\(\s*['"]([^'"]+)['"]\s*\)""", re.M)
JS_SYMBOL = re.compile(r"^(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?(?:function\*?|class|const|let|var|interface|type|enum)\s+([A-Za-z_$][\w$]*)", re.M)
PY_IMPORT = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.M)
PY_SYMBOL = re.compile(r"^(?:(?:async\s+)?def|class)\s+(\w+)|^([A-Z][A-Z0-9_]{2,})\s*(?::[^=\n]+)?=", re.M)


def build_graph(work_dir, files):
    """Dependency map of the repo, from pattern matching (no parser).

    For each file: `imports` (repo files), `packages` (external modules),
    `imported_by` (repo files) and `symbols` as (name, line) it defines.
    TypeScript/JavaScript and Python are understood; other files get empty
    entries. Dynamically built import paths are not seen.
    """
    known = set(files)
    graph = {rel: {"imports": [], "packages": [], "imported_by": [], "symbols": []} for rel in files}

    def resolve_js(rel, spec):
        base = (Path(rel).parent / spec).as_posix() if spec.startswith(".") else spec
        base = Path(base)
        # normalise "a/../b" without touching the filesystem
        parts = []
        for part in base.parts:
            if part == "..":
                if parts:
                    parts.pop()
            elif part != ".":
                parts.append(part)
        stem = "/".join(parts)
        tries = [stem] + [stem + e for e in JS_SUFFIXES] + [f"{stem}/index{e}" for e in JS_SUFFIXES]
        if stem.endswith(".js"):  # TS sources imported with a .js specifier
            tries += [stem[:-3] + ".ts", stem[:-3] + ".tsx"]
        return next((t for t in tries if t in known), None)

    def resolve_py(rel, module):
        stem = module.replace(".", "/")
        here = Path(rel).parent.as_posix()
        tries = [f"{stem}.py", f"{stem}/__init__.py"]
        if here != ".":
            tries += [f"{here}/{stem}.py", f"{here}/{stem}/__init__.py"]
        return next((t for t in tries if t in known), None)

    for rel in files:
        suffix = Path(rel).suffix.lower()
        if suffix not in JS_SUFFIXES and suffix != ".py":
            continue
        try:
            text = Path(work_dir, rel).read_text(errors="replace")
        except OSError:
            continue
        entry = graph[rel]
        entry["idents"] = set(re.findall(r"[A-Za-z_$][\w$]*", text))
        if suffix == ".py":
            specs = [a or b for a, b in PY_IMPORT.findall(text)]
            resolve, sym_re = resolve_py, PY_SYMBOL
        else:
            specs = [a or b or c for a, b, c in JS_IMPORT.findall(text)]
            resolve, sym_re = resolve_js, JS_SYMBOL
        for spec in dict.fromkeys(specs):
            target = resolve(rel, spec)
            if target and target != rel:
                entry["imports"].append(target)
            elif not target and not spec.startswith("."):
                # keep the package name, not the deep path: "@scope/pkg/x/y" -> "@scope/pkg"
                bits = spec.split("/")
                entry["packages"].append("/".join(bits[:2]) if spec.startswith("@") else bits[0].split(".")[0])
        entry["packages"] = list(dict.fromkeys(entry["packages"]))
        for m in sym_re.finditer(text):
            name = next(g for g in m.groups() if g)
            entry["symbols"].append((name, text.count("\n", 0, m.start()) + 1))
    for rel, entry in graph.items():
        for target in entry["imports"]:
            graph[target]["imported_by"].append(rel)
    return graph


def symbol_edges(graph, name, limit=8):
    """Symbol-level edges for one name: where it is defined, which files use it
    through an import of the defining file, and where the same name appears with
    no import link (a different thing, or reached some other way)."""
    def some(items):
        return ", ".join(items[:limit]) + (f", +{len(items) - limit} more" if len(items) > limit else "")
    lines = []
    definers = [(rel, line) for rel, e in graph.items() for n, line in e["symbols"] if n == name]
    for rel, line in definers:
        users = [u for u in graph[rel]["imported_by"] if name in graph[u].get("idents", ())]
        lines.append(f"defined in {rel}:{line}; used by " + (some(users) if users else "no other file (local to its file)"))
    linked = {rel for rel, _ in definers} | {u for rel, _ in definers for u in graph[rel]["imported_by"]}
    strangers = [rel for rel, e in graph.items() if name in e.get("idents", ()) and rel not in linked]
    if definers and strangers:
        lines.append("same name, no import link to a definition: " + some(strangers))
    return "\n".join(lines)


def file_uses(graph, rel, limit=12):
    """Symbols a file takes from the repo files it imports, as 'name (file)'."""
    idents = graph[rel].get("idents", ())
    used = [f"{n} ({t})" for t in graph[rel]["imports"] for n, _ in graph[t]["symbols"] if n in idents]
    return used[:limit] + ([f"+{len(used) - limit} more"] if len(used) > limit else [])


def graph_note(entry, start=None, end=None, limit=8):
    """Short lines describing a file's place in the graph, for prompts and embeddings.
    With a line range, only the symbols defined inside it are listed."""
    def some(items):
        return ", ".join(items[:limit]) + (f", +{len(items) - limit} more" if len(items) > limit else "")
    names = [n for n, line in entry["symbols"] if start is None or start <= line <= end]
    lines = []
    if names:
        lines.append("defines: " + some(names))
    if entry["imports"] or entry["packages"]:
        lines.append("imports: " + some(entry["imports"] + entry["packages"]))
    if entry["imported_by"]:
        lines.append("imported by: " + some(entry["imported_by"]))
    return "\n".join(lines)


def embed_documents(client, titled):
    """Embed (title, text) pairs as documents, in the form EmbeddingGemma 2 asks for."""
    return _embed(client, [f"title: {title or 'none'} | text: {text}" for title, text in titled])


def embed_query(client, query):
    """Embed a question about code, in the form EmbeddingGemma 2 asks for."""
    return _embed(client, [f"task: code retrieval | query: {query}"])[0]


def _embed(client, inputs):
    resp = client.post(EMBED_URL, json={"model": EMBED_MODEL, "input": inputs})
    resp.raise_for_status()
    return [d["embedding"] for d in resp.json()["data"]]


def sync(work_dir, verbose=False):
    """Bring the repo's collection up to date. Returns the collection name."""
    name = collection_name(work_dir)
    manifest_path = MANIFEST_DIR / f"{name}.json"
    with httpx.Client(timeout=600.0) as client:
        if client.get(f"{QDRANT_URL}/collections/{name}").status_code == 404:
            client.put(
                f"{QDRANT_URL}/collections/{name}",
                json={"vectors": {"size": EMBED_DIM, "distance": "Cosine"}},
            ).raise_for_status()
            manifest = {}
        else:
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

        current = list_files(work_dir)
        stale = [rel for rel in manifest if current.get(rel) != manifest[rel]]
        todo = [rel for rel in current if manifest.get(rel) != current[rel]]
        if not stale and not todo:
            return name

        if stale:
            # Drop every chunk of files that changed or disappeared before re-adding them.
            client.post(
                f"{QDRANT_URL}/collections/{name}/points/delete?wait=true",
                json={"filter": {"must": [{"key": "path", "match": {"any": stale}}]}},
            ).raise_for_status()
            for rel in stale:
                manifest.pop(rel, None)

        started = time.time()
        graph = build_graph(work_dir, current)
        pending = []  # (rel, start_line, end_line, text)
        for rel in todo:
            pending.extend((rel, s, e, t) for s, e, t in chunk_file(work_dir, rel))
        if verbose or len(pending) > EMBED_BATCH:
            print(f"ranger index: embedding {len(pending)} chunks from {len(todo)} files", file=sys.stderr)
        # Chunks still owed per file. When a file's last chunk has landed the file
        # goes into the manifest, which is saved as the run proceeds, so a run that
        # is interrupted (a crash, or Ctrl-C) resumes where it left off.
        owed = {}
        for rel, _, _, _ in pending:
            owed[rel] = owed.get(rel, 0) + 1
        for rel in todo:
            if rel not in owed:
                manifest[rel] = current[rel]
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(pending), EMBED_BATCH):
            batch = pending[i:i + EMBED_BATCH]
            # The graph note heads the embedded text so a chunk is also findable by
            # what it defines and depends on; the same facts go in the payload as tags.
            vectors = embed_documents(
                client,
                [(rel, f"{graph_note(graph[rel], s, e)}\n{text}"[:MAX_CHUNK_CHARS + 600]) for rel, s, e, text in batch],
            )
            points = [
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{rel}:{s}")),
                    "vector": vec,
                    "payload": {
                        "path": rel, "start": s, "end": e, "text": text,
                        "symbols": [n for n, line in graph[rel]["symbols"] if s <= line <= e],
                        "imports": graph[rel]["imports"],
                        "packages": graph[rel]["packages"],
                    },
                }
                for (rel, s, e, text), vec in zip(batch, vectors)
            ]
            client.put(f"{QDRANT_URL}/collections/{name}/points?wait=true", json={"points": points}).raise_for_status()
            finished = False
            for rel, _, _, _ in batch:
                owed[rel] -= 1
                if owed[rel] == 0:
                    manifest[rel] = current[rel]
                    finished = True
            if finished:
                manifest_path.write_text(json.dumps(manifest))
            if verbose:
                print(f"  {min(i + EMBED_BATCH, len(pending))}/{len(pending)} ({time.time() - started:.0f}s)", file=sys.stderr)

        # A file enters the manifest only once every one of its chunks has landed,
        # so an interrupted run re-embeds the unfinished files instead of leaving
        # silent gaps.
        manifest_path.write_text(json.dumps(manifest))
    return name


def search(work_dir, query, limit=8):
    """Closest chunks as dicts with path, start, end, text, score."""
    name = collection_name(work_dir)
    with httpx.Client(timeout=120.0) as client:
        vector = embed_query(client, query)
        resp = client.post(
            f"{QDRANT_URL}/collections/{name}/points/query",
            json={"query": vector, "limit": limit, "with_payload": True},
        )
        resp.raise_for_status()
        return [{**p["payload"], "score": p["score"]} for p in resp.json()["result"]["points"]]


def candidates_text(work_dir, hits, full=4, preview_lines=3, graph=None):
    """Format hits for the ranger prompt, best match first.

    The best `full` locations are shown whole with line numbers, read from disk so
    the numbers are current, which lets the model answer without a Read call.
    Overlapping chunks of one file are merged into a single range. The remaining
    hits are listed as a location and a few lines, as leads to follow with tools.
    """
    merged = []  # [path, start, end, best score] in rank order
    for h in hits:
        for m in merged:
            if m[0] == h["path"] and h["start"] <= m[2] + 1 and h["end"] >= m[1] - 1:
                m[1], m[2] = min(m[1], h["start"]), max(m[2], h["end"])
                break
        else:
            merged.append([h["path"], h["start"], h["end"], h["score"]])
    blocks = []
    for i, (rel, start, end, score) in enumerate(merged):
        path = Path(work_dir, rel)
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        last = end if i < full else min(end, start + preview_lines - 1)
        body = "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, min(last, len(lines)) + 1))
        label = f"{path}:{start}-{end} (similarity {score:.2f})" + ("" if i < full else " (preview)")
        note = graph_note(graph[rel]) if graph and rel in graph else ""
        uses = file_uses(graph, rel) if note else []
        if uses:
            note += "\nuses: " + ", ".join(uses)
        blocks.append(f"{label}\n{note}\n{body}" if note else f"{label}\n{body}")
    return "\n\n".join(blocks)


def symbol_matches(work_dir, query, max_symbols=6, max_lines=12, graph=None):
    """Exact-name ripgrep matches for identifiers named in the question.

    The one-hop spread beside the vector search: where each named symbol is defined,
    assigned and used, so the model can judge whether similarly named things are
    really connected. Only tokens that look like code are tried (snake_case,
    camelCase, CAPITALS, or anything in backticks), not ordinary words.
    """
    quoted = re.findall(r"`([^`]+)`", query)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)
    codey = [w for w in words if "_" in w or re.search(r"[a-z][A-Z]", w) or (w.isupper() and len(w) > 3)]
    symbols = list(dict.fromkeys(quoted + codey))[:max_symbols]
    blocks = []
    for sym in symbols:
        out = subprocess.run(
            # "--" then an explicit "." path: without a path rg reads stdin when it is not a terminal.
            ["rg", "--no-require-git", "-n", "-w", "-F", "--no-heading", "--max-columns", "200", "--", sym, "."],
            cwd=work_dir, capture_output=True, text=True,
        ).stdout.splitlines()
        out = [line.removeprefix("./") for line in out]
        edges = symbol_edges(graph, sym) if graph else ""
        if out:
            more = f"\n... {len(out) - max_lines} more matches" if len(out) > max_lines else ""
            head = f"{sym} ({len(out)} matches):\n" + (edges + "\n" if edges else "")
            blocks.append(head + "\n".join(out[:max_lines]) + more)
        else:
            blocks.append(f"{sym}: no exact match in the repository")
    return "\n\n".join(blocks)


def structure_text(work_dir, hits, query, graph, files=4):
    """The dependency edges for the best-matching files and for symbols named in
    the question, as plain labelled lines. Printed after the model's answer so
    the caller has the real graph whatever the model made of it."""
    lines = []
    seen = []
    for h in hits:
        if h["path"] not in seen:
            seen.append(h["path"])
    for rel in seen[:files]:
        if rel not in graph:
            continue
        note = graph_note(graph[rel])
        uses = file_uses(graph, rel)
        if note or uses:
            lines.append(rel)
            lines += ["  " + l for l in note.splitlines()]
            if uses:
                lines.append("  uses: " + ", ".join(uses))
    quoted = re.findall(r"`([^`]+)`", query)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)
    for sym in dict.fromkeys(quoted + words):
        edges = symbol_edges(graph, sym)
        if edges:
            lines.append(sym)
            lines += ["  " + l for l in edges.splitlines()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build or query ranger's semantic index")
    parser.add_argument("-C", "--directory", default=".", help="Repository directory")
    parser.add_argument("-q", "--query", help="Question to search for after syncing")
    parser.add_argument("--own-index", action="store_true",
                        help="Use ranger's own index even where Trailblazer watches the repo")
    args = parser.parse_args()
    work_dir = str(Path(args.directory).resolve())

    # A query against a repository Trailblazer watches is answered from its graph
    # and vectors: nothing to embed, so the answer comes back in a moment. When
    # it has no hits to give (vectors still being built, or the call failed) the
    # index below is synced and searched instead, as it always was.
    if args.query and not args.own_index and trailblazer is not None:
        status = trailblazer.watched(work_dir)
        if status:
            started = time.time()
            hits = trailblazer.search_hits(work_dir, args.query)
            if hits:
                print(f"trailblazer: {status.get('nodes', 0)} definitions "
                      f"({time.time() - started:.1f}s)", file=sys.stderr)
                for h in hits:
                    print(f"{h['score']:.3f}  {h['path']}:{h['start']}-{h['end']}")
                return
            print(f"{PROG}: trailblazer watches the repo but has no hits for this question "
                  "(vectors still building, or its search failed); using ranger's own index",
                  file=sys.stderr)

    started = time.time()
    sync(work_dir, verbose=True)
    print(f"index up to date ({time.time() - started:.1f}s)", file=sys.stderr)
    if args.query:
        for h in search(work_dir, args.query):
            print(f"{h['score']:.3f}  {h['path']}:{h['start']}-{h['end']}")


if __name__ == "__main__":
    main()
