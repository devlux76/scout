#!/usr/bin/env python3
"""Trailblazer over HTTP: scout's and ranger's candidates, for a watched repo.

Trailblazer keeps a definition graph and one vector per definition for every
repository it watches, and it is always current. Its answers are used as they
are; nothing here builds or refreshes an index.

Nothing here raises. A missing endpoint file, a service that is not running, a
timeout, a bad token or any non-2xx answer all come back as None, and the caller
reports that Trailblazer cannot answer. The one answer worth telling apart is
Trailblazer's own "this repository is not watched" (409), which is NOT_WATCHED
rather than None.

    call(op, body, timeout=10)          any operation, decoded JSON or None
    watched(work_dir)                    the status dict, or None
    ensure_watched(work_dir)             the same, starting the service and asking
                                         it to watch the repository when needed
    search_hits(work_dir, query)         hits in index.py's shape, [] on failure
    where_lines(work_dir, name)          declarations, [] when there are none
    uses_edges(work_dir, ref)            users of a definition, [] when there are none
    structure_text(work_dir, hits, q)    the appendix, in index.py's style

The graph questions answer [] only for a real empty answer and None when the
service could not answer at all, so that a caller never turns a dead service
into the claim that a name is declared nowhere.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# The skill that is running (scout or ranger share this module); names its messages.
PROG = Path(sys.argv[0]).stem if Path(sys.argv[0]).stem in ("scout", "ranger") else "ranger"

import httpx

STATE_DIR_ENV = "TRAILBLAZER_STATE_DIR"
TIMEOUT = 10            # seconds per operation
# A search embeds the question on the embedder Trailblazer's own indexing shares,
# so it waits behind that queue: 12.3 s measured 2026-10-22 with 8,437 vectors
# pending. Past the plain TIMEOUT that read as "no hits", so it is a backstop.
SEARCH_TIMEOUT = 120
MAX_HIT_CHARS = 1800    # keeps one definition inside the embedder's context, as index.py does
MAX_USERS = 8           # users listed per definition, then "+N more"
MAX_SYMBOLS = 6         # symbols taken from the question
STRUCTURE_FILES = 4     # best hit definitions whose users are listed
SERVICE = "trailblazer.service"
# A backstop, not a wait (design §13.2): the service announces itself by publishing
# its endpoint file once the API listens, so reaching this means the start went
# wrong. Measured 2026-10-22: the endpoint was published 1.5 s after the start.
START_BACKSTOP_S = 300


class _NotWatched:
    """Trailblazer's 409: the repository is not one it watches."""

    def __repr__(self):
        return "NOT_WATCHED"


NOT_WATCHED = _NotWatched()


def state_dir():
    """Where Trailblazer keeps its endpoint file ($TRAILBLAZER_STATE_DIR wins)."""
    override = os.environ.get(STATE_DIR_ENV)
    if override:
        return Path(override)
    return Path.home() / ".local/share/trailblazer"


def endpoint():
    """(url, token) from the endpoint file, or None when it is not there."""
    try:
        data = json.loads((state_dir() / "endpoint.json").read_text())
    except Exception:
        return None
    url, token = data.get("url"), data.get("token")
    if not url or not token:
        return None
    return url.rstrip("/"), token


def call(op, body, timeout=TIMEOUT):
    """POST <url>/<op> with the body; the decoded JSON, NOT_WATCHED, or None."""
    ep = endpoint()
    if ep is None:
        return None
    url, token = ep
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{url}/{op}",
                json=body,
                headers={"authorization": f"Bearer {token}"},
            )
    except Exception:
        return None
    if resp.status_code == 409:
        return NOT_WATCHED
    if not 200 <= resp.status_code < 300:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def watched(work_dir):
    """Trailblazer's status for this repository, or None if it cannot answer."""
    status = call("status", {"repo": str(work_dir)})
    if not isinstance(status, dict) or status.get("error"):
        return None
    return status


def watch(work_dir):
    """Ask Trailblazer to watch this repository; True once it has agreed."""
    answer = call("watch", {"repo": str(work_dir)})
    return isinstance(answer, dict) and answer.get("watched") is True


def start_service():
    """Start Trailblazer when it is stopped and wait for its endpoint file.

    True once it has published one. A service that is running but not answering
    is left alone: it is busy or wedged, and a restart would throw away whatever
    it was indexing. The wait is on the file event, set up before the start so the
    publish cannot be missed; the timeout only catches a start that went wrong.
    """
    active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", SERVICE])
    if active.returncode == 0:
        print(f"{PROG}: Trailblazer is running but not answering; leaving it alone", file=sys.stderr)
        return False
    folder = state_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        waiter = subprocess.Popen(
            ["inotifywait", "-e", "moved_to,create", "--include", r"/endpoint\.json$",
             "-t", str(START_BACKSTOP_S), str(folder)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except OSError as e:
        print(f"{PROG}: cannot wait for Trailblazer ({e}); not starting it", file=sys.stderr)
        return False
    setup = []
    for line in waiter.stderr:
        setup.append(line)
        if "Watches established" in line:
            break
    else:
        waiter.wait()
        # inotifywait's own words first: "Too many open files" is the host's
        # inotify instance limit, and only the operator can raise it.
        detail = "".join(setup).strip() or "inotifywait said nothing"
        print(f"{PROG}: cannot wait for Trailblazer (inotifywait could not set up its watches: "
              f"{detail}); not starting it; raise fs.inotify.max_user_instances if it ran out",
              file=sys.stderr)
        return False
    print(f"{PROG}: starting Trailblazer", file=sys.stderr)
    started = subprocess.run(["systemctl", "--user", "start", SERVICE])
    if started.returncode != 0:
        waiter.kill()
        waiter.wait()
        print(f"{PROG}: Trailblazer did not start (systemctl failed)", file=sys.stderr)
        return False
    if waiter.wait() != 0:
        print(f"{PROG}: anomaly: Trailblazer started but published no endpoint within "
              f"{START_BACKSTOP_S}s; see journalctl --user -u {SERVICE}", file=sys.stderr)
        return False
    return True


def ensure_watched(work_dir):
    """Trailblazer's status for this repository, as watched() gives it, after
    starting a stopped service and asking it to watch an unwatched repository.
    None when it still cannot answer; the caller then has no candidates."""
    body = {"repo": str(work_dir)}
    status = call("status", body)
    if status is None and start_service():
        status = call("status", body)
    if status is NOT_WATCHED:
        print(f"{PROG}: Trailblazer does not watch this repository; asking it to", file=sys.stderr)
        if watch(work_dir):
            status = call("status", body)
    if not isinstance(status, dict) or status.get("error"):
        return None
    return status


def read_range(work_dir, path, start, end):
    """Lines start..end of a file (1-indexed, inclusive), read from disk."""
    try:
        lines = Path(work_dir, path).read_text(errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[max(start - 1, 0):end])


def node_line(node):
    """One definition as 'path:start-end qualified'."""
    return f"{node.get('path')}:{node.get('start_line')}-{node.get('end_line')} {node.get('qualified')}"


def edge_line(edge):
    """One use as 'path:line qualified (kind[, uncertain][, provenance])'.

    Provenance says how the graph inferred an edge it could not resolve
    directly, e.g. Trailblazer's 'default_module' for a call through an
    injected module (`mod(st, :workspace, Workspace)`); it is kept so such an
    edge is never read as a definite call."""
    kind = edge.get("kind") or "uses"
    if edge.get("uncertain"):
        kind += ", uncertain"
    provenance = edge.get("provenance")
    if isinstance(provenance, str) and provenance.strip():
        kind += ", " + provenance.strip().replace("_", " ")
    return f"{edge.get('node', {}).get('path')}:{edge.get('line')} {edge.get('node', {}).get('qualified')} ({kind})"


def search_hits(work_dir, query, limit=8):
    """Trailblazer's closest definitions as index.py's hits:
    path, start, end, score, text (plus the node's id, kind and qualified, which
    structure_text needs to ask who uses it). Empty list on any failure."""
    found = call("search", {"repo": str(work_dir), "text": query, "limit": limit},
                 timeout=SEARCH_TIMEOUT)
    if not isinstance(found, list):
        return []
    hits = []
    for hit in found:
        node = hit.get("node") if isinstance(hit, dict) else None
        if not isinstance(node, dict) or not node.get("path"):
            continue
        start = node.get("start_line") or 1
        end = node.get("end_line") or start
        text = read_range(work_dir, node["path"], start, end)[:MAX_HIT_CHARS]
        hits.append({
            "path": node["path"],
            "start": start,
            "end": end,
            "score": float(hit.get("score") or 0.0),
            "text": text,
            "id": node.get("id"),
            "kind": node.get("kind"),
            "qualified": node.get("qualified"),
        })
    return hits


def where_lines(work_dir, name):
    """Where a name is declared: Trailblazer answers with bare node summaries.
    None when the graph could not answer — which is not the same as a name that
    is declared nowhere."""
    found = call("where", {"repo": str(work_dir), "name": name})
    if not isinstance(found, list):
        return None
    return [n for n in found if isinstance(n, dict) and n.get("path")]


def uses_edges(work_dir, ref):
    """Who uses a definition, given its id or its name. One entry per edge into it.
    None when the graph could not answer, for the reason above."""
    field = "id" if isinstance(ref, int) else "name"
    found = call("uses", {"repo": str(work_dir), field: ref})
    if not isinstance(found, list):
        return None
    return [e for e in found if isinstance(e, dict) and isinstance(e.get("node"), dict)]


QUOTED = re.compile(r"`([^`]+)`")
DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def symbols_in(query, limit=MAX_SYMBOLS):
    """Names in the question that could be code: in backticks, qualified as
    Foo.bar, snake_case or camelCase. Ordinary words are left out."""
    quoted = QUOTED.findall(query)
    words = [w for w in WORD.findall(query) if "_" in w or re.search(r"[a-z][A-Z]", w)]
    dotted = [d for d in DOTTED.findall(query) if d not in words]
    return list(dict.fromkeys(quoted + dotted + words))[:limit]


def _users(work_dir, ref):
    """The users of one definition, indented and capped; None when the graph could
    not answer, which is not a definition nobody uses."""
    edges = uses_edges(work_dir, ref)
    if edges is None:
        return None
    lines = [edge_line(e) for e in edges[:MAX_USERS]]
    if len(edges) > MAX_USERS:
        lines.append(f"... +{len(edges) - MAX_USERS} more users")
    return lines


def structure_text(work_dir, hits, query, files=STRUCTURE_FILES):
    """The dependency edges for the best-matching definitions and for the symbols
    the question names, as plain labelled lines in index.structure_text's style.
    Appended to the model's answer, so the caller has the graph as Trailblazer
    computed it whatever the model made of it.

    A graph that cannot answer says so, or is left out; it never reports that a
    definition is unused or a name undeclared, because "not asked" and "none" are
    not the same fact.
    """
    try:
        return _structure_text(work_dir, hits, query, files)
    except Exception:
        return ""


def _structure_text(work_dir, hits, query, files):
    lines = []
    chosen = []  # [(key, hit)] for the best definitions, in rank order
    for h in hits:
        key = h.get("id", (h["path"], h["start"]))
        if key not in [k for k, _ in chosen]:
            chosen.append((key, h))
        if len(chosen) == files:
            break
    for key, h in chosen:
        lines.append(f"{h.get('qualified') or h['path']}  {h['path']}:{h['start']}-{h['end']}")
        users = _users(work_dir, key) if isinstance(key, int) else []
        if users is None:
            lines.append("  users unknown: graph unavailable")
        elif not users:
            lines.append("  no other definition uses it")
        else:
            lines.append("  used by:")
            lines += ["    " + u for u in users]

    for sym in symbols_in(query):
        nodes = where_lines(work_dir, sym)
        if not nodes:
            # A name the graph could not look up is left out, not declared nowhere.
            continue
        lines.append(sym)
        lines += ["  declared: " + node_line(n) for n in nodes[:MAX_USERS]]
        users = _users(work_dir, nodes[0].get("id")) if nodes[0].get("id") else []
        if users is None:
            lines.append("  users unknown: graph unavailable")
        elif not users:
            lines.append("  used by: nothing")
        else:
            lines.append("  used by:")
            lines += ["    " + u for u in users]
    return "\n".join(lines)