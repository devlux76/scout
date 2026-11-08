#!/usr/bin/env python3
"""scout — retrieval only: candidate locations and the code graph for a question.

No model runs here. Candidates come from Trailblazer's definition graph and
vectors, for a repository it watches; without it scout says so and exits 1.
Exact-name matches for symbols in the question are added, and the graph around
the best hits follows. ranger.py hands the same output to its model to rerank
and analyse.

Usage:
    python3 scout.py -C <repo> -q "<question>"
    python3 scout.py -C <repo> -q "<question>" --verbose
"""

import argparse
import sys
from pathlib import Path

PROG = Path(sys.argv[0]).stem if Path(sys.argv[0]).stem in ("scout", "ranger") else "scout"

UNSET = object()  # "the caller did not ask", so Trailblazer's status is asked once


def skill_modules():
    """index.py and trailblazer.py, which live beside this file."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import index
    import trailblazer
    return index, trailblazer


def trailblazer_status(work_dir, watched=UNSET, tb=None):
    """Trailblazer's status for this repository: asked once per run by main() and
    handed to every caller that needs it."""
    if watched is not UNSET:
        return watched
    return (tb or skill_modules()[1]).watched(work_dir)


def trailblazer_ready(work_dir):
    """Trailblazer's status for this repository, starting the service and asking it
    to watch the repository first when needed; None when it cannot answer, and
    then there are no candidates. Asked once per run."""
    try:
        _index, trailblazer = skill_modules()
        return trailblazer.ensure_watched(work_dir)
    except Exception as e:
        print(f"{PROG}: Trailblazer unavailable ({e})", file=sys.stderr)
        return None


def candidates_with_source(work_dir, query, verbose, watched=UNSET):
    """Candidate locations as (prompt text, structure appendix, source).

    The source is "trailblazer" when Trailblazer answered, and None when it does
    not watch the repository or cannot answer at all: then the only output is a
    hint on stderr, and there are no candidates. `watched` is Trailblazer's status,
    when the caller has it.

    Trailblazer's definition graph and vectors are used as they are, so the
    candidates are ready in seconds. When it watches the repository but has no
    hits, the hint says whether its vectors are still being built; the empty
    answer is reported, not papered over with a second index.
    """
    try:
        index, trailblazer = skill_modules()
    except Exception as e:
        print(f"scout: cannot load its modules ({e})", file=sys.stderr)
        return "", "", None
    watched = trailblazer_status(work_dir, watched, trailblazer)
    if watched is None:
        print(f"scout: Trailblazer is not watching {work_dir} or is unavailable; start it "
              f"(systemctl --user start trailblazer) or run: trailblazer watch {work_dir}",
              file=sys.stderr)
        return "", "", None

    hits = trailblazer.search_hits(work_dir, query)
    pending = watched.get("pending_vectors", 0)
    if not hits and (pending > 0 or watched.get("syncing") or watched.get("deferred")):
        print(f"scout: Trailblazer is still building vectors for this repo "
              f"(pending_vectors={pending}); results are incomplete", file=sys.stderr)
    source = "trailblazer"
    text = index.candidates_text(work_dir, hits) if hits else ""
    appendix = trailblazer.structure_text(work_dir, hits, query) if hits else ""

    if verbose:
        print(f"  candidates from {source}", file=sys.stderr)
        for h in hits:
            print(f"  {source}: {h['score']:.3f} {h['path']}:{h['start']}-{h['end']}", file=sys.stderr)
    names = index.symbol_matches(work_dir, query)
    if names:
        text += "\n\nExact-name matches for symbols in the question (ripgrep, path:line:text):\n\n" + names
    return text, appendix, source


def index_candidates(work_dir, query, verbose, watched=UNSET):
    """Candidate locations for the question as (prompt text, structure appendix),
    or ("", "") if Trailblazer cannot answer. The compatible wrapper over
    candidates_with_source for callers that do not ask which source answered."""
    text, appendix, _source = candidates_with_source(work_dir, query, verbose, watched)
    return text, appendix


def main():
    parser = argparse.ArgumentParser(description="Candidate locations and the code graph for a question, without a model")
    parser.add_argument("-C", "--directory", default=".", help="Repository directory")
    parser.add_argument("-q", "--query", required=True, help="Question about the codebase")
    parser.add_argument("--verbose", action="store_true", help="Print the source and hits to stderr")
    args = parser.parse_args()

    work_dir = str(Path(args.directory).resolve())
    if not Path(work_dir).is_dir():
        print(f"Error: {work_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    candidates, structure, source = candidates_with_source(
        work_dir, args.query, args.verbose, trailblazer_ready(work_dir))
    if source is None:
        print("scout: no candidates (see the hint above)", file=sys.stderr)
        sys.exit(1)
    if not candidates:
        print("Trailblazer answered but found no candidate locations for this question",
              file=sys.stderr)
        sys.exit(2)
    print("--- candidates, best match first ---\n" + candidates)
    if structure:
        print("\n--- structure ---\n" + structure)


if __name__ == "__main__":
    main()
