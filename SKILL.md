---
name: scout
description: Retrieval only, no model - for a question about a codebase, prints the best-matching locations (file:line ranges), exact-name matches for symbols in the question, and the code graph around them (who imports, calls or uses what), in seconds. From Trailblazer's graph and vectors, for a repo Trailblazer watches; exits 1 with a hint otherwise. Use as the first step before reading, editing, reviewing or debugging code you are not already certain about.
allowed-tools: Bash(python3 *scout/scout.py*)
---

# scout

Reads Trailblazer's graph and vectors and returns the candidates with a graph: no model runs. For a question about a repository it prints the candidate locations, best match first, with their text, then the structure around them. This is the layer ranger builds on; use ranger when the locations alone do not answer the question.

```bash
python3 /home/mirai/.claude/skills/scout/scout.py -C <repo> -q "<question>"
python3 /home/mirai/.claude/skills/scout/scout.py -C <repo> -q "<question>" --verbose   # source and scores on stderr
```

## Where the answer comes from

- **Trailblazer only.** Its definition graph and vectors are used as they are: scout embeds and syncs nothing. If the service is stopped, scout starts it and waits for it to publish its endpoint; if it does not watch the repo, scout asks it to. A service that is running but not answering is left alone (a restart would throw away its indexing).
- **Needs Trailblazer watching the repo.** When Trailblazer cannot answer, scout prints `Trailblazer is not watching <repo> or is unavailable; start it (systemctl --user start trailblazer) or run: trailblazer watch <repo>` and exits 1. While a newly watched repo's vectors are still being built, an empty answer is reported as `still building vectors (pending_vectors=N)`: the results are incomplete until they finish.
- **Exact names.** Symbols named in the question are also found with ripgrep.

Exit status:

- 0: candidates printed.
- 1: Trailblazer is not watching the repo or cannot answer (the hint is on stderr).
- 2: Trailblazer answered but found no candidate locations for the question.

## Services

Trailblazer `trailblazer` is required. If a run cannot connect:

```bash
systemctl --user start trailblazer
```

Source: `~/projects/scout`. The Claude and agents skill directories link here; make Scout changes in this repository. Ranger imports the same retrieval interfaces and its installer copies the helpers from here.

Run the tests with `python3 -m unittest discover -s ~/projects/scout/tests`. Check the shared graph with `trailblazer status --repo ~/projects/scout`; `pending_vectors: 0` means every indexed definition has an embedding.
