---
name: scout
description: Retrieval only, no model - for a question about a codebase, prints the best-matching locations (file:line ranges), exact-name matches for symbols in the question, and the code graph around them (who imports, calls or uses what), in seconds. From Trailblazer's graph and vectors when it watches the repo, from ranger's own semantic index otherwise. Use as the first step before reading, editing, reviewing or debugging code you are not already certain about.
allowed-tools: Bash(python3 *scout/scout.py*)
---

# scout

Finds embeddings and returns a graph: no model runs. For a question about a repository it prints the candidate locations, best match first, with their text, then the structure around them. This is the layer ranger builds on; use ranger when the locations alone do not answer the question.

```bash
python3 /home/mirai/.claude/skills/scout/scout.py -C <repo> -q "<question>"
python3 /home/mirai/.claude/skills/scout/scout.py -C <repo> -q "<question>" --verbose   # source and scores on stderr
```

## Where the answer comes from

- **Trailblazer first.** When Trailblazer watches the repo its definition graph and vectors are used as they are: nothing is embedded or synced. If the service is stopped, scout starts it and waits for it to publish its endpoint; if it does not watch the repo, scout asks it to. A service that is running but not answering is left alone (a restart would throw away its indexing).
- **Its own index otherwise.** EmbeddingGemma 2 embeddings in a local Qdrant (honours `.gitignore`); only changed files are re-embedded, but the first run on a new repo embeds everything, which takes a while (sojourney, 160 files: about 30 minutes). The same happens while Trailblazer's vectors for a newly watched repo are still being built.
- **Exact names.** Symbols named in the question are also found with ripgrep.

Exit status:

- 0: candidates printed.
- 1: no index source could answer (Trailblazer not watching or unavailable and ranger's own index could not answer).
- 2: a source answered but found no candidate locations for the question.

## Services

Embeddings `ranger-embed` (port 8091), Qdrant container `ranger-qdrant` (6333), Trailblazer `trailblazer` (optional). If a run cannot connect:

```bash
systemctl --user start ranger-embed trailblazer
docker start ranger-qdrant
```

Source: `~/projects/scout`. The Claude and agents skill directories link here; make Scout changes in this repository. Ranger imports the same retrieval interfaces and its installer copies the helpers from here.

Run the local index tests with `python3 -m unittest discover -s ~/projects/scout/tests`. Check the shared graph with `trailblazer status --repo ~/projects/scout`; `pending_vectors: 0` means every indexed definition has an embedding.
