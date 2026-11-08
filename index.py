#!/usr/bin/env python3
"""Rendering of Trailblazer's hits for scout and ranger (bug:3df441f).

Trailblazer is the only retrieval source: its search hits are the candidates.
This module turns them into prompt text: the best locations with their numbered
source lines (candidates_text), and the exact-name ripgrep matches for symbols
the question names (symbol_matches). Nothing here embeds, stores or searches.
"""

import re
import subprocess
from pathlib import Path


def candidates_text(work_dir, hits, full=4, preview_lines=3):
    """Format Trailblazer's hits for the ranger prompt, best match first.

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
        blocks.append(f"{label}\n{body}")
    return "\n\n".join(blocks)


def symbol_matches(work_dir, query, max_symbols=6, max_lines=12):
    """Exact-name ripgrep matches for identifiers named in the question.

    Where each named symbol is defined, assigned and used, so the model can judge
    whether similarly named things are really connected. Only tokens that look
    like code are tried (snake_case, camelCase, CAPITALS, or anything in
    backticks), not ordinary words.
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
        if out:
            more = f"\n... {len(out) - max_lines} more matches" if len(out) > max_lines else ""
            blocks.append(f"{sym} ({len(out)} matches):\n" + "\n".join(out[:max_lines]) + more)
        else:
            blocks.append(f"{sym}: no exact match in the repository")
    return "\n\n".join(blocks)
