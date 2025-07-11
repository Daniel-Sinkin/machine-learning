from __future__ import annotations

"""gpt_cleaner.py
A tiny CLI utility to scrub ChatGPT‐generated text files.

Removes / Replaces
------------------
* **Separator lines** that start with `# ───` **or** `# ---` (any length).
* Replaces fancy arrows with ASCII equivalents – e.g. `→` → `->`, `←` → `<-`.
* Strips **all emoji** Unicode code-points.

Usage
-----
```bash
python gpt_cleaner.py input.txt [output.txt]
```
If *output* is omitted the file is cleaned **in-place**.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Final

# ─────────────────────────  config  ──────────────────────────
ARROW_REPLACEMENTS: Final[dict[str, str]] = {
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "⇒": "=>",
    "⇐": "<=",
    "⇔": "<=>",
}

# Lines that begin with: optional spaces, then '#', then spaces, then either '─' or '-' repeated
SEPARATOR_RE = re.compile(
    r"^\s*#\s*[\u2500-\u257F\-]{3,}"
)  # unicode box-drawing or simple hyphens

EMOJI_RE = re.compile(
    r"["  # opening char-class
    r"\U0001F300-\U0001F5FF"  # symbols & pictographs, etc.
    r"\U0001F600-\U0001F64F"  # emoticons
    r"\U0001F680-\U0001F6FF"  # transport & map
    r"\U0001F1E0-\U0001F1FF"  # flags
    r"\U00002700-\U000027BF"  # dingbats
    r"\U000024C2-\U0001F251"  # enclosed
    r"]+",
    flags=re.UNICODE,
)

# ──────────────────────────  logic  ──────────────────────────


def _clean_lines(lines: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        # 1. Drop separator lines
        if SEPARATOR_RE.match(line):
            continue

        # 2. Replace fancy arrows
        for fancy, ascii_ in ARROW_REPLACEMENTS.items():
            if fancy in line:
                line = line.replace(fancy, ascii_)

        # 3. Strip emojis
        line = EMOJI_RE.sub("", line)
        line = line.replace("—", "-")
        cleaned.append(line)
    return cleaned


# ──────────────────────────  cli  ───────────────────────────


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean ChatGPT artifacts from a text file.")
    p.add_argument("input", type=Path, help="input text file")
    p.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="optional output file; defaults to overwrite input",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    in_p: Path = args.input
    out_p: Path = args.output or in_p

    if not in_p.exists():
        sys.exit(f"[gpt_cleaner] Error: {in_p} does not exist")

    text = in_p.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    cleaned = _clean_lines(text)
    out_p.write_text("".join(cleaned), encoding="utf-8")
    print(f"[gpt_cleaner] Cleaned -> {out_p}")


if __name__ == "__main__":
    main()
