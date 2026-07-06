#!/usr/bin/env python3
"""Body word count for report.md.

Excluded, per the brief's counting rules: YAML front matter, the appendix,
raw LaTeX blocks, margin sidenotes (^[...]), headings, figure/caption lines,
and pipe-table rows.
"""
import re
import sys

src = open("report.md").read()
src = re.sub(r"^---\n.*?\n---\n", "", src, count=1, flags=re.S)  # YAML
src = re.sub(r"\n# Appendix.*$", "", src, flags=re.S)            # appendix
src = re.sub(r"```\{=latex\}.*?```", "", src, flags=re.S)        # raw blocks
src = re.sub(r"\^\[.*?\]", "", src, flags=re.S)                  # sidenotes
lines = [
    l
    for l in src.splitlines()
    if not l.startswith("#")
    and not l.startswith("![")
    and not l.lstrip().startswith("|")
]
text = "\n".join(lines).replace("*", "").replace("`", "")
n = len([t for t in text.split() if re.search(r"[A-Za-z0-9]", t)])
if "--bare" in sys.argv:
    print(n)
else:
    print(f"body words (excl. headings, captions, tables, sidenotes, appendix): {n}")
