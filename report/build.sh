#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker build -q -t isp-ex3-report . >/dev/null
# body word count, injected into the title block as a subtitle
words=$(python3 wordcount.py --bare)
# the pandoc image's ENTRYPOINT is pandoc itself
docker run --rm -v "$PWD:/work" -w /work isp-ex3-report \
  report.md -o report.pdf --pdf-engine=xelatex \
  -M subtitle="(words: ${words})"
echo "report.pdf built"
python3 wordcount.py
