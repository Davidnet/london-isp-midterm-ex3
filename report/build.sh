#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker build -q -t isp-ex3-report . >/dev/null
# the subtitle (with its declared word count) lives in report.md's YAML block;
# the pandoc image's ENTRYPOINT is pandoc itself
docker run --rm -v "$PWD:/work" -w /work isp-ex3-report \
  report.md -o report.pdf --pdf-engine=xelatex
echo "report.pdf built"
python3 wordcount.py
