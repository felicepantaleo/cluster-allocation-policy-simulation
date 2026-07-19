#!/bin/bash
# Render the PMC deck to HTML and PDF after editing the markdown.
# Usage: bash slides/render.sh   (from anywhere; ~15 s total)
set -euo pipefail
cd "$(dirname "$0")"
# node needs the LCG runtime libraries (system node is too old)
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el8-gcc11-opt/setup.sh 2>/dev/null
MARP=./node_modules/.bin/marp
[ -x "$MARP" ] || { echo "marp-cli missing: run npm install @marp-team/marp-cli here"; exit 1; }
$MARP ngt_allocation_problem.marp.md --theme-set themes/cern-ngt.css \
      --html --allow-local-files -o ngt_allocation_problem.html
$MARP ngt_allocation_problem.marp.md --theme-set themes/cern-ngt.css \
      --allow-local-files --browser firefox --pdf -o ngt_allocation_problem.pdf
ls -la ngt_allocation_problem.html ngt_allocation_problem.pdf
