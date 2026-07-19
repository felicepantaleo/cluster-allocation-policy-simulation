#!/bin/bash
# Render the PMC deck to HTML and PDF after editing the markdown.
# Usage: bash slides/render.sh   (from anywhere; ~15 s total)
cd "$(dirname "$0")" || exit 1

# node needs the LCG runtime; the LCG setup script is NOT strict-mode
# clean (unset vars, nonzero inner returns), so source it before set -e
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el8-gcc11-opt/setup.sh
set -e

MARP=./node_modules/.bin/marp
[ -x "$MARP" ] || { echo "marp-cli missing: run npm install @marp-team/marp-cli here"; exit 1; }

# a leftover headless-firefox/marp from an interrupted render blocks new
# PDF conversions: clean them up first (patterns split to avoid matching
# this script's own command line)
for pid in $(ps -u "$USER" -o pid=,args= | grep -E "bin/mar[p] |firefox.*-headles[s]" | awk '{print $1}'); do
    kill -9 "$pid" 2>/dev/null && echo "killed stale render process $pid"
done

$MARP ngt_allocation_problem.marp.md --theme-set themes/cern-ngt.css \
      --html --allow-local-files -o ngt_allocation_problem.html
timeout 120 $MARP ngt_allocation_problem.marp.md --theme-set themes/cern-ngt.css \
      --allow-local-files --browser firefox --pdf -o ngt_allocation_problem.pdf \
    || { echo "PDF step timed out; re-run once (stale processes were cleaned)"; exit 1; }
ls -la ngt_allocation_problem.html ngt_allocation_problem.pdf
