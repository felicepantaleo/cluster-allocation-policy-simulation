#!/bin/bash
# Discover what NGT data is reachable in monit-grafana org 105 with an
# authenticated session: dashboards, datasources, and a few test PromQL
# queries against the NGT datasource. Read-only.
# Usage: monit_probe.sh <cookie-jar> <outdir>
set -euo pipefail
J=$1
OUT=$2
mkdir -p "$OUT"
G="https://monit-grafana.cern.ch"
C() { curl -sS -c "$J" -b "$J" -H "X-Grafana-Org-Id: 105" "$@"; }

echo "== user / orgs"
C "$G/api/user" | tee "$OUT/user.json" | head -c 200; echo
C -X POST "$G/api/user/using/105" >/dev/null 2>&1 || true

echo "== datasources in org 105"
C "$G/api/datasources" > "$OUT/datasources.json" || true
head -c 300 "$OUT/datasources.json"; echo

echo "== dashboard search: gpu / ngt"
C "$G/api/search?query=gpu&limit=30" > "$OUT/search_gpu.json"
C "$G/api/search?query=ngt&limit=30" > "$OUT/search_ngt.json"
head -c 400 "$OUT/search_ngt.json"; echo
