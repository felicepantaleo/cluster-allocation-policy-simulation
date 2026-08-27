# Reproducing the study end to end

This is the full recipe: collect the real NGT cluster data, derive the
trace, attribute users to working packages, regenerate every plot in the
orbit gallery, run the simulator and the real-trace replay, and rebuild the
slide deck. Every command is a real script in this repo.

The raw dump and the derived trace contain usernames, so `data/` is
gitignored and nothing under it is committed. Only code, configs and the
committed writeups in `docs/` are versioned.

## 0. Environment

The system Python is 3.6; use a 3.12 virtualenv.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt          # numpy matplotlib pyyaml pytest
.venv/bin/pip install openpyxl mplhep               # xlsx parsing, CMS plot style
.venv/bin/python -m pytest tests/ -q                # 21 invariant tests, must pass
```

External tools used by the data-collection and publishing steps: `curl`
(SSO login and Prometheus queries), `ldapsearch` (name to username via
xldap.cern.ch, anonymous bind from inside CERN), and `xrdcp` from an LCG
view (`source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el8-gcc11-opt/setup.sh`)
for uploading the gallery to orbit.

## 1. Where the real data comes from

Direct `kubectl` listing across namespaces is Forbidden for a normal user.
The real data instead comes from the cluster monitoring: CERN MONIT
Prometheus, which scrapes kube-state-metrics and DCGM on the NGT cluster and
is queryable through the monit-grafana datasource proxy with the user's own
read-only SSO identity. Org 105, long-term-store datasource
`prometheus-lts` (uid `afn82ui46w9vke`). Retention is about 33 days.

### 1a. Authenticate

Needs a valid Kerberos ticket (`kinit`) and one CERN 6-digit OTP. The login
completes a Kerberos SPNEGO handshake to auth.cern.ch and then submits the
OTP; the Grafana session lands in a cookie jar.

```bash
kinit                                               # if you have no ticket
bash tools/monit_login.sh <OTP> /tmp/monit.jar      # verify: prints login OK
```

Grafana rotates `grafana_session` every ~10 min, so the jar must stay live
across a long pull. `tools/extract_monit_trace.py` drives every query with
`curl` against the same jar (which keeps it current) and silently re-logs in
via the jar's Keycloak cookies if the session drops mid-pull. A pasted
static cookie does NOT survive the first rotation; always log in with the
script.

### 1b. Extract the raw metrics

```bash
.venv/bin/python tools/extract_monit_trace.py \
    --cookies /tmp/monit.jar --out data/monit --days 40
```

Pulls the full metric set (pod lifecycle, resource requests and limits, DCGM
GPU utilization / framebuffer / power, per-pod CPU / memory / network,
node capacity / conditions / info, namespace labels) as per-day JSONL
chunks under `data/monit/`. It is resumable: each (metric, day) chunk is
cached and skipped if present, so an interrupted pull continues where it
stopped. About 2.5 GB for 33 days.

Gotcha baked into the tool: the long-term store is downsampled, so any
`rate()` query needs a window of at least 10 minutes (5 minutes returns
empty). The committed queries already use `[10m]`.

## 2. Derive the simulator trace

```bash
.venv/bin/python tools/monit_to_trace.py --raw data/monit --out data/derived
```

Writes `data/derived/requests.jsonl` (one Request per pod instance) and
`data/derived/cordons.jsonl`. Critical step: a pod NAME is reused across
recreations, so the converter splits each name into instances on
`kube_pod_created` value changes; skipping this corrupts every statistic.
The `observed` block on each request carries the measured wait, outcome,
node placement and running/pending intervals; the utilization profile comes
from the DCGM samples, so held-but-idle time separates from active time.

## 3. Attribute users to working packages

Two manual inputs the maintainer supplies:

- the WP membership spreadsheet (`NGT_ALL.xlsx`, budget sheet with WP1..WP4
  and management tabs),
- the STEAM Academy participant list (`participants.csv`), whose accounts
  are excluded from every scheduling statistic.

```bash
# names -> usernames via xldap LDAP, intersected with observed namespaces;
# a user in several WPs is assigned WP2 > WP3 > WP1 > WP4 > management
.venv/bin/python tools/map_users_wp.py --xlsx NGT_ALL.xlsx \
    --raw data/monit --out data/derived/user_wp.json

# tag STEAM participants (excluded), then department rules for the rest:
# EP/CMS -> WP3, EP/ATL -> WP2, */SFT and IT/* -> WP1
.venv/bin/python tools/classify_users.py \
    --steam-csv participants.csv --derived data/derived
```

`classify_users.py` prints the still-unclassified users to
`data/derived/remaining_to_classify.md`; tag those by hand in
`data/derived/user_wp.json` (a flat `{username: {"wp": "WP3", ...}}` map).
Attribution reached 99.4% of GPU-hours this way.

## 4. Node inventory (the cordon study's "all nodes")

```bash
.venv/bin/python tools/node_inventory.py --raw data/monit --out data/derived
```

Writes `data/derived/all_nodes.csv` and `all_nodes.md`: every node with GPU
model, capacity, `cordon_pct` (fraction of the window cordoned, integrated)
and `cordon_episodes` (distinct cordon periods, samples merged within 1 h).

## 5. Regenerate the measured-problem gallery

All read `data/derived` (and `data/monit` directly for the DCGM/cordon
plots), exclude STEAM, and write PNG plus a vector SVG copy.

```bash
G=results/gallery/ngt_allocation_problem
.venv/bin/python tools/problem_gallery.py    --derived data/derived --out $G
.venv/bin/python tools/cordon_timeline.py    --raw data/monit       --out $G
.venv/bin/python tools/gpu_vs_pod_idle.py    --raw data/monit --derived data/derived --out $G
.venv/bin/python tools/per_user_timeline.py  --derived data/derived --out $G/per_user
```

Notes that matter for correctness:
- The cordon count plot (`09_cordons`) reads the raw cordon samples and
  merges per node with a 1 h gap, counting GPU-pool nodes only. The derived
  `cordons.jsonl` splits a sustained cordon into per-day-file intervals that
  touch at the 12:00 extraction boundary; an open/close counter breaks on
  those shared boundaries and collapses a continuous cordon into spikes, so
  the plot does not use it.
- `gpu_vs_pod_idle.py` splits GPU-idle held time into CPU-active (real work
  on the pod's cores) versus fully pod-idle, using
  `container_cpu_usage_seconds_total`.

### Publish to orbit (optional)

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el8-gcc11-opt/setup.sh
D=root://eosuser.cern.ch//eos/user/f/fpantale/www/felice-website/orbit/ngt_allocation_problem
for f in $G/*.png; do xrdcp -f -s "$f" "$D/$(basename $f)"; done
xrdcp -r -f -s --parallel 4 $G/per_user "$D/"
```

Shareable link: `https://felice.web.cern.ch/orbit/?path=%2Fngt_allocation_problem`.

## 6. Simulate and replay

Synthetic calibrated run (baseline plus candidate policies on a generated
trace), used before real data was available:

```bash
.venv/bin/python -m clustersim.run --config config/phase1.yaml --out results/phase1
```

Real-trace replay: the measured month through the full policy set, with a
fidelity check of the FCFS baseline against observed waits. This is the
headline result.

```bash
.venv/bin/python tools/replay_real.py --derived data/derived --out results/replay
# results/replay/comparison.md: metrics + principles scorecard, six policies
```

Conversion rules (each an explicit assumption) are documented in the header
of `tools/replay_real.py`. The proposed policy is the `multi_budget_queue`
column: one interactive session per member (a new session supersedes the
old), multi-GPU beyond a monthly interactive allowance runs as batch behind
a WP fair-share priority queue.

Sweeps and scenarios:

```bash
.venv/bin/python -m clustersim.sweep         --config config/phase1.yaml --out results/sweep_users
.venv/bin/python -m clustersim.sweep_reserve --config config/phase1.yaml --out results/sweep_reserve
.venv/bin/python -m clustersim.sweep_mig     --config config/phase1.yaml --out results/sweep_mig
.venv/bin/python -m clustersim.realistic_week --config config/phase1.yaml   # writes scenarios/realistic_week.yaml
for s in greedy_week session_rush training_bursts realistic_week; do
  .venv/bin/python -m clustersim.scenario --scenario scenarios/$s.yaml \
      --config config/phase1.yaml --out results/scenarios/$s
done
```

## 7. Slide deck

```bash
cd slides && npm install @marp-team/marp-cli        # once, into slides/node_modules
cp $G/svg/*.svg $G/*.png slides/plots/               # refresh figures
bash slides/render.sh                                # HTML + PDF, ~15 s
```

`render.sh` sources the LCG view for node, cleans any stale headless-marp
processes, and passes `--no-stdin`.

## Refreshing with a newer month

Re-run from step 1b onward with a fresh OTP; the 30-day analysis window is
anchored to the newest request, so numbers shift slightly as the window
slides. `tools/replay_real.py` and the gallery clip to that same window, so
they stay internally consistent, but slide captions carrying absolute
numbers must be updated to the new run (the method, replay table, idle
totals, greediness counts and cloud-cost figures are the ones that move).
