# cluster-allocation-policy-simulation

Discrete-event simulator of resource allocation on the CERN NGT Kubernetes
cluster, built for the PMC scheduling-policy study. It replays an allocation
request trace (synthetic for now, the real event log later) against a model
of the cluster and pluggable scheduling policies, and compares policies A/B
on the same trace and seed.

Governing allocation principles: `PRINCIPLES.md` (priority-ordered).
Phase 1 findings: `docs/phase1-findings.md`.

**Scheduling proposal** (the policy to adopt, grounded in how HPC centers and
datacenters schedule shared GPUs, with the fix demonstrated on the real
trace): `docs/proposal-scheduling.md`.

**Reproduce the whole study end to end** (collect the real cluster data from
CERN MONIT, derive the trace, attribute users to working packages, rebuild
every gallery plot, run the simulator and the real-trace replay, render the
deck): `docs/REPRODUCE.md`.

## Layout

    clustersim/
      eventloop.py    deterministic heap-based event loop
      cluster.py      pools, nodes, granularity rules, NUMA check, cordons
      trace.py        trace schema (JSONL) and I/O
      tracegen.py     synthetic trace generator (calibrated, config-driven)
      engine.py       simulation mechanics: pending set, gaming siblings,
                      patience, reclaim resubmission, snapshots
      policies/       one file per policy behind a small interface
      metrics.py      windowed wait/fairness/utilization/WP-share metrics
      plots.py        comparison plots
      run.py          experiment runner CLI
    config/           experiment configs (cluster topology, workload, policies)
    tests/            invariant tests
    results/          runner output (gitignored, regenerated from config)
    docs/             committed phase writeups and their plots

## Policies implemented

| name | what it models |
|---|---|
| `fcfs_pending` | current NGT behavior: no queue, no quota, Pending pods placed first-fit as capacity frees, multi-Pending gaming in the trace |
| `idle_reclaim` | reap allocations whose GPU utilization stays below a threshold longer than T; reaped work resubmits after a reaction delay |
| `ngt_principles` | PRINCIPLES.md P1-P5: guaranteed 1-GPU member tier with reserved headroom, WP fair share on charged GPU-hours, multi-GPU time cap, intra-WP recycling |
| `ngt_principles_reclaim` | P1-P5 plus idle reclaim |
| `planning_cycle` | P6 on top of P1-P5: tiered declaration epochs (12:00 daily for long jobs, 3 decision points per day for jobs up to 8 h) |

## Why a custom event loop instead of SimPy

The simulation is scheduler-centric: everything happens at discrete events
(submit, release, cordon change, reclaim timer, planning epoch) where a
policy scans the Pending set against cluster state. That is a heap of
timestamped callbacks, about 30 lines, with bit-reproducible replay for a
given trace and seed and zero extra dependencies. SimPy's
process-and-interrupt model would wrap each allocation in a generator
process and still need explicit condition plumbing to wake the scheduler on
every capacity change, adding a dependency without removing any code that
matters here.

## Trace schema

A trace is a directory: `meta.json`, `requests.jsonl`, `cordons.jsonl`.
Time is seconds since epoch, t=0 is a Monday 00:00 CEST. Each request
carries: id, `group_id` (K multi-Pending gaming copies share one group; the
engine cancels siblings when one starts), user, working package, kind,
submit time, pool, resources (GPUs or MIG slices, vCPU, memory), hold
duration, a piecewise `(duration_s, gpu_util)` profile separating active
from held-but-idle time, and patience (cancel Pending after this long).
`observed` holds real-log fields the simulator does not consume (observed
wait, outcome, placement) for validating simulation against reality. When
admin access to the real event log lands, a converter writes this same
schema and everything downstream runs unchanged.

## Run

    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m pytest tests/ -q
    .venv/bin/python -m clustersim.run --config config/phase1.yaml --out results/phase1

The runner generates (or reuses, add `--regen-trace` to force) the trace,
runs every policy in the config on it, and writes per-policy records,
metrics, validation checks, plots and `comparison.md`. Same config, same
output, byte for byte.

Sweeps: `python -m clustersim.sweep` (waiting time vs number of users),
`python -m clustersim.sweep_reserve` (1-GPU tier reserve sizing),
`python -m clustersim.sweep_mig` (MIG carve sizing).

## Scenario mode

    .venv/bin/python -m clustersim.scenario --scenario scenarios/greedy_week.yaml \
        --config config/phase1.yaml --out results/scenarios/greedy_week

A scenario YAML scripts a week of requests explicitly (user, WP, pool,
GPUs, submit time as "Tue 09:30", hold, utilization profile), optionally
with its own small topology, and runs it under selected policies at
30-minute reporting granularity. Output: one per-user timeline per policy
(pending, held with GPU count, reclaim/cap and give-up markers, colored by
WP) plus an occupancy strip and a summary table. Shipped scenarios:
`scenarios/greedy_week.yaml` (one 8-GPU week-long hoard vs trainings and
member sessions), `scenarios/session_rush.yaml` (twelve members opening
9:00 sessions against two trainings), `scenarios/training_bursts.yaml`
(bursts of few-hour multi-GPU trainings under FCFS, fair share, and the
planning cycle). The schema is documented in `clustersim/scenario.py`.

`scenarios/realistic_week.yaml` is the full-scale case: the current GPU
topology and 58 users in the calibrated composition (26 trainers, 26
devs, 6 hoarders across the WPs), 727 requests over one week, cordon
windows included. It is generated (and regenerable with different seeds
or rates) by `python -m clustersim.realistic_week --config
config/phase1.yaml`, and stays a plain editable YAML.

## Assumptions vs ground truth

The config tags each block: G = ground truth from the PMC kickoff (pool
shapes, the Saturday saturation data point, the observed cordoned fraction,
gaming and idle-hold behaviors), A = assumption pending real data (the MIG
carve, the CPU node-count split, user counts and rates, GPU charge factors,
patience). Every A-tagged number is a named config knob so recalibration
against the real trace is a config change, not a code change.
