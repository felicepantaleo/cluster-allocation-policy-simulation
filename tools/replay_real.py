"""Replay the real 30-day trace through the policy set.

    python tools/replay_real.py --derived data/derived --out results/replay

Conversion rules (each an explicit, stated assumption):
- submit/hold/wait: measured. Sim t=0 is the midnight before the earliest
  pod creation, so time-of-day (planning epochs) is preserved.
- patience: for requests observed to give up, the observed give-up time;
  for satisfied requests patience is right-censored, replayed as infinite
  (replayed users are at least as patient as observed).
- utilization profile: measured DCGM segments; allocations without DCGM
  coverage (MIG, AMD, gaps) replay as fully active, so idle-reclaim only
  ever acts on MEASURED idleness (conservative for reclaim benefits).
  Profiles shorter than the hold are padded with active time.
- never-started requests get hold and profile sampled (seeded) from
  started requests of the same kind and GPU count; unplaced full-GPU
  requests are assigned the pool most used by same-size requests.
- kind by measured behavior: >=90% idle and >=48 h hold = hoard; MIG or
  short single-GPU = dev; otherwise train. CPU-flavor and cloud T4 pods
  are excluded.
- topology: pools sized from peak observed concurrency; real cordon
  windows are mapped onto replay nodes per pool in name order.
- no gaming reconstruction: each real request replays as one request.

Outputs comparison.md (metrics + principles scorecard) and a
replay-fidelity check of the FCFS baseline against observed waits.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clustersim.cluster import Cluster, PoolSpec
from clustersim.engine import Engine
from clustersim.metrics import compute
from clustersim.policies import make_policy
from clustersim.trace import CordonEvent, Request
from clustersim import principles
from clustersim.run import comparison_table

H = 3600.0
GPU_POOLS = ["h100nvl", "h100sxm", "l40s", "amd", "mig3g", "mig1g"]
WP_TARGETS = {"WP1": 0.30, "WP2": 0.30, "WP3": 0.30, "WP4": 0.10}
CHARGE = {"h100nvl": 1.0, "h100sxm": 1.0, "l40s": 0.35, "amd": 0.6,
          "mig3g": 0.40, "mig1g": 0.15}
POLICIES = {
    "fcfs_pending": {},
    "idle_reclaim": {"util_thresh": 0.05, "idle_after_s": 1800},
    "ngt_principles": {"reserve": {"h100nvl": 4, "l40s": 1},
                       "multi_gpu_cap_h": 24},
    "ngt_principles_reclaim": {"reserve": {"h100nvl": 4, "l40s": 1},
                               "multi_gpu_cap_h": 24,
                               "util_thresh": 0.05, "idle_after_s": 1800},
    # PMC variant (Felice): no reclaim; multi-GPU jobs are batch-only
    # (submitted, executed, exit at completion: the profile collapses to
    # its active envelope, workless parks are never submitted) behind a
    # priority queue (guaranteed 1-GPU tier first, then WP fair share).
    # Single-GPU interactive sessions stay exactly as observed.
    "batch_multi_queue": {"_policy": "ngt_principles", "_batch": True,
                          "reserve": {"h100nvl": 4, "l40s": 1},
                          "multi_gpu_cap_h": 1e6,
                          "max_interactive_per_user": 1},
    # as batch_multi_queue, but each member keeps a monthly budget of
    # interactive multi-GPU GPU-hours (debugging allowance); multi-GPU
    # requests that fit the remaining budget stay interactive, the rest
    # are submitted as batch jobs
    "multi_budget_queue": {"_policy": "ngt_principles",
                           "_budget_gpu_h": 96.0,
                           "reserve": {"h100nvl": 4, "l40s": 1},
                           "multi_gpu_cap_h": 1e6,
                           "max_interactive_per_user": 1},
    # PROPOSED policy (docs/proposal-scheduling.md): one free GPU per member
    # (never charged, served first, swap-at-start); every GPU beyond the first
    # charged on held time and ordered by the WP fair-share factor F=2^(-U/S)
    # over a 7-day half-life window (member's own decayed usage breaks ties
    # within a WP); no reclaim and no batchify, so held idle time is priced
    # rather than removed; multi-GPU is a 7-day renewable lease replayed as the
    # recorded hold.
    "ngt_proposal": {"_policy": "ngt_proposal",
                     "reserve": {"h100nvl": 4, "l40s": 1},
                     "usage_window_h": 168.0, "halflife_h": 168.0,
                     "multi_gpu_cap_h": 1e9,
                     "max_interactive_per_user": 1},
    # the same proposed policy WITH a modelled behavioral response to the
    # held-time charge: members stop holding idle GPUs beyond the free one, so
    # multi-GPU allocations shed their idle time (active envelope). This is the
    # upper bound of the pricing effect the fixed-behaviour replay cannot show.
    "ngt_proposal_behavioral": {"_policy": "ngt_proposal", "_batch": True,
                                "reserve": {"h100nvl": 4, "l40s": 1},
                                "usage_window_h": 168.0, "halflife_h": 168.0,
                                "multi_gpu_cap_h": 1e9,
                                "max_interactive_per_user": 1},
}


def budgetify(requests: list[Request], budget_h: float):
    """Per member: interactive multi-GPU holds are kept as observed while
    they fit in a sliding 30-day budget of held GPU-hours; requests that
    do not fit are batchified (active envelope, exit at end)."""
    import dataclasses
    ledger: dict[str, list] = defaultdict(list)  # user -> [(t, gpu_h)]
    out, kept, converted, prevented = [], 0, 0, 0
    for r in sorted(requests, key=lambda r: r.submit_time):
        if r.gpus <= 1:
            out.append(r)
            continue
        w0 = r.submit_time - 30 * 86400
        used = sum(h for t, h in ledger[r.user] if t >= w0)
        held_h = r.duration_s * r.gpus / H
        if used + held_h <= budget_h:
            ledger[r.user].append((r.submit_time, held_h))
            out.append(r)
            kept += 1
            continue
        active = [(d, u) for d, u in r.profile if u >= 0.05]
        dur = sum(d for d, _ in active)
        if dur < 600.0:
            prevented += 1
            continue
        out.append(dataclasses.replace(r, duration_s=dur, profile=active))
        converted += 1
    return out, kept, converted, prevented


def batchify(requests: list[Request]) -> tuple[list[Request], int, float]:
    """Multi-GPU requests become batch jobs: active segments only, exit at
    end. Multi-GPU holds with under 10 min of real work would never be
    submitted as batch jobs and are dropped (prevented parking)."""
    import dataclasses
    out, prevented, freed_h = [], 0, 0.0
    for r in requests:
        if r.gpus <= 1:
            out.append(r)
            continue
        active = [(d, u) for d, u in r.profile if u >= 0.05]
        dur = sum(d for d, _ in active)
        freed_h += (r.duration_s - dur) * r.gpus / H
        if dur < 600.0:
            prevented += 1
            continue
        out.append(dataclasses.replace(r, duration_s=dur, profile=active))
    return out, prevented, freed_h
TIERS = [{"max_h": 8, "decisions_per_day": 3},
         {"max_h": 100000, "decisions_per_day": 1}]


def classify(r) -> str:
    prof = r["profile"]
    tot = sum(d for d, _ in prof) if prof else 0.0
    idle = sum(d for d, u in prof if u < 0.05) if prof else 0.0
    if tot > 0 and idle / tot >= 0.9 and r["duration_s"] >= 48 * H:
        return "hoard"
    if r["pool"].startswith("mig") or (r["gpus"] == 1 and r["duration_s"] < 24 * H):
        return "dev"
    return "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", default="data/derived")
    ap.add_argument("--out", default="results/replay")
    args = ap.parse_args()
    der = Path(args.derived)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260718)

    raw = [json.loads(l) for l in open(der / "requests.jsonl")]
    wp_map = json.loads((der / "user_wp.json").read_text())
    raw = [r for r in raw if r["gpus"] > 0
           and r["pool"] not in ("cloud_t4", "cpu", "unknown")
           and wp_map.get(r["user"], {}).get("wp") != "STEAM"]

    END = max(r["submit_time"] for r in raw) + 60
    T0 = math.floor(min(r["submit_time"] for r in raw) / 86400) * 86400
    WIN0 = math.ceil((END - 30 * 86400 - T0) / 86400) * 86400  # measured month

    # pool sizing from peak observed concurrency
    started_raw = [r for r in raw if r["observed"]["outcome"] == "started"]
    peak = {}
    for p in GPU_POOLS:
        evs = []
        for r in started_raw:
            if r["pool"] == p:
                for a, b in r["observed"]["running_intervals"]:
                    # trim the 5-min phase quantization so back-to-back pods
                    # do not overlap and inflate the apparent concurrency
                    evs += [(a, r["gpus"]), (max(b - 301, a), -r["gpus"])]
        lvl = mx = 0
        for _, d in sorted(evs):
            lvl += d
            mx = max(mx, lvl)
        peak[p] = mx
    # SXM replays as partial (the real scheduler shares SXM nodes). CPU and
    # memory are NON-BINDING everywhere: reality placed every request the
    # moment GPUs were free (including 100 vCPU / 800 GB pods), so real
    # node allocatable exceeds the per-GPU share numbers and the replayed
    # constraint is the GPU/slice count alone.
    BIG = 1e9
    mig3 = max(peak["mig3g"], 8)
    mig1 = max(peak["mig1g"], 14)
    pools = [
        PoolSpec("h100nvl", max(12, math.ceil(peak["h100nvl"] / 8)), 8,
                 BIG, BIG, "partial"),
        PoolSpec("h100sxm", max(6, math.ceil(peak["h100sxm"] / 4)), 4,
                 BIG, BIG, "partial"),
        PoolSpec("l40s", max(7, math.ceil(peak["l40s"] / 4)), 4,
                 BIG, BIG, "partial"),
        PoolSpec("amd", max(1, math.ceil(peak["amd"] / 8)), 8, BIG, BIG,
                 "partial"),
        PoolSpec("mig3g", 1, mig3, BIG, BIG, "partial"),
        PoolSpec("mig1g", 1, mig1, BIG, BIG, "partial"),
    ]
    print("pool sizing (peak observed):", peak)

    # empirical samplers for never-started requests
    by_class = defaultdict(list)
    for r in started_raw:
        by_class[(r["kind"], min(r["gpus"], 8))].append(r)
    pool_pop = defaultdict(lambda: defaultdict(int))
    for r in started_raw:
        pool_pop[min(r["gpus"], 8)][r["pool"]] += 1

    requests: list[Request] = []
    for r in raw:
        obs = r["observed"]
        dur = r["duration_s"]
        prof = [(d, u) for d, u in r["profile"]]
        pool = r["pool"]
        if obs["outcome"] != "started" or dur <= 0:
            donors = by_class.get((r["kind"], min(r["gpus"], 8))) or started_raw
            donor = donors[rng.integers(len(donors))]
            dur = max(donor["duration_s"], 600.0)
            prof = [(d, u) for d, u in donor["profile"]]
            if pool in ("unplaced", "gpu_unknown"):
                pops = pool_pop[min(r["gpus"], 8)]
                pool = max(pops, key=pops.get) if pops else "h100nvl"
        tot = sum(d for d, _ in prof)
        if tot < dur - 60:
            prof = prof + [(dur - tot, 1.0)]  # uncovered time counts active
        elif not prof:
            prof = [(dur, 1.0)]
        wpe = wp_map.get(r["user"], {}).get("wp", "")
        requests.append(Request(
            request_id=r["request_id"], group_id=r["request_id"],
            user=r["user"], kind=classify({**r, "duration_s": dur,
                                           "profile": prof, "pool": pool}),
            wp=wpe if wpe.startswith("WP") else "",
            submit_time=r["submit_time"] - T0, pool=pool, gpus=r["gpus"],
            vcpus=r["vcpus"] or 8.0 * r["gpus"],
            mem_gb=r["mem_gb"] or 40.0 * r["gpus"],
            duration_s=dur, profile=prof,
            patience_s=(obs["wait_s"] if obs["outcome"] == "cancelled"
                        and obs["wait_s"] > 60 else 1e12),
        ))

    # cordons: map real node names to replay nodes per pool, name order
    pool_of = lambda n: ("h100nvl" if "h100-nvl" in n else
                         "h100sxm" if "h100-sxm" in n else
                         "l40s" if "l40s" in n else
                         "amd" if ("mi300" in n or "w7900" in n) else None)
    cords_raw = [json.loads(l) for l in open(der / "cordons.jsonl")]
    real_nodes = sorted({c["node_id"] for c in cords_raw
                         if pool_of(c["node_id"].lower())})
    node_map = {}
    counters = defaultdict(int)
    npool = {p.name: p.num_nodes for p in pools}
    for n in real_nodes:
        p = pool_of(n.lower())
        i = counters[p]
        counters[p] += 1
        if i < npool.get(p, 0):
            node_map[n] = f"{p}-{i:02d}"
    cordons = [CordonEvent(max(c["time"] - T0, 0.0), node_map[c["node_id"]],
                           c["cordoned"])
               for c in sorted(cords_raw, key=lambda c: c["time"])
               if c["node_id"] in node_map]

    horizon = END - T0
    metrics_by, cards = {}, {}
    batch_note = ""
    for pname, params in POLICIES.items():
        params = dict(params)
        engine_policy = params.pop("_policy", pname)
        run_reqs = requests
        if params.pop("_batch", False):
            run_reqs, prevented, freed_h = batchify(requests)
            batch_note = (f"batch_multi_queue: {prevented} workless "
                          f"multi-GPU parks never submitted; "
                          f"{freed_h:.0f} held GPU-h shed by exit-at-end")
            print(batch_note)
        budget = params.pop("_budget_gpu_h", None)
        if budget is not None:
            run_reqs, kept, converted, prevented = budgetify(requests, budget)
            batch_note += (f"\n\nmulti_budget_queue ({budget:.0f} GPU-h/month "
                           f"interactive multi-GPU per member): {kept} "
                           f"multi-GPU holds stay interactive, {converted} "
                           f"run as batch, {prevented} workless parks "
                           "never submitted")
            print(batch_note.splitlines()[-1])
        params.setdefault("wp_targets", WP_TARGETS)
        params.setdefault("charge_factors", CHARGE)
        engine = Engine(
            cluster=Cluster(pools), policy=make_policy(engine_policy, params),
            requests=run_reqs, cordons=list(cordons), horizon_s=horizon,
            seed=42, snapshot_interval_s=1800.0,
            resubmit_reaction_median_s=600.0, resubmit_reaction_sigma=0.8,
            resubmit_patience_s=6 * H)
        engine.run()
        m = compute(engine.records, engine.snapshots, engine.requests_by_id,
                    GPU_POOLS, WIN0, horizon,
                    charge_factors=CHARGE, wp_targets=WP_TARGETS)
        metrics_by[pname] = m
        cards[pname] = principles.scorecard(
            engine.records, engine.requests_by_id, GPU_POOLS, WIN0, horizon,
            wp_targets=WP_TARGETS, charge_factors=CHARGE, cap_h=24.0,
            planning_tiers=TIERS if pname == "planning_cycle" else None)
        if pname == "fcfs_pending":
            sim_w = {rec["request_id"]: rec["wait_s"]
                     for rec in engine.records
                     if rec.get("record") != "allocation"
                     and rec["outcome"] == "started"}
            both = [(sim_w[r["request_id"]], r["observed"]["wait_s"])
                    for r in started_raw if r["request_id"] in sim_w]
            sw = np.array([b[0] for b in both]) / 60
            ow = np.array([b[1] for b in both]) / 60
            fidelity = (f"replay fidelity (FCFS vs observed, n={len(both)}): "
                        f"sim median/p95 wait {np.median(sw):.0f}/"
                        f"{np.percentile(sw,95):.0f} min vs observed "
                        f"{np.median(ow):.0f}/{np.percentile(ow,95):.0f} min")
            print(fidelity)
            req_by_id = {r.request_id: r for r in requests}
            worst = sorted(
                ((sim_w[r["request_id"]] - r["observed"]["wait_s"], r)
                 for r in started_raw if r["request_id"] in sim_w),
                key=lambda x: -x[0])[:15]
            for dwait, r in worst:
                rq = req_by_id[r["request_id"]]
                print(f"  worst: {rq.pool} gpus={rq.gpus} vcpu={rq.vcpus:.0f} "
                      f"mem={rq.mem_gb:.0f} dur={rq.duration_s/3600:.0f}h "
                      f"sim+{dwait/3600:.1f}h obs={r['observed']['wait_s']/60:.0f}m "
                      f"submit_d={rq.submit_time/86400:.1f}")
        print(f"{pname}: started {m['started']}/{m['logical_jobs']}, "
              f"wait p95 {m['wait_overall']['p95_min']:.0f} min")

    report = ["# Real-trace replay: 30 measured days through the policy set",
              "",
              "Trace: data/derived (MONIT extraction), conversion rules in "
              "tools/replay_real.py. " + fidelity,
              "", batch_note, "",
              comparison_table(metrics_by, GPU_POOLS), "",
              principles.render_md(cards), ""]
    (out / "comparison.md").write_text("\n".join(report))
    print(f"wrote {out}/comparison.md")


if __name__ == "__main__":
    main()
