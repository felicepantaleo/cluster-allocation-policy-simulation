"""Scenario mode: a scripted week of requests at 30-minute reporting
granularity.

    python -m clustersim.scenario --scenario scenarios/greedy_week.yaml \
        --config config/phase1.yaml --out results/scenarios/greedy_week

A scenario YAML declares the week explicitly: who submits (user and working
package), what (pool, GPUs, hold, utilization profile), and when
("Tue 09:30"). The runner simulates it under each policy listed in the
scenario and produces one per-user timeline figure per policy plus a
summary table, so individual behavior (a greedy week-long 8-GPU hold, a
9:00 session rush, short trainings) is visible request by request rather
than statistically.

Scenario schema (all times "Day HH:MM", week starts Mon 00:00):

    name: greedy-week
    description: one line
    horizon_days: 7            # optional, default 7
    cluster: {pools: {...}}    # optional topology override, same schema as
                               # the main config; default: main config pools
    policies:                  # policies to run; params merged over the
      fcfs_pending: {}         # main config's params for that policy
      ngt_principles_reclaim: {reserve: {h100nvl: 2}}
    requests:
      - user: greta
        wp: WP2
        pool: h100nvl
        gpus: 8
        time: Mon 09:00        # or a list of times, one request each
        hold_h: 168
        profile: {pattern: [[1, 0.7], [23, 0.02]], repeat: 7}
        # or profile: [[2, 0.8], [1, 0.02]]  (hours, util) flat list
        # or kind: train | dev | hoard  (generated profile, seeded)
        patience_h: 48         # optional, default 168
    cordons:                   # optional
      - {node: h100nvl-01, from: Wed 08:00, to: Thu 20:00}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .engine import Engine
from .policies import make_policy
from .run import build_cluster
from .trace import CordonEvent, Request
from . import plots, tracegen

H = 3600.0
DAYS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def parse_time(spec: str) -> float:
    day, hhmm = spec.split()
    hh, mm = hhmm.split(":")
    return DAYS[day] * 86400.0 + int(hh) * H + int(mm) * 60.0


def build_profile(spec: dict, duration_s: float, rng) -> list:
    if "profile" in spec:
        p = spec["profile"]
        if isinstance(p, dict):
            segs = [(float(h) * H, float(u)) for h, u in p["pattern"]]
            segs = segs * int(p.get("repeat", 1))
        else:
            segs = [(float(h) * H, float(u)) for h, u in p]
        total = sum(d for d, _ in segs)
        if abs(total - duration_s) > 1.0:
            segs = [(d * duration_s / total, u) for d, u in segs]
        return segs
    kind = spec.get("kind", "train")
    gen = {"train": tracegen.trainer_profile, "dev": tracegen.dev_profile,
           "hoard": tracegen.hoard_profile}[kind]
    return gen(rng, duration_s)


def build_requests(scn: dict, seed: int) -> list[Request]:
    rng = np.random.default_rng(seed)
    requests = []
    for i, spec in enumerate(scn["requests"]):
        times = spec["time"] if isinstance(spec["time"], list) else [spec["time"]]
        duration_s = float(spec["hold_h"]) * H
        for j, t in enumerate(times):
            rid = f"{spec['user']}-{i:02d}{chr(97 + j)}"
            requests.append(Request(
                request_id=rid, group_id=rid,
                user=spec["user"], kind=spec.get("kind", "train"),
                wp=spec["wp"], submit_time=parse_time(t),
                pool=spec["pool"], gpus=int(spec.get("gpus", 1)),
                vcpus=float(spec.get("vcpus",
                                     tracegen.POOL_GPU_RATIO.get(
                                         spec["pool"], (8.0, 40.0))[0]
                                     * max(spec.get("gpus", 1), 1))),
                mem_gb=float(spec.get("mem_gb",
                                      tracegen.POOL_GPU_RATIO.get(
                                          spec["pool"], (8.0, 40.0))[1]
                                      * max(spec.get("gpus", 1), 1))),
                duration_s=duration_s,
                profile=build_profile(spec, duration_s, rng),
                patience_s=float(spec.get("patience_h", 168.0)) * H,
            ))
    return requests


def build_cordons(scn: dict) -> list[CordonEvent]:
    events = []
    for c in scn.get("cordons", []):
        events.append(CordonEvent(parse_time(c["from"]), c["node"], True))
        events.append(CordonEvent(parse_time(c["to"]), c["node"], False))
    return events


def summary_rows(engine: Engine) -> list[dict]:
    per_user: dict[tuple, dict] = {}
    for r in engine.records:
        key = (r["wp"], r["user"])
        d = per_user.setdefault(key, {
            "wp": r["wp"], "user": r["user"], "requests": 0, "started": 0,
            "wait_min": 0.0, "held_gpu_h": 0.0, "active_gpu_h": 0.0,
            "reclaims": 0, "cancelled": 0})
        if r.get("record") == "allocation":
            d["held_gpu_h"] += r["held_gpu_s"] / H
            d["active_gpu_h"] += r["used_gpu_s"] / H
            if r["end_reason"] in ("reclaimed", "time_capped"):
                d["reclaims"] += 1
        else:
            if r["resubmit_of"] is None:
                d["requests"] += 1
            if r["outcome"] == "started":
                d["started"] += 1
                d["wait_min"] += r["wait_s"] / 60.0
            elif r["outcome"] == "cancelled_patience":
                d["cancelled"] += 1
    return [per_user[k] for k in sorted(per_user)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    scn = yaml.safe_load(Path(args.scenario).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    horizon_s = scn.get("horizon_days", 7) * 86400.0
    cluster_cfg = scn.get("cluster", config["cluster"])
    requests = build_requests(scn, config["seed"])
    cordons = build_cordons(scn)

    md = [f"# Scenario: {scn['name']}", "", scn.get("description", "").strip(), ""]
    for pname, overrides in scn["policies"].items():
        params = dict(config["policies"].get(pname, {}))
        params.update(overrides or {})
        params.setdefault("wp_targets", config.get("wp_targets"))
        params.setdefault("charge_factors", config.get("gpu_charge_factor"))
        engine = Engine(
            cluster=build_cluster(cluster_cfg),
            policy=make_policy(pname, params),
            requests=requests, cordons=cordons, horizon_s=horizon_s,
            seed=config["engine"]["seed"],
            snapshot_interval_s=scn.get("snapshot_interval_s", 1800.0),
            resubmit_reaction_median_s=config["engine"]["resubmit_reaction_median_s"],
            resubmit_reaction_sigma=config["engine"]["resubmit_reaction_sigma"],
            resubmit_patience_s=config["engine"]["resubmit_patience_s"],
        )
        engine.run()
        with open(out / f"{pname}_records.jsonl", "w") as f:
            for r in engine.records:
                f.write(json.dumps(r) + "\n")
        gpu_pools = [p for p, s in cluster_cfg["pools"].items()
                     if s["gpus_per_node"] > 0]
        plots.scenario_timeline(
            scn["name"], pname, engine.records, engine.snapshots,
            gpu_pools, horizon_s / 86400.0, out / f"{pname}_timeline.png")

        md += [f"## {pname}", "",
               "| user | WP | requests | started | cancelled | total wait (min) | "
               "held GPU-h | active GPU-h | reclaim/cap events |",
               "|---|---|---|---|---|---|---|---|---|"]
        for d in summary_rows(engine):
            md.append(f"| {d['user']} | {d['wp']} | {d['requests']} | "
                      f"{d['started']} | {d['cancelled']} | {d['wait_min']:.0f} | "
                      f"{d['held_gpu_h']:.0f} | {d['active_gpu_h']:.1f} | "
                      f"{d['reclaims']} |")
        md.append("")
        print(f"{scn['name']}: {pname} done")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}/summary.md and timelines")


if __name__ == "__main__":
    main()
