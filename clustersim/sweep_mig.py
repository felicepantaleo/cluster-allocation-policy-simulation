"""MIG carve sizing: 1-GPU session capacity vs user count.

    python -m clustersim.sweep_mig --config config/phase1.yaml --out results/sweep_mig

The reserve sweep showed the guaranteed-tier bottleneck at scale is not the
full-GPU pools but the MIG slices where interactive sessions land. This
sweep varies how many H100 NVL nodes are carved into MIG (each carved node:
12x 3g.47gb plus 14x 1g.12gb slices, following the base config's per-node
split) against the user count, with the reserve fixed small per the reserve
sweep. Carving removes 8 full GPUs per node from the multi-GPU pool, so the
two panels of the output are the two sides of the partition question: MIG
session service quality vs the price paid by capped multi-GPU jobs.

The trace is generated once per (scale, seed) from the base topology and is
identical across carve settings; cordon events for NVL nodes that do not
exist in a smaller-NVL variant are ignored by the engine.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from .metrics import compute
from .run import run_policy
from .sweep import scaled_config, total_users
from . import plots, tracegen

H = 3600.0
POLICY = "ngt_principles_reclaim"


def carved_config(cfg: dict, n_carve: int) -> dict:
    """Topology with n_carve NVL nodes split into MIG slices. Each carved
    node is a separate Node with its own cordon exposure; concentrating all
    slices in one node would make one maintenance window zero the entire
    interactive pool (which is exactly the failure mode this sweep probes)."""
    out = copy.deepcopy(cfg)
    pools = out["cluster"]["pools"]
    base_nvl = 12  # physical NVL nodes including carved ones
    pools["h100nvl"]["num_nodes"] = base_nvl - n_carve
    for pool, per_node in (("mig3g", {"gpus_per_node": 12, "vcpu_per_node": 138,
                                      "mem_per_node": 1080}),
                           ("mig1g", {"gpus_per_node": 14, "vcpu_per_node": 46,
                                      "mem_per_node": 360})):
        pools[pool]["num_nodes"] = n_carve
        for k, v in per_node.items():
            pools[pool][k] = v
    return out


def extend_mig_cordons(cordons, config: dict, max_carve: int, seed: int):
    """Replace the base trace's MIG cordon events with streams for
    max_carve carved nodes (same distribution, dedicated child rng). Runs
    with fewer carved nodes ignore events for nodes they do not have, so
    the per-node cordon realizations are identical across carve settings."""
    keep = [c for c in cordons if not c.node_id.startswith("mig")]
    rng = np.random.default_rng([seed, 424242])
    mini = {"pools": {"mig3g": {"num_nodes": max_carve},
                      "mig1g": {"num_nodes": max_carve}}}
    horizon_s = config["horizon_days"] * 24 * H
    extra = tracegen.gen_cordons(rng, mini, config["workload"]["cordons"],
                                 horizon_s)
    return keep + extra


def session_stats(engine, t0: float, t1: float) -> dict:
    """1-GPU logical-job waits split MIG vs full-GPU, plus multi-GPU."""
    groups: dict[str, list[dict]] = {}
    for r in engine.records:
        if r.get("record") != "allocation" and t0 <= r["submit_time"] < t1 \
                and r["resubmit_of"] is None:
            groups.setdefault(r["group_id"], []).append(r)
    out = {k: {"waits": [], "never": 0}
           for k in ("mig", "fullgpu_single", "multi")}
    for recs in groups.values():
        r0 = recs[0]
        if r0["pool"].startswith("mig") and r0["gpus"] == 1:
            key = "mig"
        elif r0["pool"] in ("h100nvl", "h100sxm", "l40s") and r0["gpus"] == 1:
            key = "fullgpu_single"
        elif r0["pool"] in ("h100nvl", "h100sxm", "l40s") and r0["gpus"] > 1:
            key = "multi"
        else:
            continue
        started = [r for r in recs if r["outcome"] == "started"]
        if started:
            out[key]["waits"].append(min(r["wait_s"] for r in started) / 60.0)
        else:
            out[key]["never"] += 1
    stats = {}
    for key, d in out.items():
        n = len(d["waits"]) + d["never"]
        stats[key] = {
            "n": n,
            "p95_min": float(np.percentile(d["waits"], 95)) if d["waits"] else None,
            "never_frac": d["never"] / n if n else 0.0,
        }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", default="0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--carves", default="1,2,3")
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--reserve-nvl", type=int, default=4)
    ap.add_argument("--reserve-l40s", type=int, default=1)
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scales = [float(s) for s in args.scales.split(",")]
    carves = [int(c) for c in args.carves.split(",")]
    t0 = config["warmup_days"] * 24 * H
    t1 = config["horizon_days"] * 24 * H

    results: dict[float, dict[int, list[dict]]] = {
        s: {c: [] for c in carves} for s in scales}
    users_at_scale: dict[float, int] = {}

    for scale in scales:
        for rep in range(args.replicas):
            seed = config["seed"] + 1000 * rep + round(scale * 100)
            base = scaled_config(config, scale, seed)
            users_at_scale[scale] = total_users(base)
            _, requests, cordons = tracegen.generate(base)
            cordons = extend_mig_cordons(cordons, base, max(carves), seed)
            for n_carve in carves:
                cfg = carved_config(base, n_carve)
                params = dict(cfg["policies"][POLICY])
                params["reserve"] = {"h100nvl": args.reserve_nvl,
                                     "l40s": args.reserve_l40s}
                engine = run_policy(POLICY, params, cfg, requests, cordons)
                stats = session_stats(engine, t0, t1)
                stats["seed"] = seed
                results[scale][n_carve].append(stats)
            print(f"scale {scale} ({users_at_scale[scale]} users) "
                  f"rep {rep + 1}/{args.replicas} done")

    payload = {
        "config": args.config, "policy": POLICY,
        "reserve": {"h100nvl": args.reserve_nvl, "l40s": args.reserve_l40s},
        "base_seed": config["seed"], "replicas": args.replicas,
        "scales": scales, "carves": carves,
        "users_at_scale": {str(s): users_at_scale[s] for s in scales},
        "results": {str(s): {str(c): v for c, v in by.items()}
                    for s, by in results.items()},
    }
    (out / "sweep_mig.json").write_text(json.dumps(payload, indent=2))
    plots.mig_sizing(results, users_at_scale, out / "mig_sizing.png")
    print(f"wrote {out}/sweep_mig.json, mig_sizing.png")


if __name__ == "__main__":
    main()
