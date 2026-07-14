"""Load sweep: waiting time vs number of users.

    python -m clustersim.sweep --config config/phase1.yaml --out results/sweep_users

Scales every user-class count by the given factors (per-user submission
rates unchanged, so the user count is proportional to offered load),
generates one trace per (scale, replicate seed), and runs every policy in
the config on that identical trace. Matched conditions hold within each
sweep point: all policies see the same requests and cordons. Replicates
(different trace seeds) give the spread band; the trace is not written to
disk but is fully determined by the recorded config and seed.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from .metrics import compute
from .run import run_policy
from . import plots, tracegen

H = 3600.0


def scaled_config(config: dict, scale: float, seed: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg["seed"] = seed
    for cls, spec in cfg["workload"]["users"].items():
        spec["count"] = max(1, round(spec["count"] * scale))
    return cfg


def total_users(cfg: dict) -> int:
    return sum(s["count"] for s in cfg["workload"]["users"].values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", default="0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--replicas", type=int, default=3)
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scales = [float(s) for s in args.scales.split(",")]
    gpu_pools = config["gpu_pools"]
    t0 = config["warmup_days"] * 24 * H
    t1 = config["horizon_days"] * 24 * H

    # results[policy][scale] = list of per-replicate metric dicts
    results: dict[str, dict[float, list[dict]]] = {
        p: {s: [] for s in scales} for p in config["policies"]}
    users_at_scale: dict[float, int] = {}

    for scale in scales:
        for rep in range(args.replicas):
            seed = config["seed"] + 1000 * rep + round(scale * 100)
            cfg = scaled_config(config, scale, seed)
            users_at_scale[scale] = total_users(cfg)
            _, requests, cordons = tracegen.generate(cfg)
            for pname, pparams in cfg["policies"].items():
                engine = run_policy(pname, pparams, cfg, requests, cordons)
                m = compute(engine.records, engine.snapshots,
                            engine.requests_by_id, gpu_pools, t0, t1,
                            charge_factors=cfg.get("gpu_charge_factor"),
                            wp_targets=cfg.get("wp_targets"))
                results[pname][scale].append({
                    "seed": seed,
                    "wait_mean_min": m["wait_overall"]["mean_min"],
                    "wait_p95_min": m["wait_overall"]["p95_min"],
                    "inter_p95_min": m["wait_interactive"]["p95_min"],
                    "never_started_frac": m["never_started_frac"],
                })
            print(f"scale {scale} ({users_at_scale[scale]} users) "
                  f"rep {rep + 1}/{args.replicas} done")

    payload = {
        "config": args.config,
        "base_seed": config["seed"],
        "scales": scales,
        "users_at_scale": {str(s): users_at_scale[s] for s in scales},
        "replicas": args.replicas,
        "results": {p: {str(s): v for s, v in by.items()}
                    for p, by in results.items()},
    }
    (out / "sweep.json").write_text(json.dumps(payload, indent=2))

    plots.wait_vs_users(results, users_at_scale, out / "wait_vs_users.png")

    lines = ["# Waiting time vs number of users",
             "",
             f"Config {args.config}, {args.replicas} trace seeds per point; "
             "cells are p95 wait in minutes among started logical jobs, "
             "mean over seeds (min to max). All policies share the identical "
             "trace at each point. The never-started fraction rises with "
             "load and censors these waits; it is reported in sweep.json.",
             ""]
    hdr = "| users | " + " | ".join(results) + " |"
    lines += [hdr, "|---|" + "---|" * len(results)]
    for s in scales:
        cells = []
        for p in results:
            vals = [r["wait_p95_min"] for r in results[p][s]
                    if r["wait_p95_min"] is not None]
            cells.append(f"{sum(vals) / len(vals):.0f} "
                         f"({min(vals):.0f} to {max(vals):.0f})" if vals else "n/a")
        lines.append(f"| {users_at_scale[s]} | " + " | ".join(cells) + " |")
    (out / "sweep.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/sweep.json, sweep.md, wait_vs_users.png")


if __name__ == "__main__":
    main()
