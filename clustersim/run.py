"""Experiment runner.

    python -m clustersim.run --config config/phase1.yaml --out results/phase1

Generates (or reuses) the trace, runs every policy in the config on the same
trace, writes per-policy records/snapshots/metrics, the validation checks,
comparison plots and a markdown comparison table. Everything is derived from
the config seeds, so a rerun reproduces byte-identical results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .cluster import Cluster, PoolSpec
from .engine import Engine
from .metrics import compute, idle_held_h100_probe, saturday_window_check
from .policies import make_policy
from . import principles
from .trace import read_trace, write_trace
from . import tracegen, plots

H = 3600.0


def build_cluster(cluster_cfg: dict) -> Cluster:
    pools = [
        PoolSpec(name=name, num_nodes=s["num_nodes"], gpus_per_node=s["gpus_per_node"],
                 vcpu_per_node=s["vcpu_per_node"], mem_per_node=s["mem_per_node"],
                 granularity=s["granularity"], socket_vcpu=s.get("socket_vcpu", 0.0))
        for name, s in cluster_cfg["pools"].items()
    ]
    return Cluster(pools)


def run_policy(name: str, params: dict, config: dict, requests, cordons) -> Engine:
    cluster = build_cluster(config["cluster"])
    params = dict(params or {})
    params.setdefault("wp_targets", config.get("wp_targets"))
    params.setdefault("charge_factors", config.get("gpu_charge_factor"))
    policy = make_policy(name, params)
    eng_cfg = config["engine"]
    engine = Engine(
        cluster=cluster, policy=policy, requests=requests, cordons=cordons,
        horizon_s=config["horizon_days"] * 24 * H,
        seed=eng_cfg["seed"],
        snapshot_interval_s=config["snapshot_interval_s"],
        resubmit_reaction_median_s=eng_cfg["resubmit_reaction_median_s"],
        resubmit_reaction_sigma=eng_cfg["resubmit_reaction_sigma"],
        resubmit_patience_s=eng_cfg["resubmit_patience_s"],
    )
    engine.run()
    return engine


def fmt(v, nd=1):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def comparison_table(metrics_by_policy: dict[str, dict], gpu_pools: list[str]) -> str:
    lines = ["| metric | " + " | ".join(metrics_by_policy) + " |",
             "|---|" + "---|" * len(metrics_by_policy)]

    def row(label, getter, nd=1):
        cells = [fmt(getter(m), nd) for m in metrics_by_policy.values()]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("logical jobs in window", lambda m: m["logical_jobs"], 0)
    row("jobs started", lambda m: m["started"], 0)
    row("never started (frac)", lambda m: m["never_started_frac"], 3)
    row("wait mean (min)", lambda m: m["wait_overall"]["mean_min"])
    row("wait median (min)", lambda m: m["wait_overall"]["median_min"])
    row("wait p95 (min)", lambda m: m["wait_overall"]["p95_min"])
    row("wait max (min)", lambda m: m["wait_overall"]["max_min"])
    row("1-GPU tier wait mean (min)", lambda m: m["wait_interactive"]["mean_min"])
    row("1-GPU tier wait p95 (min)", lambda m: m["wait_interactive"]["p95_min"])
    row("1-GPU tier never started", lambda m: m["interactive_never_started"], 0)
    for p in gpu_pools:
        row(f"{p} allocated GPU-h", lambda m, p=p: m["utilization_by_pool"][p]["allocated_gpu_h"], 0)
        row(f"{p} used GPU-h", lambda m, p=p: m["utilization_by_pool"][p]["used_gpu_h"], 0)
        row(f"{p} idle-held GPU-h", lambda m, p=p: m["utilization_by_pool"][p]["idle_held_gpu_h"], 0)
        row(f"{p} reclaimable idle GPU-h", lambda m, p=p: m["utilization_by_pool"][p]["reclaimable_idle_gpu_h"], 0)
    row("reclaims", lambda m: m["n_reclaims"], 0)
    row("resubmissions", lambda m: m["n_resubmits"], 0)
    row("resubmit extra wait (h)", lambda m: m["resubmit_extra_wait_h"])
    row("Jain (GPU-h satisfaction)", lambda m: m["jain_satisfaction"], 3)
    row("Jain (per-user mean wait)", lambda m: m["jain_mean_wait"], 3)
    first = next(iter(metrics_by_policy.values()))
    if first.get("wp_shares"):
        for w in first["wp_shares"]:
            row(f"{w} charged share (target {first['wp_shares'][w]['target']:.2f})",
                lambda m, w=w: m["wp_shares"][w]["share"], 3)
        row("WP share max deviation", lambda m: m["wp_share_max_dev"], 3)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regen-trace", action="store_true")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trace_dir = out / "trace"
    if args.regen_trace or not (trace_dir / "requests.jsonl").exists():
        meta, requests, cordons = tracegen.generate(config)
        write_trace(trace_dir, meta, requests, cordons)
    meta, requests, cordons = read_trace(trace_dir)
    print(f"trace: {meta['n_logical_jobs']} logical jobs, "
          f"{meta['n_requests']} requests, {meta['n_cordon_events']} cordon events")

    gpu_pools = config["gpu_pools"]
    t0 = config["warmup_days"] * 24 * H
    t1 = config["horizon_days"] * 24 * H
    win0, win1 = config["validation"]["saturday_window_h"]

    cards: dict[str, dict] = {}
    metrics_by_policy: dict[str, dict] = {}
    waits_by_policy: dict[str, list[float]] = {}
    snapshots_by_policy: dict[str, list[dict]] = {}
    util_by_policy: dict[str, dict] = {}
    allocs_by_policy: dict[str, list[dict]] = {}
    validation: dict[str, dict] = {}

    for pname, pparams in config["policies"].items():
        print(f"running policy: {pname}")
        engine = run_policy(pname, pparams, config, requests, cordons)
        pdir = out / pname
        pdir.mkdir(exist_ok=True)
        with open(pdir / "records.jsonl", "w") as f:
            for r in engine.records:
                f.write(json.dumps(r) + "\n")
        with open(pdir / "snapshots.jsonl", "w") as f:
            for s in engine.snapshots:
                f.write(json.dumps(s) + "\n")

        reclaim_t = config["policies"].get("idle_reclaim", {}).get("idle_after_s", 1800)
        m = compute(engine.records, engine.snapshots, engine.requests_by_id,
                    gpu_pools, t0, t1, idle_after_s=reclaim_t,
                    charge_factors=config.get("gpu_charge_factor"),
                    wp_targets=config.get("wp_targets"))
        metrics_by_policy[pname] = m
        (pdir / "metrics.json").write_text(json.dumps(m, indent=2))

        sat = saturday_window_check(engine.records, engine.snapshots, gpu_pools, win0, win1)
        probe_t = config["validation"]["idle_probe_h"] * H
        sat["idle_held_h100_gpus_at_probe"] = idle_held_h100_probe(
            engine.records, engine.requests_by_id, probe_t)
        validation[pname] = sat
        (pdir / "validation.json").write_text(json.dumps(sat, indent=2))

        req_recs = [r for r in engine.records if r.get("record") != "allocation"]
        groups: dict[str, list] = {}
        for r in req_recs:
            if t0 <= r["submit_time"] < t1 and r["resubmit_of"] is None:
                groups.setdefault(r["group_id"], []).append(r)
        waits = []
        for recs in groups.values():
            started = [r for r in recs if r["outcome"] == "started"]
            if started:
                waits.append(min(r["wait_s"] for r in started))
        waits_by_policy[pname] = waits
        snapshots_by_policy[pname] = engine.snapshots
        util_by_policy[pname] = m["utilization_by_pool"]
        allocs_by_policy[pname] = [r for r in engine.records
                                   if r.get("record") == "allocation"]
        card = principles.scorecard(
            engine.records, engine.requests_by_id, gpu_pools, t0, t1,
            wp_targets=config.get("wp_targets"),
            charge_factors=config.get("gpu_charge_factor"),
            cap_h=pparams.get("multi_gpu_cap_h", 24.0) if pparams else 24.0,
            planning_tiers=(pparams or {}).get(
                "tiers", [{"max_h": 8, "decisions_per_day": 3},
                          {"max_h": 100000, "decisions_per_day": 1}])
            if pname == "planning_cycle" else None)
        cards[pname] = card
        (pdir / "principles.json").write_text(json.dumps(card, indent=2))

    plots.wait_cdf(waits_by_policy, "all pools, logical jobs", out / "wait_cdf_all.png")
    plots.gpu_hours_bars(util_by_policy, gpu_pools, out / "gpu_hours.png")
    if "fcfs_pending" in allocs_by_policy and "planning_cycle" in allocs_by_policy:
        plots.user_hold_timeline(
            {"fcfs_pending": allocs_by_policy["fcfs_pending"],
             "planning_cycle": allocs_by_policy["planning_cycle"]},
            gpu_pools, config["horizon_days"], out / "heavy_holders.png")
    sat_windows = [(win0 - 168.0, win1 - 168.0), (win0, win1)]
    plots.occupancy_timeline(snapshots_by_policy, ["h100nvl", "h100sxm", "l40s"],
                             out / "occupancy.png", shade_windows_h=sat_windows)

    table = comparison_table(metrics_by_policy, gpu_pools)
    report = ["# Phase 1 comparison: " + " vs ".join(metrics_by_policy), "",
              f"Trace seed {config['seed']}, engine seed {config['engine']['seed']}, "
              f"measurement window days {config['warmup_days']} to {config['horizon_days']}.",
              "", "## Metrics", "", table, "",
              principles.render_md(cards), "",
              "## Validation vs observed Saturday data point", ""]
    for pname, v in validation.items():
        report += [f"### {pname}", "", "```json", json.dumps(v, indent=2), "```", ""]
    (out / "comparison.md").write_text("\n".join(report))
    print(f"wrote {out}/comparison.md and plots")


if __name__ == "__main__":
    main()
