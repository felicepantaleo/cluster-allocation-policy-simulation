"""Partition sizing: 1-GPU session reserve vs capped multi-GPU capacity.

    python -m clustersim.sweep_reserve --config config/phase1.yaml --out results/sweep_reserve

For each user-population scale and each reserve fraction f, runs the
ngt_principles_reclaim policy with round(f x pool GPUs) set aside for the
guaranteed 1-GPU tier on the partial-node GPU pools (H100 NVL and L40S).
Multi-GPU jobs can never enter the reserve, so the maximum capacity
assignable to time-capped multi-GPU allocations is pool minus reserve; the
sweep identifies the smallest reserve that meets the P1 wait target at each
user count and the cost this imposes on multi-GPU jobs. Within a sweep
point every fraction runs on the identical trace (same requests, same
cordons), so the reserve is the only variable.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from .metrics import compute
from .run import run_policy
from .sweep import scaled_config, total_users
from . import plots, tracegen

H = 3600.0
POLICY = "ngt_principles_reclaim"
RESERVE_POOLS = ("h100nvl", "l40s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scales", default="0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--fractions", default="0,0.05,0.1,0.15,0.2,0.3,0.4")
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--target-min", type=float, default=15.0,
                    help="P1 target: 1-GPU tier p95 wait (minutes)")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scales = [float(s) for s in args.scales.split(",")]
    fractions = [float(f) for f in args.fractions.split(",")]
    gpu_pools = config["gpu_pools"]
    t0 = config["warmup_days"] * 24 * H
    t1 = config["horizon_days"] * 24 * H
    pool_gpus = {p: config["cluster"]["pools"][p]["num_nodes"]
                 * config["cluster"]["pools"][p]["gpus_per_node"]
                 for p in RESERVE_POOLS}

    def reserve_for(frac: float) -> dict[str, int]:
        return {p: round(frac * pool_gpus[p]) for p in RESERVE_POOLS}

    # results[scale][frac] = list over replicas of metric dicts
    results: dict[float, dict[float, list[dict]]] = {
        s: {f: [] for f in fractions} for s in scales}
    users_at_scale: dict[float, int] = {}

    for scale in scales:
        for rep in range(args.replicas):
            seed = config["seed"] + 1000 * rep + round(scale * 100)
            cfg = scaled_config(config, scale, seed)
            users_at_scale[scale] = total_users(cfg)
            _, requests, cordons = tracegen.generate(cfg)
            for frac in fractions:
                params = dict(cfg["policies"][POLICY])
                params["reserve"] = reserve_for(frac)
                engine = run_policy(POLICY, params, cfg, requests, cordons)
                m = compute(engine.records, engine.snapshots,
                            engine.requests_by_id, gpu_pools, t0, t1,
                            charge_factors=cfg.get("gpu_charge_factor"),
                            wp_targets=cfg.get("wp_targets"))
                results[scale][frac].append({
                    "seed": seed,
                    "reserve": reserve_for(frac),
                    "inter_p95_min": m["wait_interactive"]["p95_min"],
                    "multi_p95_min": m["wait_multi"]["p95_min"],
                    "multi_never_started": m["multi_never_started"],
                    "never_started_frac": m["never_started_frac"],
                })
            print(f"scale {scale} ({users_at_scale[scale]} users) "
                  f"rep {rep + 1}/{args.replicas} done")

    payload = {
        "config": args.config, "policy": POLICY,
        "base_seed": config["seed"], "replicas": args.replicas,
        "scales": scales, "fractions": fractions,
        "pool_gpus": pool_gpus, "target_min": args.target_min,
        "users_at_scale": {str(s): users_at_scale[s] for s in scales},
        "results": {str(s): {str(f): v for f, v in by.items()}
                    for s, by in results.items()},
    }
    (out / "sweep_reserve.json").write_text(json.dumps(payload, indent=2))

    total_reserve = {f: sum(reserve_for(f).values()) for f in fractions}
    plots.reserve_sizing(results, users_at_scale, total_reserve,
                         args.target_min, out / "reserve_sizing.png")

    # recommendation table: smallest reserve meeting the target per scale
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    lines = ["# Reserve sizing for the 1-GPU session tier",
             "",
             f"Policy {POLICY}, target: 1-GPU tier p95 wait below "
             f"{args.target_min:.0f} min. Reserve applies to H100 NVL "
             f"({pool_gpus['h100nvl']} GPUs) and L40S ({pool_gpus['l40s']}) "
             "proportionally; multi-GPU capacity = pool minus reserve "
             "(H100 SXM stays fully multi-GPU). Mean over "
             f"{args.replicas} trace seeds.",
             "",
             "| users | min reserve meeting target (NVL+L40S GPUs) | "
             "1-GPU p95 (min) | multi-GPU p95 (min) at that reserve | "
             "max multi-GPU capacity (NVL+L40S GPUs) |",
             "|---|---|---|---|---|"]
    for s in scales:
        pick = None
        for f in sorted(fractions):
            if (m := mean([r["inter_p95_min"] for r in results[s][f]])) is not None \
                    and m <= args.target_min:
                pick = (f, m)
                break
        if pick is None:
            by_f = {f: mean([r["inter_p95_min"] for r in results[s][f]])
                    for f in fractions}
            f_best = min(by_f, key=lambda f: by_f[f])
            cell = (f"target missed at every reserve; minimum p95 "
                    f"{by_f[f_best]:.0f} at {total_reserve[f_best]} GPUs")
            multi = mean([r["multi_p95_min"] for r in results[s][f_best]])
            cap = sum(pool_gpus.values()) - total_reserve[f_best]
            lines.append(f"| {users_at_scale[s]} | {cell} | n/a | "
                         f"{multi:.0f} | {cap} |")
            continue
        f, m = pick
        multi = mean([r["multi_p95_min"] for r in results[s][f]])
        cap = sum(pool_gpus.values()) - total_reserve[f]
        lines.append(f"| {users_at_scale[s]} | {total_reserve[f]} "
                     f"(fraction {f:.2f}) | {m:.0f} | {multi:.0f} | {cap} |")
    (out / "sweep_reserve.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/sweep_reserve.json, sweep_reserve.md, reserve_sizing.png")


if __name__ == "__main__":
    main()
