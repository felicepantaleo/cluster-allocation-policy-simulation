"""Principle-by-principle scorecard (PRINCIPLES.md P1 to P5).

Scores one policy run against each principle with concrete, checkable
numbers, and renders a comparison table across policies. Verdict
thresholds are stated next to each principle; every number comes from the
run's records.

Job-done accounting follows reclaim/cap resubmission chains: a logical
job is "done" when at least 90% of its requested active GPU-seconds were
delivered across the original allocation and all resubmissions; its
completion time runs from the original submission to the end of the last
allocation in the chain. A hoarder whose one real burst ran but whose
idle parking was reaped therefore counts as done, quickly, with a low
held ratio; that distinction is the point.
"""

from __future__ import annotations


import numpy as np

from .trace import Request

H = 3600.0


def _percentile(vals, q):
    return float(np.percentile(vals, q)) if vals else None


def scorecard(records: list[dict], requests_by_id: dict[str, Request],
              gpu_pools: list[str], t0: float, t1: float,
              wp_targets: dict | None, charge_factors: dict | None,
              p1_target_min: float = 15.0, cap_h: float = 24.0,
              planning_tiers: list | None = None) -> dict:  # tiers ignored
    gpu_pools = set(gpu_pools)
    req_recs = [r for r in records if r.get("record") != "allocation"]
    alloc_recs = [r for r in records if r.get("record") == "allocation"]

    # resubmission chains: request_id -> root group_id
    root: dict[str, str] = {}
    for r in req_recs:
        if r["resubmit_of"] is None:
            root[r["request_id"]] = r["group_id"]
        else:
            root[r["request_id"]] = root.get(r["resubmit_of"], r["group_id"])
    allocs_by_root: dict[str, list[dict]] = {}
    for a in alloc_recs:
        allocs_by_root.setdefault(root.get(a["request_id"], a["request_id"]),
                                  []).append(a)

    # logical jobs submitted in the window
    groups: dict[str, list[dict]] = {}
    for r in req_recs:
        if r["resubmit_of"] is None and t0 <= r["submit_time"] < t1:
            groups.setdefault(r["group_id"], []).append(r)

    def first_start(recs):
        started = [r for r in recs if r["outcome"] == "started"]
        return min((r["wait_s"] for r in started), default=None)

    # ---------------- P1: single-GPU interactive tier, dev sessions
    dev_waits, dev_never = [], 0
    pend_events = []  # (+1 submit, -1 resolve) for waiting dev singles
    for gid, recs in groups.items():
        r0 = recs[0]
        if not (r0["gpus"] == 1 and r0["pool"] in gpu_pools and r0["kind"] == "dev"):
            continue
        w = first_start(recs)
        if w is None:
            dev_never += 1
            span = max(r["wait_s"] for r in recs)
        else:
            dev_waits.append(w)
            span = w
        if span > 0:
            pend_events.append((r0["submit_time"], +1))
            pend_events.append((r0["submit_time"] + span, -1))
    pend_events.sort()
    level = 0
    last_t = t0
    area = 0.0
    peak = 0
    for t, delta in pend_events:
        tc = min(max(t, t0), t1)
        area += level * (tc - last_t)
        last_t = tc
        level += delta
        peak = max(peak, level)
    area += level * (t1 - last_t)
    n_dev = len(dev_waits) + dev_never
    served_target = sum(1 for w in dev_waits if w <= p1_target_min * 60)
    p1 = {
        "dev_single_sessions": n_dev,
        "served_within_target_frac": served_target / n_dev if n_dev else None,
        "p95_wait_min": _percentile([w / 60 for w in dev_waits], 95),
        "never_started": dev_never,
        "mean_waiting_devs": area / (t1 - t0),
        "peak_waiting_devs": peak,
    }
    p1["verdict"] = (
        "met" if p1["p95_wait_min"] is not None
        and p1["p95_wait_min"] <= p1_target_min and n_dev
        and dev_never / n_dev < 0.05
        else "partial" if p1["p95_wait_min"] is not None
        and p1["p95_wait_min"] <= 2 * p1_target_min
        else "missed")

    # ---------------- job-done per class (incl. the hoarder question)
    by_kind: dict[str, dict] = {}
    for gid, recs in groups.items():
        r0 = recs[0]
        if r0["pool"] not in gpu_pools:
            continue
        req = requests_by_id[r0["request_id"]]
        # requested work counts only genuinely active segments; the 0.02
        # utilization of gaps and parking is not work the user asked to do
        requested_active = req.gpus * sum(
            d * u for d, u in req.profile if u >= 0.05)
        chain = allocs_by_root.get(gid, [])
        delivered = sum(a["used_gpu_s"] for a in chain)
        held = sum(a["held_gpu_s"] for a in chain)
        d = by_kind.setdefault(r0["kind"], {"n": 0, "done": 0, "t_done_h": [],
                                            "held_ratio": []})
        d["n"] += 1
        if requested_active > 0 and delivered >= 0.9 * requested_active:
            d["done"] += 1
            d["t_done_h"].append(
                (max(a["end"] for a in chain) - r0["submit_time"]) / H)
        if chain and req.gpus * req.duration_s > 0:  # started jobs only
            d["held_ratio"].append(held / (req.gpus * req.duration_s))
    job_done = {
        kind: {
            "jobs": d["n"],
            "done_frac": d["done"] / d["n"] if d["n"] else None,
            "median_time_to_done_h": _percentile(d["t_done_h"], 50),
            "p95_time_to_done_h": _percentile(d["t_done_h"], 95),
            "median_held_vs_asked": _percentile(d["held_ratio"], 50),
        } for kind, d in sorted(by_kind.items())
    }

    # ---------------- P2: WP charged shares
    p2 = None
    if wp_targets:
        cf = charge_factors or {}
        charged = {w: 0.0 for w in wp_targets}
        for a in alloc_recs:
            if a["pool"] in gpu_pools and a.get("wp"):
                c0, c1 = max(a["start"], t0), min(a["end"], t1)
                if c1 > c0:
                    frac = (c1 - c0) / (a["end"] - a["start"])
                    charged[a["wp"]] = charged.get(a["wp"], 0.0) + \
                        a["held_gpu_s"] * frac / H * cf.get(a["pool"], 1.0)
        total = sum(charged.values())
        shares = {w: v / total if total else 0.0 for w, v in charged.items()}
        max_dev = max(abs(shares[w] - wp_targets[w]) for w in wp_targets)
        p2 = {"shares": {w: round(s, 3) for w, s in sorted(shares.items())},
              "max_deviation": max_dev,
              "verdict": "met" if max_dev <= 0.05 else
                         "partial" if max_dev <= 0.10 else "missed"}

    # ---------------- P3: multi-GPU time cap
    tol = 300.0
    violations = [a for a in alloc_recs
                  if a["pool"] in gpu_pools and a["requested_gpus"] > 1
                  and t0 <= a["start"] < t1
                  and (a["end"] - a["start"]) > cap_h * H + tol]
    excess = sum((a["end"] - a["start"] - cap_h * H) * a["held_gpus"] / H
                 for a in violations)
    p3 = {"cap_h": cap_h, "multi_gpu_allocs_over_cap": len(violations),
          "excess_gpu_h_over_cap": round(excess, 1),
          "longest_multi_gpu_hold_h": round(max(
              ((a["end"] - a["start"]) / H for a in alloc_recs
               if a["pool"] in gpu_pools and a["requested_gpus"] > 1
               and t0 <= a["start"] < t1), default=0.0), 1),
          "verdict": "met" if not violations else "missed"}

    # ---------------- P4: intra-WP recycling on contended handoffs
    # approximation: a start that coincides exactly with a release in the
    # same pool is a contended handoff (the starter was waiting for it)
    releases: dict[tuple, list[str]] = {}
    for a in alloc_recs:
        if a["pool"] in gpu_pools and t0 <= a["end"] < t1 \
                and a["end_reason"] in ("completed", "reclaimed", "time_capped"):
            releases.setdefault((a["end"], a["pool"]), []).append(a.get("wp", ""))
    handoffs = same_wp = 0
    for a in alloc_recs:
        key = (a["start"], a["pool"])
        if key in releases:
            handoffs += 1
            if a.get("wp") in releases[key]:
                same_wp += 1
    p4 = {"contended_handoffs": handoffs,
          "same_wp_frac": same_wp / handoffs if handoffs else None,
          "verdict": ("met" if handoffs and same_wp / handoffs >= 0.5 else
                      "partial" if handoffs and same_wp / handoffs >= 0.35 else
                      "missed" if handoffs else "n/a")}

    # ---------------- P5: production (multi-GPU) charged to WPs
    prod = {w: 0.0 for w in (wp_targets or {})}
    unattributed = 0.0
    for a in alloc_recs:
        if a["pool"] in gpu_pools and a["requested_gpus"] > 1:
            c0, c1 = max(a["start"], t0), min(a["end"], t1)
            if c1 > c0:
                gpu_h = a["held_gpu_s"] * (c1 - c0) / (a["end"] - a["start"]) / H
                if a.get("wp"):
                    prod[a["wp"]] = prod.get(a["wp"], 0.0) + gpu_h
                else:
                    unattributed += gpu_h
    p5 = {"multi_gpu_gpu_h_by_wp": {w: round(v, 0) for w, v in sorted(prod.items())},
          "unattributed_gpu_h": round(unattributed, 1),
          "verdict": "met" if unattributed == 0 else "missed"}

    return {"P1_interactive_guarantee": p1, "P2_wp_shares": p2,
            "P3_multi_gpu_cap": p3, "P4_intra_wp_recycling": p4,
            "P5_production_attribution": p5,
            "job_done_by_class": job_done}


def render_md(cards: dict[str, dict], p1_target_min: float = 15.0) -> str:
    pols = list(cards)
    lines = ["## Principles scorecard", "",
             f"P1 target: 1-GPU dev session served within {p1_target_min:.0f} "
             "min. Job-done: at least 90% of requested active GPU-seconds "
             "delivered across the reclaim/resubmit chain.",
             "",
             "| principle / metric | " + " | ".join(pols) + " |",
             "|---|" + "---|" * len(pols)]

    def row(label, fn, fmt="{}"):
        cells = []
        for p in pols:
            v = fn(cards[p])
            cells.append("n/a" if v is None else
                         fmt.format(v) if not isinstance(v, float) else
                         fmt.format(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("P1 verdict", lambda c: c["P1_interactive_guarantee"]["verdict"])
    row("P1 dev 1-GPU sessions", lambda c: c["P1_interactive_guarantee"]["dev_single_sessions"])
    row("P1 served within target", lambda c: c["P1_interactive_guarantee"]["served_within_target_frac"], "{:.0%}")
    row("P1 p95 wait (min)", lambda c: c["P1_interactive_guarantee"]["p95_wait_min"], "{:.0f}")
    row("P1 never started", lambda c: c["P1_interactive_guarantee"]["never_started"])
    row("P1 mean devs waiting", lambda c: c["P1_interactive_guarantee"]["mean_waiting_devs"], "{:.2f}")
    row("P1 peak devs waiting", lambda c: c["P1_interactive_guarantee"]["peak_waiting_devs"])
    row("P2 verdict", lambda c: (c["P2_wp_shares"] or {}).get("verdict"))
    row("P2 max share deviation", lambda c: (c["P2_wp_shares"] or {}).get("max_deviation"), "{:.3f}")
    row("P3 verdict", lambda c: c["P3_multi_gpu_cap"]["verdict"])
    row("P3 multi-GPU holds over cap", lambda c: c["P3_multi_gpu_cap"]["multi_gpu_allocs_over_cap"])
    row("P3 excess GPU-h over cap", lambda c: c["P3_multi_gpu_cap"]["excess_gpu_h_over_cap"])
    row("P3 longest multi-GPU hold (h)", lambda c: c["P3_multi_gpu_cap"]["longest_multi_gpu_hold_h"])
    row("P4 verdict", lambda c: c["P4_intra_wp_recycling"]["verdict"])
    row("P4 same-WP handoff fraction", lambda c: c["P4_intra_wp_recycling"]["same_wp_frac"], "{:.0%}")
    row("P5 verdict", lambda c: c["P5_production_attribution"]["verdict"])

    kinds = sorted({k for c in cards.values() for k in c["job_done_by_class"]})
    for kind in kinds:
        row(f"{kind}: jobs done", lambda c, k=kind: (
            None if k not in c["job_done_by_class"] else
            c["job_done_by_class"][k]["done_frac"]), "{:.0%}")
        row(f"{kind}: median time to done (h)", lambda c, k=kind: (
            None if k not in c["job_done_by_class"] else
            c["job_done_by_class"][k]["median_time_to_done_h"]), "{:.1f}")
        row(f"{kind}: median held vs asked (started)", lambda c, k=kind: (
            None if k not in c["job_done_by_class"] else
            c["job_done_by_class"][k]["median_held_vs_asked"]), "{:.0%}")
    return "\n".join(lines)
