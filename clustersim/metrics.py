"""Metrics over engine output, computed on a measurement window.

All quantities are windowed to [t0, t1] (warmup excluded) and split GPU vs
CPU pools where it matters. Wait metrics are at the logical-job level: a
gaming group of K siblings counts once, with the wait of whichever sibling
started first, because that is the wait the user experienced.
"""

from __future__ import annotations

import numpy as np

from .engine import used_gpu_seconds
from .trace import Request

H = 3600.0


def jain(values: list[float]) -> float:
    v = np.array([x for x in values if np.isfinite(x)])
    if len(v) == 0 or np.all(v == 0):
        return 1.0
    return float(v.sum() ** 2 / (len(v) * (v**2).sum()))


def _percentiles(waits: list[float]) -> dict:
    if not waits:
        return {"n": 0, "mean_min": None, "median_min": None, "p95_min": None, "max_min": None}
    w = np.array(waits) / 60.0
    return {
        "n": len(w),
        "mean_min": float(w.mean()),
        "median_min": float(np.median(w)),
        "p95_min": float(np.percentile(w, 95)),
        "max_min": float(w.max()),
    }


def reclaimable_idle_gpu_seconds(req: Request, span_s: float,
                                 util_thresh: float, idle_after_s: float) -> float:
    """GPU-seconds inside idle runs beyond the first idle_after_s of each run,
    within the first span_s of the profile. This is what an idle-reclaim
    policy with threshold T could have freed."""
    total = 0.0
    run = 0.0
    t = 0.0
    for dur, util in req.profile:
        d = min(dur, max(0.0, span_s - t))
        t += dur
        if d <= 0:
            break
        if util < util_thresh:
            run += d
        else:
            total += max(0.0, run - idle_after_s)
            run = 0.0
    total += max(0.0, run - idle_after_s)
    return total * req.gpus


def compute(records: list[dict], snapshots: list[dict],
            requests_by_id: dict[str, Request], gpu_pools: list[str],
            t0: float, t1: float,
            util_thresh: float = 0.05, idle_after_s: float = 1800.0,
            charge_factors: dict | None = None,
            wp_targets: dict | None = None) -> dict:
    req_recs = [r for r in records if r.get("record") != "allocation"]
    alloc_recs = [r for r in records if r.get("record") == "allocation"]

    # ---------------- logical-job waits (submitted inside the window)
    groups: dict[str, list[dict]] = {}
    for r in req_recs:
        if t0 <= r["submit_time"] < t1 and r["resubmit_of"] is None:
            groups.setdefault(r["group_id"], []).append(r)

    waits_all: list[float] = []
    waits_interactive: list[float] = []  # P1 tier: single-GPU logical jobs
    waits_multi: list[float] = []        # multi-GPU (production-size) jobs
    waits_by_pool: dict[str, list[float]] = {}
    waits_by_user: dict[str, list[float]] = {}
    n_started = n_never = n_inter_never = n_multi_never = 0
    for gid, recs in groups.items():
        gpu_job = recs[0]["pool"] in gpu_pools
        interactive = recs[0]["gpus"] == 1 and gpu_job
        multi = recs[0]["gpus"] > 1 and gpu_job
        started = [r for r in recs if r["outcome"] == "started"]
        if started:
            first = min(started, key=lambda r: r["submit_time"] + r["wait_s"])
            n_started += 1
            waits_all.append(first["wait_s"])
            if interactive:
                waits_interactive.append(first["wait_s"])
            if multi:
                waits_multi.append(first["wait_s"])
            waits_by_pool.setdefault(first["pool"], []).append(first["wait_s"])
            waits_by_user.setdefault(first["user"], []).append(first["wait_s"])
        else:
            n_never += 1
            if interactive:
                n_inter_never += 1
            if multi:
                n_multi_never += 1

    resubs = [r for r in req_recs if r["resubmit_of"] is not None
              and t0 <= r["submit_time"] < t1]
    resub_started = [r for r in resubs if r["outcome"] == "started"]

    # ---------------- utilization per GPU pool, clipped to the window
    util = {p: {"allocated_gpu_h": 0.0, "used_gpu_h": 0.0,
                "reclaimable_idle_gpu_h": 0.0} for p in gpu_pools}
    n_reclaims = 0
    for a in alloc_recs:
        if a["end_reason"] == "reclaimed" and t0 <= a["end"] < t1:
            n_reclaims += 1
        if a["pool"] not in util:
            continue
        c0, c1 = max(a["start"], t0), min(a["end"], t1)
        if c1 <= c0:
            continue
        req = requests_by_id[a["request_id"]]
        frac = (c1 - c0) / (a["end"] - a["start"]) if a["end"] > a["start"] else 0.0
        u = util[a["pool"]]
        u["allocated_gpu_h"] += a["held_gpu_s"] * frac / H
        used = (used_gpu_seconds(req, c1 - a["start"])
                - used_gpu_seconds(req, c0 - a["start"]))
        u["used_gpu_h"] += used / H
        span = min(a["end"], t1) - a["start"]
        rec_full = reclaimable_idle_gpu_seconds(req, span, util_thresh, idle_after_s)
        rec_before = reclaimable_idle_gpu_seconds(req, max(0.0, t0 - a["start"]),
                                                  util_thresh, idle_after_s)
        u["reclaimable_idle_gpu_h"] += (rec_full - rec_before) / H
    for u in util.values():
        u["idle_held_gpu_h"] = u["allocated_gpu_h"] - u["used_gpu_h"]

    # ---------------- fairness: per-user GPU-hour satisfaction in window
    demand: dict[str, float] = {}
    got: dict[str, float] = {}
    for gid, recs in groups.items():
        r0 = recs[0]
        if r0["gpus"] <= 0 or r0["pool"] not in gpu_pools:
            continue
        req = requests_by_id[r0["request_id"]]
        demand[r0["user"]] = demand.get(r0["user"], 0.0) + req.gpus * req.duration_s / H
    for a in alloc_recs:
        if a["pool"] in gpu_pools:
            c0, c1 = max(a["start"], t0), min(a["end"], t1)
            if c1 > c0:
                frac = (c1 - c0) / (a["end"] - a["start"])
                got[a["user"]] = got.get(a["user"], 0.0) + a["requested_gpu_s"] * frac / H
    satisfaction = [min(got.get(u, 0.0) / d, 1.0) for u, d in demand.items() if d > 0]

    # ---------------- WP charged shares (principle P2/P3): charged GPU-hours
    # are held hours x GPUs x per-model correction factor
    wp_shares = None
    if wp_targets:
        cf = charge_factors or {}
        charged: dict[str, float] = {w: 0.0 for w in wp_targets}
        for a in alloc_recs:
            if a["pool"] not in gpu_pools or not a.get("wp"):
                continue
            c0, c1 = max(a["start"], t0), min(a["end"], t1)
            if c1 <= c0:
                continue
            frac = (c1 - c0) / (a["end"] - a["start"])
            charged[a["wp"]] = charged.get(a["wp"], 0.0) + (
                a["held_gpu_s"] * frac / H * cf.get(a["pool"], 1.0))
        total = sum(charged.values())
        wp_shares = {
            w: {"charged_gpu_h": round(v, 1),
                "share": v / total if total else 0.0,
                "target": wp_targets.get(w, 0.0)}
            for w, v in sorted(charged.items())
        }

    out = {
        "window_days": [t0 / 86400.0, t1 / 86400.0],
        "logical_jobs": len(groups),
        "started": n_started,
        "never_started": n_never,
        "never_started_frac": n_never / len(groups) if groups else 0.0,
        "n_reclaims": n_reclaims,
        "n_resubmits": len(resubs),
        "resubmit_extra_wait_h": sum(r["wait_s"] for r in resub_started) / H,
        "wait_overall": _percentiles(waits_all),
        "wait_interactive": _percentiles(waits_interactive),
        "interactive_never_started": n_inter_never,
        "wait_multi": _percentiles(waits_multi),
        "multi_never_started": n_multi_never,
        "wait_by_pool": {p: _percentiles(w) for p, w in sorted(waits_by_pool.items())},
        "utilization_by_pool": util,
        "wp_shares": wp_shares,
        "wp_share_max_dev": (max(abs(v["share"] - v["target"])
                                 for v in wp_shares.values()) if wp_shares else None),
        "jain_satisfaction": jain(satisfaction),
        "jain_mean_wait": jain([float(np.mean(w)) for w in waits_by_user.values()]),
        "per_user_mean_wait_min": {
            u: float(np.mean(w)) / 60.0 for u, w in sorted(waits_by_user.items())
        },
    }
    return out


def saturday_window_check(records: list[dict], snapshots: list[dict],
                          gpu_pools: list[str], win0_h: float, win1_h: float) -> dict:
    """Compare against the observed data point: Saturday 18:00 to 19:35,
    8 requests Pending 15 to 71 min across pools, zero allocatable capacity,
    about 12 of 73 nodes cordoned."""
    t0, t1 = win0_h * H, win1_h * H
    snaps = [s for s in snapshots if t0 <= s["time"] <= t1]
    free = [sum(s[f"{p}.free_allocatable_gpus"] for p in gpu_pools) for s in snaps]
    cordon = [s["cordoned_fraction"] for s in snaps]
    pend_in_window = []
    for r in records:
        if r.get("record") == "allocation":
            continue
        p0, p1 = r["submit_time"], r["submit_time"] + r["wait_s"]
        if p0 < t1 and p1 > t0 and r["outcome"] in ("started", "cancelled_patience",
                                                    "pending_at_end"):
            pend_in_window.append(r)
    waits_min = sorted(r["wait_s"] / 60.0 for r in pend_in_window)
    return {
        "window_h": [win0_h, win1_h],
        "mean_free_allocatable_gpus": float(np.mean(free)) if free else None,
        "min_free_allocatable_gpus": float(np.min(free)) if free else None,
        "mean_cordoned_fraction": float(np.mean(cordon)) if cordon else None,
        "n_requests_pending_in_window": len(pend_in_window),
        "their_waits_min": [round(w, 1) for w in waits_min],
    }


def idle_held_h100_probe(records: list[dict], requests_by_id: dict[str, Request],
                         probe_time_s: float, util_thresh: float = 0.05) -> int:
    """Number of held H100 GPUs whose instantaneous utilization at probe time
    is below threshold (the '16 H100s at 1-3 GB' dashboard observation)."""
    count = 0
    for a in records:
        if a.get("record") != "allocation" or a["pool"] not in ("h100nvl", "h100sxm"):
            continue
        if not (a["start"] <= probe_time_s < a["end"]):
            continue
        req = requests_by_id[a["request_id"]]
        off = probe_time_s - a["start"]
        t = 0.0
        u_now = 0.0
        for dur, u in req.profile:
            if t <= off < t + dur:
                u_now = u
                break
            t += dur
        if u_now < util_thresh:
            count += a["held_gpus"]
    return count
