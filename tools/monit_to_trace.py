"""Convert raw MONIT Prometheus dumps (tools/extract_monit_trace.py) into
the simulator's trace schema plus a calibration report.

    python tools/monit_to_trace.py --raw data/monit --out data/derived

Produces:
  requests.jsonl   one line per user pod, simulator Request schema; the
                   `observed` dict carries outcome, wait, node, phases
  cordons.jsonl    cordon windows per node
  report.md        headline statistics vs the synthetic calibration

Attribution: namespace = user (kubeflow profile). Working packages are not
in the cluster data; `wp` stays empty until a user-to-WP map is provided.
Pods matching platform-furniture name patterns are excluded and counted.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

H = 3600.0
INFRA_RE = re.compile(
    r"^(ml-pipeline|pvcviewer|tensorboard|nvidia-|kube-|mon-|oauth2|istio|"
    r"notebook-controller|workflow-controller|metadata-|minio|mysql|cache-)")

GPU_RESOURCES = {
    "nvidia_com_gpu": ("gpu", 1.0),
    "nvidia_com_mig_3g_47gb": ("mig3g", 1.0),
    "nvidia_com_mig_1g_12gb": ("mig1g", 1.0),
    "nvidia_com_NVL_1g_12GB": ("mig1g", 1.0),
    "amd_com_gpu": ("amd", 1.0),
}

POOL_FROM_NODE = (
    ("h100-nvl", "h100nvl"), ("h100-sxm", "h100sxm"), ("h100", "h100nvl"),
    ("l40s", "l40s"), ("mi300", "amd"), ("w7900", "amd"),
    ("t4", "cloud_t4"), ("cpu", "cpu"), ("highcore", "cpu"),
    ("highfreq", "cpu"), ("mem", "cpu"),
)


def load_metric(raw: Path, key: str) -> list[dict]:
    series = []
    for f in sorted(raw.glob(f"{key}.*.json")):
        series.extend(json.loads(f.read_text()))
    return series


def intervals_from_samples(times: list[float], step: float) -> list[tuple]:
    """Merge sample timestamps into [t0, t1] intervals, breaking on gaps."""
    if not times:
        return []
    times = sorted(times)
    out = []
    t0 = prev = times[0]
    for t in times[1:]:
        if t - prev > 2.5 * step:
            out.append((t0, prev + step))
            t0 = t
        prev = t
    out.append((t0, prev + step))
    return out


def pool_of_node(node: str) -> str:
    n = node.lower()
    for pat, pool in POOL_FROM_NODE:
        if pat in n:
            return pool
    return "unknown"


def compress_profile(times: list[float], utils: list[float],
                     step: float) -> list[list[float]]:
    """5-min utilization samples -> merged (duration_s, util) segments."""
    if not times:
        return []
    segs = []
    cur_u, cur_d = utils[0], step
    for u in utils[1:]:
        if abs(u - cur_u) < 0.05:
            cur_d += step
            cur_u = (cur_u * (cur_d - step) + u * step) / cur_d
        else:
            segs.append([cur_d, round(cur_u, 3)])
            cur_u, cur_d = u, step
    segs.append([cur_d, round(cur_u, 3)])
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/monit")
    ap.add_argument("--out", default="data/derived")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((raw / "meta.json").read_text())

    user_ns = {s["metric"]["namespace"] for s in load_metric(raw, "user_ns")}

    # a pod NAME can be reused (deleted and recreated); kube_pod_created
    # changes value at recreation, so distinct created values delimit pod
    # INSTANCES, and every other metric is sliced into instance windows
    created_events: dict[tuple, list[tuple]] = defaultdict(list)  # (t, value)
    for s in load_metric(raw, "created"):
        m = s["metric"]
        created_events[(m["namespace"], m["pod"])].extend(
            (v[0], float(v[1])) for v in s["values"])

    windows: dict[tuple, list[tuple]] = {}  # key -> [(created, w0, w1), ...]
    for key, evs in created_events.items():
        evs.sort()
        vals = []
        for _, c in evs:
            if not vals or abs(c - vals[-1]) > 1.0:
                vals.append(c)
        vals.sort()
        windows[key] = [
            (c, c, vals[i + 1] if i + 1 < len(vals) else float("inf"))
            for i, c in enumerate(vals)]

    def instance_of(key, t):
        for i, (_, w0, w1) in enumerate(windows.get(key, ())):
            if w0 - 600 <= t < w1:
                return i
        return None

    pods: dict[tuple, dict] = defaultdict(lambda: {
        "phases": defaultdict(list), "requests": defaultdict(float),
        "created": None, "start": None, "node": None})

    def inst_key(m, t):
        base = (m["namespace"], m["pod"])
        i = instance_of(base, t)
        return None if i is None else (m["namespace"], m["pod"], i)

    for base, wins in windows.items():
        for i, (c, _, _) in enumerate(wins):
            pods[(base[0], base[1], i)]["created"] = c
    for s in load_metric(raw, "phase"):
        m = s["metric"]
        for v in s["values"]:
            k = inst_key(m, v[0])
            if k:
                pods[k]["phases"][m["phase"]].append(v[0])
    for s in load_metric(raw, "start_time"):
        m = s["metric"]
        for v in s["values"]:
            k = inst_key(m, v[0])
            if k and pods[k]["start"] is None:
                pods[k]["start"] = float(v[1])
    for s in load_metric(raw, "pod_info"):
        m = s["metric"]
        if not m.get("node"):
            continue
        for v in s["values"]:
            k = inst_key(m, v[0])
            if k:
                pods[k]["node"] = m["node"]
                break
    for s in load_metric(raw, "requests"):
        m = s["metric"]
        for v in s["values"]:
            k = inst_key(m, v[0])
            if k:
                pods[k]["requests"][m["resource"]] = max(
                    pods[k]["requests"][m["resource"]], float(v[1]))

    util_by_pod: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for s in load_metric(raw, "gpu_util"):
        m = s["metric"]
        base = (m.get("namespace", ""), m.get("pod", ""))
        for t, v in s["values"]:
            i = instance_of(base, t)
            if i is not None:
                util_by_pod[(base[0], base[1], i)][t].append(float(v))

    n_infra = n_nonuser = 0
    requests_out = []
    for (ns, pod, inst), d in sorted(pods.items()):
        if ns not in user_ns:
            n_nonuser += 1
            continue
        if INFRA_RE.match(pod):
            n_infra += 1
            continue
        created = d["created"]
        if created is None:
            continue
        run_iv = intervals_from_samples(d["phases"].get("Running", []), 300)
        pend_iv = intervals_from_samples(d["phases"].get("Pending", []), 300)
        started = d["start"] or (run_iv[0][0] if run_iv else None)
        ended = run_iv[-1][1] if run_iv else None
        if started and started < created:
            started = created
        wait = (started - created) if started else (
            (pend_iv[-1][1] - created) if pend_iv else 0.0)
        outcome = ("started" if started else
                   "cancelled" if pend_iv else "unknown")
        gpus, gpu_kind = 0, None
        for res, (kind_, scale) in GPU_RESOURCES.items():
            if d["requests"].get(res):
                gpus, gpu_kind = int(d["requests"][res] * scale), kind_
        node = d["node"] or ""
        if gpu_kind in ("mig3g", "mig1g", "amd"):
            pool = gpu_kind  # resource identifies the pool, wherever placed
        else:
            pool = pool_of_node(node) if node else (gpu_kind or "unplaced")
        if pool == "gpu" or (pool == "unknown" and gpu_kind == "gpu"):
            pool = "gpu_unknown"
        samples = util_by_pod.get((ns, pod, inst), {})
        st = sorted(samples)
        profile = compress_profile(
            st, [np.mean(samples[t]) / 100.0 for t in st], 300.0)
        hold = (ended - started) if (started and ended) else 0.0
        requests_out.append({
            "request_id": f"{ns}/{pod}#{inst}",
            "group_id": f"{ns}/{pod}#{inst}",
            "user": ns, "kind": gpu_kind or "cpu", "wp": "",
            "submit_time": created,
            "pool": pool, "gpus": gpus,
            "vcpus": d["requests"].get("cpu", 0.0),
            "mem_gb": d["requests"].get("memory", 0.0) / 2**30,
            "duration_s": hold,
            "profile": profile,
            "patience_s": 0.0,
            "resubmit_of": None,
            "observed": {
                "outcome": outcome, "wait_s": wait, "node": node,
                "running_intervals": run_iv, "pending_intervals": pend_iv,
            },
        })

    cordons = []
    for s in load_metric(raw, "cordon"):
        node = s["metric"]["node"]
        for t0, t1 in intervals_from_samples([v[0] for v in s["values"]], 300):
            cordons.append({"time": t0, "node_id": node, "cordoned": True})
            cordons.append({"time": t1, "node_id": node, "cordoned": False})

    with open(out / "requests.jsonl", "w") as f:
        for r in sorted(requests_out, key=lambda r: r["submit_time"]):
            f.write(json.dumps(r) + "\n")
    with open(out / "cordons.jsonl", "w") as f:
        for c in sorted(cordons, key=lambda c: c["time"]):
            f.write(json.dumps(c) + "\n")

    # ------------------------------------------------------------- report
    gpu_reqs = [r for r in requests_out if r["gpus"] > 0]
    started = [r for r in gpu_reqs if r["observed"]["outcome"] == "started"]
    waits = np.array([r["observed"]["wait_s"] for r in started]) / 60.0
    holds = np.array([r["duration_s"] for r in started if r["duration_s"] > 0]) / H
    idle_frac = []
    held_idle_h = 0.0
    for r in started:
        prof = r["profile"]
        tot = sum(d for d, _ in prof)
        if tot > 0 and r["gpus"] > 0:
            idle = sum(d for d, u in prof if u < 0.05)
            idle_frac.append(idle / tot)
            held_idle_h += idle * r["gpus"] / H
    users = sorted({r["user"] for r in gpu_reqs})
    lines = [
        "# NGT allocation history extracted from MONIT Prometheus", "",
        f"Window: {meta['days']} days ending {meta['end']} (unix). "
        f"Pods excluded: {n_infra} platform furniture, {n_nonuser} non-user "
        "namespaces.", "",
        f"- GPU/MIG requests by users: {len(gpu_reqs)} from {len(users)} "
        "distinct users",
        f"- started: {len(started)}, cancelled while Pending: "
        f"{sum(1 for r in gpu_reqs if r['observed']['outcome'] == 'cancelled')}",
        f"- wait among started: median {np.median(waits):.1f} min, "
        f"p95 {np.percentile(waits, 95):.1f} min, max {waits.max():.0f} min"
        if len(waits) else "- no started requests found",
        f"- hold duration: median {np.median(holds):.1f} h, "
        f"p95 {np.percentile(holds, 95):.1f} h" if len(holds) else "",
        f"- allocations with GPU-utilization data: {len(idle_frac)}; "
        f"median idle fraction (util<5%): "
        f"{100 * np.median(idle_frac):.0f}%" if idle_frac else "",
        f"- idle-held GPU-hours in window (from DCGM): {held_idle_h:.0f}",
        f"- cordon windows: {len(cordons) // 2}",
        "",
        "Per-pool started requests:",
    ]
    by_pool = defaultdict(list)
    for r in started:
        by_pool[r["pool"]].append(r["observed"]["wait_s"] / 60.0)
    for pool, w in sorted(by_pool.items()):
        lines.append(f"- {pool}: {len(w)} requests, median wait "
                     f"{np.median(w):.1f} min, p95 {np.percentile(w, 95):.1f} min")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/requests.jsonl ({len(requests_out)} pods), "
          f"cordons.jsonl ({len(cordons) // 2} windows), report.md")


if __name__ == "__main__":
    main()
