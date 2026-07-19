"""Extract the NGT allocation history from CERN MONIT Prometheus (via the
authenticated monit-grafana datasource proxy) into raw per-day JSONL dumps.

    python tools/extract_monit_trace.py --cookies <jar> --out data/monit \
        --days 30 [--end <unix-ts>]

Read-only. Uses the caller's own Grafana session (SSO); no admin token.
Each (metric, day) chunk is cached as a file and skipped when present, so
the pull is resumable if the session expires midway.

Metrics pulled (LTS datasource, 5 or 10 min resolution):
  kube_pod_status_phase == 1      pod lifecycle (Pending/Running/...)
  kube_pod_created                submission timestamp
  kube_pod_start_time             scheduling timestamp (wait = start-created)
  kube_pod_info                   node placement
  kube_pod_container_resource_requests   GPUs, MIG slices, CPU, memory
  kube_node_spec_unschedulable == 1      cordon windows
  DCGM_FI_DEV_GPU_UTIL            per-GPU utilization with namespace/pod
  kube_namespace_labels (kubeflow-profile)  the user-namespace list
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

GRAFANA = "https://monit-grafana.cern.ch"
DS_UID = "afn82ui46w9vke"  # org 105 "prometheus-lts"

METRICS = {
    "phase": ('kube_pod_status_phase{phase=~"Pending|Running|Succeeded|Failed"} == 1', 300),
    "created": ("kube_pod_created", 600),
    "start_time": ("kube_pod_start_time", 600),
    "pod_info": ("kube_pod_info", 600),
    "requests": ("kube_pod_container_resource_requests", 600),
    "cordon": ("kube_node_spec_unschedulable == 1", 300),
    "gpu_util": ("DCGM_FI_DEV_GPU_UTIL", 300),
    "cpu_rate": ('sum by (namespace, pod) '
                 '(rate(container_cpu_usage_seconds_total'
                 '{container!="",container!="POD"}[10m]))', 300),
    "user_ns": ('kube_namespace_labels{label_app_kubernetes_io_part_of="kubeflow-profile"}', 3600),
    # full dump so the Grafana/Prometheus backend is no longer needed
    "gpu_fb_used": ("DCGM_FI_DEV_FB_USED", 300),
    "gpu_engine": ("DCGM_FI_PROF_GR_ENGINE_ACTIVE", 300),
    "gpu_power": ("DCGM_FI_DEV_POWER_USAGE", 300),
    "mem_usage": ('sum by (namespace, pod) '
                  '(container_memory_working_set_bytes'
                  '{container!="",container!="POD"})', 300),
    "net_rx": ('sum by (namespace, pod) '
               '(rate(container_network_receive_bytes_total[10m]))', 600),
    "net_tx": ('sum by (namespace, pod) '
               '(rate(container_network_transmit_bytes_total[10m]))', 600),
    "limits": ("kube_pod_container_resource_limits", 600),
    "pod_ready": ('kube_pod_status_ready{condition="true"} == 1', 600),
    "completion": ("kube_pod_completion_time", 600),
    "node_info": ("kube_node_info", 3600),
    "node_alloc": ("kube_node_status_allocatable", 3600),
    "node_capacity": ("kube_node_status_capacity", 3600),
    "node_condition": ('kube_node_status_condition{status="true"} == 1', 600),
    "machine_cores": ("machine_cpu_cores", 3600),
    "ns_labels": ("kube_namespace_labels", 3600),
}


def relogin(jar: str) -> bool:
    """Re-establish the Grafana session using the Keycloak SSO cookies in
    the jar (present when the jar came from tools/monit_login.sh). Returns
    True when /api/user answers with an identity afterwards."""
    subprocess.run(
        ["curl", "-sSL", "-c", jar, "-b", jar, "-o", "/dev/null",
         f"{GRAFANA}/login/generic_oauth"], check=False, timeout=60)
    out = subprocess.run(
        ["curl", "-sS", "-c", jar, "-b", jar, f"{GRAFANA}/api/user"],
        capture_output=True, text=True, timeout=60, check=False).stdout
    return '"login"' in out


def query_range(jar: str, expr: str, start: int, end: int, step: int) -> dict:
    """curl keeps the jar current across Grafana's session-token rotation
    (Set-Cookie on responses); urllib with a static header does not."""
    url = f"{GRAFANA}/api/datasources/proxy/uid/{DS_UID}/api/v1/query_range"
    cmd = ["curl", "-sS", "-c", jar, "-b", jar, "--max-time", "180",
           "--data-urlencode", f"query={expr}",
           "--data-urlencode", f"start={start}",
           "--data-urlencode", f"end={end}",
           "--data-urlencode", f"step={step}", url]
    for attempt in range(4):
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=200, check=False).stdout
        try:
            res = json.loads(out)
        except json.JSONDecodeError:
            res = {"status": "error", "raw": out[:200]}
        if res.get("status") == "success":
            return res
        if "Unauthorized" in out and relogin(jar):
            continue
        if attempt == 3:
            raise RuntimeError(f"query failed: {out[:300]}")
        time.sleep(5 * (attempt + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cookies", required=True)
    ap.add_argument("--out", default="data/monit")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--end", type=int, default=None,
                    help="unix ts of range end (default: now, hour-aligned)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if args.end:
        end = args.end
    elif meta_path.exists():  # resume: keep day boundaries of cached chunks
        end = json.loads(meta_path.read_text())["end"]
    else:
        end = int(time.time()) // 3600 * 3600
    meta_path.write_text(json.dumps(
        {"end": end, "days": args.days, "datasource": DS_UID,
         "metrics": {k: v[0] for k, v in METRICS.items()}}, indent=2))

    for day in range(args.days):
        d1 = end - day * 86400
        d0 = d1 - 86400
        for key, (expr, step) in METRICS.items():
            path = out / f"{key}.{d0}.json"
            if path.exists() and path.stat().st_size > 100:
                continue
            t0 = time.time()
            res = query_range(args.cookies, expr, d0, d1, step)
            if res.get("status") != "success":
                print(f"day -{day} {key}: FAILED {str(res)[:200]}")
                continue
            n = len(res["data"]["result"])
            path.write_text(json.dumps(res["data"]["result"]))
            print(f"day -{day} {key}: {n} series, "
                  f"{path.stat().st_size // 1024} KB, {time.time() - t0:.1f}s",
                  flush=True)
    print("extraction complete")


if __name__ == "__main__":
    main()
