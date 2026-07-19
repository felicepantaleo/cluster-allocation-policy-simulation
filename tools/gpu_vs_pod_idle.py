"""Split GPU-idle held time into CPU-active vs fully pod-idle.

    python tools/gpu_vs_pod_idle.py --raw data/monit --derived data/derived \
        --out results/gallery/ngt_allocation_problem [--cpu-thresh 1.0]

A GPU allocation whose GPU sits below 5% utilization may still be doing
real CPU work in the same pod (preprocessing, compilation, CPU-side
production on the slot's spare cores). For every DCGM-covered full-GPU
allocation, each 5-minute bin of its running time is classified as:

  gpu_active   GPU utilization >= 5%
  cpu_active   GPU idle, pod CPU rate >= threshold (cores)
  pod_idle     GPU idle and pod CPU below threshold

and GPU-hours are accumulated per class (weighted by allocated GPUs).
STEAM participants excluded; sensitivity to the CPU threshold reported at
0.5, 1 and 2 cores. Renders plot 20 for the gallery.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

H = 3600.0
POOLS = ("h100nvl", "h100sxm", "l40s")


def load_series(raw: Path, key: str) -> dict:
    """(namespace, pod) -> {sample_time: value} (mean over duplicates)."""
    out: dict = defaultdict(dict)
    for f in sorted(raw.glob(f"{key}.*.json")):
        for s in json.loads(f.read_text()):
            m = s["metric"]
            k = (m.get("namespace", ""), m.get("pod", ""))
            for t, v in s["values"]:
                cur = out[k].get(t)
                out[k][t] = float(v) if cur is None else (cur + float(v)) / 2
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/monit")
    ap.add_argument("--derived", default="data/derived")
    ap.add_argument("--out", default="results/gallery/ngt_allocation_problem")
    ap.add_argument("--cpu-thresh", type=float, default=1.0)
    args = ap.parse_args()
    raw, der, out = Path(args.raw), Path(args.derived), Path(args.out)

    wp_map = json.loads((der / "user_wp.json").read_text())
    reqs = [json.loads(l) for l in open(der / "requests.jsonl")]
    reqs = [r for r in reqs
            if wp_map.get(r["user"], {}).get("wp") != "STEAM"
            and r["pool"] in POOLS and r["gpus"] > 0
            and r["observed"]["outcome"] == "started"]

    # per-GPU utilization averaged over the pod's GPUs at each sample
    gpu_u: dict = defaultdict(lambda: defaultdict(list))
    for f in sorted(raw.glob("gpu_util.*.json")):
        for s in json.loads(f.read_text()):
            m = s["metric"]
            k = (m.get("namespace", ""), m.get("pod", ""))
            for t, v in s["values"]:
                gpu_u[k][t].append(float(v))
    cpu = load_series(raw, "cpu_rate")

    def basename(request_id):  # ns/pod#inst -> (ns, pod)
        nspod = request_id.split("#")[0]
        ns, pod = nspod.split("/", 1)
        return ns, pod

    thresholds = (0.5, 1.0, 2.0)
    split = {th: defaultdict(lambda: {"gpu_active": 0.0, "cpu_active": 0.0,
                                      "pod_idle": 0.0}) for th in thresholds}
    per_user_idle = defaultdict(lambda: {"cpu_active": 0.0, "pod_idle": 0.0})
    covered = 0
    for r in reqs:
        k = basename(r["request_id"])
        gsam = gpu_u.get(k)
        if not gsam:
            continue
        covered += 1
        csam = cpu.get(k, {})
        for a, b in r["observed"]["running_intervals"]:
            for t in gsam:
                if not (a <= t < b):
                    continue
                g = np.mean(gsam[t])
                gh = 300.0 * r["gpus"] / H
                # nearest cpu sample within 5 min
                c = csam.get(t)
                if c is None:
                    c = csam.get(t - 300, csam.get(t + 300, 0.0))
                for th in thresholds:
                    d = split[th][r["pool"]]
                    if g >= 5.0:
                        d["gpu_active"] += gh
                    elif c >= th:
                        d["cpu_active"] += gh
                    else:
                        d["pod_idle"] += gh
                if g < 5.0:
                    key = "cpu_active" if c >= args.cpu_thresh else "pod_idle"
                    per_user_idle[r["user"]][key] += gh

    th = args.cpu_thresh
    tot = {k: sum(split[th][p][k] for p in POOLS)
           for k in ("gpu_active", "cpu_active", "pod_idle")}
    tot_all = sum(tot.values())
    print(f"DCGM+CPU covered allocations: {covered}")
    for t_ in thresholds:
        s = {k: sum(split[t_][p][k] for p in POOLS)
             for k in ("gpu_active", "cpu_active", "pod_idle")}
        idle = s["cpu_active"] + s["pod_idle"]
        print(f"cpu>={t_:.1f} cores: gpu-active {s['gpu_active']:.0f} GPU-h, "
              f"gpu-idle+cpu-active {s['cpu_active']:.0f} "
              f"({100*s['cpu_active']/idle:.0f}% of idle), "
              f"pod-idle {s['pod_idle']:.0f} ({100*s['pod_idle']/idle:.0f}%)")
    top = sorted(per_user_idle.items(),
                 key=lambda kv: -kv[1]["pod_idle"])[:10]
    print("top pod-idle users (GPU-h):",
          [(u, round(v["pod_idle"])) for u, v in top])

    # ---- plot 20
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15,
                         "xaxis.labellocation": "left"})
    BLUE, ORANGE, RED = "#5790fc", "#f89c20", "#e42536"
    fig, ax = plt.subplots(figsize=(10, 6.5))
    xs = np.arange(len(POOLS))
    ga = np.array([split[th][p]["gpu_active"] for p in POOLS])
    ca = np.array([split[th][p]["cpu_active"] for p in POOLS])
    pi = np.array([split[th][p]["pod_idle"] for p in POOLS])
    ax.bar(xs, ga, 0.6, color=BLUE, label="GPU active")
    ax.bar(xs, ca, 0.6, bottom=ga, color=ORANGE,
           label=f"GPU idle, CPU active (>= {th:.0f} cores)")
    ax.bar(xs, pi, 0.6, bottom=ga + ca, color=RED, alpha=0.85,
           label="pod idle (GPU and CPU)")
    for x in xs:
        s_ = ga[x] + ca[x] + pi[x]
        ax.annotate(f"{100*pi[x]/s_:.0f}% pod-idle", xy=(x, s_),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=11)
    ax.set_xticks(xs, POOLS)
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.set_ylabel("GPU-hours in 30 days (DCGM-covered)")
    idle_tot = tot["cpu_active"] + tot["pod_idle"]
    ax.set_title(f"Of the GPU-idle held hours, "
                 f"{100*tot['cpu_active']/idle_tot:.0f}% run real CPU work; "
                 f"{100*tot['pod_idle']/idle_tot:.0f}% are fully idle pods",
                 loc="left")
    ax.text(0.995, 0.015, "NGT cluster, 18 Jun to 18 Jul 2026. "
            "Felice Pantaleo (CERN)", transform=fig.transFigure, ha="right",
            va="bottom", fontsize=10, color="#555555")
    ax.legend(fontsize=10)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "20_gpu_vs_pod_idle.png", dpi=150, bbox_inches="tight")
    (out / "svg").mkdir(exist_ok=True)
    fig.savefig(out / "svg" / "20_gpu_vs_pod_idle.svg", bbox_inches="tight")
    print("wrote 20_gpu_vs_pod_idle.png")


if __name__ == "__main__":
    main()
