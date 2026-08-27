"""Node inventory from the MONIT dump: all cluster nodes with GPU model,
capacity and cordon statistics.

    python tools/node_inventory.py --raw data/monit --out data/derived

Unions every node seen in node_info / node_capacity / cordon metrics,
labels each by GPU model from its name, sizes it from
kube_node_status_capacity, and reconstructs cordon intervals per node
(samples merged when less than 1 h apart, since the long-term store is
downsampled). Writes all_nodes.csv and a Markdown table all_nodes.md.
`cordon_pct` is the fraction of the window a node spent cordoned,
integrated over time; `cordon_episodes` counts distinct cordon periods.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

MERGE_GAP = 3600.0
SAMPLE = 300.0
ORDER = {"H100 NVL": 0, "H100 SXM": 1, "L40S": 2, "MI300X": 3, "W7900": 4,
         "RTX 6000": 5, "demo GPU": 6, "Tesla T4": 7, "CPU-only": 9}


def model(n: str) -> str:
    n = n.lower()
    for pat, lbl in (("h100-nvl", "H100 NVL"), ("h100-sxm", "H100 SXM"),
                     ("l40s", "L40S"), ("mi300x", "MI300X"), ("w7900", "W7900"),
                     ("steam-t4", "Tesla T4"), ("rtx-6000", "RTX 6000"),
                     ("demo", "demo GPU")):
        if pat in n:
            return lbl
    return "CPU-only"


def intervals(ts: list[float]) -> list[tuple]:
    ts = sorted(set(ts))
    if not ts:
        return []
    runs, a, prev = [], ts[0], ts[0]
    for t in ts[1:]:
        if t - prev > MERGE_GAP:
            runs.append((a, prev + SAMPLE))
            a = t
        prev = t
    runs.append((a, prev + SAMPLE))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/monit")
    ap.add_argument("--out", default="data/derived")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cap: dict = defaultdict(dict)
    samples: dict = defaultdict(list)
    nodes: set = set()
    for f in glob.glob(str(raw / "node_capacity.*.json")):
        for s in json.loads(Path(f).read_text()):
            m = s["metric"]
            n, r = m.get("node"), m.get("resource")
            if n:
                nodes.add(n)
                if r and s["values"]:
                    cap[n][r] = float(s["values"][-1][1])
    for f in glob.glob(str(raw / "node_info.*.json")):
        for s in json.loads(Path(f).read_text()):
            if s["metric"].get("node"):
                nodes.add(s["metric"]["node"])
    for f in glob.glob(str(raw / "cordon.*.json")):
        for s in json.loads(Path(f).read_text()):
            n = s["metric"].get("node")
            if n:
                nodes.add(n)
                samples[n] += [t for t, _ in s["values"]]

    # observation window from a dense metric
    w0 = w1 = None
    for f in glob.glob(str(raw / "gpu_util.*.json")):
        for s in json.loads(Path(f).read_text()):
            for t, _ in s["values"]:
                w0 = t if w0 is None else min(w0, t)
                w1 = t if w1 is None else max(w1, t)
    span = (w1 - w0) if w0 is not None else 1.0

    rows = []
    for n in sorted(nodes):
        iv = intervals(samples.get(n, []))
        c = cap.get(n, {})
        gpus = int(c.get("nvidia_com_gpu", 0) or c.get("amd_com_gpu", 0)
                   or c.get("nvidia_com_mig_1g_12gb", 0)
                   + c.get("nvidia_com_mig_3g_47gb", 0))
        cordoned = sum(b - a for a, b in iv)
        rows.append([n, model(n), gpus, int(c.get("cpu", 0)),
                     round(c.get("memory", 0) / 2**30), len(iv),
                     round(100 * cordoned / span, 1)])
    rows.sort(key=lambda r: (ORDER.get(r[1], 8), r[0]))

    with open(out / "all_nodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "gpu_model", "gpus", "cpu_cores", "mem_gb",
                    "cordon_episodes", "cordon_pct"])
        w.writerows(rows)

    cnt: dict = defaultdict(int)
    gp: dict = defaultdict(int)
    for r in rows:
        cnt[r[1]] += 1
        gp[r[1]] += r[2]
    L = ["# NGT cluster node inventory", "",
         f"{len(rows)} nodes, window {span/86400:.0f} days. `cordon_pct` is the "
         "fraction of the window each node spent cordoned; `cordon_episodes` "
         "counts distinct cordon periods (samples merged within 1 h).", "",
         "## Summary by type", "", "| Type | Nodes | GPU / slice units |",
         "|---|---:|---:|"]
    for k in sorted(cnt, key=lambda k: ORDER.get(k, 8)):
        L.append(f"| {k} | {cnt[k]} | {gp[k]} |")
    L += ["", "## All nodes", "",
          "| Node | GPU model | GPU/slice | CPU cores | Mem (GB) | "
          "Cordon episodes | Cordon % |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} | "
                 f"{r[5]} | {r[6]} |")
    (out / "all_nodes.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out}/all_nodes.csv and all_nodes.md ({len(rows)} nodes)")


if __name__ == "__main__":
    main()
