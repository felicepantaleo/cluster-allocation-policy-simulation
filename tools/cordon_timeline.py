"""Per-node cordon timeline: one row per node, bars over cordoned periods.

    python tools/cordon_timeline.py --raw data/monit \
        --out results/gallery/ngt_allocation_problem

Reads the raw cordon samples from the full dump and reconstructs each
node's cordon intervals (samples merged when less than 1 h apart, since the
long-term store is downsampled). Rows are grouped and coloured by GPU
model; CPU-only nodes are collapsed into a summary strip at the bottom.
Makes the chronic single-node drains (e.g. the 14-day SXM cordon) visible
next to the cluster-wide 6-8 July maintenance.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})

MERGE_GAP = 3600.0
SAMPLE = 300.0
BLUE, ORANGE, PURPLE, VIOLET, GRAY, RED = (
    "#5790fc", "#f89c20", "#964a8b", "#7a21dd", "#9c9ca1", "#e42536")


def model(n: str) -> str:
    n = n.lower()
    for pat, lbl in (("h100-nvl", "H100 NVL"), ("h100-sxm", "H100 SXM"),
                     ("l40s", "L40S"), ("mi300x", "MI300X"), ("w7900", "W7900"),
                     ("steam-t4", "Tesla T4"), ("rtx-6000", "RTX 6000"),
                     ("demo", "demo GPU")):
        if pat in n:
            return lbl
    return "CPU-only"


MODEL_COLOR = {"H100 NVL": BLUE, "H100 SXM": ORANGE, "L40S": PURPLE,
               "MI300X": VIOLET, "W7900": VIOLET, "RTX 6000": GRAY,
               "demo GPU": GRAY, "Tesla T4": GRAY, "CPU-only": GRAY}
ORDER = {"H100 NVL": 0, "H100 SXM": 1, "L40S": 2, "MI300X": 3, "W7900": 4,
         "RTX 6000": 5, "demo GPU": 6, "Tesla T4": 7, "CPU-only": 9}


def intervals(ts: list[float]) -> list[tuple]:
    ts = sorted(set(ts))
    if not ts:
        return []
    runs, run = [], [ts[0], ts[0]]
    for t in ts[1:]:
        if t - run[1] > MERGE_GAP:
            runs.append((run[0], run[1] + SAMPLE))
            run = [t, t]
        else:
            run[1] = t
    runs.append((run[0], run[1] + SAMPLE))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/monit")
    ap.add_argument("--out", default="results/gallery/ngt_allocation_problem")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)

    samples = defaultdict(list)
    for f in glob.glob(str(raw / "cordon.*.json")):
        for s in json.loads(Path(f).read_text()):
            n = s["metric"].get("node")
            if n:
                samples[n] += [t for t, _ in s["values"]]
    # observation window from a dense metric
    w0 = w1 = None
    for f in glob.glob(str(raw / "gpu_util.*.json")):
        for s in json.loads(Path(f).read_text()):
            for t, _ in s["values"]:
                w0 = t if w0 is None else min(w0, t)
                w1 = t if w1 is None else max(w1, t)
    span = w1 - w0

    cord = {n: intervals(ts) for n, ts in samples.items()}
    frac = {n: sum(b - a for a, b in iv) / span for n, iv in cord.items()}
    # only nodes ever cordoned, grouped by model, GPU pools first, and within
    # a group the most-cordoned on top
    nodes = [n for n in cord if cord[n]]
    nodes.sort(key=lambda n: (ORDER.get(model(n), 8), -frac[n], n))

    def dt(t):
        return datetime.fromtimestamp(t, timezone.utc)

    fig, ax = plt.subplots(figsize=(13, 0.30 * len(nodes) + 2.0))
    seen = set()
    for y, n in enumerate(nodes):
        m = model(n)
        c = MODEL_COLOR[m]
        for a, b in cord[n]:
            ax.barh(y, (b - a) / 86400.0, left=mdates.date2num(dt(a)),
                    height=0.7, color=c, alpha=0.9,
                    label=m if m not in seen else None)
            seen.add(m)
    short = [n.replace("ngt-003-", "").replace("ngt-", "") for n in nodes]
    ax.set_yticks(range(len(nodes)),
                  [f"{s}  ({100*frac[n]:.0f}%)" for s, n in zip(short, nodes)],
                  fontsize=7.5)
    ax.set_ylim(-0.7, len(nodes) - 0.3)
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(mdates.date2num(dt(w0)), mdates.date2num(dt(w1)))
    ax.set_xlabel("date (UTC)")
    n_chronic = sum(1 for n in nodes if frac[n] > 0.10)
    ax.set_title(f"Node cordon timeline: {n_chronic} nodes down more than "
                 f"10% of the month (percent in each label = fraction of the "
                 "window cordoned); the 6-8 Jul cluster of short bars is the "
                 "coordinated maintenance",
                 loc="left", fontsize=11.5, pad=24)
    ax.text(0.995, 0.005, "NGT cluster, 15 Jun to 18 Jul 2026. "
            "Felice Pantaleo (CERN)", transform=fig.transFigure, ha="right",
            va="bottom", fontsize=9, color="#555555")
    named = ["H100 NVL", "H100 SXM", "L40S", "MI300X", "W7900"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR[m], label=m)
               for m in named if m in seen]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=GRAY,
                                 label="CPU-only / other"))
    ax.legend(handles=handles, fontsize=9, ncol=len(handles), loc="lower left",
              bbox_to_anchor=(0, 1.005), frameon=False)
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.6)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "22_cordon_timeline.png", dpi=150, bbox_inches="tight")
    (out / "svg").mkdir(exist_ok=True)
    fig.savefig(out / "svg" / "22_cordon_timeline.svg", bbox_inches="tight")
    print(f"wrote 22_cordon_timeline.png ({len(nodes)} nodes, "
          f"window {span/86400:.1f} d)")


if __name__ == "__main__":
    main()
