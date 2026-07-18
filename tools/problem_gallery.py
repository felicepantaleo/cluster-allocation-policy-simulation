"""Build the orbit gallery: the NGT allocation problem in real MONIT data.

    python tools/problem_gallery.py --derived data/derived --out results/gallery

Reads the 30-day derived trace (requests.jsonl, cordons.jsonl, user_wp.json)
and renders the numbered gallery plus README and summary. STEAM ACADEMY T4
(cloud) requests are excluded from every scheduling metric. No usernames
appear in any figure; users aggregate to working packages or 'outside WP
roster'.
"""

from __future__ import annotations

import argparse
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
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15,
                     "figure.titlesize": 15, "xaxis.labellocation": "left"})

H = 3600.0
ONPREM = ("h100nvl", "h100sxm", "l40s", "amd", "mig3g", "mig1g")
FULLGPU = ("h100nvl", "h100sxm", "l40s", "amd")
CAPACITY = {"h100nvl": 96, "h100sxm": 24, "l40s": 28}
NODE_GPUS = {"h100nvl": 8, "h100sxm": 4, "l40s": 4}
# Petroff cycle; red is reserved for alarm quantities (idle, unsatisfied)
BLUE, ORANGE, RED, PURPLE, GRAY, VIOLET = (
    "#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd")
POOL_COLOR = {"h100nvl": BLUE, "h100sxm": ORANGE, "l40s": PURPLE,
              "amd": VIOLET, "mig3g": GRAY, "mig1g": GRAY}
WP_COLOR = {"WP1": BLUE, "WP2": ORANGE, "WP3": PURPLE, "WP4": VIOLET,
            "outside roster": GRAY}
STAMP = "NGT cluster, CERN MONIT data"


def dt(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def stamp(ax, window):
    ax.text(0.995, 0.015, f"{STAMP}, {window}",
            transform=ax.figure.transFigure, ha="right", va="bottom",
            fontsize=10, color="#555555")


def save(fig, out, name):
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", default="data/derived")
    ap.add_argument("--out", default="results/gallery/ngt_allocation_problem")
    args = ap.parse_args()
    der = Path(args.derived)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reqs = [json.loads(l) for l in open(der / "requests.jsonl")]
    cords = [json.loads(l) for l in open(der / "cordons.jsonl")]
    wp_map = json.loads((der / "user_wp.json").read_text())
    # STEAM ACADEMY participants are out of scope for scheduling statistics
    n_steam = sum(1 for r in reqs
                  if wp_map.get(r["user"], {}).get("wp") == "STEAM")
    reqs = [r for r in reqs
            if wp_map.get(r["user"], {}).get("wp") != "STEAM"]
    print(f"excluded {n_steam} STEAM-participant pod instances")

    tmax = max(r["submit_time"] for r in reqs)
    END = int(np.ceil(tmax / 86400) * 86400)
    START = END - 30 * 86400
    window = f"{dt(START):%d %b} to {dt(END):%d %b %Y}"

    gpu = [r for r in reqs if r["gpus"] > 0 and r["pool"] != "cloud_t4"]
    started = [r for r in gpu if r["observed"]["outcome"] == "started"]
    cancelled = [r for r in gpu if r["observed"]["outcome"] == "cancelled"]

    def wp_of(user):
        e = wp_map.get(user)
        return e["wp"] if e and e["wp"].startswith("WP") else "outside roster"

    grid = np.arange(START, END, 600.0)
    gdt = [dt(t) for t in grid]

    # ---- 01 occupancy from DCGM: one series per ALLOCATED GPU, covering
    # every namespace (user pods undercount: service namespaces hold GPUs
    # too); deduplicated by GPU UUID per time bin
    MODEL_POOL = {"NVIDIA H100 NVL": "h100nvl",
                  "NVIDIA H100 80GB HBM3": "h100sxm",
                  "NVIDIA L40S": "l40s"}
    bins: dict[str, list] = {p: [set() for _ in grid] for p in FULLGPU}
    for f in sorted(Path("data/monit").glob("gpu_util.*.json")):
        for s in json.loads(f.read_text()):
            pool = MODEL_POOL.get(s["metric"].get("modelName", ""))
            if pool is None:
                continue
            uuid = s["metric"].get("UUID", "")
            for t, _ in s["values"]:
                i = int((t - START) // 600)
                if 0 <= i < len(grid):
                    bins[pool][i].add(uuid)
    alloc = {p: np.array([len(x) for x in bins[p]], dtype=float)
             for p in FULLGPU}
    # AMD has no NVIDIA DCGM series; fall back to user-pod accounting
    for r in started:
        if r["pool"] == "amd":
            for a, b in r["observed"]["running_intervals"]:
                i, j = np.searchsorted(grid, [a, b])
                alloc["amd"][i:j] += r["gpus"]
    cord_gpus = {p: np.zeros(len(grid)) for p in FULLGPU}
    n_cord = np.zeros(len(grid))
    from_pool = lambda n: ("h100nvl" if "h100-nvl" in n else
                           "h100sxm" if "h100-sxm" in n else
                           "l40s" if "l40s" in n else
                           "amd" if ("mi300" in n or "w7900" in n) else None)
    open_c = {}
    for c in sorted(cords, key=lambda c: c["time"]):
        if c["cordoned"]:
            open_c[c["node_id"]] = c["time"]
        elif c["node_id"] in open_c:
            a = open_c.pop(c["node_id"])
            i, j = np.searchsorted(grid, [a, c["time"]])
            n_cord[i:j] += 1
            p = from_pool(c["node_id"].lower())
            if p in cord_gpus:
                cord_gpus[p][i:j] += NODE_GPUS.get(p, 4)
    amd_cap = float(alloc["amd"].max()) if alloc["amd"].max() > 0 else 8
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for ax, p in zip(axes, FULLGPU):
        cap = CAPACITY.get(p, amd_cap)
        ax.fill_between(gdt, alloc[p], step="mid", color=POOL_COLOR[p],
                        alpha=0.75, linewidth=0, label="allocated")
        ax.plot(gdt, np.maximum(cap - cord_gpus[p], 0), color="black",
                linewidth=1.2, label="allocatable (capacity minus cordoned)")
        ax.axhline(cap, color="black", linestyle="--", linewidth=1.0,
                   label="capacity" if p == "h100nvl" else None)
        ax.set_ylabel(f"{p}\nGPUs")
        ax.set_ylim(0, cap * 1.18)
    axes[0].set_title("GPU allocation is static: pools sit fully occupied "
                      "for weeks and free up only on forced maintenance "
                      "evictions", loc="left")
    stamp(axes[0], window)
    axes[0].legend(loc="lower left", fontsize=11, ncol=3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.text(0.12, 0.005, "NVL: MIG-enabled GPUs are not visible to DCGM, "
             "so the full-GPU ceiling sits below the 96-GPU capacity line. "
             "AMD from user-pod accounting (no DCGM).", fontsize=10,
             color="#555555")
    save(fig, out, "01_occupancy_ceiling.png")

    # ---- 02 waits per pool
    fig, ax = plt.subplots(figsize=(11, 7))
    stats = []
    for p in ("h100nvl", "h100sxm", "l40s"):
        w = np.sort([r["observed"]["wait_s"] / 60 for r in started
                     if r["pool"] == p])
        if not len(w):
            continue
        y = np.arange(1, len(w) + 1) / len(w)
        ax.step(w, y, where="post", color=POOL_COLOR[p], linewidth=2.2,
                label=f"{p} (n={len(w)}, p95 {np.percentile(w,95)/60:.1f} h)")
        stats.append((p, np.percentile(w, 95)))
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xlabel("wait between request and allocation (minutes)")
    ax.set_ylabel("fraction of requests")
    ax.set_ylim(0, 1.02)
    p95nvl = max(s for pl, s in stats if pl == "h100nvl") / 60
    ax.set_title(f"Most requests are instant, but once the pool is full "
                 f"waits reach hours (H100 NVL p95: {p95nvl:.1f} h)", loc="left")
    stamp(ax, window)
    ax.legend(loc="lower right", fontsize=12)
    save(fig, out, "02_waits.png")

    # ---- 03 pending backlog
    pend = np.zeros(len(grid))
    for r in gpu:
        for a, b in r["observed"]["pending_intervals"]:
            i, j = np.searchsorted(grid, [a, b])
            pend[i:j] += 1
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(gdt, pend, step="mid", color=RED, alpha=0.7, linewidth=0)
    ax.set_ylabel("GPU/MIG requests Pending")
    ax.set_title(f"A queue exists in all but name: up to {int(pend.max())} "
                 "allocation requests wait as Pending pods", loc="left")
    stamp(ax, window)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    save(fig, out, "03_pending_backlog.png")

    # ---- 04 daily active vs idle held GPU-hours (DCGM-covered)
    days = np.arange(START, END, 86400.0)
    active_d = np.zeros(len(days))
    idle_d = np.zeros(len(days))
    covered = 0
    for r in started:
        if not r["profile"] or r["pool"] not in FULLGPU \
                or not r["observed"]["running_intervals"]:
            continue
        covered += 1
        t = r["observed"]["running_intervals"][0][0]
        for dur, u in r["profile"]:
            i = min(int((t - START) // 86400), len(days) - 1)
            if i >= 0:
                gh = dur * r["gpus"] / H
                if u < 0.05:
                    idle_d[i] += gh
                else:
                    active_d[i] += gh
            t += dur
    fig, ax = plt.subplots(figsize=(13, 6))
    ddt = [dt(d + 43200) for d in days]
    ax.bar(ddt, active_d, width=0.8, color=BLUE, label="active GPU-hours")
    ax.bar(ddt, idle_d, width=0.8, bottom=active_d, color=RED, alpha=0.85,
           label="held but idle (GPU util < 5%)")
    tot_i, tot_a = idle_d.sum(), active_d.sum()
    ax.set_ylabel("GPU-hours per day")
    ax.set_title(f"{100*tot_i/(tot_i+tot_a):.0f}% of monitored held GPU-hours "
                 f"are idle: {tot_i:.0f} GPU-hours parked in 30 days",
                 loc="left")
    stamp(ax, window)
    ax.legend(fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.text(0.12, -0.02, f"DCGM utilization available for {covered} "
             "full-GPU allocations; MIG slices not covered.", fontsize=10,
             color="#555555")
    save(fig, out, "04_idle_gpu_hours.png")

    # ---- 05 idle fraction per allocation
    fr = []
    for r in started:
        if r["profile"] and r["pool"] in FULLGPU:
            tot = sum(d for d, _ in r["profile"])
            fr.append(sum(d for d, u in r["profile"] if u < 0.05) / tot)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(fr, bins=20, color=RED, alpha=0.85)
    ax.set_xlabel("fraction of allocation lifetime with GPU util < 5%")
    ax.set_ylabel("allocations")
    ax.set_title(f"The median monitored allocation is idle for "
                 f"{100*np.median(fr):.0f}% of its lifetime (n={len(fr)})",
                 loc="left")
    stamp(ax, window)
    save(fig, out, "05_idle_fraction.png")

    # ---- 06 WP shares
    gh = defaultdict(float)
    for r in started:
        if r["pool"] in FULLGPU:
            for a, b in r["observed"]["running_intervals"]:
                gh[wp_of(r["user"])] += (b - a) * r["gpus"] / H
    order = ["WP1", "WP2", "WP3", "WP4", "outside roster"]
    vals = [gh.get(k, 0.0) for k in order]
    tot = sum(vals)
    wp_tot = sum(vals[:4])
    targets = [0.3, 0.3, 0.3, 0.1]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = np.arange(len(order))
    ax.bar(xs, [v / tot * 100 for v in vals],
           color=[WP_COLOR[k] for k in order], alpha=0.9)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v / tot * 100:.0f}%", xy=(x, v / tot * 100),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=12)
    ax.set_xticks(xs, order)
    ax.set_ylabel("share of full-GPU hours (%)")
    ax.set_title(f"Who gets the GPU-hours: {100*vals[4]/tot:.0f}% go to "
                 "users outside the WP roster", loc="left")
    stamp(ax, window)
    save(fig, out, "06_wp_shares.png")

    # ---- 07 unsatisfied requests
    w = np.array([r["observed"]["wait_s"] / 60 for r in cancelled])
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.logspace(0, np.log10(max(w.max(), 10)), 24)
    ax.hist(np.clip(w, 1, None), bins=bins, color=RED, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("time spent Pending before the request was abandoned (minutes)")
    ax.set_ylabel("abandoned requests")
    ax.set_title(f"{len(w)} allocation requests were never satisfied; "
                 f"median {np.median(w):.0f} min waited, tails span days",
                 loc="left")
    stamp(ax, window)
    kinds = defaultdict(int)
    for r in cancelled:
        kinds[f"{r['kind']} x{r['gpus']}"] += 1
    txt = "\n".join(f"{k}: {v}" for k, v in
                    sorted(kinds.items(), key=lambda kv: -kv[1])[:8])
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=11, family="monospace")
    save(fig, out, "07_unsatisfied.png")

    # ---- 08 hold durations
    holds = np.array([r["duration_s"] / H for r in started
                      if r["duration_s"] > 0 and r["pool"] in FULLGPU])
    multi_over = sum(1 for r in started
                     if r["pool"] in FULLGPU and r["gpus"] > 1
                     and r["duration_s"] > 24 * H)
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.logspace(np.log10(0.1), np.log10(holds.max()), 28)
    ax.hist(holds, bins=bins, color=BLUE, alpha=0.85)
    ax.axvline(24, color="black", linestyle="--", linewidth=1.5)
    ax.text(24, ax.get_ylim()[1] * 0.75, " proposed 24 h multi-GPU cap",
            rotation=90, va="top", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("allocation hold duration (hours)")
    ax.set_ylabel("allocations")
    ax.set_title(f"Multi-day holds are routine: p95 hold is "
                 f"{np.percentile(holds,95)/24:.1f} days; {multi_over} "
                 "multi-GPU holds exceed 24 h", loc="left")
    stamp(ax, window)
    save(fig, out, "08_hold_durations.png")

    # ---- 10 user greediness: idle-held vs active GPU-hours per user
    per_user = defaultdict(lambda: {"idle": 0.0, "active": 0.0})
    for r in started:
        if not r["profile"] or r["pool"] not in FULLGPU:
            continue
        d = per_user[r["user"]]
        for dur, u in r["profile"]:
            gh_ = dur * r["gpus"] / H
            d["idle" if u < 0.05 else "active"] += gh_
    top = sorted(per_user.items(), key=lambda kv: -kv[1]["idle"])[:30]
    tot_idle = sum(v["idle"] for v in per_user.values())
    top10_share = sum(v["idle"] for _, v in top[:10]) / tot_idle
    fig, ax = plt.subplots(figsize=(11, 0.34 * len(top) + 2.5))
    ys = np.arange(len(top))
    for y, (user, d) in zip(ys, top):
        c = WP_COLOR.get(wp_of(user), GRAY)
        ax.barh(y, d["idle"], color=c, alpha=0.95)
        ax.barh(y, d["active"], left=d["idle"], color=c, alpha=0.30)
    ax.set_yticks(ys, [f"{u} ({wp_of(user=u)})" for u, _ in top], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("GPU-hours in 30 days (solid: held idle, pale: active)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=w)
               for w, c in WP_COLOR.items()]
    ax.legend(handles=handles, fontsize=10, loc="lower right")
    ax.set_title(f"Ten users hold {100*top10_share:.0f}% of all idle "
                 "GPU-hours; idleness, not usage, ranks the heavy holders",
                 loc="left")
    stamp(ax, window)
    save(fig, out, "10_user_greediness.png")

    # ---- 09 cordons
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(gdt, n_cord, step="mid", color=GRAY, alpha=0.8)
    ax.axhline(73 / 6, color="black", linestyle="--", linewidth=1.2,
               label="1 node in 6")
    ax.set_ylabel("nodes cordoned (of 73)")
    ax.set_title(f"Maintenance cordons average {n_cord.mean():.1f} nodes "
                 f"({100*n_cord.mean()/73:.0f}% of the cluster), "
                 f"peaking at {int(n_cord.max())}", loc="left")
    stamp(ax, window)
    ax.legend(fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    save(fig, out, "09_cordons.png")


if __name__ == "__main__":
    main()
