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
STAMP = "NGT cluster"
AUTHOR = "Felice Pantaleo (CERN)"

# cloud-equivalent rates: cheapest of AWS/GCP on-demand, July 2026
USD_GPU_H = {"h100nvl": 6.88, "h100sxm": 6.88,  # AWS p5.48xlarge / 8
             "l40s": 2.62,                      # AWS g6e.12xlarge / 4
             "amd": 6.88,                       # H100-class (not on AWS/GCP)
             "mig3g": 6.88 / 2, "mig1g": 6.88 / 7}
CHF_USD = 0.862  # exchange rate used in the NGT budget sheet
RATE_CHF = {p: v * CHF_USD for p, v in USD_GPU_H.items()}


def chf_axis(ax, pool, axis="y"):
    """Secondary axis converting GPU-hours to cloud-equivalent kCHF. A pure
    unit conversion (one pool, one rate), not a second measure."""
    r = RATE_CHF[pool] / 1000.0
    fwd, inv = (lambda v: v * r), (lambda v: v / r)
    if axis == "y":
        # the CMS style draws primary ticks on all four sides; silence the
        # primary on the side the secondary occupies so scales do not overlap
        ax.tick_params(axis="y", which="both", right=False, labelright=False)
        sec = ax.secondary_yaxis("right", functions=(fwd, inv))
    else:
        ax.tick_params(axis="x", which="both", top=False, labeltop=False)
        sec = ax.secondary_xaxis("top", functions=(fwd, inv))
    sec.tick_params(labelsize=9, colors="#555555")
    return sec


def dt(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def stamp(ax, window):
    ax.text(0.995, 0.015, f"{STAMP}, {window}. {AUTHOR}",
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
    backlog_groups = [
        ("h100nvl", ("h100nvl",), BLUE), ("h100sxm", ("h100sxm",), ORANGE),
        ("l40s", ("l40s",), PURPLE), ("amd", ("amd",), VIOLET),
        ("mig", ("mig3g", "mig1g"), GRAY),
        ("never placed", ("unplaced", "gpu_unknown"), RED)]
    pend_g = {name: np.zeros(len(grid)) for name, _, _ in backlog_groups}
    for r in gpu:
        name = next((n for n, pools_, _ in backlog_groups
                     if r["pool"] in pools_), None)
        if name is None:
            continue
        for a, b in r["observed"]["pending_intervals"]:
            i, j = np.searchsorted(grid, [a, b])
            pend_g[name][i:j] += 1
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bottom = np.zeros(len(grid))
    for name, _, c in backlog_groups:
        ax.fill_between(gdt, bottom, bottom + pend_g[name], step="mid",
                        color=c, alpha=0.8, linewidth=0, label=name)
        bottom += pend_g[name]
    ax.set_ylabel("GPU/MIG requests Pending")
    ax.set_title(f"A queue exists in all but name: up to {int(bottom.max())} "
                 "allocation requests wait as Pending pods", loc="left")
    stamp(ax, window)
    ax.legend(fontsize=10, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    save(fig, out, "03_pending_backlog.png")

    # ---- 04 daily active vs idle held GPU-hours, per pool, with the
    # cloud-equivalent kCHF conversion on the right axis
    days = np.arange(START, END, 86400.0)
    ddt = [dt(d + 43200) for d in days]
    dcgm_pools = ("h100nvl", "h100sxm", "l40s")
    act_p = {p: np.zeros(len(days)) for p in dcgm_pools}
    idl_p = {p: np.zeros(len(days)) for p in dcgm_pools}
    covered = 0
    for r in started:
        if not r["profile"] or r["pool"] not in dcgm_pools \
                or not r["observed"]["running_intervals"]:
            continue
        covered += 1
        t = r["observed"]["running_intervals"][0][0]
        for dur, u in r["profile"]:
            i = min(int((t - START) // 86400), len(days) - 1)
            if i >= 0:
                gh = dur * r["gpus"] / H
                (idl_p if u < 0.05 else act_p)[r["pool"]][i] += gh
            t += dur
    tot_i = sum(v.sum() for v in idl_p.values())
    tot_a = sum(v.sum() for v in act_p.values())
    fig, axes = plt.subplots(len(dcgm_pools), 1, figsize=(13, 11),
                             sharex=True)
    for ax, p in zip(axes, dcgm_pools):
        ax.bar(ddt, act_p[p], width=0.8, color=BLUE,
               label="active" if p == dcgm_pools[0] else None)
        ax.bar(ddt, idl_p[p], width=0.8, bottom=act_p[p], color=RED,
               alpha=0.85,
               label="held but idle (util < 5%)" if p == dcgm_pools[0] else None)
        ax.set_ylabel(f"{p}\nGPU-h per day")
        sec = chf_axis(ax, p, "y")
        sec.set_ylabel("kCHF/day (cloud eq.)", fontsize=10, color="#555555")
    idle_chf_d = sum(idl_p[p].sum() * RATE_CHF[p] for p in dcgm_pools) / 1000
    axes[0].set_title(
        f"{100*tot_i/(tot_i+tot_a):.0f}% of monitored held GPU-hours are "
        f"idle: {tot_i:.0f} GPU-hours, a {idle_chf_d:.0f} kCHF/month cloud "
        "equivalent", loc="left")
    stamp(axes[0], window)
    axes[0].legend(fontsize=11)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.text(0.12, 0.005, f"DCGM utilization available for {covered} "
             "full-GPU allocations; MIG slices and AMD not covered. "
             "Cloud rates as in the cost plot.", fontsize=10, color="#555555")
    save(fig, out, "04_idle_gpu_hours.png")

    # ---- 05 idle fraction per allocation
    fr_p = {p: [] for p in ("h100nvl", "h100sxm", "l40s")}
    for r in started:
        if r["profile"] and r["pool"] in fr_p:
            tot = sum(d for d, _ in r["profile"])
            fr_p[r["pool"]].append(
                sum(d for d, u in r["profile"] if u < 0.05) / tot)
    all_fr = [f for v in fr_p.values() for f in v]
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.linspace(0, 1, 21)
    ax.hist([fr_p[p] for p in fr_p], bins=bins, stacked=True,
            color=[POOL_COLOR[p] for p in fr_p],
            label=[f"{p} (n={len(fr_p[p])})" for p in fr_p])
    ax.set_xlabel("fraction of allocation lifetime with GPU util < 5%")
    ax.set_ylabel("allocations")
    ax.set_title(f"The median monitored allocation is idle for "
                 f"{100*np.median(all_fr):.0f}% of its lifetime "
                 f"(n={len(all_fr)})", loc="left")
    stamp(ax, window)
    ax.legend(fontsize=10, loc="upper left")
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
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.tick_params(axis="y", which="both", right=False)
    ax.set_ylabel("share of full-GPU hours (%)")
    ax.set_title(f"Who gets the GPU-hours: {100*vals[4]/tot:.0f}% go to "
                 "users outside the WP roster", loc="left")
    stamp(ax, window)
    save(fig, out, "06_wp_shares.png")

    # ---- 07 unsatisfied requests
    w = np.array([r["observed"]["wait_s"] / 60 for r in cancelled])
    canc_groups = [("h100nvl", ("h100nvl",), BLUE),
                   ("h100sxm", ("h100sxm",), ORANGE),
                   ("l40s", ("l40s",), PURPLE), ("amd", ("amd",), VIOLET),
                   ("mig", ("mig3g", "mig1g"), GRAY),
                   ("never placed", ("unplaced", "gpu_unknown"), RED)]
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.logspace(0, np.log10(max(w.max(), 10)), 24)
    per_pool_w = [np.clip([r["observed"]["wait_s"] / 60 for r in cancelled
                           if r["pool"] in pools_], 1, None)
                  for _, pools_, _ in canc_groups]
    ax.hist(per_pool_w, bins=bins, stacked=True,
            color=[c for _, _, c in canc_groups],
            label=[n for n, _, _ in canc_groups])
    ax.set_xscale("log")
    ax.legend(fontsize=10, loc="upper left")
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
    hold_p = [[r["duration_s"] / H for r in started
               if r["duration_s"] > 0 and r["pool"] == p] for p in FULLGPU]
    ax.hist(hold_p, bins=bins, stacked=True,
            color=[POOL_COLOR[p] for p in FULLGPU], label=list(FULLGPU))
    ax.axvline(24, color="black", linestyle="--", linewidth=1.5)
    ax.legend(fontsize=10, loc="upper left")
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
    # the P1 development allowance: one GPU held around the clock for the
    # whole window may legitimately sit mostly idle
    allowance = 30 * 24.0
    excess = sum(max(0.0, v["idle"] - allowance) for v in per_user.values())
    n_over = sum(1 for v in per_user.values() if v["idle"] > allowance)
    fig, ax = plt.subplots(figsize=(11, 0.34 * len(top) + 2.5))
    ys = np.arange(len(top))
    for y, (user, d) in zip(ys, top):
        c = WP_COLOR.get(wp_of(user), GRAY)
        ax.barh(y, d["idle"], color=c, alpha=0.95)
        ax.barh(y, d["active"], left=d["idle"], color=c, alpha=0.30)
    ax.axvline(allowance, color="black", linestyle="--", linewidth=1.4)
    ax.text(allowance, len(top) - 0.4, "  one dev GPU held 24/7 all month "
            f"({allowance:.0f} GPU-h)", rotation=90, va="bottom", ha="right",
            fontsize=10)
    ax.set_yticks(ys, [f"{u} ({wp_of(user=u)})" for u, _ in top], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("GPU-hours in 30 days (solid: held idle, pale: active)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=w)
               for w, c in WP_COLOR.items()]
    ax.legend(handles=handles, fontsize=10, loc="lower right")
    ax.set_title(f"{n_over} users hold more idle GPU-time than a full-time "
                 f"dev GPU; {excess:.0f} GPU-hours sit above that allowance",
                 loc="left")
    stamp(ax, window)
    save(fig, out, "10_user_greediness.png")

    # ---- 11 total held GPU-hours per user vs the single-GPU line
    tot_user = defaultdict(float)
    for r in started:
        if r["pool"] not in FULLGPU:
            continue
        for a, b in r["observed"]["running_intervals"]:
            tot_user[r["user"]] += (b - a) * r["gpus"] / H
    topt = sorted(tot_user.items(), key=lambda kv: -kv[1])[:30]
    allowance = 30 * 24.0
    n_over_t = sum(1 for v in tot_user.values() if v > allowance)
    fig, ax = plt.subplots(figsize=(11, 0.34 * len(topt) + 2.5))
    ys = np.arange(len(topt))
    for y, (user, v) in zip(ys, topt):
        ax.barh(y, v, color=WP_COLOR.get(wp_of(user), GRAY), alpha=0.95)
    ax.axvline(allowance, color="black", linestyle="--", linewidth=1.4)
    ax.text(allowance, len(topt) - 0.4, "  one GPU held 24/7 all month "
            f"({allowance:.0f} GPU-h)", rotation=90, va="bottom", ha="right",
            fontsize=10)
    ax.set_yticks(ys, [f"{u} ({wp_of(user=u)})" for u, _ in topt], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("total held GPU-hours in 30 days (full-GPU pools)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=w)
               for w, c in WP_COLOR.items()]
    ax.legend(handles=handles, fontsize=10, loc="lower right")
    ax.set_title(f"{n_over_t} users held more than one GPU-month; the top "
                 f"holder used {topt[0][1] / allowance:.1f} GPUs' worth "
                 "around the clock", loc="left")
    stamp(ax, window)
    save(fig, out, "11_user_total_hours.png")

    # ---- 12 pool usage stacked by WP
    pool_wp = defaultdict(lambda: defaultdict(float))
    for r in started:
        if r["pool"] not in ONPREM:
            continue
        for a, b in r["observed"]["running_intervals"]:
            pool_wp[r["pool"]][wp_of(r["user"])] += (b - a) * r["gpus"] / H
    order_p = [p for p in ONPREM if pool_wp.get(p)]
    wps = ["WP1", "WP2", "WP3", "WP4", "outside roster"]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = np.arange(len(order_p))
    bottom = np.zeros(len(order_p))
    for w in wps:
        vals = np.array([pool_wp[p].get(w, 0.0) for p in order_p])
        ax.bar(xs, vals, 0.62, bottom=bottom, color=WP_COLOR[w], label=w)
        bottom += vals
    for x, p in zip(xs, order_p):
        ax.annotate(f"{bottom[x]:.0f}", xy=(x, bottom[x]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=11)
    ax.set_xticks(xs, order_p)
    ax.set_ylabel("held GPU- or slice-hours in 30 days")
    dom = {p: max(pool_wp[p], key=pool_wp[p].get) for p in order_p}
    ax.set_title("Who uses which pool: H100 NVL dominates the volume; "
                 f"SXM is mostly {dom.get('h100sxm', '?')}", loc="left")
    stamp(ax, window)
    ax.legend(fontsize=10)
    fig.text(0.12, -0.02, "MIG bars count slice-hours, not GPU-hours.",
             fontsize=10, color="#555555")
    save(fig, out, "12_pool_usage_by_wp.png")

    # ---- 13 idle vs active per pool (DCGM-covered)
    pool_ia = defaultdict(lambda: {"idle": 0.0, "active": 0.0})
    for r in started:
        if not r["profile"] or r["pool"] not in FULLGPU:
            continue
        for dur, u in r["profile"]:
            pool_ia[r["pool"]]["idle" if u < 0.05 else "active"] += \
                dur * r["gpus"] / H
    order_f = [p for p in FULLGPU if pool_ia.get(p)]
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(order_f))
    act = [pool_ia[p]["active"] for p in order_f]
    idl = [pool_ia[p]["idle"] for p in order_f]
    ax.bar(xs, act, 0.6, color=BLUE, label="active GPU-hours")
    ax.bar(xs, idl, 0.6, bottom=act, color=RED, alpha=0.85,
           label="held but idle (GPU util < 5%)")
    for x, a_, i_ in zip(xs, act, idl):
        ax.annotate(f"{100*i_/(a_+i_):.0f}% idle", xy=(x, a_ + i_),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=11)
    ax.set_xticks(xs, order_f)
    ax.set_ylabel("GPU-hours in 30 days (DCGM-covered)")
    worst = max(order_f, key=lambda p: pool_ia[p]["idle"] /
                max(pool_ia[p]["idle"] + pool_ia[p]["active"], 1))
    ax.set_title("Idle holding by pool: every pool wastes most of its held "
                 f"hours; {worst} is the worst", loc="left")
    stamp(ax, window)
    ax.legend(fontsize=10)
    save(fig, out, "13_pool_idle_active.png")

    # ---- 14 waits while holding nothing (lockouts) vs top-up waits
    run_by_user = defaultdict(list)
    for r in reqs:
        if r["gpus"] > 0 and r["pool"] not in ("cloud_t4", "cpu", "unknown"):
            for iv in r["observed"].get("running_intervals", []):
                run_by_user[r["user"]].append((iv[0], iv[1], r["request_id"]))

    def covered(p0, p1, user, exclude):
        cov = sorted((max(a, p0), min(b, p1))
                     for a, b, rid in run_by_user[user]
                     if rid != exclude and min(b, p1) > max(a, p0))
        tot, cur = 0.0, None
        for a, b in cov:
            if cur is None or a > cur[1]:
                if cur:
                    tot += cur[1] - cur[0]
                cur = [a, b]
            else:
                cur[1] = max(cur[1], b)
        if cur:
            tot += cur[1] - cur[0]
        return tot / (p1 - p0)

    cats = {"locked out\n(no other GPU pod)": [], "partially covered": [],
            "top-up\n(other GPU pod running)": []}
    for r in gpu:
        w = r["observed"]["wait_s"]
        if w < 300:
            continue
        f = covered(r["submit_time"], r["submit_time"] + w, r["user"],
                    r["request_id"])
        key = list(cats)[0 if f < 0.1 else 2 if f > 0.9 else 1]
        cats[key].append(r["observed"]["outcome"] == "cancelled")
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(cats))
    got = [sum(1 for c in v if not c) for v in cats.values()]
    gave = [sum(1 for c in v if c) for v in cats.values()]
    ax.bar(xs, got, 0.55, color=BLUE, label="eventually got the GPU")
    ax.bar(xs, gave, 0.55, bottom=got, color=RED, alpha=0.85,
           label="gave up waiting")
    for x, g, u in zip(xs, got, gave):
        ax.annotate(f"{g+u}", xy=(x, g + u), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=12)
    ax.set_xticks(xs, list(cats), fontsize=12)
    ax.set_ylabel("waiting episodes over 5 min, 30 days")
    n_lock = len(cats[list(cats)[0]])
    frac_gave = gave[0] / max(n_lock, 1)
    ax.set_title(f"{n_lock} times a user waited while holding NOTHING on the "
                 f"GPUs; {100*frac_gave:.0f}% of those gave up", loc="left")
    stamp(ax, window)
    ax.legend(fontsize=11)
    save(fig, out, "14_lockout_waits.png")

    # ---- 15 greediness by pool: idle vs active per user, per pool
    pu = defaultdict(lambda: {"idle": 0.0, "active": 0.0})
    for r in started:
        if not r["profile"] or r["pool"] not in ("h100nvl", "h100sxm", "l40s"):
            continue
        for dur, u in r["profile"]:
            pu[(r["pool"], r["user"])]["idle" if u < 0.05 else "active"] += \
                dur * r["gpus"] / H
    panels = ("h100nvl", "h100sxm", "l40s")
    NTOP = 12
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(11, len(panels) * (0.36 * NTOP + 1.6)))
    allowance = 30 * 24.0
    for ax, pool in zip(axes, panels):
        rows = sorted(((u, d) for (p, u), d in pu.items() if p == pool),
                      key=lambda kv: -kv[1]["idle"])[:NTOP]
        ys = np.arange(len(rows))
        for y, (user, d) in zip(ys, rows):
            c = WP_COLOR.get(wp_of(user), GRAY)
            ax.barh(y, d["idle"], color=c, alpha=0.95)
            ax.barh(y, d["active"], left=d["idle"], color=c, alpha=0.30)
        ax.axvline(allowance, color="black", linestyle="--", linewidth=1.2)
        ax.set_yticks(ys, [f"{u} ({wp_of(user=u)})" for u, _ in rows],
                      fontsize=9)
        ax.invert_yaxis()
        tot_i = sum(d["idle"] for _, d in
                    ((k, v) for k, v in pu.items() if k[0] == pool))
        top_i = sum(d["idle"] for _, d in rows)
        ax.set_ylabel(pool, fontsize=13)
        sec = chf_axis(ax, pool, "x")
        if pool == panels[0]:
            sec.set_xlabel("kCHF (cloud equivalent)", fontsize=10,
                           color="#555555")
        ax.annotate(f"top {len(rows)} hold {100*top_i/max(tot_i,1):.0f}% of "
                    f"this pool's idle GPU-hours", xy=(0.98, 0.06),
                    xycoords="axes fraction", ha="right", fontsize=10,
                    color="#555555")
    axes[0].set_title("The heavy idle holders differ per pool "
                      "(solid: held idle, pale: active; dashed: one GPU "
                      "24/7 all month)", loc="left", pad=34)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=w)
               for w, c in WP_COLOR.items()]
    axes[0].legend(handles=handles, fontsize=9, loc="lower right", ncol=2)
    axes[-1].set_xlabel("GPU-hours in 30 days")
    stamp(axes[-1], window)
    fig.tight_layout()
    save(fig, out, "15_user_greediness_by_pool.png")

    # ---- 16 cloud-equivalent cost in CHF (cheapest of AWS/GCP, Jul 2026)
    cost_act = defaultdict(float)
    cost_idle = defaultdict(float)
    cost_unk = defaultdict(float)
    for r in started:
        if r["pool"] not in ONPREM:
            continue
        rate = USD_GPU_H[r["pool"]] * CHF_USD
        held = sum(b - a for a, b in r["observed"]["running_intervals"])
        if r["profile"] and r["pool"] in ("h100nvl", "h100sxm", "l40s"):
            for dur, u in r["profile"]:
                chf = dur * r["gpus"] / H * rate
                (cost_idle if u < 0.05 else cost_act)[r["pool"]] += chf
        else:
            cost_unk[r["pool"]] += held * r["gpus"] / H * rate
    order_c = [p for p in ONPREM
               if cost_act[p] + cost_idle[p] + cost_unk[p] > 0]
    tot_chf = sum(cost_act[p] + cost_idle[p] + cost_unk[p] for p in order_c)
    idle_chf = sum(cost_idle.values())
    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = np.arange(len(order_c))
    a_ = np.array([cost_act[p] / 1000 for p in order_c])
    i_ = np.array([cost_idle[p] / 1000 for p in order_c])
    u_ = np.array([cost_unk[p] / 1000 for p in order_c])
    ax.bar(xs, a_, 0.6, color=BLUE, label="active")
    ax.bar(xs, i_, 0.6, bottom=a_, color=RED, alpha=0.85,
           label="held but idle")
    ax.bar(xs, u_, 0.6, bottom=a_ + i_, color=GRAY, alpha=0.8,
           label="no utilization data (MIG, AMD)")
    for x in xs:
        ax.annotate(f"{a_[x]+i_[x]+u_[x]:.0f}k", xy=(x, a_[x] + i_[x] + u_[x]),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=11)
    ax.set_xticks(xs, order_c)
    ax.set_ylim(0, float((a_ + i_ + u_).max()) * 1.14)
    ax.set_ylabel("cloud-equivalent cost (kCHF per month)")
    ax.set_title(f"Renting last month's held GPU-hours from the cheapest "
                 f"public cloud: {tot_chf/1000:.0f} kCHF, of which at least "
                 f"{idle_chf/1000:.0f} kCHF sat idle", loc="left")
    stamp(ax, window)
    ax.legend(fontsize=10)
    fig.text(0.12, -0.06,
             "On-demand, cheapest of AWS/GCP, July 2026: H100 6.88 USD/GPU-h "
             "(AWS p5.48xlarge; GCP A3 10.98), L40S 2.62 USD/GPU-h (AWS\n"
             "g6e.12xlarge); MIG priced as H100 fractions (1/2, 1/7); MI300X "
             "not offered by either, priced H100-class. CHF at 0.862 (NGT\n"
             "budget rate). Egress, storage and CPU-only nodes excluded.",
             fontsize=9, color="#555555")
    save(fig, out, "16_cloud_cost_chf.png")

    # ---- 17 total held GPU-hours per user, per pool, with kCHF axis
    tot_pu = defaultdict(float)
    for r in started:
        if r["pool"] in FULLGPU:
            for a, b in r["observed"]["running_intervals"]:
                tot_pu[(r["pool"], r["user"])] += (b - a) * r["gpus"] / H
    NT = 10
    fig, axes = plt.subplots(len(FULLGPU), 1,
                             figsize=(11, len(FULLGPU) * (0.36 * NT + 1.7)))
    for ax, pool in zip(axes, FULLGPU):
        rows = sorted(((u, v) for (p, u), v in tot_pu.items() if p == pool),
                      key=lambda kv: -kv[1])[:NT]
        ys = np.arange(len(rows))
        for y, (user, v) in zip(ys, rows):
            ax.barh(y, v, color=WP_COLOR.get(wp_of(user), GRAY), alpha=0.95)
        ax.axvline(30 * 24.0, color="black", linestyle="--", linewidth=1.2)
        ax.set_yticks(ys, [f"{u} ({wp_of(user=u)})" for u, _ in rows],
                      fontsize=9)
        ax.invert_yaxis()
        ax.set_ylabel(pool, fontsize=13)
        sec = chf_axis(ax, pool, "x")
        if pool == FULLGPU[0]:
            sec.set_xlabel("kCHF (cloud equivalent)", fontsize=10,
                           color="#555555")
    axes[0].set_title("Top consumers per pool, in GPU-hours and cloud-"
                      "equivalent kCHF (dashed: one GPU 24/7 all month)",
                      loc="left", pad=34)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=w)
               for w, c in WP_COLOR.items()]
    axes[0].legend(handles=handles, fontsize=9, loc="lower right", ncol=2)
    axes[-1].set_xlabel("total held GPU-hours in 30 days")
    stamp(axes[-1], window)
    fig.tight_layout()
    save(fig, out, "17_user_total_by_pool.png")

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
