"""One timeline plot per user: GPUs held over time, split active vs idle.

    python tools/per_user_timeline.py --derived data/derived \
        --out results/gallery/ngt_allocation_problem/per_user

For every GPU user (STEAM excluded) draws GPUs held over the 30-day window
as a stacked step area: blue = GPU active (util >= 5%), red = held idle.
Pending (waiting) spans are marked as a thin band at the bottom. One PNG
per user under per_user/, named by rank; plus a contact-sheet grid of the
top holders as 23_user_timelines_top.png in the parent folder.
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
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

plt.style.use(hep.style.CMS)
GRAY = "#9c9ca1"
H = 3600.0
FULLGPU = ("h100nvl", "h100sxm", "l40s", "amd")
POOL_COLOR = {"h100nvl": "#5790fc", "h100sxm": "#f89c20",
              "l40s": "#964a8b", "amd": "#7a21dd"}
POOL_LABEL = {"h100nvl": "H100 NVL", "h100sxm": "H100 SXM",
              "l40s": "L40S", "amd": "AMD"}


def user_series(reqs, grid):
    """per-pool active[g] and idle[g] GPU counts, plus pending[g]."""
    active = {p: np.zeros(len(grid)) for p in FULLGPU}
    idle = {p: np.zeros(len(grid)) for p in FULLGPU}
    pend = np.zeros(len(grid))
    for r in reqs:
        for a, b in r["observed"].get("pending_intervals", []):
            i, j = np.searchsorted(grid, [a, b])
            pend[i:j] += 1
        ivs = r["observed"].get("running_intervals", [])
        if not ivs or r["pool"] not in FULLGPU:
            continue
        t = ivs[0][0]
        prof = r["profile"] or [(sum(b - a for a, b in ivs), 1.0)]
        for d, u in prof:
            i, j = np.searchsorted(grid, [t, t + d])
            (active if u >= 0.05 else idle)[r["pool"]][i:j] += r["gpus"]
            t += d
    return active, idle, pend


def draw(ax, gdt, active, idle, pend, title, seen=None):
    base = np.zeros(len(gdt))
    for p in FULLGPU:
        a, i = active[p], idle[p]
        if a.max() == 0 and i.max() == 0:
            continue
        lbl = POOL_LABEL[p] if seen is not None and p not in seen else None
        if seen is not None:
            seen.add(p)
        ax.fill_between(gdt, base, base + a, step="mid", color=POOL_COLOR[p],
                        linewidth=0, label=lbl)
        base = base + a
        ax.fill_between(gdt, base, base + i, step="mid", color=POOL_COLOR[p],
                        alpha=0.32, linewidth=0)
        base = base + i
    ymax = max(base.max(), 1)
    if pend.max() > 0:
        ax.fill_between(gdt, -0.12 * ymax * (pend > 0), 0, step="mid",
                        color=GRAY, linewidth=0)
    ax.set_ylim(-0.16 * ymax, ymax * 1.1)
    ax.set_title(title, fontsize=22, loc="left")
    ax.set_ylabel("GPUs", fontsize=20)
    ax.tick_params(labelsize=18)
    # GPUs are integer: integer ticks only, at most ~5 of them
    ax.yaxis.set_major_locator(
        mticker.MaxNLocator(integer=True, nbins=5, min_n_ticks=1))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", default="data/derived")
    ap.add_argument("--out",
                    default="results/gallery/ngt_allocation_problem/per_user")
    args = ap.parse_args()
    der = Path(args.derived)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wp = json.loads((der / "user_wp.json").read_text())
    reqs = [json.loads(l) for l in open(der / "requests.jsonl")]
    reqs = [r for r in reqs if wp.get(r["user"], {}).get("wp") != "STEAM"
            and r["gpus"] > 0 and r["pool"] in FULLGPU]
    tmax = max(r["submit_time"] for r in reqs)
    END = int(np.ceil(tmax / 86400) * 86400)
    START = END - 30 * 86400
    grid = np.arange(START, END, 1800.0)
    gdt = [datetime.fromtimestamp(t, timezone.utc) for t in grid]

    by_user = defaultdict(list)
    for r in reqs:
        by_user[r["user"]].append(r)

    def wp_of(u):
        e = wp.get(u)
        return e["wp"] if e and e["wp"].startswith("WP") else "outside"

    held = {}
    for u, rs in by_user.items():
        tot = 0.0
        for r in rs:
            for a, b in r["observed"].get("running_intervals", []):
                tot += (b - a) * r["gpus"] / H
        held[u] = tot
    users = [u for u in sorted(by_user, key=lambda u: -held[u]) if held[u] > 0]

    series = {}
    for u in users:
        series[u] = user_series(by_user[u], grid)

    # per-WP: individual PNGs in per_user/<WP>/ and one contact sheet per WP
    parent = out.parent
    by_wp = defaultdict(list)
    for u in users:  # already sorted by held desc
        by_wp[wp_of(u)].append(u)

    for w in ("WP1", "WP2", "WP3", "WP4", "outside"):
        wu = by_wp.get(w, [])
        if not wu:
            continue
        wdir = out / w
        wdir.mkdir(parents=True, exist_ok=True)

        def idle_h(u):
            return sum(series[u][1][p].sum() for p in FULLGPU) * 0.5

        def peak(u):
            a, i, _ = series[u]
            return int(max((sum(a[p] for p in FULLGPU)
                            + sum(i[p] for p in FULLGPU)).max(), 0))

        for rank, u in enumerate(wu):
            a, i, pe = series[u]
            idlepct = min(100, 100 * idle_h(u) / held[u]) if held[u] else 0
            fig, ax = plt.subplots(figsize=(13, 12.0))
            seen = set()
            draw(ax, gdt, a, i, pe,
                 f"{u} ({w}) - {held[u]:.0f} GPU-h held, {idlepct:.0f}% idle, "
                 f"peak {peak(u)} GPU", seen)
            ax.legend(handles=[plt.Rectangle((0,0),1,1,color=POOL_COLOR[p],label=POOL_LABEL[p]) for p in FULLGPU if a[p].max() or i[p].max()], fontsize=16, loc="upper right", ncol=4)
            ax.set_xlim(gdt[0], gdt[-1])
            fig.tight_layout()
            fig.savefig(wdir / f"{rank:03d}_{u}.png", dpi=110,
                        bbox_inches="tight")
            plt.close(fig)
        # contact sheet for this WP (all its users, 2 columns)
        n = len(wu)
        rows = (n + 1) // 2
        fig, axes = plt.subplots(max(rows, 1), 2,
                                 figsize=(19, 9.6 * max(rows, 1) + 0.5),
                                 sharex=True, squeeze=False)
        flat = axes.ravel()
        seen = set()
        for ax, u in zip(flat, wu):
            a, i, pe = series[u]
            ip = min(100, 100 * idle_h(u) / held[u]) if held[u] else 0
            draw(ax, gdt, a, i, pe, f"{u} - {held[u]:.0f} GPU-h, {ip:.0f}% idle",
                 seen)
        for ax in flat[n:]:
            ax.axis("off")
        present = [p for p in FULLGPU
                   if any(series[u][0][p].max() or series[u][1][p].max()
                          for u in wu)]
        handles = [plt.Rectangle((0, 0), 1, 1, color=POOL_COLOR[p],
                                 label=POOL_LABEL[p]) for p in present]
        handles += [plt.Rectangle((0, 0), 1, 1, color="#666", label="active"),
                    plt.Rectangle((0, 0), 1, 1, color="#666", alpha=0.32,
                                  label="held idle")]
        fig.legend(handles=handles, fontsize=22, ncol=len(handles),
                   loc="upper right", bbox_to_anchor=(0.995, 0.997))
        for ax in axes[-1]:
            ax.set_xlim(gdt[0], gdt[-1])
        wgpu = sum(held[u] for u in wu)
        widle = sum(idle_h(u) for u in wu)
        fig.suptitle(f"{w} GPU timelines: {n} users, {wgpu:.0f} GPU-h held, "
                     f"{100*widle/wgpu:.0f}% idle "
                     "(colour = pool; solid = active, pale = held idle)",
                     fontsize=28)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        fig.savefig(parent / f"23_user_timelines_{w}.png", dpi=110,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {parent}/23_user_timelines_{w}.png ({n} users)")
    print(f"wrote {len(users)} per-user timelines under {out}/<WP>/")


if __name__ == "__main__":
    main()
