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
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

plt.style.use(hep.style.CMS)
BLUE, RED, GRAY = "#5790fc", "#e42536", "#9c9ca1"
H = 3600.0
FULLGPU = ("h100nvl", "h100sxm", "l40s", "amd")


def user_series(reqs, grid):
    """active[g], idle[g], pend[g] GPU counts per grid bin for one user."""
    active = np.zeros(len(grid))
    idle = np.zeros(len(grid))
    pend = np.zeros(len(grid))
    for r in reqs:
        for a, b in r["observed"].get("pending_intervals", []):
            i, j = np.searchsorted(grid, [a, b])
            pend[i:j] += 1
        ivs = r["observed"].get("running_intervals", [])
        if not ivs:
            continue
        t = ivs[0][0]
        prof = r["profile"] or [(sum(b - a for a, b in ivs), 1.0)]
        covered = sum(d for d, _ in prof)
        for a, b in ivs:  # if profile shorter than hold, pad tail active
            pass
        for d, u in prof:
            i, j = np.searchsorted(grid, [t, t + d])
            (active if u >= 0.05 else idle)[i:j] += r["gpus"]
            t += d
    return active, idle, pend


def draw(ax, gdt, active, idle, pend, title):
    ax.fill_between(gdt, active, step="mid", color=BLUE, linewidth=0,
                    label="GPU active")
    ax.fill_between(gdt, active, active + idle, step="mid", color=RED,
                    alpha=0.85, linewidth=0, label="held idle")
    ymax = max((active + idle).max(), 1)
    if pend.max() > 0:
        ax.fill_between(gdt, -0.12 * ymax * (pend > 0), 0, step="mid",
                        color=GRAY, linewidth=0)
    ax.set_ylim(-0.16 * ymax, ymax * 1.1)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel("GPUs")
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
        for rank, u in enumerate(wu):
            a, i, pe = series[u]
            idlepct = min(100, 100 * i.sum() * 0.5 / held[u]) if held[u] else 0
            fig, ax = plt.subplots(figsize=(13, 3.0))
            draw(ax, gdt, a, i, pe,
                 f"{u} ({w}) - {held[u]:.0f} GPU-h held, {idlepct:.0f}% idle, "
                 f"peak {int((a+i).max())} GPU")
            ax.legend(fontsize=8, loc="upper right", ncol=2)
            ax.set_xlim(gdt[0], gdt[-1])
            fig.tight_layout()
            fig.savefig(wdir / f"{rank:03d}_{u}.png", dpi=110,
                        bbox_inches="tight")
            plt.close(fig)
        # contact sheet for this WP (all its users, 2 columns)
        n = len(wu)
        rows = (n + 1) // 2
        fig, axes = plt.subplots(max(rows, 1), 2,
                                 figsize=(19, 2.4 * max(rows, 1) + 0.5),
                                 sharex=True, squeeze=False)
        flat = axes.ravel()
        for ax, u in zip(flat, wu):
            a, i, pe = series[u]
            draw(ax, gdt, a, i, pe,
                 f"{u} - {held[u]:.0f} GPU-h, "
                 f"{min(100, 100*i.sum()*0.5/held[u]) if held[u] else 0:.0f}% idle")
        for ax in flat[n:]:
            ax.axis("off")
        flat[0].legend(fontsize=8, loc="upper right", ncol=2)
        for ax in axes[-1]:
            ax.set_xlim(gdt[0], gdt[-1])
        wgpu = sum(held[u] for u in wu)
        widle = sum(series[u][1].sum() * 0.5 for u in wu)
        fig.suptitle(f"{w} GPU timelines: {n} users, {wgpu:.0f} GPU-h held, "
                     f"{100*widle/wgpu:.0f}% idle "
                     "(blue = active, red = held idle)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        fig.savefig(parent / f"23_user_timelines_{w}.png", dpi=110,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {parent}/23_user_timelines_{w}.png ({n} users)")
    print(f"wrote {len(users)} per-user timelines under {out}/<WP>/")


if __name__ == "__main__":
    main()
