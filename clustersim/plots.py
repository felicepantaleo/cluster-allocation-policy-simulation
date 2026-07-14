"""Comparison plots (static PNG, light surface).

Styling follows the reference dataviz palette: categorical slots in fixed
order (blue = baseline, aqua = candidate, yellow = idle-held), thin marks,
hairline grid, no dual axes (stacked subplots instead). Sub-3:1 slots get
direct labels, and every figure ships with a companion table in the report.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
BLUE = "#2a78d6"      # slot 1
AQUA = "#1baf7a"      # slot 2
YELLOW = "#eda100"    # slot 3 (also: idle-held segment in the bar chart)
GREEN = "#008300"     # slot 4
VIOLET = "#4a3aa7"    # slot 5
YELLOW_DARK = "#c98500"

# fixed categorical order: a policy keeps its slot in every figure
SERIES = (BLUE, AQUA, YELLOW, GREEN, VIOLET)

H = 3600.0


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _fig(w=8.0, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    _style_axes(ax)
    return fig, ax


def wait_cdf(waits_by_policy: dict[str, list[float]], pool_label: str, out_png):
    """CDF of logical-job wait times, one line per policy."""
    fig, ax = _fig()
    colors = {name: c for name, c in zip(waits_by_policy, SERIES)}
    for name, waits in waits_by_policy.items():
        if not waits:
            continue
        w = np.sort(np.array(waits) / 60.0)
        y = np.arange(1, len(w) + 1) / len(w)
        ax.step(w, y, where="post", color=colors[name], linewidth=2.0, label=name)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("wait before start (minutes, symlog)", color=INK, fontsize=10)
    ax.set_ylabel("fraction of jobs", color=INK, fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Wait-time CDF, {pool_label}", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def gpu_hours_bars(util_by_policy: dict[str, dict], pools: list[str], out_png):
    """Per pool and policy: stacked used / idle-held GPU-hours; the
    reclaimable part of idle is hatched (texture, tone-on-tone)."""
    fig, ax = _fig(10.0, 5.0)
    n_pol = len(util_by_policy)
    width = 0.8 / n_pol
    xs = np.arange(len(pools)) * 1.1
    for j, (policy, util) in enumerate(util_by_policy.items()):
        off = (j - (n_pol - 1) / 2) * (width + 0.015)
        used = [util[p]["used_gpu_h"] for p in pools]
        idle = [util[p]["idle_held_gpu_h"] for p in pools]
        recl = [min(util[p]["reclaimable_idle_gpu_h"], i) for p, i in zip(pools, idle)]
        idle_rest = [i - r for i, r in zip(idle, recl)]
        ax.bar(xs + off, used, width, color=BLUE, label="used (active)" if j == 0 else None)
        ax.bar(xs + off, idle_rest, width, bottom=used, color=YELLOW,
               label="idle-held" if j == 0 else None)
        ax.bar(xs + off, recl, width, bottom=np.array(used) + np.array(idle_rest),
               color=YELLOW, hatch="///", edgecolor=YELLOW_DARK, linewidth=0.0,
               label="idle-held, reclaimable" if j == 0 else None)
    ax.set_xticks(xs, pools)
    ax.set_ylabel("GPU-hours in measurement window", color=INK, fontsize=10)
    ax.set_title("Allocated GPU-hours: used vs idle-held", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.005,
             "bars per pool, left to right: " + ", ".join(util_by_policy),
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _hold_series(allocs: list[dict], horizon_days: float):
    """Step series of GPUs held over time from allocation spans."""
    events: list[tuple[float, int]] = []
    for a in allocs:
        events.append((a["start"], a["held_gpus"]))
        events.append((a["end"], -a["held_gpus"]))
    events.sort()
    t, y = [0.0], [0]
    level = 0
    for time, delta in events:
        level += delta
        t.append(time / 86400.0)
        y.append(level)
    t.append(horizon_days)
    y.append(level)
    return np.array(t), np.array(y)


def user_hold_timeline(alloc_records_by_policy: dict[str, list[dict]],
                       gpu_pools: list[str], horizon_days: float, out_png,
                       top_n: int = 6):
    """Small multiples, one row per heavy holder: GPUs held vs time under
    two policies. Users are ranked by idle-held GPU-hours (held minus used)
    under the FIRST policy, which selects the members parking big
    allocations rather than the legitimate heavy trainers."""
    policies = list(alloc_records_by_policy)
    base = policies[0]
    pools = set(gpu_pools)

    def gpu_allocs(policy, user=None):
        return [a for a in alloc_records_by_policy[policy]
                if a["pool"] in pools and (user is None or a["user"] == user)]

    idle_held: dict[str, float] = {}
    for a in gpu_allocs(base):
        idle_held[a["user"]] = idle_held.get(a["user"], 0.0) + (
            a["held_gpu_s"] - a["used_gpu_s"])
    top = [u for u, _ in sorted(idle_held.items(), key=lambda kv: -kv[1])[:top_n]]

    colors = {name: c for name, c in zip(policies, (BLUE, VIOLET))}
    fig, axes = plt.subplots(len(top), 1, figsize=(10, 1.25 * len(top) + 1.2),
                             sharex=True, facecolor=SURFACE)
    for ax, user in zip(axes, top):
        _style_axes(ax)
        ymax = 4
        for policy in policies:
            allocs = gpu_allocs(policy, user)
            t, y = _hold_series(allocs, horizon_days)
            ax.fill_between(t, y, step="post", color=colors[policy], alpha=0.35,
                            linewidth=0)
            ax.step(t, y, where="post", color=colors[policy], linewidth=1.4)
            ymax = max(ymax, y.max() if len(y) else 0)
        ax.set_ylim(0, ymax * 1.25)
        info = next(iter(gpu_allocs(base, user)), {})
        ax.set_ylabel(f"{user}\n({info.get('kind', '?')}, {info.get('wp', '?')})",
                      rotation=0, ha="right", va="center", fontsize=9, color=INK)
        totals = [sum(a["held_gpu_s"] for a in gpu_allocs(p, user)) / H
                  for p in policies]
        ax.annotate(f"held {totals[0]:.0f} vs {totals[1]:.0f} GPU-h",
                    xy=(1.0, 0.82), xycoords="axes fraction", ha="right",
                    fontsize=8.5, color=MUTED)
    axes[0].set_title(
        f"GPUs held by the heaviest idle holders: {policies[0]} vs {policies[1]}",
        color=INK, fontsize=11, loc="left")
    handles = [plt.Line2D([], [], color=colors[p], linewidth=3, label=p)
               for p in policies]
    axes[0].legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK,
                   loc="upper right", bbox_to_anchor=(1.0, 1.6), ncol=2)
    axes[-1].set_xlabel("simulation time (days, day 0 = Monday)",
                        color=INK, fontsize=10)
    axes[-1].set_xlim(0, horizon_days)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def occupancy_timeline(snapshots_by_policy: dict[str, list[dict]],
                       gpu_pools: list[str], out_png,
                       shade_windows_h: list[tuple[float, float]] | None = None):
    """Two stacked panels sharing x: allocated vs allocatable GPUs (top),
    Pending request count (bottom). One color per policy."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for ax in (ax1, ax2):
        _style_axes(ax)
    colors = {name: c for name, c in zip(snapshots_by_policy, SERIES)}
    for name, snaps in snapshots_by_policy.items():
        t = np.array([s["time"] for s in snaps]) / 86400.0
        allocated = np.array([sum(s[f"{p}.allocated_gpus"] for p in gpu_pools) for s in snaps])
        ax1.plot(t, allocated, color=colors[name], linewidth=1.6, label=f"{name}: allocated")
        pend = [s["pending"] for s in snaps]
        ax2.plot(t, pend, color=colors[name], linewidth=1.6, label=name)
    first = next(iter(snapshots_by_policy.values()))
    t = np.array([s["time"] for s in first]) / 86400.0
    cap = np.array([sum(s[f"{p}.capacity_gpus"] for p in gpu_pools) for s in first])
    ax1.plot(t, cap, color=INK, linewidth=1.0, linestyle=":", label="total capacity")
    if shade_windows_h:
        for w0, w1 in shade_windows_h:
            for ax in (ax1, ax2):
                ax.axvspan(w0 / 24.0, w1 / 24.0, color=GRID, alpha=0.6, zorder=0)
    ax1.set_ylabel("GPUs", color=INK, fontsize=10)
    ax1.set_title("GPU occupancy and Pending backlog (shaded: Saturday validation window)",
                  color=INK, fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=3)
    ax2.set_ylabel("Pending requests", color=INK, fontsize=10)
    ax2.set_xlabel("simulation time (days, day 0 = Monday)", color=INK, fontsize=10)
    ax2.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
