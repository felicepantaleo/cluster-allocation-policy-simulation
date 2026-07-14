"""NGT principles policy (PRINCIPLES.md P1 to P5).

P1 interactive guarantee: single-GPU requests from members not already
holding a guaranteed GPU form the top tier, and only they may consume the
reserved headroom (config `reserve`, GPUs per pool kept free for the tier).
P2 WP fair share: pending order follows each WP's deficit against its
target share of charged GPU-hours over a sliding usage window.
P3 multi-GPU time cap: allocations with more than one GPU are terminated at
`multi_gpu_cap_h`; remaining active work resubmits (charged again).
P4 intra-WP recycling: when shares are near target, requests from the WP
that just released capacity go first.
P5 production attribution: charge is booked per WP (the metrics module
attributes multi-GPU production jobs to the WP, not the member).

Charged rate = held GPUs x per-model correction factor (config
`gpu_charge_factor` passed through policy params as `charge_factors`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Policy
from ..trace import Request

if TYPE_CHECKING:
    from ..cluster import Allocation
    from ..engine import Engine

H = 3600.0


class WpFairShare(Policy):
    name = "ngt_principles"

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.targets: dict[str, float] = params.get(
            "wp_targets", {"WP1": .3, "WP2": .3, "WP3": .3, "WP4": .1})
        self.charge_factors: dict[str, float] = params.get("charge_factors", {})
        self.reserve: dict[str, int] = params.get("reserve", {})
        self.cap_s = params.get("multi_gpu_cap_h", 24.0) * H
        self.window_s = params.get("usage_window_h", 72.0) * H
        self.fair_tol = params.get("fair_tolerance", 0.05)
        # usage ledger: one entry per allocation, pruned once outside window
        self._ledger: list[dict] = []
        self._running_singles: dict[str, int] = {}

    # ----------------------------------------------------------- accounting

    def _usage(self, now: float) -> dict[str, float]:
        w0 = now - self.window_s
        self._ledger = [e for e in self._ledger
                        if e["end"] is None or e["end"] > w0]
        usage = {wp: 0.0 for wp in self.targets}
        for e in self._ledger:
            end = now if e["end"] is None else e["end"]
            overlap = max(0.0, end - max(e["start"], w0))
            usage[e["wp"]] = usage.get(e["wp"], 0.0) + overlap * e["rate"]
        return usage

    def _deficits(self, now: float) -> tuple[dict[str, float], bool]:
        usage = self._usage(now)
        total = sum(usage.values())
        deficits, fair = {}, True
        for wp, target in self.targets.items():
            share = usage.get(wp, 0.0) / total if total > 0 else target
            deficits[wp] = target - share
            if abs(share - target) > self.fair_tol:
                fair = False
        return deficits, fair

    def _is_guaranteed_tier(self, req: Request) -> bool:
        return (req.gpus == 1
                and self._running_singles.get(req.user, 0) == 0)

    # ------------------------------------------------------------- ordering

    def order_pending(self, pending: list[Request], engine: "Engine") -> list[Request]:
        deficits, fair = self._deficits(engine.loop.now)
        recycle_wp = engine.last_release_wp if fair else None

        def key(r: Request):
            return (
                0 if self._is_guaranteed_tier(r) else 1,          # P1 first
                0 if (recycle_wp and r.wp == recycle_wp) else 1,  # P4
                -deficits.get(r.wp, 0.0),                         # P2
                r.submit_time, r.request_id,
            )

        return sorted(pending, key=key)

    def eligible(self, req: Request, engine: "Engine") -> bool:
        # P1 reservation: non-guaranteed requests must leave `reserve` GPUs
        # free in the pool; guaranteed singles may dip into the headroom
        res = self.reserve.get(req.pool, 0)
        if res and not self._is_guaranteed_tier(req):
            free = engine.cluster.free_gpus(req.pool, allocatable_only=True)
            if free - req.gpus < res:
                return False
        return True

    # ------------------------------------------------------------ lifecycle

    def on_start(self, alloc: "Allocation", engine: "Engine") -> None:
        req = alloc.request
        if req.gpus > 0:
            rate = alloc.held_gpus * self.charge_factors.get(req.pool, 1.0)
            self._ledger.append({"wp": req.wp, "start": alloc.start,
                                 "end": None, "alloc_id": alloc.alloc_id,
                                 "rate": rate})
        if req.gpus == 1:
            self._running_singles[req.user] = self._running_singles.get(req.user, 0) + 1
        if req.gpus > 1 and req.duration_s > self.cap_s:
            engine.loop.schedule(
                alloc.start + self.cap_s,
                lambda: engine.reclaim(alloc, reclaim_offset=self.cap_s,
                                       reason="time_capped"),
            )

    def on_end(self, alloc: "Allocation", engine: "Engine") -> None:
        req = alloc.request
        for e in self._ledger:
            if e.get("alloc_id") == alloc.alloc_id:
                e["end"] = alloc.actual_end
                break
        if req.gpus == 1:
            n = self._running_singles.get(req.user, 1) - 1
            self._running_singles[req.user] = max(n, 0)
