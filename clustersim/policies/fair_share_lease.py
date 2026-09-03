"""NGT proposal policy: a free GPU per member plus WP fair-share leases.

Realizes docs/proposal-scheduling.md:
- One free GPU per member (single-GPU), never charged, served first, at most
  one at a time with swap-at-start (principles 1, 2, 3).
- Every GPU beyond the first is charged to the working package on held wall
  time and ordered by the fair-share factor F = 2^(-U/S). U is the WP's 7-day
  half-life decayed usage share, S its target share renormalized over the WPs
  with live demand (principles 2, 5, 6). Within a WP, the member with the
  lower decayed usage goes first.
- No idle reclaim: the held-time charge prices idle holding instead.
- No hard multi-GPU cap: an allocation runs its declared duration, a 7-day
  renewable lease, replayed as the recorded hold.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .wp_fair_share import WpFairShare
from ..trace import Request

if TYPE_CHECKING:
    from ..cluster import Allocation
    from ..engine import Engine

H = 3600.0


class FairShareLease(WpFairShare):
    name = "ngt_proposal"

    def __init__(self, params: dict) -> None:
        params = dict(params)
        params.setdefault("usage_window_h", 168.0)   # 7-day window
        params.setdefault("multi_gpu_cap_h", 1e9)    # no forced cap
        params.setdefault("max_interactive_per_user", 1)
        super().__init__(params)
        self.halflife_s = params.get("halflife_h", 168.0) * H

    # ----------------------------------------------------------- accounting

    def _decayed(self, now: float, keyf) -> dict:
        """Sum charge over the window with a half-life decay, grouped by keyf.
        Contribution of one allocation is the time integral of its charge rate
        times 2^(-age / halflife)."""
        w0 = now - self.window_s
        self._ledger = [e for e in self._ledger
                        if e["end"] is None or e["end"] > w0]
        lam = math.log(2.0) / self.halflife_s
        out: dict = {}
        for e in self._ledger:
            end = now if e["end"] is None else e["end"]
            a = max(e["start"], w0)
            if end <= a:
                continue
            contrib = e["rate"] / lam * (
                math.exp(-lam * (now - end)) - math.exp(-lam * (now - a)))
            k = keyf(e)
            out[k] = out.get(k, 0.0) + contrib
        return out

    def _usage(self, now: float) -> dict:
        usage = {wp: 0.0 for wp in self.targets}
        for wp, v in self._decayed(now, lambda e: e["wp"]).items():
            usage[wp] = usage.get(wp, 0.0) + v
        return usage

    # ------------------------------------------------------------- ordering

    def order_pending(self, pending: list[Request], engine: "Engine") -> list[Request]:
        now = engine.loop.now
        usage = self._usage(now)
        total = sum(usage.values())
        active = ({r.wp for r in pending if r.gpus > 1}
                  | {wp for wp, u in usage.items() if u > 0})
        s_sum = sum(self.targets.get(wp, 0.0) for wp in active) or 1.0
        factor: dict = {}
        for wp in self.targets:
            S = self.targets.get(wp, 0.0) / s_sum
            U = usage.get(wp, 0.0) / total if total > 0 else 0.0
            factor[wp] = 2.0 ** (-U / S) if S > 0 else 0.0
        user_usage = self._decayed(now, lambda e: e["user"])

        def key(r: Request):
            return (
                0 if self._is_guaranteed_tier(r) else 1,   # free GPU first
                -factor.get(r.wp, 1.0),                     # WP fair share
                user_usage.get(r.user, 0.0),               # lighter member first
                r.submit_time, r.request_id,
            )

        return sorted(pending, key=key)

    # ------------------------------------------------------------ lifecycle

    def on_start(self, alloc: "Allocation", engine: "Engine") -> None:
        req = alloc.request
        # the free GPU (single-GPU) is never charged; everything beyond is
        if req.gpus > 1:
            rate = alloc.held_gpus * self.charge_factors.get(req.pool, 1.0)
            self._ledger.append({"wp": req.wp, "user": req.user,
                                 "start": alloc.start, "end": None,
                                 "alloc_id": alloc.alloc_id, "rate": rate})
        if req.gpus == 1:
            self._running_singles[req.user] = (
                self._running_singles.get(req.user, 0) + 1)
        # no multi-GPU cap: the 7-day renewable lease runs the recorded hold
