"""P6 planning cycle on top of the principles policy (P1 to P5).

Big allocations for the next 24 hours are declared before 12:00; at 12:00
the order is decided from priorities and quotas. Shorter jobs get more
frequent decision points: a tier with a max allocation time of 8 hours has
3 decision epochs per day, and so on per config.

Implementation: a request becomes eligible at the first decision epoch at
or after its submission (its declaration lands in the next batch). Epochs
are anchored at 12:00 and spaced 24h / decisions_per_day within each tier.
The P1 guaranteed interactive tier stays continuous and never waits for an
epoch. Ordering at an epoch is inherited from the principles policy:
guarantee first, then WP deficit, then submission order.

Combined variant classes at the bottom mix in idle reclaim.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .idle_reclaim import DEFAULT_IDLE_AFTER_S, DEFAULT_UTIL_THRESH, arm_idle_timer
from .wp_fair_share import WpFairShare
from ..trace import Request

if TYPE_CHECKING:
    from ..cluster import Allocation
    from ..engine import Engine

H = 3600.0
NOON = 12 * H


class PlanningCycle(WpFairShare):
    name = "planning_cycle"

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # tiers sorted by max duration; a job belongs to the first tier
        # whose max_h covers its duration, longest tier catches the rest
        self.tiers = sorted(
            params.get("tiers", [{"max_h": 8, "decisions_per_day": 3},
                                 {"max_h": 100000, "decisions_per_day": 1}]),
            key=lambda t: t["max_h"],
        )

    def _tier_period(self, req: Request) -> float:
        for t in self.tiers:
            if req.duration_s <= t["max_h"] * H:
                return 86400.0 / t["decisions_per_day"]
        return 86400.0 / self.tiers[-1]["decisions_per_day"]

    def next_epoch(self, req: Request) -> float:
        period = self._tier_period(req)
        return NOON + math.ceil((req.submit_time - NOON) / period) * period

    def setup(self, engine: "Engine") -> None:
        times = set()
        for tier in self.tiers:
            period = 86400.0 / tier["decisions_per_day"]
            t = NOON % period  # earliest non-negative point of the epoch grid
            while t < engine.horizon_s:
                times.add(t)
                t += period
        for t in sorted(times):
            engine.loop.schedule(t + 1.0, engine._try_schedule)

    def eligible(self, req: Request, engine: "Engine") -> bool:
        if not super().eligible(req, engine):
            return False
        if self._is_guaranteed_tier(req):
            return True  # P1 stays continuous
        return engine.loop.now >= self.next_epoch(req)


class WpFairShareReclaim(WpFairShare):
    """P1 to P5 plus idle reclaim (Maria's aggressive freeing)."""

    name = "ngt_principles_reclaim"

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.util_thresh = params.get("util_thresh", DEFAULT_UTIL_THRESH)
        self.idle_after_s = params.get("idle_after_s", DEFAULT_IDLE_AFTER_S)

    def on_start(self, alloc: "Allocation", engine: "Engine") -> None:
        super().on_start(alloc, engine)
        arm_idle_timer(alloc, engine, self.util_thresh, self.idle_after_s)
