"""Policy interface.

A policy decides two things:
- the order in which Pending requests are considered when capacity changes
  (order_pending), and
- optional lifecycle hooks (on_start) used e.g. to arm reclaim timers.

The engine owns the mechanics: pending set, placement, sibling cancellation,
patience timeouts. Policies stay small and comparable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..trace import Request

if TYPE_CHECKING:
    from ..cluster import Allocation
    from ..engine import Engine


class Policy:
    name = "base"
    placement = "spread"  # kube-scheduler LeastAllocated default

    def __init__(self, params: dict) -> None:
        self.params = params
        self.placement = params.get("placement", self.placement)

    def setup(self, engine: "Engine") -> None:
        """Called once before the run; may schedule policy events."""

    def order_pending(self, pending: list[Request], engine: "Engine") -> list[Request]:
        raise NotImplementedError

    def eligible(self, req: Request, engine: "Engine") -> bool:
        """May a Pending request be placed right now? Quota, reservation and
        planning-cycle policies gate here; capacity fit is the engine's job."""
        return True

    def on_start(self, alloc: "Allocation", engine: "Engine") -> None:
        pass

    def on_end(self, alloc: "Allocation", engine: "Engine") -> None:
        pass
