"""Idle reclaim on top of FCFS: reap allocations idle longer than T.

An allocation whose GPU utilization stays below util_thresh for longer than
idle_after_s is terminated and its resources freed. The engine models the
user-side cost: if the reaped job still had active work in its profile, the
user resubmits the remainder after a reaction delay and waits again.

Reclaim timers are armed from the request's utilization profile at start
time: for every idle run of length > idle_after_s we schedule a check at
idle_run_start + idle_after_s. The check verifies the allocation is still
running before reaping, so completed or already-reaped jobs are untouched.
Only GPU allocations are subject to reclaim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .fcfs_pending import FcfsPending

if TYPE_CHECKING:
    from ..cluster import Allocation
    from ..engine import Engine

DEFAULT_UTIL_THRESH = 0.05
DEFAULT_IDLE_AFTER_S = 1800.0


class IdleReclaim(FcfsPending):
    name = "idle_reclaim"

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.util_thresh = params.get("util_thresh", DEFAULT_UTIL_THRESH)
        self.idle_after_s = params.get("idle_after_s", DEFAULT_IDLE_AFTER_S)

    def on_start(self, alloc: "Allocation", engine: "Engine") -> None:
        arm_idle_timer(alloc, engine, self.util_thresh, self.idle_after_s)


def arm_idle_timer(alloc: "Allocation", engine: "Engine",
                   util_thresh: float, idle_after_s: float) -> None:
    """Arm the reap timer for the first idle run exceeding the threshold.
    Shared by IdleReclaim and the combined principles-plus-reclaim policy."""
    if alloc.request.gpus <= 0:
        return
    # merge consecutive below-threshold segments into idle runs
    offset = 0.0
    run_start = None
    run_len = 0.0
    for dur, util in alloc.request.profile:
        if util < util_thresh:
            if run_start is None:
                run_start = offset
            run_len += dur
        else:
            if run_start is not None and run_len > idle_after_s:
                break
            run_start, run_len = None, 0.0
        offset += dur
    if run_start is None or run_len <= idle_after_s:
        return
    # the first idle run to exceed T kills the allocation; later
    # segments never execute, so arm only this one timer
    reclaim_offset = run_start + idle_after_s
    engine.loop.schedule(
        alloc.start + reclaim_offset,
        lambda: engine.reclaim(alloc, reclaim_offset=reclaim_offset),
    )
