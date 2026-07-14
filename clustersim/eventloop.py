"""Minimal deterministic discrete-event loop.

Events are (time, sequence, callback) tuples on a heap. The sequence number
breaks ties so that simultaneous events fire in scheduling order, which makes
every run bit-reproducible for a given trace and seed.
"""

from __future__ import annotations

import heapq
from typing import Callable


class EventLoop:
    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Callable[[], None]]] = []
        self._seq = 0
        self.now = 0.0

    def schedule(self, time: float, fn: Callable[[], None]) -> None:
        if time < self.now:
            raise ValueError(f"cannot schedule in the past: {time} < {self.now}")
        heapq.heappush(self._heap, (time, self._seq, fn))
        self._seq += 1

    def run(self, until: float | None = None) -> None:
        while self._heap:
            time, _, fn = self._heap[0]
            if until is not None and time > until:
                break
            heapq.heappop(self._heap)
            self.now = time
            fn()
        if until is not None:
            self.now = max(self.now, until)
