"""Baseline: the current NGT behavior.

There is no queue and no quota. Pending pods sit in the apiserver and the
scheduler places any pod that fits whenever capacity frees. That is FCFS by
submit time with skipping: an old pod that does not fit does not block a
newer, smaller pod (no head-of-line blocking). Multi-Pending gaming lives in
the trace (sibling groups), not here; this policy just does not prevent it.
"""

from __future__ import annotations

from .base import Policy
from ..trace import Request


class FcfsPending(Policy):
    name = "fcfs_pending"

    def order_pending(self, pending: list[Request], engine=None) -> list[Request]:
        return sorted(pending, key=lambda r: (r.submit_time, r.request_id))
