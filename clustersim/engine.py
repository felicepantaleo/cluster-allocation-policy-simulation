"""Simulation engine: wires trace, cluster and policy together.

The engine owns the generic mechanics so policies stay comparable:
- pending set and scheduling scans on every capacity-changing event,
- multi-Pending gaming semantics: when one sibling of a group starts, the
  user's other Pending copies are cancelled immediately,
- patience: a request still Pending after its patience is cancelled,
- reclaim support: a policy may reap a running allocation; if the reaped job
  still had active work, the user resubmits the remainder after a reaction
  delay drawn from the engine RNG,
- periodic state snapshots for occupancy/pending time series.

Outputs are per-request outcome records and per-allocation spans, both plain
dicts ready for JSON.
"""

from __future__ import annotations

import numpy as np

from .cluster import Allocation, Cluster
from .eventloop import EventLoop
from .policies.base import Policy
from .trace import CordonEvent, Request


def used_gpu_seconds(req: Request, span_s: float) -> float:
    """GPU-seconds of actual work in the first span_s of the profile."""
    total = 0.0
    t = 0.0
    for dur, util in req.profile:
        if t >= span_s:
            break
        total += min(dur, span_s - t) * util * req.gpus
        t += dur
    return total


def remaining_profile(req: Request, offset_s: float) -> list[tuple[float, float]]:
    """Profile segments after offset_s, first segment trimmed."""
    out = []
    t = 0.0
    for dur, util in req.profile:
        seg_start, seg_end = t, t + dur
        t = seg_end
        if seg_end <= offset_s:
            continue
        out.append((seg_end - max(seg_start, offset_s), util))
    return [(d, u) for d, u in out if d > 1e-9]


class Engine:
    def __init__(
        self,
        cluster: Cluster,
        policy: Policy,
        requests: list[Request],
        cordons: list[CordonEvent],
        horizon_s: float,
        seed: int,
        snapshot_interval_s: float = 600.0,
        resubmit_reaction_median_s: float = 600.0,
        resubmit_reaction_sigma: float = 0.8,
        resubmit_patience_s: float = 6 * 3600.0,
    ) -> None:
        self.cluster = cluster
        self.policy = policy
        self.loop = EventLoop()
        self.horizon_s = horizon_s
        self.rng = np.random.default_rng(seed)
        self.resubmit_reaction_median_s = resubmit_reaction_median_s
        self.resubmit_reaction_sigma = resubmit_reaction_sigma
        self.resubmit_patience_s = resubmit_patience_s

        self.pending: dict[str, Request] = {}
        self.requests_by_id: dict[str, Request] = {r.request_id: r for r in requests}
        self.group_started: set[str] = set()
        self.records: list[dict] = []
        self.snapshots: list[dict] = []
        self._resub_seq = 0
        self._snapshot_interval = snapshot_interval_s

        for req in requests:
            self.loop.schedule(req.submit_time, lambda r=req: self._submit(r))
        for ev in cordons:
            self.loop.schedule(
                ev.time, lambda e=ev: self._cordon(e.node_id, e.cordoned)
            )
        self.loop.schedule(0.0, self._snapshot)
        self.last_release_wp: str | None = None
        self.policy.setup(self)

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        self.loop.run(until=self.horizon_s)
        now = self.horizon_s
        for req in list(self.pending.values()):
            self._record(req, outcome="pending_at_end", wait_s=now - req.submit_time)
        self.pending.clear()
        for alloc in self.cluster.allocations.values():
            if alloc.actual_end is None:
                self._record_alloc(alloc, censored=True)

    # ---------------------------------------------------------------- events

    def _submit(self, req: Request) -> None:
        now = self.loop.now
        if req.group_id in self.group_started:
            self._record(req, outcome="cancelled_sibling", wait_s=0.0)
            return
        if not self.cluster.schedulable_ever(req):
            self._record(req, outcome="unschedulable", wait_s=0.0)
            return
        self.pending[req.request_id] = req
        self.loop.schedule(
            now + req.patience_s, lambda: self._patience_expired(req.request_id)
        )
        self._try_schedule()

    def _patience_expired(self, request_id: str) -> None:
        req = self.pending.pop(request_id, None)
        if req is not None:
            self._record(req, outcome="cancelled_patience", wait_s=req.patience_s)

    def _cordon(self, node_id: str, cordoned: bool) -> None:
        self.cluster.set_cordon(node_id, cordoned)
        if not cordoned:
            self._try_schedule()

    def _end(self, alloc: Allocation) -> None:
        if alloc.actual_end is not None:
            return
        self.cluster.release(alloc, self.loop.now, "completed")
        self._record_alloc(alloc)
        self.last_release_wp = alloc.request.wp
        self.policy.on_end(alloc, self)
        self._try_schedule()

    def reclaim(self, alloc: Allocation, reclaim_offset: float,
                reason: str = "reclaimed") -> None:
        """Called by a policy timer (idle reclaim or a multi-GPU time cap).
        reclaim_offset is seconds into the allocation at which it fires."""
        if alloc.actual_end is not None:
            return
        now = self.loop.now
        self.cluster.release(alloc, now, reason)
        self._record_alloc(alloc)
        self.last_release_wp = alloc.request.wp
        self.policy.on_end(alloc, self)
        req = alloc.request
        rest = remaining_profile(req, reclaim_offset)
        # a user who resubmits after a reap comes back to work: drop the
        # leading idle (it was the reason for the reap, not future behavior)
        while rest and rest[0][1] < 0.05:
            rest.pop(0)
        active_left = req.gpus * sum(d * u for d, u in rest if u >= 0.05)
        if active_left > 60.0 * max(req.gpus, 1):
            delay = float(
                self.rng.lognormal(
                    mean=np.log(self.resubmit_reaction_median_s),
                    sigma=self.resubmit_reaction_sigma,
                )
            )
            self._resub_seq += 1
            resub = Request(
                request_id=f"{req.request_id}.r{self._resub_seq}",
                group_id=f"{req.group_id}.r{self._resub_seq}",
                user=req.user,
                kind=req.kind,
                wp=req.wp,
                submit_time=now + delay,
                pool=req.pool,
                gpus=req.gpus,
                vcpus=req.vcpus,
                mem_gb=req.mem_gb,
                duration_s=sum(d for d, _ in rest),
                profile=rest,
                patience_s=self.resubmit_patience_s,
                resubmit_of=req.request_id,
            )
            if resub.submit_time < self.horizon_s:
                self.requests_by_id[resub.request_id] = resub
                self.loop.schedule(resub.submit_time, lambda: self._submit(resub))
        self._try_schedule()

    # ------------------------------------------------------------ scheduling

    def _try_schedule(self) -> None:
        now = self.loop.now
        for req in self.policy.order_pending(list(self.pending.values()), self):
            if req.request_id not in self.pending:
                continue  # cancelled as a sibling earlier in this pass
            if req.group_id in self.group_started:
                del self.pending[req.request_id]
                self._record(req, outcome="cancelled_sibling", wait_s=now - req.submit_time)
                continue
            if not self.policy.eligible(req, self):
                continue
            node = self.cluster.find_node(req, self.policy.placement)
            if node is None:
                continue  # no head-of-line blocking: keep scanning smaller pods
            del self.pending[req.request_id]
            alloc = self.cluster.allocate(req, node, now)
            self.group_started.add(req.group_id)
            self._record(
                req, outcome="started", wait_s=now - req.submit_time,
                node_id=node.node_id, alloc_id=alloc.alloc_id,
            )
            self.loop.schedule(alloc.planned_end, lambda a=alloc: self._end(a))
            self.policy.on_start(alloc, self)
            # cancel the user's other Pending copies of the same logical job
            for sib_id, sib in list(self.pending.items()):
                if sib.group_id == req.group_id:
                    del self.pending[sib_id]
                    self._record(
                        sib, outcome="cancelled_sibling", wait_s=now - sib.submit_time
                    )

    # ------------------------------------------------------------- recording

    def _record(self, req: Request, outcome: str, wait_s: float, **extra) -> None:
        rec = {
            "request_id": req.request_id,
            "group_id": req.group_id,
            "user": req.user,
            "kind": req.kind,
            "wp": req.wp,
            "pool": req.pool,
            "gpus": req.gpus,
            "vcpus": req.vcpus,
            "mem_gb": req.mem_gb,
            "submit_time": req.submit_time,
            "resubmit_of": req.resubmit_of,
            "outcome": outcome,
            "wait_s": wait_s,
        }
        rec.update(extra)
        self.records.append(rec)

    def _record_alloc(self, alloc: Allocation, censored: bool = False) -> None:
        end = alloc.actual_end if not censored else self.horizon_s
        span = end - alloc.start
        req = alloc.request
        self.records.append({
            "record": "allocation",
            "alloc_id": alloc.alloc_id,
            "request_id": req.request_id,
            "user": req.user,
            "kind": req.kind,
            "wp": req.wp,
            "pool": req.pool,
            "node_id": alloc.node_id,
            "start": alloc.start,
            "end": end,
            "end_reason": alloc.end_reason if not censored else "running_at_end",
            "held_gpus": alloc.held_gpus,
            "requested_gpus": req.gpus,
            "held_gpu_s": alloc.held_gpus * span,
            "requested_gpu_s": req.gpus * span,
            "used_gpu_s": used_gpu_seconds(req, span),
        })

    def _snapshot(self) -> None:
        now = self.loop.now
        snap = {"time": now, "pending": len(self.pending),
                "cordoned_fraction": self.cluster.cordoned_fraction()}
        for name in self.cluster.pools:
            pool_nodes = self.cluster.by_pool[name]
            cap = sum(n.pool.gpus_per_node for n in pool_nodes)
            allocatable_free = self.cluster.free_gpus(name, allocatable_only=True)
            snap[f"{name}.capacity_gpus"] = cap
            snap[f"{name}.free_allocatable_gpus"] = allocatable_free
            snap[f"{name}.allocated_gpus"] = sum(
                n.pool.gpus_per_node - n.gpus_free for n in pool_nodes
            )
            snap[f"{name}.pending"] = sum(
                1 for r in self.pending.values() if r.pool == name
            )
        self.snapshots.append(snap)
        nxt = now + self._snapshot_interval
        if nxt <= self.horizon_s:
            self.loop.schedule(nxt, self._snapshot)
