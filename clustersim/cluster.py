"""Resource model: pools, nodes, fit checks, allocation bookkeeping.

Granularity rules modeled per pool:
- partial: a request takes a slice of a node (H100 NVL, L40S, MIG slices).
- whole_node_multi_gpu: single-GPU requests share a node, multi-GPU requests
  take the whole node (H100 SXM, NVLink mesh).
- whole_node: all-or-nothing whole node (CPU flavors).

The one-socket NUMA constraint applies to whole-node CPU flavors: a request
whose vcpus exceed one socket's capacity is unschedulable, ever.

Cordoning a node stops new placements; running allocations are unaffected,
matching kubectl cordon semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .trace import Request

GRANULARITIES = ("partial", "whole_node_multi_gpu", "whole_node")


@dataclass
class PoolSpec:
    name: str
    num_nodes: int
    gpus_per_node: int          # 0 for CPU flavors; slice count for MIG pools
    vcpu_per_node: float
    mem_per_node: float         # GB
    granularity: str
    socket_vcpu: float = 0.0    # NUMA cap for whole_node pools; 0 disables check

    def __post_init__(self) -> None:
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"unknown granularity {self.granularity}")


@dataclass
class Node:
    node_id: str
    pool: PoolSpec
    gpus_free: int = 0
    vcpu_free: float = 0.0
    mem_free: float = 0.0
    cordoned: bool = False
    running: set = field(default_factory=set)  # alloc_ids

    @property
    def empty(self) -> bool:
        return not self.running

    def fits(self, req: Request) -> bool:
        """Capacity check only; caller filters on cordon state and pool name."""
        g = self.pool.granularity
        if g == "whole_node":
            return self.empty
        if g == "whole_node_multi_gpu" and req.gpus > 1:
            return self.empty and req.gpus <= self.pool.gpus_per_node
        return (
            req.gpus <= self.gpus_free
            and req.vcpus <= self.vcpu_free
            and req.mem_gb <= self.mem_free
        )


@dataclass
class Allocation:
    alloc_id: int
    request: Request
    node_id: str
    start: float
    planned_end: float
    # resources actually held (whole node can exceed the request)
    held_gpus: int
    held_vcpu: float
    held_mem: float
    actual_end: float | None = None
    end_reason: str | None = None  # completed | reclaimed


class Cluster:
    def __init__(self, pools: list[PoolSpec]) -> None:
        self.pools: dict[str, PoolSpec] = {p.name: p for p in pools}
        self.nodes: dict[str, Node] = {}
        self.by_pool: dict[str, list[Node]] = {p.name: [] for p in pools}
        for p in pools:
            for i in range(p.num_nodes):
                n = Node(
                    node_id=f"{p.name}-{i:02d}",
                    pool=p,
                    gpus_free=p.gpus_per_node,
                    vcpu_free=p.vcpu_per_node,
                    mem_free=p.mem_per_node,
                )
                self.nodes[n.node_id] = n
                self.by_pool[p.name].append(n)
        self._next_alloc = 0
        self.allocations: dict[int, Allocation] = {}

    def schedulable_ever(self, req: Request) -> bool:
        """Could this request ever fit on an empty node of its pool?"""
        pool = self.pools.get(req.pool)
        if pool is None:
            return False
        if pool.granularity == "whole_node":
            if pool.socket_vcpu and req.vcpus > pool.socket_vcpu:
                return False  # one-socket NUMA alignment
            return req.vcpus <= pool.vcpu_per_node and req.mem_gb <= pool.mem_per_node
        return (
            req.gpus <= pool.gpus_per_node
            and req.vcpus <= pool.vcpu_per_node
            and req.mem_gb <= pool.mem_per_node
        )

    def find_node(self, req: Request, strategy: str = "spread") -> Node | None:
        """Pick a node for the request among uncordoned candidates.

        spread: most free GPUs first (kube-scheduler LeastAllocated default).
        pack:   fewest free GPUs first (bin-packing, reduces fragmentation).
        Ties break on node_id for determinism.
        """
        candidates = [
            n for n in self.by_pool.get(req.pool, ())
            if not n.cordoned and n.fits(req)
        ]
        if not candidates:
            return None
        sign = -1 if strategy == "spread" else 1
        return min(candidates, key=lambda n: (sign * n.gpus_free, sign * n.vcpu_free, n.node_id))

    def allocate(self, req: Request, node: Node, now: float) -> Allocation:
        pool = node.pool
        whole = pool.granularity == "whole_node" or (
            pool.granularity == "whole_node_multi_gpu" and req.gpus > 1
        )
        if whole:
            held_g, held_c, held_m = node.gpus_free, node.vcpu_free, node.mem_free
            node.gpus_free, node.vcpu_free, node.mem_free = 0, 0.0, 0.0
        else:
            held_g, held_c, held_m = req.gpus, req.vcpus, req.mem_gb
            node.gpus_free -= held_g
            node.vcpu_free -= held_c
            node.mem_free -= held_m
        alloc = Allocation(
            alloc_id=self._next_alloc,
            request=req,
            node_id=node.node_id,
            start=now,
            planned_end=now + req.duration_s,
            held_gpus=held_g,
            held_vcpu=held_c,
            held_mem=held_m,
        )
        self._next_alloc += 1
        self.allocations[alloc.alloc_id] = alloc
        node.running.add(alloc.alloc_id)
        return alloc

    def release(self, alloc: Allocation, now: float, reason: str) -> None:
        if alloc.actual_end is not None:
            return
        node = self.nodes[alloc.node_id]
        node.running.discard(alloc.alloc_id)
        node.gpus_free += alloc.held_gpus
        node.vcpu_free += alloc.held_vcpu
        node.mem_free += alloc.held_mem
        alloc.actual_end = now
        alloc.end_reason = reason

    def set_cordon(self, node_id: str, cordoned: bool) -> None:
        self.nodes[node_id].cordoned = cordoned

    # observability helpers -------------------------------------------------

    def free_gpus(self, pool_name: str, allocatable_only: bool = True) -> int:
        return sum(
            n.gpus_free for n in self.by_pool[pool_name]
            if not (allocatable_only and n.cordoned)
        )

    def cordoned_fraction(self) -> float:
        n = len(self.nodes)
        return sum(1 for node in self.nodes.values() if node.cordoned) / n if n else 0.0
