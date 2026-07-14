"""Trace schema and JSONL I/O.

A trace directory contains:
  meta.json      generator parameters, seed, epoch convention
  requests.jsonl one Request per line, sorted by submit_time
  cordons.jsonl  one CordonEvent per line, sorted by time

Time convention: seconds since sim epoch, where t=0 is a Monday 00:00 local
(CEST). Hour-of-week = (t / 3600) mod 168; Saturday 18:00 is hour 138.

The same schema is the adapter point for the real cluster event log: when
admin access lands, a converter writes requests.jsonl/cordons.jsonl from the
log and everything downstream runs unchanged. Fields the simulator does not
consume but the real log provides (observed wait, outcome, placement) go into
the `observed` dict for validation against simulated results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SECONDS_PER_HOUR = 3600.0
HOURS_PER_WEEK = 168


@dataclass
class Request:
    request_id: str
    group_id: str            # logical job; K gaming siblings share one group_id
    user: str
    kind: str                # train | dev | hoard | cpu
    submit_time: float
    pool: str
    gpus: int                # GPU count, MIG slice count, or 0 for CPU flavors
    vcpus: float
    mem_gb: float
    duration_s: float        # hold time from start to voluntary release
    wp: str = ""             # working package (WP1..WP4) charged for the job
    # piecewise GPU utilization: list of [segment_duration_s, util_frac],
    # sums to duration_s; the ground truth behind held-vs-used accounting
    profile: list = field(default_factory=list)
    patience_s: float = 1e12  # cancel if still Pending after this long
    resubmit_of: str | None = None  # set on engine-generated resubmissions
    observed: dict = field(default_factory=dict)  # real-log fields, unused by sim

    def active_gpu_seconds(self) -> float:
        return self.gpus * sum(d * u for d, u in self.profile)


@dataclass
class CordonEvent:
    time: float
    node_id: str
    cordoned: bool


def write_trace(out_dir: Path, meta: dict, requests: list[Request],
                cordons: list[CordonEvent]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    with open(out_dir / "requests.jsonl", "w") as f:
        for r in sorted(requests, key=lambda r: (r.submit_time, r.request_id)):
            f.write(json.dumps(asdict(r)) + "\n")
    with open(out_dir / "cordons.jsonl", "w") as f:
        for c in sorted(cordons, key=lambda c: (c.time, c.node_id)):
            f.write(json.dumps(asdict(c)) + "\n")


def read_trace(trace_dir: Path) -> tuple[dict, list[Request], list[CordonEvent]]:
    meta = json.loads((trace_dir / "meta.json").read_text())
    requests = []
    with open(trace_dir / "requests.jsonl") as f:
        for line in f:
            d = json.loads(line)
            d["profile"] = [tuple(seg) for seg in d["profile"]]
            requests.append(Request(**d))
    cordons = []
    with open(trace_dir / "cordons.jsonl") as f:
        for line in f:
            cordons.append(CordonEvent(**json.loads(line)))
    return meta, requests, cordons
