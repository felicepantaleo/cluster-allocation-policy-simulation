from .base import Policy
from .fair_share_lease import FairShareLease
from .fcfs_pending import FcfsPending
from .idle_reclaim import IdleReclaim
from .planning_cycle import PlanningCycle, WpFairShareReclaim
from .wp_fair_share import WpFairShare

REGISTRY = {
    "fcfs_pending": FcfsPending,
    "idle_reclaim": IdleReclaim,
    "ngt_principles": WpFairShare,
    "ngt_principles_reclaim": WpFairShareReclaim,
    "planning_cycle": PlanningCycle,
    "ngt_proposal": FairShareLease,
}


def make_policy(name: str, params: dict | None = None) -> Policy:
    return REGISTRY[name](params or {})
