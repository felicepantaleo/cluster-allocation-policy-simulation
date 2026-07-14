import numpy as np
import pytest

from clustersim.cluster import Cluster, PoolSpec
from clustersim.engine import Engine, remaining_profile, used_gpu_seconds
from clustersim.policies import make_policy
from clustersim.trace import CordonEvent, Request

H = 3600.0


def small_cluster():
    return Cluster([
        PoolSpec("gpu", num_nodes=2, gpus_per_node=4, vcpu_per_node=92,
                 mem_per_node=720, granularity="partial"),
        PoolSpec("sxm", num_nodes=1, gpus_per_node=4, vcpu_per_node=60,
                 mem_per_node=960, granularity="whole_node_multi_gpu"),
        PoolSpec("cpu", num_nodes=1, gpus_per_node=0, vcpu_per_node=380,
                 mem_per_node=768, granularity="whole_node", socket_vcpu=190),
    ])


def req(rid, pool, gpus=1, vcpus=10, mem=50, t=0.0, dur=2 * H, profile=None,
        group=None, user="u0", patience=1e12, kind="train"):
    return Request(
        request_id=rid, group_id=group or rid, user=user, kind=kind,
        submit_time=t, pool=pool, gpus=gpus, vcpus=vcpus, mem_gb=mem,
        duration_s=dur, profile=profile or [(dur, 0.9)], patience_s=patience,
    )


def run(cluster, requests, cordons=(), policy="fcfs_pending", params=None, horizon=48 * H):
    eng = Engine(cluster, make_policy(policy, params or {}), requests,
                 list(cordons), horizon_s=horizon, seed=1)
    eng.run()
    return eng


def outcomes(eng):
    return {r["request_id"]: r for r in eng.records if r.get("record") != "allocation"}


def test_numa_unschedulable():
    eng = run(small_cluster(), [req("a", "cpu", gpus=0, vcpus=200, mem=100)])
    assert outcomes(eng)["a"]["outcome"] == "unschedulable"


def test_whole_node_exclusive():
    r1 = req("a", "cpu", gpus=0, vcpus=100, mem=100, t=0)
    r2 = req("b", "cpu", gpus=0, vcpus=100, mem=100, t=60, dur=H)
    eng = run(small_cluster(), [r1, r2])
    o = outcomes(eng)
    # second whole-node job waits for the first to finish
    assert o["b"]["outcome"] == "started"
    assert o["b"]["wait_s"] == pytest.approx(2 * H - 60)


def test_sxm_multi_gpu_takes_whole_node():
    r1 = req("a", "sxm", gpus=2, vcpus=30, mem=400, t=0)
    r2 = req("b", "sxm", gpus=1, vcpus=15, mem=200, t=60, dur=H)
    eng = run(small_cluster(), [r1, r2])
    o = outcomes(eng)
    assert o["b"]["wait_s"] > 0  # single-GPU pod cannot share the held node


def test_partial_pool_shares_node():
    r1 = req("a", "gpu", gpus=3, t=0)
    r2 = req("b", "gpu", gpus=1, t=60)
    eng = run(small_cluster(), [r1, r2])
    assert outcomes(eng)["b"]["wait_s"] == 0.0


def test_fcfs_no_holb():
    # nodes hold 4+3 of 8 GPUs; a Pending 4-GPU job cannot fit, but a later
    # 1-GPU job takes the free GPU (no head-of-line blocking)
    rs = [req("a", "gpu", gpus=4, t=0, dur=4 * H),
          req("b", "gpu", gpus=3, t=10, dur=4 * H),
          req("c", "gpu", gpus=4, t=20, dur=H),
          req("d", "gpu", gpus=1, t=30, dur=H)]
    eng = run(small_cluster(), rs)
    o = outcomes(eng)
    assert o["c"]["wait_s"] > 0
    assert o["d"]["wait_s"] == 0.0


def test_sibling_cancellation():
    sib1 = req("a-0", "gpu", gpus=1, t=0, group="a")
    sib2 = req("a-1", "sxm", gpus=1, vcpus=15, mem=240, t=0, group="a")
    eng = run(small_cluster(), [sib1, sib2])
    o = outcomes(eng)
    started = {r["outcome"] for r in (o["a-0"], o["a-1"])}
    assert started == {"started", "cancelled_sibling"}


def test_patience_cancel():
    blocker = req("a", "cpu", gpus=0, vcpus=100, mem=100, t=0, dur=10 * H)
    waiter = req("b", "cpu", gpus=0, vcpus=100, mem=100, t=60, patience=H)
    eng = run(small_cluster(), [blocker, waiter])
    assert outcomes(eng)["b"]["outcome"] == "cancelled_patience"


def test_cordon_blocks_placement():
    cords = [CordonEvent(time=0.0, node_id="gpu-00", cordoned=True),
             CordonEvent(time=0.0, node_id="gpu-01", cordoned=True)]
    eng = run(small_cluster(), [req("a", "gpu", gpus=1, t=10, patience=H)], cords)
    assert outcomes(eng)["a"]["outcome"] == "cancelled_patience"


def test_idle_reclaim_reaps_and_resubmits():
    # profile: 1h active, then 6h idle, then 1h active
    prof = [(H, 0.9), (6 * H, 0.02), (H, 0.9)]
    r = req("a", "gpu", gpus=2, t=0, dur=8 * H, profile=prof)
    eng = run(small_cluster(), [r], policy="idle_reclaim",
              params={"util_thresh": 0.05, "idle_after_s": 1800})
    allocs = [x for x in eng.records if x.get("record") == "allocation"]
    reaped = [a for a in allocs if a["end_reason"] == "reclaimed"]
    assert len(reaped) == 1
    assert reaped[0]["end"] == pytest.approx(H + 1800)
    o = outcomes(eng)
    resubs = [rec for rec in o.values() if rec["resubmit_of"] == "a"]
    assert len(resubs) == 1 and resubs[0]["outcome"] == "started"


def test_idle_reclaim_ignores_active_job():
    r = req("a", "gpu", gpus=2, t=0, dur=8 * H, profile=[(8 * H, 0.9)])
    eng = run(small_cluster(), [r], policy="idle_reclaim")
    allocs = [x for x in eng.records if x.get("record") == "allocation"]
    assert allocs[0]["end_reason"] == "completed"


def test_profile_helpers():
    r = req("a", "gpu", gpus=2, dur=3 * H, profile=[(H, 0.5), (2 * H, 0.0)])
    assert used_gpu_seconds(r, 3 * H) == pytest.approx(H * 0.5 * 2)
    assert used_gpu_seconds(r, 0.5 * H) == pytest.approx(0.5 * H * 0.5 * 2)
    rest = remaining_profile(r, 0.5 * H)
    assert rest[0] == pytest.approx((0.5 * H, 0.5))
    assert sum(d for d, _ in rest) == pytest.approx(2.5 * H)


def test_determinism():
    from clustersim import tracegen
    cfg = {
        "seed": 7, "horizon_days": 2,
        "cluster": {"pools": {
            "h100nvl": {"num_nodes": 2, "gpus_per_node": 8, "vcpu_per_node": 184,
                        "mem_per_node": 1440, "granularity": "partial"},
        }},
        "workload": {
            "users": {"trainers": {"count": 2, "jobs_per_day": 2},
                      "devs": {"count": 0, "jobs_per_day": 0},
                      "hoarders": {"count": 0, "jobs_per_day": 0},
                      "cpu_batch": {"count": 0, "jobs_per_day": 0}},
            "diurnal": {"work_hours": [9, 18],
                        "weights": {"work": 1.0, "evening": 0.45, "night": 0.12},
                        "weekend_factor": 0.35},
            "gaming": {"prob": 0.0, "prob_dev": 0.0, "prob_hoard": 0.0,
                       "k_probs": {2: 1.0}},
            "patience": {"median_h": 4.0, "sigma": 1.0},
            "cordons": {"mean_gap_h": 180, "duration_median_h": 36,
                        "duration_sigma": 0.5},
        },
    }
    m1, r1, c1 = tracegen.generate(cfg)
    m2, r2, c2 = tracegen.generate(cfg)
    assert [x.request_id for x in r1] == [x.request_id for x in r2]
    assert [x.submit_time for x in r1] == [x.submit_time for x in r2]
    # trainers only target h100nvl in this shrunken topology, so force pool
    reqs = [x for x in r1 if x.pool == "h100nvl"]
    cluster1 = Cluster([PoolSpec("h100nvl", 2, 8, 184.0, 1440.0, "partial")])
    cluster2 = Cluster([PoolSpec("h100nvl", 2, 8, 184.0, 1440.0, "partial")])
    e1 = Engine(cluster1, make_policy("fcfs_pending", {}), reqs, c1, 2 * 86400.0, seed=3)
    e2 = Engine(cluster2, make_policy("fcfs_pending", {}), reqs, c2, 2 * 86400.0, seed=3)
    e1.run(), e2.run()
    assert e1.records == e2.records


def wp_req(rid, wp, gpus, t, dur=2 * H, user=None, pool="gpu", profile=None):
    return Request(
        request_id=rid, group_id=rid, user=user or f"u-{rid}", kind="train",
        wp=wp, submit_time=t, pool=pool, gpus=gpus, vcpus=10.0 * gpus,
        mem_gb=50.0 * gpus, duration_s=dur,
        profile=profile or [(dur, 0.9)], patience_s=1e12,
    )


def test_reserve_blocks_multi_gpu_but_not_singles():
    params = {"reserve": {"gpu": 2}, "wp_targets": {"WP1": 0.5, "WP2": 0.5}}
    rs = [wp_req("a", "WP1", 4, t=0),          # leaves 4 free, ok (4-4 >= ... 8-4 >= 2)
          wp_req("b", "WP1", 4, t=10),         # would leave 0 < reserve 2
          wp_req("c", "WP2", 1, t=20, user="alice")]  # guaranteed tier, may dip in
    eng = run(small_cluster(), rs, policy="ngt_principles", params=params)
    o = outcomes(eng)
    assert o["a"]["wait_s"] == 0.0
    assert o["b"]["wait_s"] > 0
    assert o["c"]["wait_s"] == 0.0


def test_wp_deficit_ordering():
    params = {"wp_targets": {"WP1": 0.5, "WP2": 0.5}, "fair_tolerance": 0.01}
    # one node pool: WP1 runs 2h, then two full-node jobs pend; the WP2 job
    # submitted later must start first (WP1 has consumed all recent hours)
    cluster = Cluster([PoolSpec("gpu", 1, 4, 92.0, 720.0, "partial")])
    rs = [wp_req("a", "WP1", 4, t=0),
          wp_req("b", "WP1", 4, t=10),
          wp_req("c", "WP2", 4, t=20)]
    eng = Engine(cluster, make_policy("ngt_principles", params), rs, [],
                 horizon_s=48 * H, seed=1)
    eng.run()
    o = outcomes(eng)
    assert o["c"]["wait_s"] < o["b"]["wait_s"]


def test_multi_gpu_time_cap():
    params = {"multi_gpu_cap_h": 24, "wp_targets": {"WP1": 1.0}}
    r = wp_req("a", "WP1", 2, t=0, dur=30 * H, profile=[(30 * H, 0.9)])
    eng = run(small_cluster(), [r], policy="ngt_principles", params=params,
              horizon=80 * H)
    allocs = [x for x in eng.records if x.get("record") == "allocation"]
    capped = [a for a in allocs if a["end_reason"] == "time_capped"]
    assert len(capped) == 1
    assert capped[0]["end"] == pytest.approx(24 * H)
    o = outcomes(eng)
    resubs = [rec for rec in o.values() if rec["resubmit_of"] == "a"]
    assert len(resubs) == 1  # remaining 6h of work comes back


def test_single_gpu_not_capped():
    params = {"multi_gpu_cap_h": 24, "wp_targets": {"WP1": 1.0}}
    r = wp_req("a", "WP1", 1, t=0, dur=30 * H, profile=[(30 * H, 0.9)])
    eng = run(small_cluster(), [r], policy="ngt_principles", params=params,
              horizon=80 * H)
    allocs = [x for x in eng.records if x.get("record") == "allocation"]
    assert allocs[0]["end_reason"] == "completed"


def test_planning_cycle_epochs():
    params = {"wp_targets": {"WP1": 1.0},
              "tiers": [{"max_h": 8, "decisions_per_day": 3},
                        {"max_h": 100000, "decisions_per_day": 1}]}
    # 20h job submitted Monday 13:00 waits for Tuesday 12:00 (epoch grid);
    # 1-GPU job stays continuous
    rs = [wp_req("big", "WP1", 4, t=13 * H, dur=20 * H),
          wp_req("dev", "WP1", 1, t=13 * H, dur=2 * H, user="bob")]
    eng = run(small_cluster(), rs, policy="planning_cycle", params=params,
              horizon=72 * H)
    o = outcomes(eng)
    assert o["dev"]["wait_s"] == 0.0
    assert o["big"]["wait_s"] == pytest.approx(36 * H - 13 * H + 1.0, abs=2.0)


def test_planning_cycle_short_tier():
    params = {"wp_targets": {"WP1": 1.0},
              "tiers": [{"max_h": 8, "decisions_per_day": 3},
                        {"max_h": 100000, "decisions_per_day": 1}]}
    # 4h job submitted Monday 13:00: 8h tier epochs at 04, 12, 20 -> starts 20:00
    rs = [wp_req("short", "WP1", 4, t=13 * H, dur=4 * H)]
    eng = run(small_cluster(), rs, policy="planning_cycle", params=params,
              horizon=72 * H)
    o = outcomes(eng)
    assert o["short"]["wait_s"] == pytest.approx(7 * H + 1.0, abs=2.0)
