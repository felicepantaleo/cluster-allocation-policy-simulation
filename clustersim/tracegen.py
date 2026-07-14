"""Synthetic trace generator.

Generates the workload the observed NGT behaviors imply, with every knob in
the config so the trace can be recalibrated when the real event log arrives:

- four user classes: trainers (high-utilization multi-GPU jobs), devs
  (interactive, bursty, low utilization, long idle tails), hoarders (grab
  GPUs and hold them near-idle for days because release is unsafe), and CPU
  batch users on the whole-node flavors,
- nonhomogeneous Poisson arrivals with a working-hours peak, evening and
  night shoulders, and a weekend factor,
- multi-Pending gaming: with probability gaming.prob a logical job is
  submitted as K copies across eligible pools (one group_id; the engine
  cancels siblings when one starts),
- per-node cordon windows tuned to a time-average cordoned fraction of about
  1/6 (exponential gaps, lognormal durations).

Utilization profiles are piecewise (duration_s, util) segments; they are the
ground truth that separates held-but-idle from active GPU time.
"""

from __future__ import annotations

import numpy as np

from .trace import CordonEvent, Request

H = 3600.0

# per-GPU (or per-slice) vcpu / mem_gb requested, by pool
POOL_GPU_RATIO = {
    "h100nvl": (23.0, 180.0),
    "h100sxm": (15.0, 240.0),
    "l40s": (31.0, 360.0),
    "mig3g": (8.0, 40.0),
    "mig1g": (4.0, 12.0),
}

GAMING_POOLS = {
    "train": ["h100nvl", "h100sxm", "l40s"],
    "hoard": ["h100nvl", "h100sxm", "l40s"],
    "dev": ["h100nvl", "l40s"],
}


def lognormal(rng, median: float, sigma: float, lo: float, hi: float) -> float:
    return float(np.clip(rng.lognormal(np.log(median), sigma), lo, hi))


def hour_weight(hour_of_week: float, cfg: dict) -> float:
    dow = int(hour_of_week // 24)
    hod = hour_of_week % 24
    w0, w1 = cfg["work_hours"]
    if w0 <= hod < w1:
        w = cfg["weights"]["work"]
    elif 7 <= hod < w0 or w1 <= hod < 23:
        w = cfg["weights"]["evening"]
    else:
        w = cfg["weights"]["night"]
    if dow >= 5:
        w *= cfg["weekend_factor"]
    return w


def sample_arrivals(rng, jobs_per_day: float, horizon_s: float, diurnal: dict) -> list[float]:
    """Poisson arrivals per hour bucket, shaped by the diurnal profile and
    normalized so the average rate equals jobs_per_day."""
    hours = int(horizon_s // H)
    weights = np.array([hour_weight(h % 168, diurnal) for h in range(hours)])
    mean_w = np.mean([hour_weight(h, diurnal) for h in range(168)])
    rates = (jobs_per_day / 24.0) * weights / mean_w
    times = []
    counts = rng.poisson(rates)
    for h, k in enumerate(counts):
        times.extend(h * H + rng.uniform(0, H, size=k))
    return sorted(times)


# ------------------------------------------------------------------ profiles

def trainer_profile(rng, duration: float) -> list:
    setup = rng.uniform(300, 900)
    tail = lognormal(rng, 2400, 0.8, 300, 6 * H)
    if setup + tail > 0.8 * duration:
        scale = 0.8 * duration / (setup + tail)
        setup, tail = setup * scale, tail * scale
    util = rng.uniform(0.65, 0.95)
    return [(setup, 0.02), (duration - setup - tail, util), (tail, 0.02)]


def dev_profile(rng, duration: float) -> list:
    tail = min(lognormal(rng, 2 * H, 0.6, 600, 8 * H), 0.6 * duration)
    body = duration - tail
    segs, t = [], 0.0
    while t < body:
        burst = min(rng.uniform(600, 1800), body - t)
        segs.append((burst, rng.uniform(0.25, 0.5)))
        t += burst
        if t >= body:
            break
        gap = min(rng.uniform(1800, 7200), body - t)
        segs.append((gap, 0.02))
        t += gap
    segs.append((tail, 0.02))
    return segs


def hoard_profile(rng, duration: float) -> list:
    warm = min(1800.0, 0.1 * duration)
    return [(warm, rng.uniform(0.3, 0.6)), (duration - warm, 0.02)]


def cpu_profile(duration: float) -> list:
    return [(duration, 1.0)]


# ------------------------------------------------------------------- classes

def gen_class_requests(rng, cls: str, cfg: dict, cluster_cfg: dict,
                       horizon_s: float, diurnal: dict, gaming: dict,
                       patience: dict, wp_probs: dict) -> list[Request]:
    wps = list(wp_probs)
    wp_p = list(wp_probs.values())
    reqs: list[Request] = []
    for u in range(cfg["count"]):
        user = f"{cls[0]}{u:02d}"
        wp = str(rng.choice(wps, p=wp_p))
        urng = np.random.default_rng(rng.integers(0, 2**63))
        for i, t in enumerate(sample_arrivals(urng, cfg["jobs_per_day"], horizon_s, diurnal)):
            gid = f"{user}-{i:04d}"
            reqs.extend(
                make_job(urng, cls, user, wp, gid, t, cluster_cfg, gaming, patience)
            )
    return reqs


def make_job(rng, cls: str, user: str, wp: str, gid: str, t: float,
             cluster_cfg: dict, gaming: dict, patience: dict) -> list[Request]:
    if cls == "trainers":
        pool = rng.choice(["h100nvl", "h100sxm", "l40s"], p=[0.55, 0.25, 0.20])
        gpus_choices = {
            "h100nvl": ([1, 2, 4, 8], [0.25, 0.20, 0.30, 0.25]),
            "h100sxm": ([1, 2, 4], [0.20, 0.30, 0.50]),
            "l40s": ([1, 2, 4], [0.50, 0.30, 0.20]),
        }[pool]
        gpus = int(rng.choice(gpus_choices[0], p=gpus_choices[1]))
        duration = lognormal(rng, 8 * H, 1.0, 0.5 * H, 72 * H)
        profile = trainer_profile(rng, duration)
        kind, game_p = "train", gaming["prob"]
    elif cls == "devs":
        pool = rng.choice(["mig3g", "mig1g", "h100nvl", "l40s"], p=[0.35, 0.30, 0.25, 0.10])
        gpus = 1
        duration = lognormal(rng, 5 * H, 0.9, 0.5 * H, 24 * H)
        profile = dev_profile(rng, duration)
        kind = "dev"
        game_p = gaming["prob_dev"] if pool in GAMING_POOLS["dev"] else 0.0
    elif cls == "hoarders":
        # parked jupyter sessions land on 1g MIG slices too
        pool = rng.choice(["h100nvl", "h100sxm", "l40s", "mig1g"], p=[0.5, 0.15, 0.2, 0.15])
        gpus = 1 if pool == "mig1g" else int(rng.choice([1, 2, 4], p=[0.3, 0.3, 0.4]))
        duration = lognormal(rng, 60 * H, 0.7, 24 * H, 168 * H)
        profile = hoard_profile(rng, duration)
        kind, game_p = "hoard", gaming["prob_hoard"]
    else:  # cpu_batch
        flavor = rng.choice(
            ["cpuintensive", "highcore384", "highfreq", "memintensive"],
            p=[0.50, 0.20, 0.15, 0.15],
        )
        spec = cluster_cfg["pools"][flavor]
        vcpus = float(np.floor(rng.uniform(0.3, 1.0) * spec["socket_vcpu"]))
        mem = 0.5 * vcpus / spec["vcpu_per_node"] * spec["mem_per_node"]
        duration = lognormal(rng, 6 * H, 1.0, 1 * H, 48 * H)
        pat = lognormal(rng, patience["median_h"] * H, patience["sigma"], 900, 48 * H)
        return [Request(
            request_id=gid, group_id=gid, user=user, kind="cpu", wp=wp,
            submit_time=t, pool=flavor, gpus=0, vcpus=vcpus, mem_gb=mem,
            duration_s=duration, profile=cpu_profile(duration), patience_s=pat,
        )]

    pat_median = patience["median_h"] * H if kind != "hoard" else 12 * H
    pat = lognormal(rng, pat_median, patience["sigma"], 900, 48 * H)

    pools = [pool]
    if gpus <= 4 and rng.uniform() < game_p:
        k = int(rng.choice(list(gaming["k_probs"].keys()),
                           p=list(gaming["k_probs"].values())))
        others = [p for p in GAMING_POOLS[kind] if p != pool]
        rng.shuffle(others)
        pools += others[: k - 1]

    out = []
    for j, p in enumerate(pools):
        cpu_r, mem_r = POOL_GPU_RATIO[p]
        out.append(Request(
            request_id=f"{gid}-{j}", group_id=gid, user=user, kind=kind, wp=wp,
            submit_time=t, pool=p, gpus=gpus,
            vcpus=cpu_r * gpus, mem_gb=mem_r * gpus,
            duration_s=duration, profile=profile, patience_s=pat,
        ))
    return out


# ------------------------------------------------------------------- cordons

def gen_cordons(rng, cluster_cfg: dict, cordon_cfg: dict, horizon_s: float) -> list[CordonEvent]:
    events = []
    for pool_name, spec in cluster_cfg["pools"].items():
        for i in range(spec["num_nodes"]):
            node_id = f"{pool_name}-{i:02d}"
            t = float(rng.exponential(cordon_cfg["mean_gap_h"] * H))
            while t < horizon_s:
                dur = lognormal(rng, cordon_cfg["duration_median_h"] * H,
                                cordon_cfg["duration_sigma"], 2 * H, 7 * 24 * H)
                events.append(CordonEvent(time=t, node_id=node_id, cordoned=True))
                if t + dur < horizon_s:
                    events.append(CordonEvent(time=t + dur, node_id=node_id, cordoned=False))
                t += dur + float(rng.exponential(cordon_cfg["mean_gap_h"] * H))
    return events


# --------------------------------------------------------------------- entry

def generate(config: dict) -> tuple[dict, list[Request], list[CordonEvent]]:
    wl = config["workload"]
    seed = config["seed"]
    horizon_s = config["horizon_days"] * 24 * H
    rng = np.random.default_rng(seed)
    wp_probs = wl.get("wp_user_probs") or config.get(
        "wp_targets", {"WP1": .3, "WP2": .3, "WP3": .3, "WP4": .1})
    requests: list[Request] = []
    for cls in ("trainers", "devs", "hoarders", "cpu_batch"):
        requests.extend(gen_class_requests(
            rng, cls, wl["users"][cls], config["cluster"], horizon_s,
            wl["diurnal"], wl["gaming"], wl["patience"], wp_probs,
        ))
    cordons = gen_cordons(rng, config["cluster"], wl["cordons"], horizon_s)
    meta = {
        "seed": seed,
        "horizon_days": config["horizon_days"],
        "epoch": "t=0 is Monday 00:00 CEST; hour-of-week 138 is Saturday 18:00",
        "n_requests": len(requests),
        "n_logical_jobs": len({r.group_id for r in requests}),
        "n_cordon_events": len(cordons),
    }
    return meta, requests, cordons
