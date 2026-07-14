# Phase 1 findings: current policy vs the NGT principles policies

Simulation of 14 days on the synthetic trace (seed 20260714, engine seed
42), week 1 warmup, week 2 measured: 1901 logical jobs in the measurement
window, five policies on the identical trace. Full table:
`docs/phase1/comparison.md`. All numbers below are from that table; the
trace is synthetic, calibrated to the observed Saturday data point, so treat
absolute values as indicative and A/B deltas as the result.

## Calibration against the observed data point

Under the baseline policy the simulated second Saturday 18:00 to 19:35
shows: H100 NVL, H100 SXM and 3g-MIG at zero free allocatable capacity
(residual slack only in the assumed 1g-MIG carve and L40S fragments), 10 to
13 requests Pending at any instant (observed: 8), mean cordoned fraction
0.176 (observed: about 12/73 = 0.16), and 15 idle-held H100 GPUs at a
weekday probe (dashboard observation: 16). Simulated in-window waits reach
into hours where the observation reported 15 to 71 minutes; whether the
real tail is heavier than one snapshot suggests is a question for the real
event log.

## Headline results

1. The current policy wastes about one fifth of the busiest pool. Of
   11159 allocated H100 NVL GPU-hours in the week, 4678 are idle-held and
   2393 of those sit in idle stretches longer than 30 minutes, i.e.
   reclaimable without touching any active job. L40S is worse in relative
   terms (1198 of 3487 allocated GPU-hours reclaimable).

2. Idle reclaim alone (30 min threshold) converts most of that waste into
   throughput: jobs started rise from 1494 to 1577, never-started fraction
   drops from 21.4% to 17.0%, and the guaranteed-tier (single-GPU) p95 wait
   falls from 164 to 49 minutes. The cost is churn: about 2055 reaps and
   1246 resubmissions per week, concentrated on interactive sessions whose
   natural gaps exceed 30 minutes.

3. The principles policy without reclaim underdelivers on P1: reserving 8
   NVL plus 4 L40S GPUs for the 1-GPU member tier improves its p95 wait
   only to 123 minutes, because parked near-idle single-GPU sessions are
   themselves guaranteed-tier and eat the reserved headroom. A guarantee
   without reclaim gets parked on.

4. Principles plus reclaim is the strongest candidate. The 1-GPU member
   tier reaches p95 = 11.7 minutes (P1's "little waiting", delivered), WP
   charged shares land within 1.5 points of the 30/30/30/10 target (P2, vs
   4.1 points under the current policy), multi-GPU jobs respect the 24 h cap
   with remainders resubmitted (P3), and overall p95 wait is the best of all
   five policies (217 vs 277 minutes baseline) at equal-best throughput.

5. The planning cycle, as parameterized here, trades throughput for share
   precision. WP shares are essentially exact (max deviation 1.2 points)
   but 47% of jobs never start: jobs longer than 8 h wait for the daily
   12:00 decision while synthetic users cancel after a median 4 h patience,
   and a member's second concurrent session is outside the P1 guarantee so
   it also waits for an epoch. Two caveats before judging P6 on this
   number: real users facing a declared queue would adapt their patience
   (the patience model is the weakest assumption in the trace), and epoch
   frequency is a free parameter. A hybrid worth simulating next: planning
   epochs only for multi-GPU production jobs (P5/P6), continuous scheduling
   plus reclaim for everything else.

## Plots

- `phase1/wait_cdf_all.png`: wait-time CDF per policy, logical jobs.
- `phase1/gpu_hours.png`: used vs idle-held vs reclaimable GPU-hours per
  pool and policy.
- `phase1/occupancy.png`: GPU occupancy and Pending backlog over the two
  weeks (Saturday validation window shaded).
- `phase1/heavy_holders.png`: GPUs held over time by the six heaviest idle
  holders (ranked by held-minus-used GPU-hours under the current policy),
  current policy vs planning cycle. Under FCFS-Pending the top hoarders
  keep 4 to 11 GPUs continuously for multiple days (h04: 1826 held GPU-h in
  two weeks); under declared requests with fixed decision points, quotas
  and the 24 h multi-GPU cap the same users drop to 264 to 649 held GPU-h,
  a factor 2 to 4.5 less, and their holds turn over at epoch boundaries
  instead of persisting.

## Waiting time vs number of users

`phase1/wait_vs_users.png` and `phase1/sweep.md` (user counts scaled 0.5x
to 1.5x of the calibrated 82, per-user rates fixed, 3 trace seeds per
point, all policies on the identical trace at each point):

- For all continuous policies the overall p95 wait grows roughly linearly
  with the user count (FCFS: 91 min at 41 users to 452 min at 123). Idle
  reclaim buys a near-constant offset, not a slope change.
- The 1-GPU member tier is where policies separate. Under the current
  policy its p95 wait grows from 10 to 305 minutes over the sweep; the
  reserve alone (ngt_principles) barely helps at high load (253 min at 123
  users) because parked singles absorb the headroom. With reserve plus
  reclaim the tier stays essentially flat: 3 to 55 minutes across a 3x
  load range. P1 survives load growth only with both mechanisms.
- The planning cycle's wait is set by the decision cadence, not by
  contention: p95 sits at 420 to 500 minutes at every load, and even at
  half load 44% of jobs never start under the 4 h median patience
  assumption. Epoch frequency, not capacity, is its binding constraint.
- Caveat: waits are among started jobs, and the never-started fraction
  rises with load (FCFS: 6.5% at 41 users, 44% at 123), so high-load p95
  values are censored from above; sweep.json carries the fractions.

## Sizing the 1-GPU session pool vs the multi-GPU pool

Two sweeps under the recommended policy (principles plus reclaim) answer
how much capacity to dedicate to 1-GPU sessions and how much can remain
for time-capped multi-GPU allocations, as the user base grows.

Reserved full-GPU headroom (`phase1/reserve_sizing.png`,
`phase1/sweep_reserve.md`; reserve swept 0 to 46 of the 116 NVL+L40S GPUs
at each user count):

- The reserve should stay small at every tested user count. Up to 62 users
  even reserve zero meets the 15 min P1 target, because reclaim alone
  keeps enough capacity turning over. Beyond that, the optimum is 0 to 12
  GPUs, and larger reserves make the 1-GPU tier WORSE, not better: at 123
  users the tier's p95 rises from 53 min at a 5-GPU reserve to 93 min at
  46 GPUs. Two mechanisms: reserved GPUs idle whenever no guaranteed-tier
  request wants them, and a member's second concurrent session (outside
  the per-member guarantee) is pushed out together with the multi-GPU
  jobs.
- Multi-GPU jobs pay monotonically for every reserved GPU (p95 645 to 901
  min at 123 users going from 0 to 46 reserved), so oversizing the
  reserve is a pure loss. The maximum multi-GPU pool is therefore
  essentially everything: all of H100 SXM plus NVL and L40S minus at most
  a dozen GPUs.

MIG carve for interactive sessions (`phase1/mig_sizing.png`; 1 to 3
separate NVL nodes carved, 8 full GPUs lost per node): slice capacity is
not what binds; node-level cordon exposure is. Tracing cancelled sessions
showed they pend inside maintenance windows of the carved node (one 59 h
cordon zeroed the whole interactive pool while, outside cordons, even the
smallest carve had free slices at every tested load). Redundancy fixes it:
going from one to two carved nodes drops MIG sessions that never start
from 9 to 33% down to 1.2 to 5.7% at every user count, and a third node
adds little (0 to 1.3%). The multi-GPU price of the second node is within
seed noise (p95 shifts of order plus or minus 30 to 60 min on a 430 to
690 min baseline).

Sizing recommendation, combining the sweeps: dedicate two MIG-carved
nodes (2 x 26 slices, on separate machines) to 1-GPU sessions plus at
most a 5 to 12 GPU reserve on the full-GPU pools, and leave everything
else, about 100 NVL plus L40S GPUs and all 24 SXM GPUs, as the maximum
pool for time-capped multi-GPU allocations. That configuration holds the
P1 tier at or near its target across the whole 41 to 123 user range;
growing the user base further calls for re-running the two sweeps, not
for larger reserves.

## What this does not yet cover

Cordon-fraction sensitivity sweep, the gaming externality sweep (K-way
Pending vs everyone else's wait), gang/topology-aware NVLink scheduling,
AMD pools, declared-wait metrics for the planning cycle (wait measured from
the declaration deadline), and everything real-trace: all queued behind the
model-and-metrics checkpoint. The sensitivity of the phase 1 conclusions to
the A-tagged workload assumptions (rates, patience, charge factors) is the
first thing to test once the real event log is available.
