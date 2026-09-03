# Proposal: a fair-share scheduling policy for the NGT GPU cluster

Author: Felice Pantaleo. For discussion in the PMC.

## 1. Summary

The NGT cluster runs first come first served. A session is a bare Pod. There
is no queue, no quota, and no time limit. This proposal replaces that with the
policy that every large shared GPU center already runs: a small guaranteed
interactive tier, a time-limited batch tier for everything larger, and
fair-share priority across the working packages accounted over a rolling
window. The design is not novel. The same three parts run at NERSC, Harvard
FASRC, Stanford, Princeton, the KU cluster, and inside NVIDIA Run:ai. This
document states the policy, shows that it matches real-world practice, and
shows on the measured NGT trace that it fixes the problems we recorded.

## 2. The measured problem

All numbers below are from the 30 to 40 day MONIT extraction of the real
cluster and the replay in `results/replay/comparison.md`. Reproduction:
`docs/REPRODUCE.md`.

- Demand exists and waits are long. 1948 GPU and MIG requests from 144 users
  in 40 days. Under the current policy the NVL pool wait is p95 930 minutes.
  29 single-GPU requests never started at all.
- Allocations are held idle. 62777 idle-held GPU-hours per month. For
  allocations with DCGM data the median idle fraction is 100 percent. Only 20
  percent of GPU-idle hours carry any CPU work. Hold duration is median 8.2 h
  but p95 326 h.
- Requests carry no maximum time. A member requests an allocation without
  declaring how long it is needed, so nothing bounds a hold and the scheduler
  cannot estimate when a GPU frees. This is a root gap: it is the reason the
  cluster has no queue and no back pressure.
- There is no back pressure on hoarding. The current policy has multi-GPU
  holds that run for weeks. The longest single multi-GPU hold in replay is
  665 h. This is the direct result of no walltime limit and no queue.
- Users hold sessions defensively. 57 of 112 users ran more than one pod at
  the same time, which is insurance behaviour against not getting a GPU back.

The one line conclusion: because a user cannot be sure to get a GPU back,
users never release GPUs, so the cluster looks full while sitting idle.

## 3. Design principles

These are the principles this proposal implements, in priority order. Lower
number wins on conflict.

1. Every member can hold one interactive session of at most one GPU. A member
   who needs more interactive GPUs or nodes can get them by agreement with the
   working-package leader, at the cost of priority.
2. The interactive tier is reliable. A member does not have to release the
   session in the evening. A member can be confident of getting one in the
   morning.
3. The interactive session is terminated automatically after an idle period.
4. Any larger request (more than one GPU, or long-running) is a batch request
   with a maximum allocation time, so training runs finish and the GPUs turn
   over.
5. Accounting is per working package over the last 7 days.
6. As a working package uses more in that window, its members get lower
   priority for new allocations beyond the one guaranteed session. Priority
   plus accounting together stop anyone holding resources forever, while still
   allowing multiple or larger long-lived allocations when capacity is free.
7. Specific use cases can get a reserved slice of the farm, only after PMC
   approval.
8. A shared entry node lets any member be inside the cluster without allocating
   resources. Its cores are shared by all users. A process that pins a CPU at
   100 percent for more than a set time is killed.

## 4. How real centers do this

This section is the deep-research basis. Every mechanism our principles need
is standard, with a named config knob and a center that runs it.

### 4.1 A guaranteed interactive tier is a dedicated partition with a per-user cap

The way to guarantee "a GPU in the morning" is a partition reserved for
interactive work with a per-user cap of one. The partition size divided by the
cap is the number of members served at once.

- NERSC Perlmutter runs an `interactive` QOS backed by a standing node
  reservation. It is engineered to deliver nodes within 6 minutes, and the
  request auto-cancels after 6 minutes if none are free, so it never queues
  forever and never hoards. Per-user cap 4 nodes, walltime 4 h, high priority.
  (docs.nersc.gov/jobs/policy, docs.nersc.gov/jobs/interactive)
- Harvard FASRC runs a no-cost interactive tier (`gpu_test`) that is exempt
  from fair-share accounting, capped at 2 jobs and a short walltime.
  (docs.rc.fas.harvard.edu/kb/fairshare)
- Princeton Della offers a single-GPU `mig` partition as the interactive
  target. (researchcomputing.princeton.edu/systems/della)

The sizing rule for NGT: reserve N GPUs for the interactive tier, cap each
member at one, and N is the number of members served at once. N is set near
the expected simultaneous morning demand, not the full roster of about 130
members. NERSC reserves a fraction of the machine, not all of it.

### 4.2 A hard walltime limit is the prerequisite for a working queue

Every production GPU queue enforces a maximum walltime, and applies a default
when the user omits it. Slurm states plainly that backfill "is difficult
without reasonable time limit estimates" because the start time of a pending
job depends on when running jobs end. With unbounded jobs a scheduler cannot
estimate turnaround, so a queue degenerates back to first come first served.
This is exactly the NGT failure mode.

Real maxima: NERSC Perlmutter `regular` 48 h; OLCF Frontier 2 to 24 h by job
size; ALCF Polaris `preemptable` up to 72 h; JUWELS booster 24 h; LUMI
standard-g 48 h. The safe pattern is a hard cap plus a default, and requeue on
timeout with a warning signal so the job can checkpoint (Slurm
`--signal=TERM@120`, `--requeue`).

### 4.3 Interactive small vs batch large is an explicit, enforced split

The universal rule is that interactive work is single and small and time or
idle bounded, while heavy or multi-device work must go to batch with a hard
walltime. The justification is physical: an interactive session is held for
human latency, so it must be small and reclaimable, while training is
throughput work that a scheduler can pack, preempt, and turn over. NERSC
encodes this as separate QOS (`interactive` short and capped vs `regular` long
and exclusive). University policies cap the short queue at 1 GPU and force
multi-GPU work to batch.

### 4.4 Idle sessions are culled, and the workspace is kept

Every managed interactive platform culls idle sessions. The important property
is that culling frees the GPU but preserves the workspace, so the user loses
no data. Signals and typical values:

- Kubeflow Notebooks: `ENABLE_CULLING`, `CULL_IDLE_TIME` (default 1440 min),
  signal is kernel activity; culling scales the pod to zero but keeps the PVC.
- JupyterHub idle culler: `cull.timeout` (common default 3600 s),
  `cull.maxAge` for a hard cap.
- NVIDIA Run:ai: idle GPU time limit per workload type, signal is GPU
  utilization measured in 30 s windows.

For NGT the right signal is GPU utilization near zero (we already extract
this from DCGM on full GPUs), with a timeout long enough to survive overnight
so an owned-but-idle session is not killed too early.

### 4.5 Accounting per project over a rolling window, priority falls with usage

This is the exact mechanism principles 5 and 6 ask for. It is the Slurm
fair-share factor.

- The classic fair-share factor is `F = 2^(-U/S)`, where `U` is the
  normalized decayed usage of the account and `S` is its assigned share. When
  usage equals share, `F = 0.5`. Under-use gives `F` above 0.5. Over-use
  drives `F` toward 0. So the more a working package used recently, the lower
  the priority of its next request. This is principle 6, verbatim.
- The "last 7 days" is a standard knob. Slurm `PriorityDecayHalfLife` defaults
  to exactly 7 days: usage now counts full, at 7 days half, at 14 days a
  quarter. (slurm.schedmd.com/priority_multifactor)
- A decayed window is preferred over a hard 7-day reset. A hard reset creates
  a cliff: at the reset instant every project's usage jumps to zero, the order
  flips, and users time submissions around the boundary. Exponential decay is
  a rolling window with no cliff and nothing to game. Recommendation: account
  the "last 7 days" as a 7-day half-life decay, not a rectangular window.
- The tree form (Slurm Fair Tree, the default) guarantees that if working
  package A has a higher factor than B, every member of A ranks above every
  member of B. A heavy package cannot interleave its users above a light
  package. This is what makes per-WP fairness hold across users.
- This is in production for GPUs today. The KU cluster runs `f =
  2^(-usage/shares)` with a 7-day half-life. NVIDIA Run:ai time-based
  fair-share uses a one-week window by default. Harvard FASRC uses a 3-day
  half-life per lab.

On the theory: the multi-resource generalization is Dominant Resource
Fairness (Ghodsi et al., NSDI 2011). When GPUs are the one binding resource,
which is the NGT case, DRF collapses to weighted max-min on GPU count, so we
do not need the full DRF vector: equalizing weighted per-WP GPU shares gives
the identical result. DRF also gives strategy-proofness, so a package cannot
win more GPUs by padding its CPU or memory requests.

### 4.6 Burst when idle, yield when busy

The behaviour "keep more than one long-lived session when capacity is free,
but drop in priority as your package uses more" is the guaranteed-vs-
opportunistic split every hyperscaler runs. Google Borg has priority bands
(production above batch above best-effort); production is never preempted by
production, best-effort runs in reclaimed slack and is killed first when
production needs it (Verma et al., EuroSys 2015). NVIDIA Run:ai is the closest
match to this proposal: deserved quota is guaranteed and non-preemptible, over
quota is only for preemptible workloads, and when a project below its share
needs GPUs the scheduler preempts over-quota work newest first.

Mapping to NGT: the one guaranteed session per member is the protected tier
(never preempted). Extra concurrent or larger allocations are opportunistic
and run while GPUs are free. The fair-share factor sets the order in which
those opportunistic allocations are admitted and, if the cluster fills, the
order in which they yield.

### 4.7 Reservations for specific use cases, by consensus

Centers reserve a slice for a group or an event, and they gate it behind
approval. Slurm `scontrol create reservation` scopes nodes to named accounts
or users, with recurring flags. On Kubernetes the same is a labelled node set
with a taint, exposed to one team through a dedicated Kueue `ResourceFlavor`
and `ClusterQueue`. The governance norm is explicit: Utah CHPC requires the
PI to request it and the group allocation to cover it; Caltech caps
reservations at 2 weeks and 3 per year. This supports principle 7: reserve
only after PMC approval, time-boxed.

## 5. The proposed NGT policy

### 5.1 Interactive tier (principles 1, 2, 3)

- One interactive session per member, at most one GPU. A validating webhook
  counts the member's running interactive pods and rejects a second. A
  per-user cap of one on a dedicated interactive tier is the NERSC Perlmutter
  `interactive` QOS model (docs.nersc.gov/jobs/policy).
- Opening a new session supersedes the member's old one (swap at start). The
  system never terminates another member's work to make room.
- A member who needs more than one interactive GPU, or more than one node, can
  request it with the working-package leader's agreement. This allocation is
  not part of the guarantee. It is charged to the working package's 7-day
  fair-share, so it lowers the package's priority for further allocations, and
  it yields first when the cluster fills. This is the burst-when-idle,
  yield-when-busy tier (4.6) applied to interactive work: the WP leader gates
  it and the fair-share cost self-limits it. It is in production as the NVIDIA
  Run:ai over-quota tier, where in-quota work is guaranteed and over-quota work
  is preemptible.
- The tier is sized to be reliably available. Reserve a headroom of GPUs and
  MIG slices for it, spread over at least two physical nodes so one cordon
  does not empty it. Because most interactive work fits a MIG slice, a single
  H100 serves several members. A dedicated single-GPU interactive slice
  target is in production at Princeton Della (`mig` partition,
  researchcomputing.princeton.edu/systems/della) and Harvard FASRC
  (`gpu_test`, docs.rc.fas.harvard.edu).
- Idle culling on GPU utilization near zero, with an overnight-tolerant
  timeout. Culling on GPU utilization is the NVIDIA Run:ai idle GPU time
  limit (developer.nvidia.com/blog, run-ai-docs.nvidia.com). Culling frees the
  GPU and keeps the user's persistent volume, so the member does not have to
  release the session before leaving and finds the tier free in the morning.
  Scaling an idle notebook to zero while keeping its volume is the Kubeflow
  Notebooks culler (`CULL_IDLE_TIME`, preserves the PVC).

### 5.2 Batch tier (principle 4)

- Any request for more than one GPU, or any long-running request, is a batch
  job with a declared maximum allocation time. It starts, runs, and exits at
  the end. There is no interactive multi-GPU hold. The maximum time is
  mandatory at submission; if omitted, a default cap applies. This closes the
  present-day gap that requests carry no time at all (2). A hard walltime on
  batch GPU work is universal: NERSC Perlmutter 48 h, OLCF Frontier 2 to 24 h,
  ALCF Polaris up to 72 h, JUWELS 24 h, LUMI 48 h; a default when the user
  omits the time is the Slurm `DefaultTime`.
- A hard walltime cap with a default, requeue on timeout, and a warning signal
  before the kill so the job checkpoints. This is the standard training
  pattern and it is what makes the queue and fairness work at all (4.2). The
  requeue-and-checkpoint pattern is documented by Slurm (`--requeue`,
  `--signal=TERM@120`) and by NVIDIA Run:ai for preemptible training.

### 5.3 Accounting and priority (principles 5, 6)

- Charge each delivered GPU-hour to the member's working package. Charge is
  wall time times GPU count times a per-model factor, applied at the end,
  which is the standard service-unit model (NERSC, OLCF, ALCF, TACC all charge
  node-hours or GPU-hours times a per-queue factor).
- Account per working package over the last 7 days as a 7-day half-life decay
  (4.5), not a hard reset. A 7-day half-life is the Slurm `PriorityDecayHalfLife`
  default (slurm.schedmd.com), is run verbatim by the KU Community Cluster
  (docs.crc.ku.edu), and is the NVIDIA Run:ai time-based fair-share default
  one-week window (developer.nvidia.com/blog).
- Order new non-guaranteed requests by the per-WP fair-share factor `F =
  2^(-U/S)`, and within a package by the member's own decayed usage. This
  factor is the Slurm classic fair-share formula, in production per lab at
  Harvard FASRC and per account at the KU cluster. The one guaranteed
  single-GPU session per member is exempt from this ordering and is always
  served. Exempting the guaranteed interactive tier from fair-share
  accounting is the FASRC `gpu_test` model.
- Renormalize the target shares `S` over the working packages that have live
  demand. This is a measured requirement, not a detail: WP4 consumed nothing
  in the real month, so a fixed 30/30/30/10 target makes every fair-share
  variant chase an unreachable 10 percent and show a worse WP deviation than
  the current policy in replay. Targets must renormalize over active packages.

### 5.4 Reservations (principle 7)

Reserve part of the farm for a specific use case only after PMC approval and
time-boxed, as a labelled, tainted node set exposed through a dedicated queue.
Reservations gated behind PI approval and time-boxed are the norm at Utah CHPC
(PI request, allocation must cover the reservation) and Caltech HPC (max 2
weeks, 3 per year).

### 5.5 Shared entry node (principle 8)

Provide one shared node that any member can enter without an allocation. Its
purpose is presence in the cluster, not computation: editing code, submitting
batch jobs, moving data, small tasks. All cores are shared, there is no GPU and
no reserved slot. A fair-use limit kills any process that holds a CPU at 100
percent for more than a set time, so one user cannot degrade the node for the
others. This is the standard HPC login node with a fair-use enforcer: every
center runs login nodes that forbid heavy compute, and the enforcement is
cgroups-based. The Arbiter2 tool (Utah CHPC) does exactly this: it caps each
user's CPU share on a shared node, applies escalating penalties, and kills the
processes of a user who keeps exceeding the limit. It runs at Utah CHPC and
Brown OSCAR.

This node also removes a driver of GPU hoarding measured on NGT: today a member
holds a GPU pod partly to keep a foothold in the cluster (57 of 112 users ran
more than one pod at once). A free shared entry point removes that reason, so
GPU pods are held only for GPU work.

## 6. Evidence that this works on the real trace

Replayed on the measured month (`results/replay/comparison.md`, column
`batch_multi_queue`, the batch-only realization of this policy), against the
current policy (`fcfs_pending`):

| metric | current (fcfs) | proposed (batch queue) |
|---|---|---|
| wait p95 (min) | 965 | 0 |
| 1-GPU tier served within 15 min | 88% | 100% |
| 1-GPU sessions never started | 29 | 0 |
| NVL idle-held GPU-h | 37978 | 14051 |
| requests satisfied | 95.7% | 99.5% |
| longest multi-GPU hold (h) | 665 | batch, exits at end |

The interactive guarantee is met for 100 percent of the 866 single-GPU dev
sessions with zero wait, idle holding is cut by roughly two thirds, and no
other member's work is terminated. The only terminated allocations are a
member's own superseded sessions.

One honest caveat from the replay: because multi-GPU work becomes batch that
exits at the end, the batch column delivers about 47 percent of the "dev jobs
done" measure under fixed-behaviour replay. This is an artifact of replaying
recorded behaviour that assumed indefinite interactive holds; with a declared
walltime users checkpoint and resubmit, which the fixed trace cannot model. It
is fair across all intervention variants and does not change the ranking.

## 7. Implementation on the NGT stack

Assessed in `docs/implementability.md`. Nothing here is enforceable from user
space; all of it needs cluster-admin deployment. The stack already runs an
admission webhook and an idle-culling alert, so the pieces are in class.

- Queue, per-WP quota, borrowing and reclaim: Kueue. One `ResourceFlavor` per
  GPU model, one `ClusterQueue` per working package in one cohort.
  `nominalQuota` is the WP guaranteed share, `borrowingLimit` and
  `lendingLimit` bound greedy tenants, `fairSharing.enable: true` gives the
  weighted dominant-resource-share ordering, and `reclaimWithinCohort` lets a
  WP take back its guarantee. This is the k8s realization of 4.5 and 4.6. The
  guaranteed-quota-plus-over-quota-fair-share model it implements is the one
  NVIDIA Run:ai runs in production, open-sourced as the KAI Scheduler.
- Batch walltime: native Job `activeDeadlineSeconds` from the declared time,
  plus `ttlSecondsAfterFinished` for cleanup.
- One interactive session per member with swap: a validating webhook plus a
  small controller that deletes the superseded session. Same complexity as the
  webhook already deployed.
- Interactive single-GPU cap: `LimitRange max.nvidia.com/gpu: 1` and a
  `ResourceQuota` on `requests.nvidia.com/gpu` in the interactive namespace.
- 7-day fair-share accounting: a small controller that reads the monitoring
  backend (this study proved the data is queryable with user privileges only)
  and feeds the decayed per-WP usage into the queue ordering.
- Shared entry node: one node open to all members, no GPU, with a per-user
  cgroup CPU cap and a fair-use killer. Kubernetes limits CPU per user through
  cgroups already; the sustained-100-percent killer is the Arbiter2 model.

Two constraints from the hardware and telemetry:

- L40S GPUs do not support MIG (Ada Lovelace lacks the hardware). Small
  guaranteed interactive slices use MIG on the H100 pools; on L40S use MPS or
  time-slicing instead.
- The cluster exports no per-MIG-slice utilization today. Any idleness signal
  on MIG sessions is blind until DCGM MIG profiling is enabled. Full-GPU pools
  are fully covered. This is the one hard telemetry gap.

## 8. Parameters for the PMC to fix

- N, the number of GPUs reserved for the interactive tier, and its split
  between full GPUs and MIG slices.
- The idle-cull timeout for interactive sessions.
- The batch walltime cap and its default.
- Whether interactive allocations beyond the one-GPU guarantee need a hard cap,
  on top of the working-package leader's agreement and the fair-share cost.
- The CPU-time threshold and the kill time for the shared entry node.
- The fair-share target shares, and the rule for renormalizing over active
  working packages.
- The per-model charge factors.
- The fair-share half-life. The proposal recommends 7 days, matching the
  request and the Slurm and Run:ai defaults.

## 9. References

Fair-share and accounting: Slurm multifactor priority, classic fair-share,
Fair Tree, and slurm.conf (`PriorityDecayHalfLife`, `PriorityUsageResetPeriod`,
`TRESBillingWeights`), slurm.schedmd.com. KU Community Cluster fairshare,
docs.crc.ku.edu. NVIDIA Run:ai time-based fair-share,
developer.nvidia.com/blog. Harvard FASRC fairshare, docs.rc.fas.harvard.edu.

Interactive tier and reservations: NERSC Perlmutter policy and interactive
jobs, docs.nersc.gov. ALCF Polaris, docs.alcf.anl.gov. LUMI partitions,
docs.lumi-supercomputer.eu. JUWELS batch system, apps.fz-juelich.de. Slurm
reservations, slurm.schedmd.com/reservations. Utah CHPC and Caltech HPC
reservation policies. Shared entry node fair-use enforcement: Arbiter2
(github.com/chpc-uofu/arbiter2, Utah CHPC, PEARC 2019), also at Brown OSCAR.

Theory: Ghodsi et al., "Dominant Resource Fairness", NSDI 2011. Hindman et
al., "Mesos", NSDI 2011. Verma et al., "Large-scale cluster management at
Google with Borg", EuroSys 2015. Mahajan et al., "Themis", NSDI 2020.
Narayanan et al., "Gavel", OSDI 2020. Zhao et al., "HiveD", OSDI 2020.

Kubernetes enforcement: Kueue (kueue.sigs.k8s.io), Volcano (volcano.sh),
NVIDIA KAI Scheduler, NVIDIA k8s-device-plugin and GPU Operator (MIG,
time-slicing, MPS), Kubernetes DRA (kubernetes.io), Kubeflow and JupyterHub
idle culling.
