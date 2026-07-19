# Can the candidate policies actually be enforced on the NGT cluster?

Assessment against the stack as it runs today: users create bare pods via
kubectl; an admission webhook already mutates requests (nodeSelector
tolerations, full-slot CPU injection); an idle-culling mechanism already
exists (alert threshold 24 h); DCGM telemetry covers full GPUs but exports
no per-MIG-slice utilization (measured: zero slice series); there is no
queueing system deployed. Everything below requires cluster-admin
deployment; nothing is enforceable from user space.

## Off the shelf (standard Kubernetes components)

- Batch queue with priorities for multi-GPU jobs: Kueue (or Volcano).
  Multi-GPU work becomes a Job admitted through a queue; a validating
  webhook rejects bare multi-GPU pods. Standard, well supported.
- WP fair-share quotas and intra-WP recycling (P2, P4): Kueue
  ClusterQueues per WP in one cohort with borrowing. Needs the user-to-WP
  mapping maintained as namespace labels (the mapping exists; keeping it
  current is process, not technology).
- Exit-at-end enforcement (P3 as batch): Job semantics plus
  activeDeadlineSeconds from a declared walltime. Native.
- P1 guarantee headroom: a dedicated interactive quota partition (or the
  MIG pools, spread over two physical nodes for cordon redundancy).
- Gang scheduling for NVLink jobs: supported by Kueue/Volcano.

## Small custom development

- One interactive session per member with swap-at-start: a validating
  webhook counting the member's running interactive pods plus a tiny
  controller that deletes the superseded session. Same complexity class
  as the webhook already running on the cluster.
- Monthly interactive multi-GPU budget: no Kubernetes-native object
  expresses cumulative GPU-hours per month (quotas are concurrent, not
  cumulative). Needs a small accounting controller that queries the
  monitoring backend and feeds the admission decision. The accounting
  data demonstrably exists (this study extracted it with user privileges
  only).
- Charge factors (P3 accounting): pure reporting from existing
  telemetry; no scheduler involvement.

## Genuinely blocked today

- Idle DETECTION on MIG slices: the cluster exports no per-slice
  utilization, so any idleness-based mechanism (reclaim, idle culling
  tighter than pod-level heuristics, idle accounting) cannot see MIG
  sessions until DCGM MIG profiling is enabled. Full-GPU pools are fully
  covered.

## Architecturally awkward

- The P6 planning cycle (admission decisions at fixed epochs): no
  standard component implements timed admission rounds; it would be a
  bespoke controller pausing and releasing the queue. It is also the
  worst performer in every replay. Recommend deprioritizing.

## Bottom line

The preferred realization (one interactive session per member with swap,
batch-only multi-GPU beyond a monthly interactive budget, priority queue
with WP fair share) is implementable with Kueue plus one custom webhook
and one small accounting controller. The single hard technical gap is
MIG-slice utilization telemetry; the practical gap is that all of it
needs cluster-admin engagement.
