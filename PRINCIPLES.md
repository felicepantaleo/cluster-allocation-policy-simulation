# NGT allocation principles (priority order)

These are the governing principles for the allocation policy the PMC study
evaluates, in strict priority order: when two principles conflict, the lower
number wins. Each principle maps to a simulation mechanism and a metric, so
every policy candidate is scored against all five.

## P1. Guaranteed interactive GPU per member

Every member of NGT should be granted 1 GPU at any time, with little waiting
time.

- Simulation: single-GPU (or MIG-slice) member requests form a guaranteed
  tier that the policy must serve ahead of everything else, regardless of
  working-package shares.
- Metric: p95 wait of single-GPU member requests. Provisional target: below
  15 minutes (PMC to confirm the number). The sim also reports the capacity
  cost of the guarantee: how many GPUs must be effectively set aside.

## P2. Fair GPU-hour split across working packages

GPU-hours are distributed across the four NGT working packages as
WP1 30%, WP2 30%, WP3 30%, WP4 10%.

- Simulation: every user (and every production job, see P5) belongs to a WP;
  delivered GPU-hours are charged to that WP.
- Metric: delivered charged share per WP over the measurement window vs the
  30/30/30/10 target; report the maximum absolute deviation and the share
  time series. Fairness is across WPs first; Jain across users applies
  within a WP.

## P3. Multi-GPU allocations are time-limited and charged with a model factor

Allocating more than one GPU is possible, but for a limited amount of time.
Charged GPU time is wall time multiplied by the number of GPUs and by a
correction factor that depends on the GPU model.

- Simulation: a per-pool charge factor (config `gpu_charge_factor`, current
  values are placeholders until the PMC fixes them) and a maximum duration
  for multi-GPU allocations (policy parameter `multi_gpu_max_h`).
- Metric: charged GPU-hours per WP and per user; violations count as policy
  failures (a compliant policy never lets a multi-GPU allocation exceed the
  cap).

## P4. Intra-WP recycling when the cluster is full and shares are fair

When all resources are allocated at the fair split and the cluster is full,
a removed allocation from a WP gives access to another allocation from the
same WP.

- Simulation: on release under full-cluster fair-share conditions, the
  scheduler first considers Pending requests from the same WP before global
  order applies.
- Metric: fraction of handoffs that stayed within the releasing WP while the
  cluster was full; WP share stability during saturated periods.

## P5. Big production jobs run on behalf of the working package

Production jobs on big allocations (more than one GPU) are submitted on
behalf of the WP, not the individual.

- Simulation: multi-GPU production requests carry the WP identity as the
  accountable entity; their charge goes to the WP budget and they compete in
  the WP's share, not in the member's personal guarantee (P1 is per member
  and single-GPU only).
- Metric: share accounting per P2 with production jobs attributed to WPs;
  per-user metrics exclude WP-attributed production jobs.

## P6. Declared allocations and a tiered planning cycle

Big allocations and job launches for the following 24 hours are declared
before 12:00. At 12:00, given the priorities and the quotas, the order of
the jobs and allocations is decided. Shorter jobs follow the same pattern at
higher frequency: with a max allocation time of 8 hours they can be
submitted at 3 decision points per day, and so on down the tiers (the
shorter the allocation cap, the more frequent the submission windows).

- Simulation: a planning-cycle policy that replaces continuous first-fit
  with batch decisions at fixed epochs. Tiers are (max allocation duration,
  decisions per day): (24 h, 1 at 12:00), (8 h, 3), and further tiers as
  configured. At each epoch the batch is ordered by the principles above:
  P1 guarantees first, then WP quota headroom (P2), then declaration order.
  The P1 interactive single-GPU tier stays continuous and never waits for an
  epoch.
- Metric: wait measured from the declaration deadline (not from submission)
  for declared jobs; fraction of declared jobs granted at their epoch;
  quota adherence per epoch; and the same global wait, utilization and
  share metrics as every policy, A/B against baseline.

## Consequences for the policy roadmap

The candidate that implements these principles is a WP-quota fair-share
scheduler with an interactive guarantee tier, a multi-GPU time cap, charge
factors, and intra-WP recycling. The originally listed candidates (idle
reclaim, dev-tier separation, gang scheduling, queue plus quota) become
building blocks or comparison points, not ends in themselves. Baseline for
every comparison remains the current FCFS-via-Pending with gaming.
