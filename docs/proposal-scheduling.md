# Happy Cluster: a fair-share scheduling policy for the NGT GPU cluster

## 1. Summary

The NGT GPU cluster is a shared pool of GPU servers used by the whole project. A member gets GPUs by starting a session, which is a container (in Kubernetes terms, a Pod). Today the cluster runs first come first served, with no controls: there is no waiting line (no queue), no limit per group (no quota), and no limit on how long a session runs (no time limit). The result is that the cluster looks full but is mostly idle, members wait a long time, and some never get a GPU.

This proposal replaces the free-for-all with three rules that large GPU centres already use. First, every member always has one GPU, at no cost, released automatically when it sits idle. Second, any GPU beyond that first one is time-limited and costs the member's project some scheduling priority. Third, priority is shared fairly between the project's working packages (the sub-teams WP1 to WP4, across which the budget is split), based on how much each used in the last week.

Nothing here is new. The same mechanisms run at national supercomputers, at university HPC clusters, and inside commercial GPU schedulers. This document explains the problem we measured on NGT, states the policy, and shows on the real usage record that it fixes the problems. Appendix A documents that each rule is already standard practice at other centres. Numbers in square brackets point to the References.

## 2. The measured problem

All numbers come from the cluster's own monitoring (the MONIT system), extracted over 30 to 40 days.

- The demand is real and the waits are long. In 40 days there were 1948 GPU requests from 144 users. On the busiest pool (the H100 NVL GPUs) the slowest 5 percent of requests waited more than 930 minutes, about 15 hours. 29 requests for a single GPU never ran at all.
- GPUs are held but not used. Members held 62777 GPU-hours per month without computing on them; one GPU-hour is one GPU reserved for one hour. For the allocations where we can read the GPU meter (NVIDIA's DCGM telemetry), the typical one used the GPU essentially not at all: the median fraction of idle time is 100 percent. Only 20 percent of that idle time had even the CPU busy. A typical session was held for about 8 hours, but the longest 5 percent ran past 326 hours, about two weeks.
- Requests carry no time limit. A member asks for GPUs without saying for how long. Nothing bounds the hold, and the scheduler cannot tell when a GPU will free up. This one gap is the root cause: it is why the cluster has no working queue and nothing pushes back on over-use.
- Nothing pushes back on hoarding. Some multi-GPU sessions ran for weeks; the longest reached 665 hours. With no time limit and no queue, this is the expected outcome.
- Members hold GPUs defensively. 57 of 112 users ran more than one session at the same time, keeping spares in case they cannot get a GPU back later.

The cause and effect is simple. Because a member cannot be sure of getting a GPU back, no one lets go. So the cluster looks full while sitting idle.

## 3. Design principles

These are the rules this proposal implements, in priority order. When two conflict, the lower number wins.

1. Every member can hold one GPU at no priority cost, the free GPU. The member does not have to release it in the evening and can be confident of getting one in the morning. The work on it can be interactive or a job.
2. Every GPU beyond that first free one is charged to the member's working package, so it costs priority for all members of that package.
3. The free GPU is released automatically after a period of sitting idle.
4. Every allocation beyond the first GPU declares a maximum duration, at most 7 days, defaulting to 8 hours if none is given. It can be renewed without limit; each renewal competes again at the package's current priority. It may span several GPUs or several servers, for example a job that runs across many GPUs at once. A member can also request that its servers sit on one high-performance network switch and carry matching network cards, and add a second server, with or without a GPU, on the same switch, so the two can exchange data over the fast RDMA fabric.
5. Usage is accounted per working package over the last 7 days.
6. The more a working package used in that window, the lower the priority its members get for new allocations beyond the one free GPU. Priority and accounting together stop anyone holding resources forever, while still allowing more or larger long-lived allocations when the cluster has room.
7. A specific use case can get a reserved slice of the cluster, only after PMC approval.
8. A shared entry server lets any member be inside the cluster without holding an allocation. Its cores are shared by everyone. Any process that pins a core at 100 percent for longer than a set time is stopped.

## 4. The proposed NGT policy

### 4.1 The free GPU (principles 1, 2, 3)

- Each member may hold one free GPU, at no priority cost. The work on it can be interactive or an unattended job. An automatic admission check (a validating webhook) counts the member's free GPUs and rejects a second one.
- Starting a new free GPU replaces the member's old one (a swap at the moment of starting). The system never stops another member's work to make room.
- The free tier is sized to be reliably available. Set aside a headroom of GPUs and MIG slices for it, spread over at least two physical servers so that taking one server down for maintenance does not empty the tier. Because most single-GPU work fits on a fraction of a GPU (a MIG slice, one of the smaller isolated GPUs that a modern NVIDIA GPU can be split into), one H100 GPU serves several members; on our cluster these slices are managed by standard components [26, 27, 28]. A free single-GPU tier of this kind runs at Harvard FASRC [7].
- The free GPU is released when its GPU use stays near zero for a set time, chosen long enough to survive overnight. Releasing it frees the GPU but keeps the member's saved files (the persistent volume), so the member need not release it before leaving and finds one free in the morning. This is the same idle-GPU timeout and notebook culling that other platforms run [6, 29]. A member who needs guaranteed, uninterrupted GPU time uses a paid, time-limited allocation instead (4.2).

### 4.2 Allocations beyond the first GPU (principles 2, 4, 6)

- There is no interactive-versus-job distinction. Any allocation beyond the free GPU, whether a notebook or a training run, follows the same two rules: it declares a maximum duration, and it costs priority.
- An allocation may span several GPUs or several servers, for example a training run spread across many GPUs that must communicate (an MPI or NCCL job). A multi-server allocation is started all-or-nothing (called gang scheduling): either all its servers are granted together or none is, so no part sits idle waiting for the rest. The scheduler also places the parts close together on the fast interconnect (NVLink within a server, RDMA between servers). Standard schedulers do this: one admits a workload all-or-nothing, another groups the parts with a minimum-members rule [23, 24].
- A request can also pin its servers to one named high-performance network switch, and the member can then add more servers on the same switch, including a server with no GPU, for example a node that only loads or preprocesses data. Servers under one switch have the highest bandwidth and lowest latency between them. They can exchange data over the RDMA fabric using GPUDirect RDMA, in which the network card moves data straight in and out of GPU memory without a detour through the CPU [32].
- The switch alone is not enough for RDMA: the two servers must also carry compatible network cards. NGT has two kinds of fabric. On InfiniBand every card is from one vendor (NVIDIA), so any pair does RDMA. On RoCE (RDMA over Ethernet) the cards are a mix of NVIDIA and Broadcom; plain TCP/IP works between any pair, but RDMA, and therefore GPUDirect RDMA, works only between cards of the same vendor. So a request that needs RDMA must state the fabric and the card vendor it requires, and the scheduler must match both the switch and the vendor, not the switch alone. This is standard topology-aware scheduling, where nodes carry labels for the switch hierarchy [31]; the card vendor is simply one more label to match.
- The same rule governs MPI across GPU flavours, for example an NVIDIA H100 worker paired with an AMD MI300X worker: the workers can be launched together, but whether they talk over RDMA or fall back to TCP/IP depends on the switch and network-card match above. Running MPI across GPU flavours is under investigation on NGT [33].
- A request declares how long it needs, at most 7 days, and defaults to 8 hours if the member gives no value. This closes the present-day gap that requests carry no time at all (Section 2). A maximum plus a default is universal on GPU queues [8, 10].
- The 7-day maximum equals the one-week accounting window (4.3) on purpose: an allocation cannot outlast one window without renewing. It can be renewed without limit, but each renewal competes again at the package's current priority. So a package that used a lot in the last week renews at a lower priority. This is what stops "allocations that live forever": not a forced shutdown, but a rising cost in priority. Renewing a time-limited allocation is the same "resubmit when the time runs out" pattern every centre uses, with a warning signal before the end so long work can checkpoint.
- The allocation is charged to the member's working package for the whole time it is held, whether the GPU is busy or not (see 4.3). So holding it costs the whole package priority, and it is the first to yield when the cluster fills. This is the "use it while it is free, give it back when others need it" tier described in Appendix A.6: the priority cost limits it by itself, so no approval step is needed. It is exactly the over-quota tier of a commercial scheduler, where in-quota work is guaranteed and over-quota work can be stopped [6, 25].

### 4.3 Accounting and priority (principles 2, 5, 6)

- Charge each held GPU-hour to the member's working package, except each member's one free GPU. The charge is the time held, times the number of GPUs, times a per-model weight (a GPU-hour on a fast GPU counts more than on a slow one). This is the standard "service unit" used at major centres [8]. The charge is on the time held, not the time used, so holding a GPU idle still costs the package priority. This is the key point: it replaces "reclaiming" idle GPUs by force with a price on holding them. The 62777 idle GPU-hours per month we measured would carry a cost, without the cluster ever having to stop anyone's work.
- Account each working package's usage over the last 7 days as a 7-day half-life decay (see Appendix A.5), not a weekly reset [4, 5, 6].
- Order requests beyond the free GPU by the per-package fair-share score `F = 2^(-U/S)`, and, within a package, serve the member who used least first. This is the standard fair-share formula [2, 5, 7]. Each member's one free GPU is outside this ordering and is always served, the same exemption a free interactive tier gets elsewhere [7].
- Set each package's target share `S` only over the packages that actually have demand. This matters in practice: in the measured month WP4 used nothing, so a fixed target of 30/30/30/10 would make the ordering chase a 10 percent share that no one is asking for. The targets must be renormalised over the packages with live demand.
- The requested duration does not change the priority. Priority comes from the package's recent usage `U`, not from how long a request asks for. One 7-day request for 2 GPUs and seven daily 24-hour requests for 2 GPUs deliver the same 336 GPU-hours, so they cost the same priority. The only difference is that a shorter request is easier to fit into a gap and so tends to start sooner (see backfill, Appendix A.2). This rewards honest, short requests and gives no advantage to splitting a job up. A long request gets no free ride either: it is charged the whole time it is held, so its package's priority keeps falling while it runs, and it is the first to yield when the cluster fills and its package is over its share.

What `U` and `S` are, and how to set them for NGT:

- `S` is each working package's target share. As an example, a split of 30/30/30/10 across WP1 to WP4, renormalised over the packages with live demand so the shares add to 1. With WP4 idle, the active split is WP1, WP2, WP3 at 0.333 each. The real split should follow the budget or person-power assigned to each package.
- `U` is each package's share of the GPU-hours actually delivered in the last 7 days, with the 7-day half-life weighting, normalised so the shares add to 1. The free GPU and the excluded pools (the STEAM Academy training GPUs) are left out of `U`.
- Recompute both every few minutes (Slurm recomputes every 5 minutes by default).

A worked example, using the measured month. In that month WP3 ran heavy: it took about 0.37 of the delivered GPU-hours against a 0.30 target. Take a snapshot with equal targets `S = 0.333` and recent usage `U` of {WP1 0.28, WP2 0.30, WP3 0.42}:

| Working package | Target S | Recent usage U | Priority score F = 2^(-U/S) |
|---|---|---|---|
| WP1 | 0.333 | 0.28 | 0.56 |
| WP2 | 0.333 | 0.30 | 0.54 |
| WP3 | 0.333 | 0.42 | 0.42 |

A new request from WP1, which is under its share, beats one from WP3, which is over its share, by 0.56 to 0.42. So WP3's heavier recent use puts its next request behind WP1 and WP2. As WP3 slows down, its `U` decays over the following week and its score climbs back. The free GPU is exempt, so a member of WP3 still gets one GPU with no wait. These numbers are only an illustration; the real `U` is computed from the usage record.

### 4.4 Reservations (principle 7)

Reserve part of the cluster for a specific use case only after PMC approval, and only for a fixed period. In practice this is a set of servers labelled for one team and reachable only through that team's queue. Requiring approval and a time limit is the norm at other centres, where the group's leader must request it, its allocation must cover it, and each reservation is limited to a fixed period such as two weeks [14, 15].

### 4.5 Shared entry server (principle 8)

Provide one shared server that any member can log in to without holding an allocation. Its purpose is to be inside the cluster, not to compute: editing code, submitting jobs, moving data, small tasks. Its cores are shared by everyone, it has no GPU, and it reserves nothing. A fair-use limit stops any process that pins a core at 100 percent for longer than a set time, so one person cannot slow the server for everyone else. This is the standard "login node" that every centre runs, with the usual protection that heavy compute is not allowed there. A dedicated tool does exactly this: it caps each user's share of the cores, warns, and then stops the processes of anyone who keeps going over [16, 17].

This server also removes one cause of GPU hoarding we measured: today a member holds a GPU session partly just to keep a foothold in the cluster (57 of 112 users ran more than one session at once). A free place to be inside the cluster removes that reason, so GPU sessions are held only for GPU work.

## 5. Parameters for the PMC to fix

- N, the number of GPUs reserved for the free tier, and how it is split between full GPUs and MIG slices.
- The target share `S` per working package, based on the budget or person-power assigned to each.
- How long a free GPU may sit idle before it is released.
- The maximum allocation duration (proposed: 7 days) and the default when a request gives none (proposed: 8 hours).
- Whether allocations beyond the one free GPU also need a hard cap on the number of GPUs, on top of the priority cost.
- The CPU-time threshold and the timeout for the shared entry server.
- The rule for renormalising the target shares over the packages with live demand.
- The per-model GPU-hour weights.
- The fair-share half-life. The proposal recommends 7 days, matching the common defaults.

## Appendix A. How real centres solve each piece

This appendix documents how large shared GPU centres already solve each part of the problem. Every mechanism in the policy above is standard, and for each we cite a centre that runs it. Numbers in square brackets point to the References.

### A.1 A reliable "GPU in the morning" is a small reserved pool with a one-per-person limit

To promise every member a GPU on demand, centres set aside a group of GPUs for short, interactive use, and let each person take only one at a time. The size of that pool, divided by one-per-person, is the number of people served at the same time.

- NERSC (the US national centre) reserves nodes for an interactive queue and delivers a GPU within 6 minutes. If none is free within 6 minutes the request is dropped rather than left waiting, so the pool is never hoarded [8, 9].
- Harvard FASRC runs a free interactive queue that does not count against a group's fair share [7].

The sizing rule for NGT: reserve N GPUs for this free tier and cap each member at one, so N members are served at once. Set N near the number of people who typically want a GPU at the same time in the morning, not the full roster of about 130. Because a single interactive task usually fits on a MIG slice (Section 4.1), one physical GPU serves several people at once.

### A.2 A time limit on every job is what makes a queue work

Every production GPU queue makes a job declare a maximum run time (its "walltime"), and fills in a default if the user does not [1]. The reason is mechanical. To start a small job early in a gap without delaying a large job already waiting, the scheduler must know when the running jobs will finish. This gap-filling is called backfill. With no time limits the scheduler cannot predict anything, so the queue collapses back to first come first served. That is exactly the NGT situation.

Real limits at other centres run from one to three days, that is 24 to 72 hours [8, 10, 11, 12]. Jobs that need longer save their state to disk at intervals (a "checkpoint") and resubmit, so a time limit never loses work.

### A.3 The time limit matters more than the "interactive" or "batch" label

Many centres keep two separate queues: a small, short, interactive one, and a large, long, batch one. The useful part of that split is not the label but the time limit that every job carries, because the time limit is what lets the scheduler plan, turn work over, and account for it. This proposal keeps the time limit and drops the label. Beyond the one free GPU, a request simply states how long it needs and pays for it in priority, whether the member is typing at it or running an unattended job.

### A.4 An idle interactive session is stopped automatically, and its work is kept

Every interactive platform stops sessions that sit idle, and, importantly, it keeps the user's files. One product stops a session when its GPU has been idle for a set time [6]. Notebook platforms stop an idle notebook but keep its disk (the "persistent volume"), so the user loses nothing and simply reopens the session later [29, 30]. For NGT the natural idle signal is GPU use near zero, which we already measure, with a timeout long enough to survive overnight so that a paused-but-owned session is not stopped too soon.

### A.5 Charge each project, count only the last week, and lower the priority of heavy users

The standard way to share fairly between groups is a "fair-share" rule. Give each project a target share of the machine. Track how much it actually used recently. The more it used, the lower the priority of its next request; as it eases off, its priority recovers.

Slurm, the most common HPC scheduler, writes this as a priority score `F = 2^(-U/S)` [1, 2]. Here `U` is the project's recent usage as a fraction of the total, and `S` is its target fraction. If a project used exactly its target, `F = 0.5`. If it used less, `F` rises above 0.5 and the project goes first. If it used more, `F` falls toward 0 and the project waits. A tree form of the rule keeps this ordering consistent from projects down to individual members [3].

"Recent" is set by a half-life. Usage right now counts in full; usage a week ago counts half; two weeks ago a quarter. The common default half-life is exactly 7 days [4]. A smooth decay is better than wiping the record clean every week, because a hard weekly reset lets people time their large jobs for just after the reset.

This is not theoretical. It runs in production for GPUs today, from a 7-day half-life on the KU cluster [5], to a one-week window in a commercial scheduler [6], to a 3-day half-life at Harvard FASRC [7]. The underlying theory, Dominant Resource Fairness [18], handles sharing several resource types at once; because GPUs are the only scarce resource here, it reduces to simply balancing GPU-hours between projects, and it cannot be gamed by padding a request with extra CPU or memory. Fair scheduling of shared GPU clusters is an active research field [20, 21, 22].

### A.6 Let a project use spare GPUs when the machine is quiet, and give them back when others need them

When the cluster is empty, a project should be able to run more than its share. When others arrive, it should give those extra GPUs back first. This "use it while it is free" idea is how every large operator runs. Google's Borg system protects production jobs and stops opportunistic ones first when capacity is needed [19]. Commercial GPU schedulers guarantee each project its quota and let extra work run only if it can be stopped ("preempted") later [6]. On NGT the one free GPU per member is always protected. Everything beyond it is opportunistic: the fair-share score decides who gets the spare GPUs, and who gives them back first when the cluster fills.

### A.7 A slice can be set aside for a special need, but only by agreement

Centres can reserve part of the machine for a project or an event, and they always require approval first [13]. On our system (Kubernetes) this is done by labelling a set of servers for one team. Two centres require a formal request and time-box the reservation, for example to two weeks at a time [14, 15]. NGT should do the same: reserve a slice only after PMC approval, and only for a fixed period.

## References

1. Slurm multifactor priority. https://slurm.schedmd.com/priority_multifactor.html
2. Slurm classic fairshare algorithm. https://slurm.schedmd.com/classic_fair_share.html
3. Slurm Fair Tree algorithm. https://slurm.schedmd.com/fair_tree.html
4. Slurm configuration, PriorityDecayHalfLife and TRESBillingWeights. https://slurm.schedmd.com/slurm.conf.html
5. KU Community Cluster fairshare, 7-day half-life. https://docs.crc.ku.edu/how-to/fairshare-priority/
6. NVIDIA Run:ai time-based fairshare, one-week window. https://developer.nvidia.com/blog/ensuring-balanced-gpu-allocation-in-kubernetes-clusters-with-time-based-fairshare/
7. Harvard FASRC fairshare and job accounting. https://docs.rc.fas.harvard.edu/kb/fairshare/
8. NERSC Perlmutter queues and charges. https://docs.nersc.gov/jobs/policy/
9. NERSC interactive jobs. https://docs.nersc.gov/jobs/interactive/
10. ALCF Polaris running jobs and queues. https://docs.alcf.anl.gov/polaris/running-jobs/
11. LUMI Slurm partitions. https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/partitions/
12. JUWELS batch system. https://apps.fz-juelich.de/jsc/hps/juwels/batchsystem.html
13. Slurm advanced resource reservations. https://slurm.schedmd.com/reservations.html
14. Utah CHPC general cluster policies. https://www.chpc.utah.edu/documentation/policies/2.1GeneralHPCClusterPolicies.php
15. Caltech HPC reservations. https://www.hpc.caltech.edu/documentation/faq/how-do-i-go-about-getting-a-reservation-on-the-cluster
16. Arbiter2, cgroups login-node governor (Utah). https://github.com/chpc-uofu/arbiter2
17. Arbiter2 at Brown OSCAR. https://docs.ccv.brown.edu/oscar/connecting-to-oscar/ssh/arbiter2
18. Ghodsi et al., Dominant Resource Fairness, NSDI 2011. https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types
19. Verma et al., Large-scale cluster management at Google with Borg, EuroSys 2015. https://research.google/pubs/large-scale-cluster-management-at-google-with-borg/
20. Mahajan et al., Themis: Fair and Efficient GPU Cluster Scheduling, NSDI 2020. https://www.usenix.org/conference/nsdi20/presentation/mahajan
21. Narayanan et al., Gavel: Heterogeneity-Aware Cluster Scheduling, OSDI 2020. https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak
22. Zhao et al., HiveD: Sharing a GPU Cluster with Guarantees, OSDI 2020. https://www.usenix.org/conference/osdi20/presentation/zhao-hanyu
23. Kueue concepts. https://kueue.sigs.k8s.io/docs/concepts/
24. Volcano documentation. https://volcano.sh/en/docs/
25. NVIDIA KAI Scheduler. https://github.com/NVIDIA/KAI-Scheduler
26. NVIDIA k8s device plugin, MIG, time-slicing, MPS. https://github.com/NVIDIA/k8s-device-plugin
27. NVIDIA GPU Operator. https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html
28. Kubernetes Dynamic Resource Allocation. https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/
29. Kubeflow notebook culling, ENABLE_CULLING and CULL_IDLE_TIME. https://awslabs.github.io/kubeflow-manifests/docs/deployment/configure-notebook-culling/
30. JupyterHub idle culler. https://github.com/jupyterhub/jupyterhub-idle-culler
31. Kueue topology-aware scheduling. https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/
32. NVIDIA GPUDirect RDMA. https://docs.nvidia.com/cuda/gpudirect-rdma/index.html
33. NGT support tracker, work item 72, MPI across GPU flavours. https://gitlab.cern.ch/mlops/ngt/project/support/-/work_items/72
