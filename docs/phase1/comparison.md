# Phase 1 comparison: fcfs_pending vs idle_reclaim vs ngt_principles vs ngt_principles_reclaim vs planning_cycle

Trace seed 20260714, engine seed 42, measurement window days 7 to 14.

## Metrics

| metric | fcfs_pending | idle_reclaim | ngt_principles | ngt_principles_reclaim | planning_cycle |
|---|---|---|---|---|---|
| logical jobs in window | 1901 | 1901 | 1901 | 1901 | 1901 |
| jobs started | 1494 | 1577 | 1501 | 1577 | 1002 |
| never started (frac) | 0.214 | 0.170 | 0.210 | 0.170 | 0.473 |
| wait mean (min) | 58.7 | 43.4 | 47.9 | 37.0 | 138.7 |
| wait median (min) | 0.0 | 0.0 | 0.0 | 0.0 | 69.8 |
| wait p95 (min) | 276.6 | 252.3 | 262.6 | 217.5 | 453.5 |
| wait max (min) | 2107.1 | 2107.1 | 2854.1 | 2107.1 | 2107.1 |
| 1-GPU tier wait mean (min) | 34.1 | 14.8 | 23.9 | 7.1 | 67.9 |
| 1-GPU tier wait p95 (min) | 164.2 | 49.2 | 123.1 | 11.7 | 309.9 |
| 1-GPU tier never started | 117 | 78 | 109 | 68 | 223 |
| h100nvl allocated GPU-h | 11159 | 9984 | 9918 | 9451 | 5917 |
| h100nvl used GPU-h | 6481 | 7277 | 6001 | 6913 | 3246 |
| h100nvl idle-held GPU-h | 4678 | 2707 | 3917 | 2538 | 2671 |
| h100nvl reclaimable idle GPU-h | 2393 | 0 | 1690 | 0 | 1261 |
| h100sxm allocated GPU-h | 3000 | 2980 | 2925 | 2770 | 1509 |
| h100sxm used GPU-h | 1732 | 1894 | 1731 | 1799 | 819 |
| h100sxm idle-held GPU-h | 1269 | 1085 | 1194 | 971 | 690 |
| h100sxm reclaimable idle GPU-h | 320 | 0 | 469 | 0 | 221 |
| l40s allocated GPU-h | 3487 | 2993 | 2594 | 2497 | 1676 |
| l40s used GPU-h | 1624 | 2199 | 1533 | 1778 | 863 |
| l40s idle-held GPU-h | 1862 | 794 | 1061 | 718 | 813 |
| l40s reclaimable idle GPU-h | 1198 | 0 | 424 | 0 | 408 |
| mig3g allocated GPU-h | 743 | 375 | 757 | 375 | 665 |
| mig3g used GPU-h | 61 | 61 | 62 | 61 | 54 |
| mig3g idle-held GPU-h | 683 | 314 | 694 | 314 | 611 |
| mig3g reclaimable idle GPU-h | 414 | 0 | 420 | 0 | 374 |
| mig1g allocated GPU-h | 1320 | 504 | 1223 | 505 | 957 |
| mig1g used GPU-h | 89 | 81 | 87 | 81 | 69 |
| mig1g idle-held GPU-h | 1232 | 423 | 1136 | 424 | 888 |
| mig1g reclaimable idle GPU-h | 876 | 0 | 776 | 0 | 599 |
| reclaims | 0 | 2055 | 0 | 2068 | 0 |
| resubmissions | 0 | 1246 | 38 | 1304 | 11 |
| resubmit extra wait (h) | 0.0 | 53.7 | 34.1 | 40.7 | 5.1 |
| Jain (GPU-h satisfaction) | 0.941 | 0.827 | 0.924 | 0.829 | 0.817 |
| Jain (per-user mean wait) | 0.599 | 0.542 | 0.633 | 0.562 | 0.636 |
| WP1 charged share (target 0.30) | 0.259 | 0.323 | 0.272 | 0.313 | 0.288 |
| WP2 charged share (target 0.30) | 0.338 | 0.248 | 0.321 | 0.285 | 0.294 |
| WP3 charged share (target 0.30) | 0.311 | 0.314 | 0.316 | 0.308 | 0.309 |
| WP4 charged share (target 0.10) | 0.092 | 0.114 | 0.091 | 0.094 | 0.109 |
| WP share max deviation | 0.041 | 0.052 | 0.028 | 0.015 | 0.012 |

## Validation vs observed Saturday data point

### fcfs_pending

```json
{
  "window_h": [
    306.0,
    307.583
  ],
  "mean_free_allocatable_gpus": 12.0,
  "min_free_allocatable_gpus": 7.0,
  "mean_cordoned_fraction": 0.17567567567567569,
  "n_requests_pending_in_window": 21,
  "their_waits_min": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    66.5,
    97.8,
    112.9,
    145.1,
    180.2,
    205.2,
    235.5,
    249.5,
    317.5,
    337.0,
    338.4,
    353.2,
    382.9,
    384.5,
    761.9
  ],
  "idle_held_h100_gpus_at_probe": 15
}
```

### idle_reclaim

```json
{
  "window_h": [
    306.0,
    307.583
  ],
  "mean_free_allocatable_gpus": 33.0,
  "min_free_allocatable_gpus": 31.0,
  "mean_cordoned_fraction": 0.17567567567567569,
  "n_requests_pending_in_window": 30,
  "their_waits_min": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    57.1,
    66.5,
    77.9,
    97.8,
    180.7,
    235.5,
    353.2,
    382.9,
    384.5
  ],
  "idle_held_h100_gpus_at_probe": 7
}
```

### ngt_principles

```json
{
  "window_h": [
    306.0,
    307.583
  ],
  "mean_free_allocatable_gpus": 29.8,
  "min_free_allocatable_gpus": 25.0,
  "mean_cordoned_fraction": 0.17567567567567569,
  "n_requests_pending_in_window": 25,
  "their_waits_min": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    13.2,
    34.0,
    94.8,
    97.7,
    98.6,
    112.9,
    117.6,
    145.3,
    187.8,
    214.8,
    229.0,
    272.6,
    291.5,
    385.2,
    447.0,
    501.5,
    670.5,
    713.9,
    922.1
  ],
  "idle_held_h100_gpus_at_probe": 14
}
```

### ngt_principles_reclaim

```json
{
  "window_h": [
    306.0,
    307.583
  ],
  "mean_free_allocatable_gpus": 56.5,
  "min_free_allocatable_gpus": 51.0,
  "mean_cordoned_fraction": 0.17567567567567569,
  "n_requests_pending_in_window": 33,
  "their_waits_min": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    66.5,
    97.8,
    112.9,
    171.8,
    209.5,
    217.5,
    235.5,
    257.1,
    265.8,
    1004.5
  ],
  "idle_held_h100_gpus_at_probe": 2
}
```

### planning_cycle

```json
{
  "window_h": [
    306.0,
    307.583
  ],
  "mean_free_allocatable_gpus": 97.0,
  "min_free_allocatable_gpus": 94.0,
  "mean_cordoned_fraction": 0.17567567567567569,
  "n_requests_pending_in_window": 62,
  "their_waits_min": [
    0.0,
    27.1,
    42.3,
    46.5,
    56.3,
    65.3,
    97.8,
    98.6,
    112.9,
    123.2,
    134.3,
    135.6,
    144.8,
    151.9,
    151.9,
    154.0,
    159.9,
    184.5,
    189.0,
    189.0,
    189.0,
    196.0,
    199.1,
    199.1,
    199.1,
    209.4,
    211.9,
    228.1,
    235.5,
    239.4,
    253.6,
    257.1,
    273.3,
    291.5,
    296.1,
    296.9,
    306.7,
    309.5,
    309.5,
    321.7,
    321.8,
    345.3,
    346.2,
    355.9,
    360.0,
    360.0,
    360.0,
    360.5,
    363.2,
    385.2,
    413.1,
    413.1,
    417.1,
    466.7,
    477.0,
    479.2,
    670.5,
    809.8,
    828.0,
    843.3,
    875.4,
    899.4
  ],
  "idle_held_h100_gpus_at_probe": 16
}
```
