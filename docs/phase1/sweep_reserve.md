# Reserve sizing for the 1-GPU session tier

Policy ngt_principles_reclaim, target: 1-GPU tier p95 wait below 15 min. Reserve applies to H100 NVL (88 GPUs) and L40S (28) proportionally; multi-GPU capacity = pool minus reserve (H100 SXM stays fully multi-GPU). Mean over 3 trace seeds.

| users | min reserve meeting target (NVL+L40S GPUs) | 1-GPU p95 (min) | multi-GPU p95 (min) at that reserve | max multi-GPU capacity (NVL+L40S GPUs) |
|---|---|---|---|---|
| 41 | 0 (fraction 0.00) | 0 | 161 | 116 |
| 62 | 0 (fraction 0.00) | 6 | 347 | 116 |
| 82 | target missed at every reserve; minimum p95 15 at 0 GPUs | n/a | 478 | 116 |
| 102 | target missed at every reserve; minimum p95 44 at 12 GPUs | n/a | 562 | 104 |
| 123 | target missed at every reserve; minimum p95 53 at 5 GPUs | n/a | 645 | 111 |
