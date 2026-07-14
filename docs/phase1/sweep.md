# Waiting time vs number of users

Config config/phase1.yaml, 3 trace seeds per point; cells are p95 wait in minutes among started logical jobs, mean over seeds (min to max). All policies share the identical trace at each point. The never-started fraction rises with load and censors these waits; it is reported in sweep.json.

| users | fcfs_pending | idle_reclaim | ngt_principles | ngt_principles_reclaim | planning_cycle |
|---|---|---|---|---|---|
| 41 | 91 (81 to 107) | 57 (52 to 62) | 85 (81 to 90) | 66 (53 to 74) | 423 (403 to 442) |
| 62 | 206 (173 to 242) | 182 (137 to 234) | 194 (142 to 232) | 176 (116 to 230) | 451 (416 to 472) |
| 82 | 288 (258 to 330) | 246 (226 to 275) | 292 (269 to 307) | 262 (212 to 304) | 458 (455 to 462) |
| 102 | 376 (345 to 392) | 333 (325 to 345) | 379 (344 to 408) | 323 (309 to 337) | 480 (456 to 514) |
| 123 | 452 (412 to 490) | 398 (385 to 409) | 452 (426 to 491) | 426 (370 to 476) | 496 (476 to 531) |
