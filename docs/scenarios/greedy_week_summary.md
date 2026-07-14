# Scenario: greedy-week

One greedy 8-GPU week-long hold vs daily few-hour trainings and 1-GPU member sessions from all four working packages, on 16 NVL GPUs.

## fcfs_pending

| user | WP | requests | started | cancelled | total wait (min) | held GPU-h | active GPU-h | reclaim/cap events |
|---|---|---|---|---|---|---|---|---|
| sam | WP1 | 5 | 5 | 0 | 0 | 40 | 3.5 | 0 |
| tom | WP1 | 5 | 5 | 0 | 0 | 120 | 85.4 | 0 |
| greta | WP2 | 1 | 1 | 0 | 0 | 1272 | 63.5 | 0 |
| mia | WP2 | 5 | 5 | 0 | 0 | 40 | 3.5 | 0 |
| ada | WP3 | 3 | 3 | 0 | 270 | 48 | 37.9 | 0 |
| leo | WP3 | 5 | 5 | 0 | 0 | 40 | 3.5 | 0 |
| lea | WP4 | 2 | 2 | 0 | 660 | 32 | 22.5 | 0 |

## ngt_principles_reclaim

| user | WP | requests | started | cancelled | total wait (min) | held GPU-h | active GPU-h | reclaim/cap events |
|---|---|---|---|---|---|---|---|---|
| sam | WP1 | 5 | 15 | 0 | 0 | 15 | 3.0 | 15 |
| tom | WP1 | 5 | 5 | 0 | 0 | 120 | 85.4 | 0 |
| greta | WP2 | 1 | 7 | 0 | 84 | 84 | 39.8 | 7 |
| mia | WP2 | 5 | 15 | 0 | 0 | 15 | 3.0 | 15 |
| ada | WP3 | 3 | 3 | 0 | 0 | 48 | 37.9 | 0 |
| leo | WP3 | 5 | 15 | 0 | 0 | 15 | 3.0 | 15 |
| lea | WP4 | 2 | 2 | 0 | 0 | 32 | 22.5 | 0 |

