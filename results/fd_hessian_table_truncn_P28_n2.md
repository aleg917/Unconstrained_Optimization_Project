# Truncated Newton (matrix-free) — P28, n = 2

_Variably Dimensioned (H densa, no CPR)_

| start | k | scaled | stencil | grad.norm | iters/max | success | rate | time (s) | note |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 0 | centered | 6.51e-12 | 7/500 | yes | 1.68 | 0.00 | grad_abs |
| 0 | 8 | 0 | centered | 2.23e-09 | 7/500 | yes | 1.68 | 0.00 | grad_abs |
| 0 | 12 | 0 | centered | 5.88e-07 | 11/500 | yes | 1.00 | 0.00 | grad_abs |
| 0 | 4 | 1 | centered | 6.51e-12 | 7/500 | yes | 1.68 | 0.00 | grad_abs |
| 0 | 8 | 1 | centered | 2.23e-09 | 7/500 | yes | 1.68 | 0.00 | grad_abs |
| 0 | 12 | 1 | centered | 5.88e-07 | 11/500 | yes | 1.00 | 0.00 | grad_abs |
| 1 | 4 | 0 | centered | 1.63e-07 | 6/500 | yes | 1.69 | 0.00 | grad_abs |
| 1 | 8 | 0 | centered | 1.66e-07 | 6/500 | yes | 1.69 | 0.00 | grad_abs |
| 1 | 12 | 0 | centered | 9.48e-07 | 16/500 | yes | 0.90 | 0.00 | grad_abs |
| 1 | 4 | 1 | centered | 1.63e-07 | 6/500 | yes | 1.69 | 0.00 | grad_abs |
| 1 | 8 | 1 | centered | 1.62e-07 | 6/500 | yes | 1.69 | 0.00 | grad_abs |
| 1 | 12 | 1 | centered | 8.65e-07 | 15/500 | yes | 0.86 | 0.00 | grad_abs |

## Aggregato: media tra successi

| k | scaled | stencil | n. successi | avg grad.norm | avg iters | avg rate | avg time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 0 | centered | 2 / 2 | 8.16e-08 | 6.5 | 1.69 | 0.00 |
| 4 | 1 | centered | 2 / 2 | 8.16e-08 | 6.5 | 1.69 | 0.00 |
| 8 | 0 | centered | 2 / 2 | 8.40e-08 | 6.5 | 1.69 | 0.00 |
| 8 | 1 | centered | 2 / 2 | 8.21e-08 | 6.5 | 1.69 | 0.00 |
| 12 | 0 | centered | 2 / 2 | 7.68e-07 | 13.5 | 0.95 | 0.00 |
| 12 | 1 | centered | 2 / 2 | 7.27e-07 | 13.0 | 0.93 | 0.00 |