# Truncated Newton (matrix-free) — P16, n = 1000

_Banded Trigonometric (H diagonale, 1 colore CPR)_

| start | k | scaled | stencil | grad.norm | iters/max | success | rate | time (s) | note |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 0 | centered | 7.23e-08 | 19/200 | yes | 1.50 | 0.02 | grad_abs |
| 0 | 8 | 0 | centered | 2.03e-07 | 20/200 | yes | 1.45 | 0.02 | grad_abs |
| 0 | 12 | 0 | centered | 1.10e-04 | 200/200 | no | 0.05 | 8.62 | max_iter |
| 0 | 4 | 1 | centered | 2.36e-06 | 200/200 | no | 1.00 | 0.84 | max_iter |
| 0 | 8 | 1 | centered | 2.01e-07 | 19/200 | yes | 1.38 | 0.02 | grad_abs |
| 0 | 12 | 1 | centered | 1.81e-05 | 200/200 | no | 0.00 | 11.57 | max_iter |

## Aggregato: media tra successi

| k | scaled | stencil | n. successi | avg grad.norm | avg iters | avg rate | avg time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 0 | centered | 1 / 1 | 7.23e-08 | 19.0 | 1.50 | 0.02 |
| 4 | 1 | centered | 0 / 1 | - | - | - | - |
| 8 | 0 | centered | 1 / 1 | 2.03e-07 | 20.0 | 1.45 | 0.02 |
| 8 | 1 | centered | 1 / 1 | 2.01e-07 | 19.0 | 1.38 | 0.02 |
| 12 | 0 | centered | 0 / 1 | - | - | - | - |
| 12 | 1 | centered | 0 / 1 | - | - | - | - |