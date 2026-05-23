# Truncated Newton (matrix-free) — P16, n = 2

_Banded Trigonometric (H diagonale, 1 colore CPR)_

| start | k | scaled | stencil | grad.norm | iters/max | success | rate | time (s) | note |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 0 | centered | 8.11e-13 | 4/200 | yes | 3.24 | 0.00 | grad_abs |
| 0 | 8 | 0 | centered | 1.99e-09 | 4/200 | yes | 3.24 | 0.00 | grad_abs |
| 0 | 12 | 0 | centered | 3.81e-07 | 6/200 | yes | 1.00 | 0.00 | grad_abs |
| 0 | 4 | 1 | centered | 8.00e-13 | 4/200 | yes | 3.24 | 0.00 | grad_abs |
| 0 | 8 | 1 | centered | 5.82e-09 | 4/200 | yes | 3.24 | 0.00 | grad_abs |
| 0 | 12 | 1 | centered | 3.50e-07 | 7/200 | yes | 1.00 | 0.00 | grad_abs |

## Aggregato: media tra successi

| k | scaled | stencil | n. successi | avg grad.norm | avg iters | avg rate | avg time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 0 | centered | 1 / 1 | 8.11e-13 | 4.0 | 3.24 | 0.00 |
| 4 | 1 | centered | 1 / 1 | 8.00e-13 | 4.0 | 3.24 | 0.00 |
| 8 | 0 | centered | 1 / 1 | 1.99e-09 | 4.0 | 3.24 | 0.00 |
| 8 | 1 | centered | 1 / 1 | 5.82e-09 | 4.0 | 3.24 | 0.00 |
| 12 | 0 | centered | 1 / 1 | 3.81e-07 | 6.0 | 1.00 | 0.00 |
| 12 | 1 | centered | 1 / 1 | 3.50e-07 | 7.0 | 1.00 | 0.00 |