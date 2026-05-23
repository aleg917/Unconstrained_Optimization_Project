# Modified Newton — P16, n = 1000

_Banded Trigonometric (H diagonale, 1 colore CPR)_

| start | k | scaled | stencil | grad.norm | iters/max | success | rate | time (s) | note |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 0 | centered | 2.79e-13 | 7/200 | yes | 2.25 | 0.49 | grad_abs |
| 0 | 8 | 0 | centered | 2.02e-07 | 7/200 | yes | 2.64 | 0.53 | grad_abs |
| 0 | 12 | 0 | centered | 8.66e-07 | 11/200 | yes | 1.00 | 0.78 | grad_abs |
| 0 | 4 | 1 | centered | 2.08e-07 | 7/200 | yes | 1.92 | 0.48 | grad_abs |
| 0 | 8 | 1 | centered | 7.37e-08 | 7/200 | yes | 2.64 | 0.45 | grad_abs |
| 0 | 12 | 1 | centered | 1.65e-07 | 6/200 | yes | 2.84 | 0.41 | grad_abs |

## Aggregato: media tra successi

| k | scaled | stencil | n. successi | avg grad.norm | avg iters | avg rate | avg time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 0 | centered | 1 / 1 | 2.79e-13 | 7.0 | 2.25 | 0.49 |
| 4 | 1 | centered | 1 / 1 | 2.08e-07 | 7.0 | 1.92 | 0.48 |
| 8 | 0 | centered | 1 / 1 | 2.02e-07 | 7.0 | 2.64 | 0.53 |
| 8 | 1 | centered | 1 / 1 | 7.37e-08 | 7.0 | 2.64 | 0.45 |
| 12 | 0 | centered | 1 / 1 | 8.66e-07 | 11.0 | 1.00 | 0.78 |
| 12 | 1 | centered | 1 / 1 | 1.65e-07 | 6.0 | 2.84 | 0.41 |