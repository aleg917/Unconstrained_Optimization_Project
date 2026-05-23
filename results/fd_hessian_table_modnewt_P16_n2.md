# Modified Newton — P16, n = 2

_Banded Trigonometric (H diagonale, 1 colore CPR)_

| start | k | scaled | stencil | grad.norm | iters/max | success | rate | time (s) | note |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 0 | centered | 1.21e-14 | 5/200 | yes | 3.16 | 0.00 | grad_abs |
| 0 | 8 | 0 | centered | 1.66e-12 | 5/200 | yes | 3.16 | 0.00 | grad_abs |
| 0 | 12 | 0 | centered | 9.27e-08 | 6/200 | yes | 1.00 | 0.00 | grad_abs |
| 0 | 4 | 1 | centered | 5.57e-09 | 5/200 | yes | 2.86 | 0.00 | grad_abs |
| 0 | 8 | 1 | centered | 1.21e-14 | 5/200 | yes | 3.16 | 0.00 | grad_abs |
| 0 | 12 | 1 | centered | 5.78e-11 | 5/200 | yes | 3.08 | 0.00 | grad_abs |

## Aggregato: media tra successi

| k | scaled | stencil | n. successi | avg grad.norm | avg iters | avg rate | avg time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 0 | centered | 1 / 1 | 1.21e-14 | 5.0 | 3.16 | 0.00 |
| 4 | 1 | centered | 1 / 1 | 5.57e-09 | 5.0 | 2.86 | 0.00 |
| 8 | 0 | centered | 1 / 1 | 1.66e-12 | 5.0 | 3.16 | 0.00 |
| 8 | 1 | centered | 1 / 1 | 1.21e-14 | 5.0 | 3.16 | 0.00 |
| 12 | 0 | centered | 1 / 1 | 9.27e-08 | 6.0 | 1.00 | 0.00 |
| 12 | 1 | centered | 1 / 1 | 5.78e-11 | 5.0 | 3.08 | 0.00 |