# Modified Newton — P28, n = 2

_Variably Dimensioned (H densa, no CPR)_

| start | k | scaled | stencil | grad.norm | iters/max | success | rate | time (s) | note |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 0 | centered | 2.09e-11 | 7/500 | yes | 1.68 | 0.00 | grad_abs |
| 0 | 8 | 0 | centered | 8.07e-12 | 7/500 | yes | 1.68 | 0.00 | grad_abs |
| 0 | 12 | 0 | centered | 5.38e-09 | 7/500 | yes | 1.68 | 0.00 | grad_abs |
| 0 | 4 | 1 | centered | 2.09e-11 | 7/500 | yes | 1.68 | 0.00 | grad_abs |
| 0 | 8 | 1 | centered | 5.81e-12 | 7/500 | yes | 1.68 | 0.00 | grad_abs |
| 0 | 12 | 1 | centered | 2.69e-09 | 7/500 | yes | 1.68 | 0.00 | grad_abs |
| 1 | 4 | 0 | centered | 5.07e-07 | 6/500 | yes | 1.26 | 0.00 | grad_abs |
| 1 | 8 | 0 | centered | 5.07e-07 | 6/500 | yes | 1.26 | 0.00 | grad_abs |
| 1 | 12 | 0 | centered | 2.80e-07 | 6/500 | yes | 1.26 | 0.00 | grad_abs |
| 1 | 4 | 1 | centered | 5.07e-07 | 6/500 | yes | 1.26 | 0.00 | grad_abs |
| 1 | 8 | 1 | centered | 5.07e-07 | 6/500 | yes | 1.26 | 0.00 | grad_abs |
| 1 | 12 | 1 | centered | 1.15e-07 | 6/500 | yes | 1.26 | 0.00 | grad_abs |

## Aggregato: media tra successi

| k | scaled | stencil | n. successi | avg grad.norm | avg iters | avg rate | avg time (s) |
|---|---|---|---|---|---|---|---|
| 4 | 0 | centered | 2 / 2 | 2.54e-07 | 6.5 | 1.47 | 0.00 |
| 4 | 1 | centered | 2 / 2 | 2.54e-07 | 6.5 | 1.47 | 0.00 |
| 8 | 0 | centered | 2 / 2 | 2.53e-07 | 6.5 | 1.47 | 0.00 |
| 8 | 1 | centered | 2 / 2 | 2.53e-07 | 6.5 | 1.47 | 0.00 |
| 12 | 0 | centered | 2 / 2 | 1.43e-07 | 6.5 | 1.47 | 0.00 |
| 12 | 1 | centered | 2 / 2 | 5.87e-08 | 6.5 | 1.47 | 0.00 |