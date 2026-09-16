# MCMC tuning matrix — bias / dispersion / coverage on converged replicates


## nt3_strong (case core_042, N = 120 per condition)

| Condition | non-conv. % | n conv. | bias_median | IQR | MAD | SD cap120 | optimism % | coverage % | CI overflow % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MCMC A | 30.0 | 84 | 46.2 | 124.8 | 80.7 | 32.6 | 75 | 82.1 | 84.5 |
| MCMC B | 12.5 | 105 | 48.6 | 125.9 | 83.9 | 31.4 | 77 | 82.9 | 89.5 |
| MCMC C | 21.7 | 94 | 48.9 | 130.3 | 86.8 | 30.5 | 80 | 81.9 | 91.5 |
| MCMC D | 16.7 | 100 | 57.8 | 118.2 | 95.0 | 30.9 | 78 | 83.0 | 89.0 |
| MCMC E | 2.5 | 117 | 48.2 | 123.3 | 81.2 | 31.8 | 74 | 82.9 | 89.7 |
| two_stage_conjugate | 0.0 (fail) | 120 | -7.0 | 63.7 | 37.2 | 35.1 | 42 | 84.2 | 0.0 |

## nt2_strong (case core_015, N = 120 per condition)

| Condition | non-conv. % | n conv. | bias_median | IQR | MAD | SD cap120 | optimism % | coverage % | CI overflow % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MCMC A | 70.8 | 35 | 192.1 | 278.9 | 136.8 | 11.2 | 100 | 91.4 | 100.0 |
| MCMC B | 42.5 | 69 | 172.1 | 179.3 | 131.0 | 17.4 | 96 | 85.5 | 98.6 |
| MCMC C | 68.3 | 38 | 187.6 | 385.3 | 180.6 | 17.0 | 92 | 94.7 | 100.0 |
| MCMC D | 55.0 | 54 | 174.8 | 446.5 | 176.7 | 13.2 | 100 | 81.5 | 100.0 |
| MCMC E | 19.2 | 97 | 160.9 | 201.8 | 150.6 | 17.6 | 97 | 87.6 | 99.0 |
| two_stage_conjugate | 100.0 (fail, N_CONDS_TOO_LOW) | 0 | — | — | — | — | — | — | — |

## nt2_accurate (case core_013, N = 120 per condition)

| Condition | non-conv. % | n conv. | bias_median | IQR | MAD | SD cap120 | optimism % | coverage % | CI overflow % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MCMC A | 65.0 | 42 | 54.3 | 106.7 | 72.2 | 28.2 | 90 | 100.0 | 92.9 |
| MCMC E | 13.3 | 104 | 52.5 | 101.8 | 71.2 | 30.7 | 85 | 99.0 | 92.3 |
| two_stage_conjugate | 100.0 (fail, N_CONDS_TOO_LOW) | 0 | — | — | — | — | — | — | — |

## nt4_strong (case core_069, N = 120 per condition)

| Condition | non-conv. % | n conv. | bias_median | IQR | MAD | SD cap120 | optimism % | coverage % | CI overflow % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MCMC A | 20.0 | 96 | 6.0 | 39.9 | 26.2 | 25.8 | 61 | 80.2 | 51.0 |
| MCMC E | 0.0 | 120 | 5.5 | 37.7 | 25.6 | 25.8 | 59 | 81.7 | 51.7 |
| two_stage_conjugate | 0.0 (fail) | 120 | -6.0 | 33.0 | 22.2 | 25.6 | 44 | 93.3 | 0.0 |

## nt3_accurate (case core_040, N = 120 per condition)

| Condition | non-conv. % | n conv. | bias_median | IQR | MAD | SD cap120 | optimism % | coverage % | CI overflow % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MCMC A | 16.7 | 100 | 16.2 | 63.7 | 42.0 | 31.4 | 70 | 91.0 | 79.0 |
| MCMC E | 0.8 | 119 | 17.2 | 58.7 | 44.2 | 31.1 | 70 | 91.6 | 80.7 |
| two_stage_conjugate | 0.0 (fail) | 120 | -17.1 | 30.3 | 22.4 | 27.5 | 26 | 93.3 | 0.0 |

MCMC point estimate = posterior mean of t90 samples (as in the production run); bias relative to t90_true = 61.6224 months. two_stage_conjugate row: replicates 0-119 of the same case from estimator_results.parquet (stored estimate, implementation-capped at 120).