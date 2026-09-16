
### core|n_t=3|prior=strong

| Estimator | n | impl-capped | SD cap60 | SD cap120 | SD cap240 | SD uncapped | median | IQR | MAD | P5–P95 | SD(log t90) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `two_stage_conjugate` | 8918 | 14.5% | 14.44 | 32.46 | 60.50 | 59703.38 | -2.66 | 47.04 | 32.13 | 3877.76 | 1.541 |
| `mcmc` | 493 | 0.0% | 7.79 | 31.41 | 73.94 | inf | 36.42 | 129.07 | 69.13 | 823.39 | 25.649 |
| `classical_ols_multi_temp` | 8165 | 0.0% | 16.55 | 36.87 | 67.45 | 1.5e+06 | -5.10 | 64.89 | 40.93 | 528.75 | 1.201 |
| `classical_ich_q1e` | 8452 | 0.0% | 8.54 | 23.02 | 39.43 | 1443.41 | -1.42 | 23.75 | 16.75 | 115.99 | 0.526 |

### core|n_t=3|prior=accurate

| Estimator | n | impl-capped | SD cap60 | SD cap120 | SD cap240 | SD uncapped | median | IQR | MAD | P5–P95 | SD(log t90) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `two_stage_conjugate` | 8941 | 12.1% | 16.04 | 32.25 | 61.55 | 28069.22 | -12.85 | 39.60 | 27.84 | 2507.77 | 1.483 |
| `mcmc` | 523 | 0.0% | 10.25 | 31.80 | 61.79 | 4.9e+92 | 14.88 | 67.89 | 43.79 | 325.28 | 15.145 |
| `classical_ols_multi_temp` | 8108 | 0.0% | 16.74 | 36.92 | 67.09 | 115357.81 | -5.14 | 65.35 | 40.88 | 514.88 | 1.215 |
| `classical_ich_q1e` | 8400 | 0.0% | 8.43 | 23.02 | 40.02 | 2092.78 | -1.12 | 22.90 | 16.31 | 119.00 | 0.536 |

### core|n_t=4|prior=strong

| Estimator | n | impl-capped | SD cap60 | SD cap120 | SD cap240 | SD uncapped | median | IQR | MAD | P5–P95 | SD(log t90) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `two_stage_conjugate` | 8995 | 10.3% | 13.23 | 28.14 | 52.02 | 40507.89 | -3.79 | 31.32 | 22.85 | 328.29 | 1.045 |
| `mcmc` | 734 | 0.0% | 8.00 | 29.13 | 62.46 | 231.74 | 12.09 | 60.03 | 32.44 | 416.10 | 0.749 |
| `classical_ols_multi_temp` | 8975 | 0.0% | 14.36 | 33.45 | 59.77 | 11219.65 | -3.59 | 52.55 | 34.23 | 318.57 | 0.950 |
| `classical_ich_q1e` | 8432 | 0.0% | 8.35 | 22.99 | 40.48 | 368.26 | -1.42 | 23.70 | 16.61 | 123.37 | 0.528 |

### robustness_all

| Estimator | n | impl-capped | SD cap60 | SD cap120 | SD cap240 | SD uncapped | median | IQR | MAD | P5–P95 | SD(log t90) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `two_stage_conjugate` | 17220 | 17.1% | 17.67 | 37.09 | 60.31 | 3983.86 | -0.13 | 64.21 | 41.51 | 292.53 | 0.985 |
| `mcmc` | 1497 | 0.0% | 12.99 | 32.91 | 60.09 | inf | 44.73 | 82.97 | 60.24 | 292.80 | 19.752 |
| `classical_ols_multi_temp` | 19044 | 0.0% | 18.05 | 42.96 | 89.34 | 142411.90 | 44.85 | 214.57 | 107.08 | 1247.42 | 1.368 |
| `classical_ich_q1e` | 19821 | 0.0% | 9.82 | 23.00 | 36.76 | 1294.50 | 5.00 | 27.49 | 19.22 | 104.01 | 0.448 |

All rows use the uncapped point estimate (two-stage reconstructed from k_mean); 'impl-capped' is the fraction of successful replicates whose stored two-stage estimate sat at the 120-month implementation cap.
