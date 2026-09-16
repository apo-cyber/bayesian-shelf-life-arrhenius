# Stage-1 uncertainty propagation — central cell (n_T = 3, prior strong), 8918 successful replicates, B = 400

| Second stage | n | bias_median | IQR | MAD | SD cap120 | optimism % | coverage % | median log-width 95 % | > 120 mo % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| published | 8918 | -2.66 | 47.0 | 32.1 | 32.46 | 47 | 90.2 | 2.30 | 14.5 |
| no_floor | 8918 | -2.76 | 46.9 | 32.4 | 32.54 | 47 | 89.6 | 2.31 | 14.4 |
| unweighted | 8918 | 0.68 | 56.0 | 37.1 | 34.23 | 51 | 89.7 | 2.35 | 18.1 |
| mc_propagation | 8918 | -1.90 | 42.0 | 29.6 | 30.70 | 47 | 97.9 | 3.10 | 12.1 |
| mcmc_joint | 493 | 36.42 | 129.1 | 69.1 | 31.41 | 73 | 82.6 | 1.94 | 40.4 |

Stage-1 diagnostics: 5 % SE floor active in 42.7 % of replicates (0.51 of 3 temperatures on average); max SE_k/k > 0.15 (delta-method warning) in 90.7 %.

Point-estimate agreement with `published` (|rel. diff|): no_floor: median 0.00 %, P95 3.9 %; unweighted: median 22.61 %, P95 7509.6 %; mc_propagation: median 5.97 %, P95 88.7 %

## n_pts = 3 (stage-1 residual df = 1)

| Second stage | n | bias_median | IQR | MAD | SD cap120 | optimism % | coverage % | median log-width 95 % | > 120 mo % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| published | 2957 | -2.79 | 52.7 | 35.0 | 34.03 | 47 | 84.8 | 2.22 | 16.4 |
| no_floor | 2957 | -2.71 | 53.5 | 35.5 | 34.20 | 47 | 83.6 | 2.22 | 16.3 |
| unweighted | 2957 | -0.05 | 60.1 | 38.3 | 35.07 | 50 | 86.2 | 2.34 | 19.9 |
| mc_propagation | 2957 | -2.13 | 43.5 | 30.7 | 31.37 | 47 | 97.9 | 3.48 | 12.1 |

## n_pts = 4 (stage-1 residual df = 2)

| Second stage | n | bias_median | IQR | MAD | SD cap120 | optimism % | coverage % | median log-width 95 % | > 120 mo % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| published | 2981 | -2.23 | 49.2 | 33.4 | 33.01 | 47 | 90.3 | 2.34 | 15.3 |
| no_floor | 2981 | -2.53 | 49.1 | 33.7 | 33.07 | 47 | 89.8 | 2.35 | 15.3 |
| unweighted | 2981 | 0.58 | 58.4 | 38.3 | 34.68 | 50 | 90.3 | 2.41 | 18.7 |
| mc_propagation | 2981 | -1.47 | 44.4 | 31.2 | 31.37 | 48 | 97.5 | 3.06 | 13.5 |

## n_pts = 6 (stage-1 residual df = 4)

| Second stage | n | bias_median | IQR | MAD | SD cap120 | optimism % | coverage % | median log-width 95 % | > 120 mo % |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| published | 2980 | -2.88 | 39.9 | 28.8 | 30.21 | 46 | 95.4 | 2.33 | 11.7 |
| no_floor | 2980 | -2.89 | 39.9 | 28.8 | 30.22 | 46 | 95.2 | 2.33 | 11.7 |
| unweighted | 2980 | 1.16 | 51.3 | 35.0 | 32.93 | 52 | 92.4 | 2.31 | 15.8 |
| mc_propagation | 2980 | -2.06 | 38.7 | 27.6 | 29.30 | 47 | 98.4 | 2.67 | 10.9 |