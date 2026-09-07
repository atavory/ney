# Current Score Selector vs Weighted-Residual Selector

Entries are percent MSE gain, shown as `current -> weighted-residual`.

| benchmark | setting | AIPW | selective ML | Ma DR-BC | C-TMLE |
|---|---:|---:|---:|---:|---:|
| Kang-Schafer | KS CC, n=200 | 24.1 -> 25.1 | -0.3 -> 0.0 | 11.1 -> 16.3 | -0.9 -> 13.0 |
| Kang-Schafer | KS CC, n=1000 | 32.9 -> 30.5 | -0.2 -> -0.1 | 30.9 -> 47.1 | 0.0 -> 30.4 |
| Kang-Schafer | KS CI, n=200 | 27.3 -> 27.5 | 0.3 -> 0.1 | 25.0 -> 13.4 | 0.0 -> 9.7 |
| Kang-Schafer | KS CI, n=1000 | 26.4 -> 26.8 | 0.0 -> 0.0 | 41.8 -> 52.7 | 0.0 -> 25.9 |
| Kang-Schafer | KS IC, n=200 | 25.2 -> 25.1 | 17.0 -> 15.7 | 2.0 -> -0.8 | 1.8 -> 3.1 |
| Kang-Schafer | KS IC, n=1000 | 13.5 -> 12.6 | 10.7 -> 9.1 | 7.2 -> 12.5 | 0.0 -> 14.2 |
| Kang-Schafer | KS II, n=200 | 26.2 -> 26.5 | 30.0 -> 32.4 | 4.4 -> 12.3 | 0.0 -> 6.7 |
| Kang-Schafer | KS II, n=1000 | 16.4 -> 14.2 | 32.9 -> 38.3 | 5.6 -> 15.8 | 0.0 -> 16.3 |
| IHDP | IHDP A, strength=0 | 1.4 -> 1.4 | -0.6 -> -0.6 | 2.5 -> 0.7 | 0.0 -> 0.0 |
| IHDP | IHDP A, strength=3 | 4.5 -> 4.9 | 0.5 -> 0.7 | 1.6 -> 4.6 | 0.0 -> 0.0 |
| IHDP | IHDP B, strength=0 | 3.8 -> 4.0 | -0.6 -> -0.4 | 4.3 -> 5.3 | 0.0 -> 0.0 |
| IHDP | IHDP B, strength=3 | 4.8 -> 4.6 | 0.3 -> 0.2 | -1.5 -> -0.2 | 0.0 -> 0.0 |
| ACIC 2016 | ACIC 2016 A, strength=0 | -0.7 -> -0.8 | 0.4 -> -0.4 | 0.2 -> 0.1 | 0.0 -> 0.0 |
| ACIC 2016 | ACIC 2016 A, strength=3 | 1.2 -> 1.9 | 0.4 -> 1.3 | 0.3 -> 0.4 | 0.0 -> 0.1 |
| ACIC 2016 | ACIC 2016 B, strength=0 | 1.1 -> 1.1 | 1.2 -> 1.3 | 1.0 -> 2.7 | 0.0 -> 0.0 |
| ACIC 2016 | ACIC 2016 B, strength=3 | 0.7 -> 0.8 | -1.3 -> -1.5 | 0.6 -> -0.4 | 0.0 -> 0.0 |
| ACIC 2017 | ACIC 2017 A, strength=0 | 5.0 -> 5.0 | 0.9 -> 1.6 | 1.3 -> 1.9 | 0.0 -> 0.0 |
| ACIC 2017 | ACIC 2017 A, strength=3 | 2.0 -> 2.1 | 0.6 -> 1.9 | 0.0 -> 1.2 | 0.0 -> 0.0 |
| ACIC 2017 | ACIC 2017 B, strength=0 | -0.5 -> -0.5 | -2.1 -> -2.2 | -0.6 -> -0.6 | 0.0 -> 0.0 |
| ACIC 2017 | ACIC 2017 B, strength=3 | 0.9 -> 0.7 | 0.5 -> -0.1 | 1.1 -> -1.4 | 0.0 -> 0.0 |
| Twins | Twins A, strength=0 | 0.0 -> 0.0 | 6.2 -> 2.1 | 1.2 -> 0.2 | 0.0 -> 0.0 |
| Twins | Twins A, strength=3 | -0.1 -> -0.1 | 12.6 -> 18.9 | -0.6 -> 0.5 | 0.0 -> 0.0 |
| Twins | Twins B, strength=0 | 1.2 -> 1.2 | 6.0 -> 2.4 | 0.0 -> 0.1 | 0.0 -> 0.0 |
| Twins | Twins B, strength=3 | 0.1 -> 0.1 | 3.1 -> 3.9 | -0.5 -> -0.4 | 0.0 -> 0.0 |
