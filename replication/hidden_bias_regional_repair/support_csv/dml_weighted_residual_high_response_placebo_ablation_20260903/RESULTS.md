# Weighted-Residual Selector High-Response Placebo Ablation

This bundle replays the existing low-response and high-response placebo
ablation runs, but replaces the original score-selected gamma by
held-out weighted-residual gamma selection.  It does not refit the
upstream nuisances or overwrite the earlier placebo bundle.

## Summary

- AIPW: low-response 6.060% [4.609, 7.332], high-response 0.253% [-0.093, 0.600], low-minus-high 5.807% [4.345, 7.082], activation low/high 23.6%/25.2%.
- selective ML: low-response 1.598% [0.306, 2.791], high-response 0.432% [0.029, 0.807], low-minus-high 1.166% [-0.156, 2.407], activation low/high 29.0%/29.4%.
- Ma DR-BC: low-response 3.781% [2.605, 4.832], high-response 0.772% [0.206, 1.281], low-minus-high 3.009% [1.699, 4.230], activation low/high 23.6%/25.2%.
- C-TMLE: low-response 3.553% [2.805, 4.261], high-response -0.345% [-0.537, -0.151], low-minus-high 3.898% [3.099, 4.649], activation low/high 25.3%/25.9%.
