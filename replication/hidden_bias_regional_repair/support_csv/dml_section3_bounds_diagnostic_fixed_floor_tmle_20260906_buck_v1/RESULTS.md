# Section 3 Bounds Diagnostic

This retrospective diagnostic evaluates the numerical quantities in
Subsections 3.2 and 3.3 on the paper-facing benchmark fits. It uses true
`pi` and `mu` from the simulation designs and is not a selection rule.

The variance and bias inequalities are checked on the empirical
distribution of the generated covariates, using the additive unit noise
variance in the benchmark data generators.

## Overall

- Rows: 2304
- Nonzero selected-gamma share: 0.247
- Harm share: 0.130
- Active-move harm share: 0.525
- Variance-bound violations: 0
- Squared-bias-bound violations: 0
- Variance sufficient-condition share: 0.000
- Positive weighted-risk-improvement share: 0.213
- Active positive weighted-risk-improvement share: 0.860
- Positive average-variance-reduction share: 0.217
- Active positive average-variance-reduction share: 0.877
- Positive hidden-squared-bias-reduction share: 0.148
- Active positive hidden-squared-bias-reduction share: 0.598
- Mean hidden squared-bias change: 0.412655
- Mean average-variance reduction: 0.768106

## By Expert

- tmle: rows 2304, variance violations 0, bias violations 0, variance certified 0.000, active 0.247, active harm 0.525, active positive weighted-risk improvement 0.860, active positive variance reduction 0.877, active positive squared-bias reduction 0.598
