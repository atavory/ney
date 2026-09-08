# Section 3 Bounds Diagnostic

This retrospective diagnostic evaluates the numerical quantities in
Subsections 3.2 and 3.3 on the paper-facing benchmark fits. It uses true
`pi` and `mu` from the simulation designs and is not a selection rule.

The variance and bias inequalities are checked on the empirical
distribution of the generated covariates, using the additive unit noise
variance in the benchmark data generators.

## Overall

- Rows: 9216
- Nonzero selected-gamma share: 0.291
- Harm share: 0.108
- Active-move harm share: 0.369
- Variance-bound violations: 0
- Squared-bias-bound violations: 0
- Variance sufficient-condition share: 0.000
- Positive weighted-risk-improvement share: 0.265
- Active positive weighted-risk-improvement share: 0.908
- Positive average-variance-reduction share: 0.205
- Active positive average-variance-reduction share: 0.703
- Positive hidden-squared-bias-reduction share: 0.219
- Active positive hidden-squared-bias-reduction share: 0.751
- Mean hidden squared-bias change: 1300.6
- Mean average-variance reduction: 62965.1

## By Expert

- aipw: rows 2304, variance violations 0, bias violations 0, variance certified 0.000, active 0.274, active harm 0.368, active positive weighted-risk improvement 0.875, active positive variance reduction 0.891, active positive squared-bias reduction 0.693
- ctmle: rows 2304, variance violations 0, bias violations 0, variance certified 0.000, active 0.266, active harm 0.320, active positive weighted-risk improvement 0.992, active positive variance reduction 0.074, active positive squared-bias reduction 0.910
- cui_selective_ml: rows 2304, variance violations 0, bias violations 0, variance certified 0.000, active 0.352, active harm 0.411, active positive weighted-risk improvement 0.895, active positive variance reduction 0.887, active positive squared-bias reduction 0.722
- ma_dr_bc: rows 2304, variance violations 0, bias violations 0, variance certified 0.000, active 0.274, active harm 0.365, active positive weighted-risk improvement 0.875, active positive variance reduction 0.891, active positive squared-bias reduction 0.693
