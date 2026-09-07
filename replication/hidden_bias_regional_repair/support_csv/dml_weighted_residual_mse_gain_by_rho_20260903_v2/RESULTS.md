# Weighted-Residual Selector MSE Gain by Rho

This diagnostic joins selected-candidate MSE gains to the response-model
misspecification index rho_p used in Section 3.  It is retrospective:
rho_p is computed from the known true response probability in the
simulation designs and was not used to select the repair.

Primary endpoint: the gamma selected by held-out weighted residual loss.  The original score-selected endpoint is not used in this bundle.

Theory-facing residual endpoint: true response-weighted residual
square, E[(1-pi)/pi * (m_hat-mu)^2], computed for the reference
outcome prediction and for the selected repaired outcome prediction.
The residual-square delta is reference minus selected, so positive
values mean the repaired outcome prediction reduced this quantity.

For IHDP, ACIC 2016, ACIC 2017, and Twins, labels A and B denote the
two preexisting known-truth variants in the released experiment, each
at strengths 0 and 3.  The internal design names remain in the CSVs.

Interpretation: rho_p is a severity covariate, not a sufficient
condition for improvement.  The repair also needs residual signal that
the selected candidate can exploit.

## Pooled Readout

- Setting-method rows: 96
- Mean rho: 1.865
- Mean selected-candidate gain: 6.693%
- Mean weighted residual-square reduction: 4.577%
- Mean weighted residual-square delta: 58.6272
- Wins / losses / crossing zero: 34 / 0 / 62
- OLS slope of gain on rho: 0.864 percentage points per rho unit
- OLS slope of weighted residual-square reduction on rho: -0.791 percentage points per rho unit
- Pearson / Spearman: 0.111 / 0.070

## Rho Bins

- bin 1: rho 0.205-0.486, mean gain 6.010%, mean weighted-residual reduction 8.146%, mean weighted-residual delta 104.755, wins/losses/crossing 9/0/15
- bin 2: rho 0.496-1.703, mean gain 3.372%, mean weighted-residual reduction 2.460%, mean weighted-residual delta 57.7475, wins/losses/crossing 6/0/18
- bin 3: rho 1.703-3.109, mean gain 10.737%, mean weighted-residual reduction 5.452%, mean weighted-residual delta 37.1926, wins/losses/crossing 13/0/11
- bin 4: rho 3.325-4.192, mean gain 6.653%, mean weighted-residual reduction 2.252%, mean weighted-residual delta 34.814, wins/losses/crossing 6/0/18

## By Expert

- aipw: mean rho 3.167, mean gain 8.949%, mean weighted-residual reduction 3.653%, mean weighted-residual delta 35.9927, wins/losses/crossing 11/0/13, gain slope -1.239, residual slope -2.292
- ctmle: mean rho 0.446, mean gain 4.979%, mean weighted-residual reduction 6.037%, mean weighted-residual delta 49.6657, wins/losses/crossing 8/0/16, gain slope -109.468, residual slope -131.600
- cui_selective_ml: mean rho 0.682, mean gain 5.186%, mean weighted-residual reduction 4.966%, mean weighted-residual delta 112.858, wins/losses/crossing 8/0/16, gain slope 14.380, residual slope 9.457
- ma_dr_bc: mean rho 3.167, mean gain 7.658%, mean weighted-residual reduction 3.653%, mean weighted-residual delta 35.9927, wins/losses/crossing 7/0/17, gain slope -3.948, residual slope -2.292

## Largest Gains

- ma_dr_bc on KS CI, n=1000: 52.674% gain, 27.042% weighted-residual reduction, weighted-residual delta 175.358, rho 2.569
- ma_dr_bc on KS CC, n=1000: 47.069% gain, 25.401% weighted-residual reduction, weighted-residual delta 154.565, rho 2.503
- cui_selective_ml on KS II, n=1000: 38.250% gain, 54.378% weighted-residual reduction, weighted-residual delta 690.087, rho 1.215
- cui_selective_ml on KS II, n=200: 32.438% gain, 25.331% weighted-residual reduction, weighted-residual delta 696.074, rho 1.676
- aipw on KS CC, n=1000: 30.501% gain, 25.401% weighted-residual reduction, weighted-residual delta 154.565, rho 2.503
- ctmle on KS CC, n=1000: 30.424% gain, 35.587% weighted-residual reduction, weighted-residual delta 226.096, rho 0.359
- aipw on KS CI, n=200: 27.510% gain, 10.613% weighted-residual reduction, weighted-residual delta 134.001, rho 3.351
- aipw on KS CI, n=1000: 26.836% gain, 27.042% weighted-residual reduction, weighted-residual delta 175.358, rho 2.569

## Largest Losses

- cui_selective_ml on ACIC 2017 B, strength=0: -2.170% gain, -1.564% weighted-residual reduction, weighted-residual delta -0.0139833, rho 0.510
- cui_selective_ml on ACIC 2016 B, strength=3: -1.463% gain, 2.315% weighted-residual reduction, weighted-residual delta 0.0289218, rho 0.542
- ma_dr_bc on ACIC 2017 B, strength=3: -1.397% gain, -1.424% weighted-residual reduction, weighted-residual delta -0.0276266, rho 4.135
- ma_dr_bc on KS IC, n=200: -0.807% gain, 7.492% weighted-residual reduction, weighted-residual delta 106.564, rho 3.394
- aipw on ACIC 2016 A, strength=0: -0.770% gain, -0.489% weighted-residual reduction, weighted-residual delta -0.00806117, rho 4.192
- cui_selective_ml on IHDP A, strength=0: -0.633% gain, -0.236% weighted-residual reduction, weighted-residual delta -0.000388403, rho 0.499
- ma_dr_bc on ACIC 2017 B, strength=0: -0.564% gain, -0.781% weighted-residual reduction, weighted-residual delta -0.0127741, rho 4.128
- aipw on ACIC 2017 B, strength=0: -0.470% gain, -0.781% weighted-residual reduction, weighted-residual delta -0.0127741, rho 4.128
