# Section 4 Same-Sample Selection Penalty

This derived diagnostic reads the Section 3 bound replay rows for the
paper-facing benchmark fits.  It compares actual same-sample squared
error with the fresh-risk decomposition `bias^2 + A` for the same
selected gamma.  The paper's primary estimand first computes each
setting-by-method gain and then averages those gains with equal weight.
The pooled ratio-of-means appears only as a labelled sensitivity check.

## Equal-setting result

- Primary experts: same-sample MSE gain 6.693%, fresh-risk gain 4.899%, gain gap 1.794 percentage points [0.180, 3.703], active share 0.291, same-sample harm 0.108, fresh-risk harm 0.051.
- Fixed-floor TMLE: same-sample MSE gain -7.670%, fresh-risk gain 3.741%, gain gap -11.411 percentage points [-21.510, -2.887], active share 0.247, same-sample harm 0.130, fresh-risk harm 0.054.
- Across the 96 primary setting-by-method cells, the mean absolute gain gap is 3.695 percentage points and the range is [-27.874, 63.523].
- Kang--Schafer contributes 460 of 570 fixed-floor TMLE active moves (80.7%).

## Pooled sensitivity (not the paper estimand)

- Primary experts: same-sample MSE gain 18.741%, fixed-candidate risk gain 18.729%, gap 0.012 percentage points.
- Fixed-floor TMLE: same-sample MSE gain -21.372%, fixed-candidate risk gain 7.240%, gap -28.612 percentage points.

## By Method

- aipw: same-sample MSE gain 8.949%, fresh-risk gain 7.002%, gain gap 1.947 percentage points, active 0.274, same-sample harm 0.101, fresh-risk harm 0.050.
- ctmle: same-sample MSE gain 4.979%, fresh-risk gain 4.848%, gain gap 0.131 percentage points, active 0.266, same-sample harm 0.085, fresh-risk harm 0.033.
- cui_selective_ml: same-sample MSE gain 5.186%, fresh-risk gain 0.742%, gain gap 4.444 percentage points, active 0.352, same-sample harm 0.145, fresh-risk harm 0.068.
- ma_dr_bc: same-sample MSE gain 7.658%, fresh-risk gain 7.002%, gain gap 0.655 percentage points, active 0.274, same-sample harm 0.102, fresh-risk harm 0.050.
- tmle: same-sample MSE gain -7.670%, fresh-risk gain 3.741%, gain gap -11.411 percentage points, active 0.247, same-sample harm 0.130, fresh-risk harm 0.054.
