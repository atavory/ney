# Section 4 Same-Sample Selection Penalty

This derived diagnostic reads the Section 3 bound replay rows for the
paper-facing benchmark fits.  It compares actual same-sample squared
error with the fresh-risk decomposition `bias^2 + A` for the same
selected gamma.

## Overall

- Primary experts: same-sample MSE gain 18.741%, fresh-risk gain 18.729%, gain gap 0.012 percentage points, active share 0.291, same-sample harm 0.108, fresh-risk harm 0.051.
- Fixed-floor TMLE: same-sample MSE gain -21.372%, fresh-risk gain 7.240%, gain gap -28.612 percentage points, active share 0.247, same-sample harm 0.130, fresh-risk harm 0.054.

## By Method

- aipw: same-sample MSE gain 25.525%, fresh-risk gain 17.103%, gain gap 8.422 percentage points, active 0.274, same-sample harm 0.101, fresh-risk harm 0.050.
- ctmle: same-sample MSE gain 9.637%, fresh-risk gain 9.276%, gain gap 0.361 percentage points, active 0.266, same-sample harm 0.085, fresh-risk harm 0.033.
- cui_selective_ml: same-sample MSE gain 17.073%, fresh-risk gain 18.729%, gain gap -1.656 percentage points, active 0.352, same-sample harm 0.145, fresh-risk harm 0.068.
- ma_dr_bc: same-sample MSE gain 12.951%, fresh-risk gain 17.103%, gain gap -4.152 percentage points, active 0.274, same-sample harm 0.102, fresh-risk harm 0.050.
- tmle: same-sample MSE gain -21.372%, fresh-risk gain 7.240%, gain gap -28.612 percentage points, active 0.247, same-sample harm 0.130, fresh-risk harm 0.054.
