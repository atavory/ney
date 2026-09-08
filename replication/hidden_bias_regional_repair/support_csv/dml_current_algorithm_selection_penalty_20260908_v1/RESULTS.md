# Current Algorithm 1 Selection-Penalty Experiment

This non-overwriting experiment runs the current full-data cross-fitted
weighted-residual selector.  It compares the actual same-sample returned
estimator with the gamma-zero baseline and with the selected candidate's
fresh-sample risk.

## Summary

- correct_response_clear, n=2000, reps=4000: same-sample MSE gain 37.072%, fresh-risk gain 33.322%, oracle fresh-risk gain 33.324%, mean same-sample penalty -5.21112e-05 (-0.030x selected fresh risk), fresh harm 0.000, same-sample harm 0.376, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
- correct_response_clear, n=8000, reps=1500: same-sample MSE gain 34.103%, fresh-risk gain 33.544%, oracle fresh-risk gain 33.544%, mean same-sample penalty 1.74194e-05 (0.040x selected fresh risk), fresh harm 0.000, same-sample harm 0.387, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
- correct_response_clear, n=32000, reps=400: same-sample MSE gain 33.730%, fresh-risk gain 33.601%, oracle fresh-risk gain 33.601%, mean same-sample penalty 4.92684e-06 (0.045x selected fresh risk), fresh harm 0.000, same-sample harm 0.378, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
- correct_response_near_tie, n=2000, reps=4000: same-sample MSE gain -0.308%, fresh-risk gain 0.358%, oracle fresh-risk gain 0.789%, mean same-sample penalty 3.77206e-05 (0.024x selected fresh risk), fresh harm 0.060, same-sample harm 0.285, selected shares gamma0=0.445, gamma0.25=0.107, gamma0.5=0.261, gamma1=0.188.
- correct_response_near_tie, n=8000, reps=1500: same-sample MSE gain 0.840%, fresh-risk gain 0.883%, oracle fresh-risk gain 0.983%, mean same-sample penalty 6.77207e-06 (0.017x selected fresh risk), fresh harm 0.002, same-sample harm 0.472, selected shares gamma0=0.029, gamma0.25=0.027, gamma0.5=0.170, gamma1=0.773.
- correct_response_near_tie, n=32000, reps=400: same-sample MSE gain 0.129%, fresh-risk gain 1.051%, oracle fresh-risk gain 1.052%, mean same-sample penalty 6.88374e-06 (0.070x selected fresh risk), fresh harm 0.000, same-sample harm 0.492, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
- misspecified_response_tradeoff, n=2000, reps=4000: same-sample MSE gain 29.822%, fresh-risk gain 33.948%, oracle fresh-risk gain 33.948%, mean same-sample penalty -0.00011894 (-0.062x selected fresh risk), fresh harm 0.000, same-sample harm 0.401, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
- misspecified_response_tradeoff, n=8000, reps=1500: same-sample MSE gain 31.909%, fresh-risk gain 37.631%, oracle fresh-risk gain 37.631%, mean same-sample penalty -1.20145e-05 (-0.025x selected fresh risk), fresh harm 0.000, same-sample harm 0.393, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
- misspecified_response_tradeoff, n=32000, reps=400: same-sample MSE gain 49.226%, fresh-risk gain 48.598%, oracle fresh-risk gain 48.598%, mean same-sample penalty -6.86868e-06 (-0.058x selected fresh risk), fresh harm 0.000, same-sample harm 0.372, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
- clipped_response_tail, n=2000, reps=4000: same-sample MSE gain 86.767%, fresh-risk gain 87.803%, oracle fresh-risk gain 87.803%, mean same-sample penalty 0.000166911 (0.128x selected fresh risk), fresh harm 0.000, same-sample harm 0.112, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
- clipped_response_tail, n=8000, reps=1500: same-sample MSE gain 95.787%, fresh-risk gain 96.565%, oracle fresh-risk gain 96.565%, mean same-sample penalty 7.35059e-05 (0.226x selected fresh risk), fresh harm 0.000, same-sample harm 0.015, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
- clipped_response_tail, n=32000, reps=400: same-sample MSE gain 99.011%, fresh-risk gain 99.108%, oracle fresh-risk gain 99.108%, mean same-sample penalty 9.60968e-06 (0.118x selected fresh risk), fresh harm 0.000, same-sample harm 0.000, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
