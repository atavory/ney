# Honest-Split Selected-Gamma Experiment

This non-overwriting experiment uses a train/validation/estimation split.
The reference outcome model and residual direction are fit on training
data, gamma is selected on validation data by the one-SE weighted-residual
rule, and the returned AIPW estimate is computed on an untouched
estimation sample.

## Summary

- correct_response_clear, train/validation/estimation=1000/500/500, reps=3000: MSE gain 33.128%, fresh-risk gain 32.831%, oracle fresh-risk gain 33.170%, fresh violations 0/3000, estimation squared-error over RHS 691/3000, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.058, gamma1=0.942.
- correct_response_clear, train/validation/estimation=4000/2000/2000, reps=1200: MSE gain 33.541%, fresh-risk gain 33.441%, oracle fresh-risk gain 33.441%, fresh violations 0/1200, estimation squared-error over RHS 333/1200, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
- correct_response_clear, train/validation/estimation=16000/8000/8000, reps=300: MSE gain 31.557%, fresh-risk gain 33.576%, oracle fresh-risk gain 33.576%, fresh violations 0/300, estimation squared-error over RHS 85/300, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
- correct_response_near_tie, train/validation/estimation=1000/500/500, reps=3000: MSE gain 0.666%, fresh-risk gain 0.303%, oracle fresh-risk gain 0.850%, fresh violations 0/3000, estimation squared-error over RHS 854/3000, selected shares gamma0=0.632, gamma0.25=0.048, gamma0.5=0.092, gamma1=0.228.
- correct_response_near_tie, train/validation/estimation=4000/2000/2000, reps=1200: MSE gain 0.469%, fresh-risk gain 0.646%, oracle fresh-risk gain 0.984%, fresh violations 0/1200, estimation squared-error over RHS 383/1200, selected shares gamma0=0.299, gamma0.25=0.078, gamma0.5=0.203, gamma1=0.419.
- correct_response_near_tie, train/validation/estimation=16000/8000/8000, reps=300: MSE gain 0.482%, fresh-risk gain 1.007%, oracle fresh-risk gain 1.064%, fresh violations 0/300, estimation squared-error over RHS 98/300, selected shares gamma0=0.017, gamma0.25=0.010, gamma0.5=0.193, gamma1=0.780.
- misspecified_response_tradeoff, train/validation/estimation=1000/500/500, reps=3000: MSE gain 33.504%, fresh-risk gain 32.573%, oracle fresh-risk gain 33.039%, fresh violations 0/3000, estimation squared-error over RHS 233/3000, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.076, gamma1=0.924.
- misspecified_response_tradeoff, train/validation/estimation=4000/2000/2000, reps=1200: MSE gain 36.179%, fresh-risk gain 34.217%, oracle fresh-risk gain 34.240%, fresh violations 0/1200, estimation squared-error over RHS 122/1200, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.003, gamma1=0.997.
- misspecified_response_tradeoff, train/validation/estimation=16000/8000/8000, reps=300: MSE gain 38.827%, fresh-risk gain 37.657%, oracle fresh-risk gain 37.657%, fresh violations 0/300, estimation squared-error over RHS 41/300, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
- clipped_response_tail, train/validation/estimation=1000/500/500, reps=3000: MSE gain 65.262%, fresh-risk gain 66.544%, oracle fresh-risk gain 67.037%, fresh violations 0/3000, estimation squared-error over RHS 12/3000, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.043, gamma1=0.957.
- clipped_response_tail, train/validation/estimation=4000/2000/2000, reps=1200: MSE gain 87.674%, fresh-risk gain 88.082%, oracle fresh-risk gain 88.090%, fresh violations 0/1200, estimation squared-error over RHS 4/1200, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.001, gamma1=0.999.
- clipped_response_tail, train/validation/estimation=16000/8000/8000, reps=300: MSE gain 96.801%, fresh-risk gain 96.615%, oracle fresh-risk gain 96.615%, fresh violations 0/300, estimation squared-error over RHS 0/300, selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000.
