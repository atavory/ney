# Weighted-Residual Selector Response-Bin Action and Reward Decomposition

This diagnostic decomposes two quantities by true response-probability
quartiles within each generated sample.  Bin 1 is the lowest-response
quartile and bin 4 is the highest-response quartile.

Action is where the repair moves the fitted outcome surface:
(m_selected - m_ref)^2, with both raw and response-weighted shares
reported.  Reward is where that movement reduces true
response-weighted residual-square:
E[(1-pi)/pi * (m_hat-mu)^2].  Each bin entry is an unconditional
contribution, so the four bins add back to the corresponding total.

This is the direct check for the low-response story.  The rho diagnostic
is only a response-model misspecification severity check.
The supported claim is about magnitude: the positive average
response-weighted residual-square delta is concentrated in the
lowest-response quartile.  It is not a claim that low-response bins
improve more often in every setting; the positive/negative/zero counts
are reported separately.

## All Experts

- bin 1 (lowest_response): mean pi 0.245, action share 0.422 raw / 0.854 weighted, reward share 0.921, positive/negative/zero 37/44/15
- bin 2 (response_bin_2): mean pi 0.627, action share 0.213 raw / 0.097 weighted, reward share 0.057, positive/negative/zero 33/48/15
- bin 3 (response_bin_3): mean pi 0.731, action share 0.169 raw / 0.035 weighted, reward share 0.016, positive/negative/zero 33/48/15
- bin 4 (highest_response): mean pi 0.843, action share 0.197 raw / 0.014 weighted, reward share 0.006, positive/negative/zero 31/50/15

## By Expert

- aipw, bin 1: mean pi 0.245, action share 0.414 raw / 0.854 weighted, reward share 0.963, positive/negative/zero 9/15/0
- aipw, bin 2: mean pi 0.627, action share 0.205 raw / 0.094 weighted, reward share 0.028, positive/negative/zero 6/18/0
- aipw, bin 3: mean pi 0.731, action share 0.177 raw / 0.037 weighted, reward share 0.009, positive/negative/zero 6/18/0
- aipw, bin 4: mean pi 0.843, action share 0.204 raw / 0.015 weighted, reward share 0.001, positive/negative/zero 4/20/0
- ctmle, bin 1: mean pi 0.245, action share 0.385 raw / 0.839 weighted, reward share 0.922, positive/negative/zero 8/1/15
- ctmle, bin 2: mean pi 0.627, action share 0.213 raw / 0.102 weighted, reward share 0.050, positive/negative/zero 8/1/15
- ctmle, bin 3: mean pi 0.731, action share 0.187 raw / 0.042 weighted, reward share 0.020, positive/negative/zero 8/1/15
- ctmle, bin 4: mean pi 0.843, action share 0.215 raw / 0.017 weighted, reward share 0.008, positive/negative/zero 8/1/15
- cui_selective_ml, bin 1: mean pi 0.245, action share 0.443 raw / 0.859 weighted, reward share 0.895, positive/negative/zero 11/13/0
- cui_selective_ml, bin 2: mean pi 0.627, action share 0.220 raw / 0.098 weighted, reward share 0.078, positive/negative/zero 13/11/0
- cui_selective_ml, bin 3: mean pi 0.731, action share 0.154 raw / 0.031 weighted, reward share 0.020, positive/negative/zero 13/11/0
- cui_selective_ml, bin 4: mean pi 0.843, action share 0.182 raw / 0.012 weighted, reward share 0.008, positive/negative/zero 15/9/0
- ma_dr_bc, bin 1: mean pi 0.245, action share 0.414 raw / 0.854 weighted, reward share 0.963, positive/negative/zero 9/15/0
- ma_dr_bc, bin 2: mean pi 0.627, action share 0.205 raw / 0.094 weighted, reward share 0.028, positive/negative/zero 6/18/0
- ma_dr_bc, bin 3: mean pi 0.731, action share 0.177 raw / 0.037 weighted, reward share 0.009, positive/negative/zero 6/18/0
- ma_dr_bc, bin 4: mean pi 0.843, action share 0.204 raw / 0.015 weighted, reward share 0.001, positive/negative/zero 4/20/0
