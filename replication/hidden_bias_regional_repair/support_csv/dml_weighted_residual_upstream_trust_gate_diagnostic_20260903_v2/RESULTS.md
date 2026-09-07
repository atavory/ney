# Upstream Trust Gate Diagnostic

This retrospective diagnostic forces the repair to stand down when an
estimator-agnostic upstream diagnostic fails.  It does not inspect the
method name and it does not refit the repair path.

## Upstream Metrics

- aipw: floor sensitivity 3.407 SE, residual ESS 0.044, floor share 0.140, max residual-weight share 0.027, score tail 14.448
- ctmle: floor sensitivity 1.270 SE, residual ESS 0.223, floor share 0.333, max residual-weight share 0.006, score tail 5.495
- cui_selective_ml: floor sensitivity 0.000 SE, residual ESS 0.136, floor share 0.005, max residual-weight share 0.086, score tail 9.772
- ma_dr_bc: floor sensitivity 4.281 SE, residual ESS 0.044, floor share 0.140, max residual-weight share 0.027, score tail 13.814
- tmle: floor sensitivity 0.434 SE, residual ESS 0.044, floor share 0.140, max residual-weight share 0.027, score tail 14.361

## Best Rules

- veto_stable_floor_sens_le_1_floor_share_ge_0.05_resid_ess_le_0.2: AIPW 25.920%, selective 17.136%, C-TMLE 9.587%, fixed-floor TMLE -0.210% [-0.700%, 0.000%], Ma 12.735%, AIPW/selective/Ma retention 1.015/1.004/0.983, TMLE activation 0.5%, pass=True
- veto_stable_floor_sens_le_0.75_floor_share_ge_0.05_resid_ess_le_0.2: AIPW 26.036%, selective 17.136%, C-TMLE 9.587%, fixed-floor TMLE -0.299% [-0.794%, -0.005%], Ma 12.823%, AIPW/selective/Ma retention 1.020/1.004/0.990, TMLE activation 1.3%, pass=True
- veto_stable_floor_sens_le_1.25_floor_share_ge_0.05_resid_ess_le_0.2: AIPW 25.393%, selective 17.136%, C-TMLE 9.621%, fixed-floor TMLE -0.210% [-0.712%, 0.000%], Ma 12.769%, AIPW/selective/Ma retention 0.995/1.004/0.986, TMLE activation 0.3%, pass=True
- veto_stable_floor_sens_le_1_floor_share_ge_0.1_resid_ess_le_0.2: AIPW 25.920%, selective 17.073%, C-TMLE 9.587%, fixed-floor TMLE -0.319% [-0.899%, 0.035%], Ma 12.735%, AIPW/selective/Ma retention 1.015/1.000/0.983, TMLE activation 1.3%, pass=True
- veto_stable_floor_sens_le_0.75_floor_share_ge_0.1_resid_ess_le_0.2: AIPW 26.036%, selective 17.073%, C-TMLE 9.587%, fixed-floor TMLE -0.408% [-1.023%, -0.004%], Ma 12.823%, AIPW/selective/Ma retention 1.020/1.000/0.990, TMLE activation 2.0%, pass=True
- veto_stable_floor_sens_le_1.25_floor_share_ge_0.1_resid_ess_le_0.2: AIPW 25.394%, selective 17.073%, C-TMLE 9.621%, fixed-floor TMLE -0.319% [-0.904%, 0.036%], Ma 12.769%, AIPW/selective/Ma retention 0.995/1.000/0.986, TMLE activation 1.1%, pass=True
- veto_stable_floor_sens_le_0.5_floor_share_ge_0.05_resid_ess_le_0.2: AIPW 25.731%, selective 17.136%, C-TMLE 9.637%, fixed-floor TMLE -0.477% [-1.458%, 0.346%], Ma 12.945%, AIPW/selective/Ma retention 1.008/1.004/1.000, TMLE activation 4.3%, pass=True
- veto_stable_floor_sens_le_0.5_floor_share_ge_0.1_resid_ess_le_0.2: AIPW 25.731%, selective 17.073%, C-TMLE 9.637%, fixed-floor TMLE -0.594% [-1.614%, 0.261%], Ma 12.945%, AIPW/selective/Ma retention 1.008/1.000/1.000, TMLE activation 4.9%, pass=True
- veto_stable_floor_sens_le_0.5_floor_share_ge_0.05: AIPW 25.731%, selective 17.136%, C-TMLE 8.370%, fixed-floor TMLE -0.477% [-1.457%, 0.354%], Ma 12.945%, AIPW/selective/Ma retention 1.008/1.004/1.000, TMLE activation 4.3%, pass=True
- veto_stable_floor_sens_le_0.5_floor_share_ge_0.1: AIPW 25.731%, selective 17.073%, C-TMLE 8.370%, fixed-floor TMLE -0.594% [-1.606%, 0.246%], Ma 12.945%, AIPW/selective/Ma retention 1.008/1.000/1.000, TMLE activation 4.9%, pass=True
- veto_stable_floor_sens_le_0.5_floor_share_ge_0.15_resid_ess_le_0.2: AIPW 25.738%, selective 17.073%, C-TMLE 9.637%, fixed-floor TMLE -1.205% [-3.907%, 1.670%], Ma 12.945%, AIPW/selective/Ma retention 1.008/1.000/1.000, TMLE activation 13.0%, pass=True
- veto_stable_floor_sens_le_0.75_floor_share_ge_0.15_resid_ess_le_0.2: AIPW 26.084%, selective 17.073%, C-TMLE 9.587%, fixed-floor TMLE -1.250% [-3.914%, 1.604%], Ma 12.869%, AIPW/selective/Ma retention 1.022/1.000/0.994, TMLE activation 11.7%, pass=True
- veto_stable_floor_sens_le_1_floor_share_ge_0.15_resid_ess_le_0.2: AIPW 26.057%, selective 17.073%, C-TMLE 9.587%, fixed-floor TMLE -1.234% [-3.976%, 1.618%], Ma 12.792%, AIPW/selective/Ma retention 1.021/1.000/0.988, TMLE activation 11.4%, pass=True
- veto_stable_floor_sens_le_1.25_floor_share_ge_0.15_resid_ess_le_0.2: AIPW 25.826%, selective 17.073%, C-TMLE 9.621%, fixed-floor TMLE -1.234% [-4.000%, 1.495%], Ma 12.732%, AIPW/selective/Ma retention 1.012/1.000/0.983, TMLE activation 11.3%, pass=True
- veto_stable_floor_sens_le_0.75_floor_share_ge_0.05: AIPW 26.036%, selective 17.136%, C-TMLE 6.309%, fixed-floor TMLE -0.299% [-0.799%, -0.009%], Ma 12.823%, AIPW/selective/Ma retention 1.020/1.004/0.990, TMLE activation 1.3%, pass=True

## Method Readout for veto_stable_floor_sens_le_1_floor_share_ge_0.05_resid_ess_le_0.2

- aipw: current 25.525%, gated 25.920% [19.428%, 32.435%], activation 23.7% vs 27.4%, harm 8.6% vs 10.1%
- ctmle: current 9.637%, gated 9.587% [7.254%, 12.160%], activation 26.4% vs 26.6%, harm 8.5% vs 8.5%
- cui_selective_ml: current 17.073%, gated 17.136% [10.850%, 23.370%], activation 34.7% vs 35.2%, harm 14.2% vs 14.5%
- ma_dr_bc: current 12.951%, gated 12.735% [6.434%, 18.779%], activation 26.7% vs 27.4%, harm 10.0% vs 10.2%
- tmle: current -21.372%, gated -0.210% [-0.700%, 0.000%], activation 0.5% vs 24.7%, harm 0.2% vs 13.0%
