# Cross-fitted nuisance accuracy and response dependence

## Scope

This diagnostic regenerates the current 24 benchmark settings:
eight Kang--Schafer settings, plus four settings each for IHDP, ACIC 2016,
ACIC 2017, and Twins.  Each setting is run for 32 replications with five
response-propensity bins.

The two nuisance fits are the cross-fitted response score `p_hat` and the
cross-fitted observed-outcome prediction `m_hat`, both using XGBoost and
three-fold cross-fitting.  The two nuisance fits use the same held-out folds,
stratified by rank bins of true `pi`, so each fold has comparable
response-propensity support.  Bins are formed two ways: by `p_hat`, which is
observable, and by true `pi`, which is a simulation diagnostic.

The expert-reference check additionally fits AIPW, C-TMLE, selective ML,
Ma DR-BC, and the fixed-floor TMLE diagnostic for 16 replications on the same
24 settings.  It records each reference outcome's true error by response bin,
the selected response score, and the fraction of observations pinned at the
0.05 floor.

## Dependence on response probability

The relevant question is not whether `p_hat` has lower absolute error than
`m_hat`.  The quantities have different scales.  The diagnostic asks whether
`m_hat` degrades more as response becomes scarce.

The primary fit is

```text
normalized squared true error = alpha + beta * (1 - true pi).
```

The normalization divides each nuisance error by its own setting-level average
error before fitting.  Thus the slope compares dependence on response scarcity,
not the absolute scale of response and outcome errors.

| dataset | settings | beta p | beta m | beta m - beta p | R2 p | R2 m |
|---|---:|---:|---:|---:|---:|---:|
| ACIC 2016 | 4 | 0.133 | 0.447 | 0.314 | 0.02 | 0.25 |
| ACIC 2017 | 4 | 0.083 | 0.405 | 0.322 | 0.01 | 0.23 |
| IHDP | 4 | -0.298 | 0.083 | 0.381 | 0.07 | 0.45 |
| Kang--Schafer | 8 | 0.015 | 1.445 | 1.430 | 0.00 | 0.38 |
| Twins | 4 | -0.628 | 0.372 | 1.000 | 0.18 | 0.42 |
| all | 24 | -0.113 | 0.700 | 0.813 | 0.05 | 0.35 |
| non-Kang--Schafer | 16 | -0.177 | 0.327 | 0.504 | 0.07 | 0.34 |

This is the cleanest version of the diagnostic.  The response-model error is
nearly flat, and in several sources decreases, as response gets scarce.  The
outcome-model error rises with response scarcity in every dataset average.
The outcome curve also has meaningfully higher fitted response-dependence.

| dataset | settings | low response | high response | low p nRMSE | high p nRMSE | low m nRMSE | high m nRMSE | low p CV skill | low m CV skill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACIC 2016 | 4 | 20.6% | 87.2% | 0.89 | 0.76 | 0.80 | 0.73 | 0.10 | 0.31 |
| ACIC 2017 | 4 | 20.6% | 87.2% | 0.88 | 0.77 | 0.80 | 0.73 | 0.11 | 0.32 |
| IHDP | 4 | 20.7% | 87.6% | 0.71 | 0.71 | 1.48 | 1.47 | 0.23 | -0.27 |
| Kang--Schafer | 8 | 18.6% | 81.9% | 1.00 | 0.99 | 0.55 | 0.37 | -0.34 | -5.84 |
| Twins | 4 | 5.1% | 87.4% | 0.39 | 0.48 | 0.95 | 0.84 | -0.30 | -0.16 |

`p nRMSE` is RMSE of `p_hat` against true `pi`, divided by the setting-level
standard deviation of true `pi`.  `m nRMSE` is RMSE of `m_hat` against true
`mu`, divided by the setting-level standard deviation of true `mu`.  CV skill
compares the model's observed cross-validated loss with a binwise constant
predictor; positive values improve on the binwise constant predictor.

The low-minus-high dependence summary is:

| dataset | p low-response excess | m low-response excess | m-minus-p excess |
|---|---:|---:|---:|
| ACIC 2016 | 0.13 | 0.08 | -0.05 |
| ACIC 2017 | 0.11 | 0.07 | -0.04 |
| IHDP | 0.00 | 0.01 | 0.00 |
| Kang--Schafer | 0.01 | 0.18 | 0.17 |
| Twins | -0.09 | 0.11 | 0.20 |

Across all 24 settings, `m_hat` has larger low-response excess than `p_hat`
on the normalized truth scale: 0.104 versus 0.029.  Excluding Kang--Schafer,
the same comparison is 0.068 versus 0.039.  This supports the qualitative
claim that outcome prediction is more response-sensitive, but the effect is
not uniform by dataset.

## Expert-reference check

The expert-reference output shows the same broad low-response pattern for the
reference outcome predictions.  For fixed-floor TMLE, the 0.05 floor pins many
low-response observations: about 50% in ACIC, IHDP, and Kang--Schafer, and
about 79% in Twins.  However, fixed-floor TMLE does not have a uniquely worse
low-response outcome-error ratio than C-TMLE in this diagnostic.  Their
low/high normalized outcome-error ratios are similar:

| dataset | TMLE ratio | C-TMLE ratio | TMLE low-bin floor fraction |
|---|---:|---:|---:|
| ACIC 2016 | 1.09 | 1.11 | 50.3% |
| ACIC 2017 | 1.08 | 1.09 | 53.8% |
| IHDP | 0.97 | 1.01 | 52.2% |
| Kang--Schafer | 1.42 | 1.47 | 47.6% |
| Twins | 1.13 | 1.14 | 79.0% |

So the fixed-floor TMLE loss in the Section 4 MSE table is not explained by a
larger low-response outcome prediction error alone.  The more plausible
mechanism remains interaction between the fixed floor, the score/weighting
geometry, and the repair selector's tendency to move when the reference score
looks improvable.

## Reading

The paper can use this as empirical motivation for relying on response-weighted
residual checks: response is observed for every unit, and outcome learning is
at least as response-sensitive as response-score learning in the aggregate.
It should not claim that `p_hat` is uniformly more accurate than `m_hat`, and
it should not attribute fixed-floor TMLE's MSE behavior solely to the 0.05 cap.
