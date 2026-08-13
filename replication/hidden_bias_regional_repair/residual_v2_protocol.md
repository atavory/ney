# Frozen residual-v2 evaluation protocol

Frozen before inspecting any residual-v2 simulation output on 2026-08-12.

**Superseded before execution.** An independent audit established that the
certified residual constructor applied its correction globally even though
the manuscript described it as regional. No residual-v2 simulation was run
under this draft. The operative protocol is
`regional_residual_v2_protocol.md`.

## Candidate construction

The repair remains an outcome-residual repair. It does not project the
reference influence score onto an unrestricted mean-zero control.

- For a missing-outcome mean, fit `Y - m0(X)` among responders on folds that
  exclude the prediction fold, using the MAR variance-risk weight
  `(1-p(X))/p(X)^2`. Set `m_gamma(X) = m0(X) + gamma h(X)`.
- For the published Ma ATT design, fit `Delta Y - m0(X)` among comparison
  units on folds that exclude the prediction fold, using the treated-target
  transport weight `p(X)/(1-p(X))`. Set
  `m_gamma(X) = m0(X) + gamma h(X)`.
- Use the same damping path for both applications:
  `gamma in {0, .01, .025, .05, .1, .25, .5, 1}`.

## Endpoint and score

- For TMLE, obtain each candidate endpoint through the exact AIPW-score
  contrast induced by `m_gamma`, around the frozen TMLE endpoint. Evaluate
  candidate risk with the full AIPW score of the targeted outcome model.
- For Ma, recompute the published DR-BC boundary-corrected functional for
  every `m_gamma`. Evaluate candidate risk by adding the candidate-minus-base
  direct-score contrast to the complete Ma reference score. This retains the
  reference propensity and outcome first-stage influence terms and changes
  only the candidate outcome model.
- Center every candidate at the same reference-score mean. Candidate-specific
  recentering is forbidden because it could hide a nonzero endpoint change.

## Selection rule

Select a nonzero gamma only when its paired held-out reduction in squared
score loss exceeds `2.83` standard errors. This is the fixed one-sided
Bonferroni threshold for three held-out folds and seven nonzero gamma values.
Among eligible values, choose the smallest mean squared score loss, breaking
ties toward the smaller gamma. Otherwise return gamma zero.

The selector and all preprocessing decisions are shared by TMLE and Ma. The
rule may stand down for either estimator. No estimator-specific fallback to
influence projection is permitted.

## Evaluation

- Use three folds and XGBoost, matching the certified breadth experiments.
- Run structural and identity tests before simulations.
- Use seed ranges disjoint from every development and certified run.
- Smoke tests may check execution only. Do not revise the rule after viewing
  their accuracy results.
- Evaluate TMLE on the previously certified Kang--Schafer and published Cui
  designs. Evaluate Ma on published DGPs 2 and 3.
- Report reference and repaired squared error, activation, selected gamma,
  score-risk change, and individual-harm rate. Report null and adverse results
  as generated.
