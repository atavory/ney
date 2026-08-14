# Frozen regional-residual-v2 evaluation protocol

Frozen before inspecting any regional-residual-v2 simulation output on
2026-08-12. This protocol supersedes `residual_v2_protocol.md` before that
draft was executed.

**Historical protocol as of 2026-08-14.** It is retained to reproduce the
regional-residual diagnostic, but it is not the paper's prospective universal
rule. It evaluates different experts on different native datasets and cannot
support a Cartesian cross-expert claim. The 2.83-SE normal-theory gate also
requires exact paired comparison and tail-calibration diagnostics before it
can be called conservative. Current work is documented in
`UNIFIED_REPAIR_RESEARCH_LOG_20260814.md`.

## Region and candidate construction

The repair changes the outcome model only inside a supported, estimated
low-response region. It never substitutes a score-projection adapter.

- Construct the region with the already frozen
  `estimated_residual_lowp_supported` detector: rank cross-fitted residual
  evidence within the lowest estimated-response decile, consider the fixed
  25%, 50%, 75%, and 100% prefixes, require at least 30 responders, and return
  the empty region when no prefix clears the fixed detector threshold.
- For a missing-outcome mean, fit `Y - m0(X)` among responders on folds that
  exclude the prediction fold, using `(1-p(X))/p(X)^2` weights.
- For the published Ma ATT design, treat comparison units as responders. Fit
  `Delta Y - m0(X)` among comparison units on folds that exclude the
  prediction fold, using the equivalent response-probability weight
  `p_treat(X)/(1-p_treat(X))^2`.
- In both cases define
  `m_gamma(X) = m0(X) + gamma h(X) 1{X in G}`. An empty region makes every
  candidate identical to the reference.
- Use the same damping path in both applications:
  `gamma in {0, .01, .025, .05, .1, .25, .5, 1}`.

## Endpoint and score

- For TMLE, obtain each candidate endpoint through the exact AIPW-score
  contrast induced by `m_gamma`, around the frozen TMLE endpoint. Evaluate
  candidate risk with the full AIPW score of the targeted outcome model.
- For Ma, recompute the published DR-BC boundary-corrected functional for
  every `m_gamma`. Evaluate candidate risk by adding the candidate-minus-base
  direct-score contrast to the complete Ma reference score. This retains the
  reference first-stage influence terms and changes only the candidate
  outcome model.
- Center every candidate at the same reference-score mean. Candidate-specific
  recentering is forbidden.

## Selection rule

Select a nonzero gamma only when its paired held-out reduction in squared
score loss exceeds `2.83` standard errors, the fixed one-sided Bonferroni
threshold for three held-out folds and seven nonzero gamma values. Among
eligible values, choose the smallest mean loss, breaking ties toward the
smaller gamma. Otherwise return gamma zero.

The region rule, residual construction, damping path, and score-risk gate are
shared by TMLE and Ma. No estimator-specific fallback is permitted.

## Evaluation

- Use three folds and XGBoost.
- Run structural and identity tests before simulations.
- Use seed ranges disjoint from all development and certified runs.
- Smoke tests check execution only and cannot change the rule.
- Evaluate TMLE on the certified Kang--Schafer and published Cui designs.
  Evaluate Ma on published DGPs 2 and 3.
- Report reference and repaired squared error, activation, region stand-down,
  selected gamma, score-risk change, and individual-harm rate, including null
  or adverse results.
