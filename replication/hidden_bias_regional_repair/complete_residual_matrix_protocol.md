# Complete residual-rule comparison protocol

Frozen before scientific output on 2026-08-13.

## Question

Choose one outcome-residual adapter for every compatible upstream estimator.
The comparison crosses two rules with the complete applicable estimator and
design matrix. It never selects a rule separately by estimator.

The two rules differ only in support:

- **Global residual:** apply the cross-fitted residual direction to every unit.
- **Regional residual:** apply the same direction only inside the frozen,
  responder-supported low-response proposal.

Both rules use the same nuisance fits, residual weights, candidate path,
held-out score-risk gate, scalar shrinkage, observations, folds, and seeds.
Neither rule uses an influence-projection fallback.

## Missing-outcome matrix

Each rule is crossed with plain TMLE, adaptive C-TMLE, AIPW, and faithful Cui
selective ML on every missing-outcome cell:

- Kang--Schafer: four specification combinations at `n = 200, 1000` (8 cells).
- Published Cui scenarios: two scenarios at `n = 250, 500, 1000, 2000`
  (8 cells).
- Placement stress: one shared null and three nonzero strengths in each of the
  aligned, partial, and disjoint geometries (10 cells).
- Public real covariates: digits and breast cancer, each in wider/partial and
  aligned geometries at strengths `0, 1, 2` (12 cells).
- Regional-shift anchor: strengths `0, 3, 5, 8` (4 cells).

This gives 42 design cells, 4 estimators, and 2 rules: 336 estimator--rule
cells. Each cell contains 96 paired replications (24 chunks of 4), for 32,256
missing-outcome rows. Every estimator and rule receives the exact same sample
seed within a design cell and replication.

The missing-outcome rules use three folds, XGBoost 3.4.0 nuisance learners,
50 whole-procedure bootstrap draws per replication, the candidate path
`{0,.01,.025,.05,.1,.25,.5,1}`, the fixed 2.83-SE held-out score-risk gate,
and final scalar shrinkage with `c = 2`.

## Ma matrix

Both residual scopes are applied to the published Ma DR-BC ATT estimator on
DGPs 2 and 3. Each DGP--rule cell contains 384 replications at `n = 10,000`,
for 1,536 rows. The candidate recomputes Ma's boundary-corrected functional;
it does not substitute an ordinary AIPW endpoint. It uses the same gamma path
and 2.83-SE held-out full-score-risk gate as the missing-outcome comparison.

## Seeds

- Breadth designs: base `3,900,000,000`; offsets depend only on design cell
  and chunk.
- Published Cui scenarios: base `4,000,000,000`; offsets depend only on design
  cell and chunk.
- Ma DGPs: base `4,100,000,000`; offsets depend only on DGP, chunk, and
  replication.

These ranges are disjoint from development and earlier releases. Rule and
estimator labels never enter a seed.

## Reporting and decision

The release retains every row, failure, empty proposal, and exact stand-down.
For each rule, estimator, and design family it reports:

- unrepaired and repaired absolute MSE;
- paired relative MSE change with a stratified bootstrap interval;
- activation and individual-harm rates;
- detected-region and repair-support mass;
- the complete per-cell ranking of unrepaired and repaired estimators.

We will inspect both complete matrices before choosing one rule. We will not
mix scopes across estimators. Because this matrix is used to choose the rule,
its winning result is not labeled confirmatory. After the choice, the selected
single rule must pass a fresh-seed confirmation before it replaces the paper's
current results.

Execution-only smoke tests used separate seeds and cannot enter any summary.
