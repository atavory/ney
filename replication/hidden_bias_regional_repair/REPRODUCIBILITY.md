# EJS Reproducibility Contract

This file is the public run contract for the EJS Section 4 empirical package.
It resolves which artifacts support the submitted numbers, what can be checked
from this repository alone, and what requires archived raw inputs.

## Authoritative Section 4 Bundle

The current paper-facing Section 4 source is:

```text
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/
```

That bundle is a replay of the archived Aug. 14 candidate paths.  It keeps the
same upstream fits, response predictions, fitted residual directions, and
candidate grid, then selects the reported candidate by held-out
response-weighted residual loss.

Do not use these older bundles for current manuscript numbers:

```text
support_csv/dml_unified_cartesian_global_residual_20260814/
support_csv/dml_section4_release_20260812_v1/
support_csv/dml_regional_residual_v2_20260812/
```

They remain audit trail and historical comparison only.

## Public and Archived Inputs

This public repository contains enough compact data to regenerate and verify
the submitted tables:

```text
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/weighted_residual_gamma_replication_rows.csv
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/weighted_residual_gamma_setting_summary.csv
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/weighted_residual_gamma_primary_table.csv
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/weighted_residual_gamma_primary_summary.csv
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/weighted_residual_gamma_method_summary.csv
```

The public package also contains the scripts that render the manuscript tables
and verify the paper values.

The public package also contains compact companion diagnostics used by
Section 4 and the empirical appendix:

```text
support_csv/dml_weighted_residual_mse_gain_by_rho_20260903_v2/
support_csv/dml_weighted_residual_high_response_placebo_ablation_20260903/
support_csv/dml_weighted_residual_augmented_gamma_grid_ablation_20260903/
support_csv/dml_weighted_residual_upstream_trust_gate_diagnostic_20260903_v2/
support_csv/dml_section3_bounds_diagnostic_20260908_theorem_fit_public_v1/
support_csv/dml_section3_bounds_diagnostic_fixed_floor_tmle_20260908_theorem_fit_public_v1/
support_csv/dml_selected_gamma_theorem_simulation_20260908_v2/
support_csv/dml_honest_split_selected_gamma_20260908_v2/
support_csv/dml_current_algorithm_selection_penalty_20260908_v1/
support_csv/dml_section4_same_sample_selection_penalty_20260908_v2/
```

The Section 4 same-sample bridge reports a 0.012 percentage-point gap between
actual same-sample MSE gain and fixed-candidate replay gain for the four
primary expert families: 18.741% versus 18.729%.  The appendix fixed-floor
TMLE arm is the exception: -21.372% actual same-sample gain versus 7.240%
fixed-candidate replay gain.

A full fresh refit from raw data is not self-contained in this GitHub package.
The original shard archives and the raw IHDP, ACIC, and Twins covariate files
are tracked in the project data archive and Manifold records.  They are not
needed to verify the submitted numbers from the compact public rows, but they
are needed to rerun all nuisance fitting from scratch.

## Benchmark Matrix

The primary table has 24 benchmark settings per expert family:

```text
Kang--Schafer: 8 settings
IHDP:          4 settings
ACIC 2016:     4 settings
ACIC 2017:     4 settings
Twins:         4 settings
```

Each setting has 96 paired replications.  The public row file therefore has
2,304 setting-replication groups.  With five expert families including the
appendix fixed-floor TMLE diagnostic, it contains 11,520 rows.

The four primary expert families are AIPW, C-TMLE, selective ML, and Ma DR-BC.
Fixed-floor TMLE is an appendix diagnostic, not a primary comparator.

## Data-Generating Processes

Kang--Schafer uses the standard four-covariate Gaussian simulation.  The
outcome mean is

```text
mu = 210 + 27.4 z1 + 13.7 (z2 + z3 + z4)
```

and the response probability is

```text
pi = sigmoid(-z1 + 0.5 z2 - 0.25 z3 - 0.1 z4).
```

The four Kang--Schafer nuisance regimes are labels for the estimator inputs,
not separate datasets:

```text
CC: correct outcome covariates, correct response covariates
CI: correct outcome covariates, incorrect response covariates
IC: incorrect outcome covariates, correct response covariates
II: incorrect outcome covariates, incorrect response covariates
```

Each regime is run at `n=200` and `n=1000`.

IHDP, ACIC 2016, ACIC 2017, and Twins are semi-synthetic known-truth
benchmarks: they use real covariate matrices and synthetic response/outcome
laws so MSE is observable.  For each of these datasets, the paper reports
source settings A and B at signal strengths 0 and 3.  In the source code, A
maps to the `*_semisynth` design and B maps to the `*_misaligned` design.  The
observable dataset category in the paper remains the dataset name; A/B are
known-truth source settings.

For IHDP, ACIC 2016, ACIC 2017, and Twins, the constructors standardize the
real covariates, define a low-response subset from a score's bottom 15%, draw
responses from a Bernoulli probability equal to `epsilon` inside that subset
and a clipped high-response function outside it, add a dataset-specific
synthetic signal at strength 0 or 3, and set truth to the population mean of
the synthetic outcome surface.

Supplementary public-covariate checks for digits, breast cancer, diabetes, and
wine are retained in companion bundles.  They are not part of the 24-setting
primary table.

## Repair Algorithm Used for Section 4

For each expert family and setting, the run starts from cross-fitted outcome
and response predictions `(m0, p)`.  The repair learns one residual direction
`h` from responders using the weighted squared residual objective

```text
mean over R=1 of ((1 - p_hat) / p_hat^2) * (Y - m0)^2.
```

It then evaluates the candidate path

```text
m_gamma = m0 + gamma h,    gamma in {0, 0.25, 0.5, 1}.
```

The reported replay selects `gamma` by held-out response-weighted residual
loss:

```text
mean over R=1 of ((1 - p_hat) / p_hat^2) * (Y - m_gamma)^2.
```

The zero candidate is always eligible.  A nonzero candidate is eligible only
when its held-out improvement over `gamma=0` exceeds one estimated standard
error.  Among eligible candidates, the selected candidate is the one with the
smallest held-out weighted residual loss, with ties resolved toward the
smaller `gamma`.

The current paper-facing endpoint is the selected candidate from this replay.
Historical scalar-shrinkage and centered-score fields in older raw archives are
not current Section 4 parameters.

## Learners and Cross-Fitting

The Aug. 14 candidate paths use three-fold cross-fitting and XGBoost nuisance
learners.  The source records XGBoost version 3.4.0.  The maintained runner
constructs the XGBoost nuisance learners as:

```text
XGBRegressor(random_state=seed, n_jobs=1, verbosity=0)
XGBClassifier(random_state=seed, n_jobs=1, verbosity=0)
```

If XGBoost is unavailable, the code has histogram-gradient-boosting fallbacks,
but the paper-facing run used XGBoost.

Response models use stratified folds with shuffled splits.  Outcome and
residual models use fold labels drawn from the replication seed.  Residual
models train only on responders and use weights `(1 - p_hat) / p_hat^2`.

## Expert Families

AIPW uses the usual observed-outcome mean score

```text
m(X) + R / p(X) * (Y - m(X)).
```

C-TMLE uses the implementation in the runner: it selects its response floor
collaboratively before repair and then receives the same residual candidate
path.

Selective ML uses the in-driver two-fold mixed-minimax selector.  Its
propensity learner library is `logistic_l1`, `random_forest`, and
`gradient_boosting`; its outcome learner library is `lasso`, `random_forest`,
and `gradient_boosting`.

Ma DR-BC uses the in-driver same-target bias-corrected trimmed doubly robust
implementation with a singleton trimming threshold 0.05, correction order 1,
and shifted-Legendre sieve degree 3.

Fixed-floor TMLE uses a singleton 0.05 response floor without collaborative
floor selection.  It is reported as an appendix diagnostic.

## Seeds and Replication Counts

The archived Aug. 14 candidate paths use `seed_base=1800000000`, 24 chunks,
and 4 replications per chunk for each method-setting combination.  The
benchmark expansion bundles use seed `20260831`, 480 or 1,440 jobs depending
on the dataset group, and 4 replications per job.

Inside the runner, the per-replication seed is computed from the base seed,
sample size, response floor, signal strength, replication index, design name,
and response-design name:

```text
seed = seed0
     + n * 1009
     + int(epsilon * 1000) * 100003
     + int(round(strength * 100)) * 10007
     + rep * 37
     + design_seed_offset(design)
     + mar_seed_offset(mar_design)
```

The replay row file records `seed0`, `rep`, `design`, `n`, and `strength` for
each row.

## Aggregation and Intervals

The main manuscript table is an equal-setting table.  It lists each of the 24
dataset/settings separately and reports the MSE gain for each expert family in
that setting.

The manuscript summary table averages those 24 setting gains equally for each
primary expert family and reports counts of positive settings, negative
settings, and settings whose interval is above zero.

The `weighted_residual_gamma_method_summary.csv` and
`weighted_residual_gamma_primary_summary.csv` files are equal-replication
diagnostics.  They are useful checks, but they are not the main manuscript
headline.

Setting-level intervals use paired percentile resampling of the 96 paired
replications within the setting.  The table-rendering script uses 5,000 draws
for intervals it has to fill from the public replay rows, and preserves
already committed setting intervals when they are present.

## Verification

From the public replication directory, run:

```bash
python3 scripts/verify_section4_manuscript.py \
  --data-root . \
  --paper-root /path/to/overleaf-paper
```

The verifier checks the current primary bundle, required companion bundles,
generated TeX files, and manuscript references.  A passing verifier means the
paper is synchronized with the public compact artifact package; it does not
claim that the public package contains every raw input needed for a full fresh
refit.
