# Cross-fitted nuisance accuracy by response

This diagnostic regenerates the benchmark settings and fits the
cross-fitted response score p_hat and outcome prediction m_hat.
Cross-fitting uses the same folds for p_hat and m_hat, stratified
by rank bins of true pi so the held-out folds have comparable
response-propensity support.  True pi is used here only because
these are known-truth simulations.
Rows are binned both by p_hat, which is observable in applications,
and by true pi, which is available only in these known-truth
simulations.

The response model is scored against the observed response label
and, as a simulation diagnostic, against true pi.  The outcome
model is scored against observed Y among responders and, as a
simulation diagnostic, against true mu on all units in the bin.
The p and m losses live on different scales; the normalized RMSE
columns divide by the setting-level standard deviation of true pi
or true mu and are mainly for within-setting shape checks.

reps: 32
bins: 5
folds: 3
learner: xgboost
propensity_learner: xgboost
support_data: /tmp/dml_real_benchmark_support_data
settings: 24

Output files:

- per_bin_reps.csv: one row per setting, replication, binning rule, and bin.
- bin_summary.csv: averages over replications for each setting and bin.
- dataset_summary.csv: averages over settings and replications by dataset and bin.
- low_high_summary.csv: lowest-bin versus highest-bin summaries by setting.
- dependence_summary.csv: per-setting low/high ratios and slopes versus response probability.
- dataset_dependence_summary.csv: dataset-level averages of the dependence diagnostics.
- normalized_error_fit_summary.csv: per-setting fits of normalized squared true error against response scarcity.
- dataset_normalized_error_fit_summary.csv: dataset-level averages of the normalized error fits.
- reference_error_per_bin_reps.csv: expert-reference true-error summaries by response bin.
- reference_error_bin_summary.csv: expert-reference averages by setting and response bin.
- reference_error_low_high_summary.csv: expert-reference low/high response-bin contrasts.
- reference_error_dataset_summary.csv: expert-reference dataset-level averages.
