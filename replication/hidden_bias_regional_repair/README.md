# Hidden Bias and Residual Repair for Low-Response AIPW

This directory is the public replication package for the EJS manuscript.  The
current Section 4 tables use the weighted-residual gamma-selection replay on a
24-setting benchmark matrix.

## Current Paper Source

The paper-facing primary source is:

```text
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/
```

It replays the archived Aug. 14 candidate paths, keeps the same upstream fits
and candidate grid, and selects the candidate by held-out response-weighted
residual loss.  The primary table contains eight Kang--Schafer settings and
four settings for each of IHDP, ACIC 2016, ACIC 2017, and Twins.

Primary 24-setting equal-setting readout:

```text
AIPW:          mean gain +8.95%, positive 20/24, negative 4/24, interval above zero 11/24
selective ML:  mean gain +5.19%, positive 16/24, negative 8/24, interval above zero 8/24
Ma DR-BC:      mean gain +7.66%, positive 18/24, negative 6/24, interval above zero 7/24
C-TMLE:        mean gain +4.98%, positive 9/24,  negative 0/24, interval above zero 8/24
```

The generated files consumed by the paper are:

```text
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/section4_unified_family_table.tex
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/section4_unified_summary_table.tex
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/section4_fixed_floor_tmle_diagnostic_table.tex
support_csv/dml_weighted_residual_high_response_placebo_ablation_20260903/section4_weighted_residual_high_response_placebo_ablation_table.tex
support_csv/dml_weighted_residual_mse_gain_by_rho_20260903_v2/section4_response_bin_action_reward_figure.tex
```

## Current Diagnostics

The current companion bundles are:

```text
support_csv/dml_weighted_residual_mse_gain_by_rho_20260903_v2/
support_csv/dml_weighted_residual_high_response_placebo_ablation_20260903/
support_csv/dml_weighted_residual_augmented_gamma_grid_ablation_20260903/
support_csv/dml_weighted_residual_upstream_trust_gate_diagnostic_20260903_v2/
support_csv/dml_section3_bounds_diagnostic_20260906_buck_v1/
support_csv/dml_section3_bounds_diagnostic_fixed_floor_tmle_20260906_buck_v1/
```

The high-response placebo diagnostic compares the selected low-response
support to a matched high-response support under the weighted-residual
selector.  The response-bin diagnostic decomposes where the global repair acts
and pays off: the lowest true-response quartile has 42.2% of raw action, 85.4%
of response-weighted action, and 92.1% of response-weighted residual-square
reward.

The Section 3 bounds diagnostics replay the selected candidates against known
simulation truth.  For the four primary expert families they report zero
variance-bound violations and zero squared-bias-bound violations over 9,216
rows.  The fixed-floor TMLE diagnostic also has zero bound violations, but its
active moves have much weaker hidden-squared-bias improvement and frequent
realized harm.

The fixed-floor TMLE arm is appendix-only.  The upstream-trust gate bundle is a
retrospective diagnostic that shows fixed-floor TMLE can be made to stand down;
it is not part of the submitted Algorithm 1 unless the protocol is changed.

## Scripts

Current paper-facing scripts:

```text
scripts/dml_render_weighted_residual_gamma_tables.py
scripts/dml_weighted_residual_gamma_selection_ablation.py
scripts/dml_weighted_residual_high_response_placebo_ablation.py
scripts/dml_weighted_residual_augmented_gamma_grid_ablation.py
scripts/dml_upstream_trust_gate_diagnostic.py
scripts/dml_mse_gain_by_rho.py
scripts/dml_section3_bounds_diagnostic.py
scripts/dml_weighted_residual_rho_helper.py
scripts/verify_section4_manuscript.py
```

The compact verifier uses only the Python standard library:

```bash
python3 scripts/verify_section4_manuscript.py \
  --data-root . \
  --paper-root /path/to/overleaf-paper
```

## Historical Sources

The Aug. 14 centered-score selected-candidate reconstruction remains in:

```text
support_csv/dml_unified_cartesian_global_residual_20260814/
```

It supplies candidate-path provenance and historical comparison only.  It is
not the current manuscript readout.  Older mixed-release, regional-residual,
scalar-damping, and no-shrinkage bundles are retained as audit trail and should
not be pooled into the current Section 4 tables.
