# Public Artifact Index

This file records the public-facing code and compact data locations for the
hidden-bias residual-repair project.

For the complete EJS run contract, including the public-versus-archived input
boundary, DGP definitions, learner settings, comparator settings, replication
counts, seeds, aggregation rule, and verification command, see
`REPRODUCIBILITY.md`.

## Current EJS Section 4 Source

The current manuscript Section 4 is generated from:

```text
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/
```

This bundle replays the archived candidate paths from the Aug. 14 unified
global-residual run and selects \(\gamma\) by held-out response-weighted
residual loss.  The paper-facing table contains 24 benchmark settings per
expert family:

```text
Kang--Schafer: 8 settings
IHDP:          4 settings
ACIC 2016:     4 settings
ACIC 2017:     4 settings
Twins:         4 settings
```

The current paper-facing generated files are:

```text
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/section4_unified_family_table.tex
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/section4_unified_summary_table.tex
support_csv/dml_weighted_residual_gamma_selection_ablation_20260903/section4_fixed_floor_tmle_diagnostic_table.tex
support_csv/dml_weighted_residual_high_response_placebo_ablation_20260903/section4_weighted_residual_high_response_placebo_ablation_table.tex
support_csv/dml_weighted_residual_mse_gain_by_rho_20260903_v2/section4_response_bin_action_reward_figure.tex
```

Primary 24-setting equal-setting readout:

```text
AIPW:          +8.95%, positive 20/24, negative 4/24, interval above zero 11/24
selective ML:  +5.19%, positive 16/24, negative 8/24, interval above zero 8/24
Ma DR-BC:      +7.66%, positive 18/24, negative 6/24, interval above zero 7/24
C-TMLE:        +4.98%, positive 9/24,  negative 0/24, interval above zero 8/24
```

The fixed-floor TMLE arm is reported only in the appendix diagnostic table.
C-TMLE remains the primary TMLE comparator.

## Current Companion Diagnostics

The current companion bundles are:

```text
support_csv/dml_weighted_residual_mse_gain_by_rho_20260903_v2/
support_csv/dml_weighted_residual_high_response_placebo_ablation_20260903/
support_csv/dml_weighted_residual_augmented_gamma_grid_ablation_20260903/
support_csv/dml_weighted_residual_upstream_trust_gate_diagnostic_20260903_v2/
support_csv/dml_section3_bounds_diagnostic_20260906_buck_v1/
support_csv/dml_section3_bounds_diagnostic_fixed_floor_tmle_20260906_buck_v1/
```

Key diagnostics:

```text
Lowest response quartile: 42.2% raw action, 85.4% response-weighted action, 92.1% response-weighted reward.
High-response placebo: low-response placement dominates high-response placement for AIPW, Ma DR-BC, and C-TMLE.
Gamma 0.75 sensitivity: primary pooled gain changes by 0.561 percentage points; interval crosses zero.
Section 3 primary bounds: zero variance-bound and zero squared-bias-bound violations over 9,216 rows.
Fixed-floor TMLE bounds: zero bound violations, but active-move hidden squared bias improves only 59.8% and realized harm occurs 52.5%.
Upstream-trust gate: retrospective only; moves fixed-floor TMLE from -21.37% to -0.21% without erasing primary pooled gains.
```

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

Verify the public package against a paper checkout with:

```bash
python3 scripts/verify_section4_manuscript.py \
  --data-root . \
  --paper-root /path/to/overleaf-paper
```

## Historical Sources

The old centered-score selected-candidate reconstruction remains in:

```text
support_csv/dml_unified_cartesian_global_residual_20260814/
```

It supplies fitted residual directions and candidate-path provenance.  It is
not the current manuscript readout.  Older mixed-release, regional-residual,
scalar-damping, and no-shrinkage bundles remain archived for audit and should
not be pooled into current Section 4 outputs.

The raw shard tarballs are larger than the compact public package and remain
recoverable from the data Overleaf and Manifold records maintained with the
project.
