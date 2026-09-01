# Public Artifact Index

This file records the public-facing code and data locations for the
hidden-bias regional-repair project.

## Current EJS Section 4 Source

The current manuscript Section 4 is generated from the unified global-residual
bundle plus the 2026-08-31 benchmark expansions:

- `support_csv/dml_unified_cartesian_global_residual_20260814/`
- `support_csv/dml_real_benchmark_expansion_20260831/`
- `support_csv/dml_real_benchmark_acic2017_20260831/`
- `support_csv/dml_real_benchmark_twins_20260831/`

This is a compact public reconstruction of the Aug. 14 v3 Cartesian run.  It
contains 170 expert-by-setting summaries, 30 family summaries, generated TeX
for the manuscript macros and Kang--Schafer rows, reconstruction provenance,
and public checksums.  The primary manuscript table combines its eight
Kang--Schafer settings with IHDP, ACIC 2016, ACIC 2017, and Twins expansion
rows for 24 benchmark settings per expert family.

The paper-facing scripts are:

- `scripts/validated_reference_transfer.py`
  - Main experiment runner.  Its `if_residual` branch fits the residual
    direction by responder-only weighted residual regression and chooses the
    damping value with the centered fitted-score loss used by the Aug. 14
    experiment source snapshot.  The `validation_risk=balanced_mse` metadata
    field is retained as a run label, but this branch's gate is the score-loss
    gate in the source.
  - SHA256:
    `3ab6bc1d0077316fe397d6cc97304b6e62d76eca969d9cf7a67a5a835b38db7e`
- `scripts/recreate_unified_cartesian_global_residual.py`
  - SHA256:
    `a0676b1bcf3d854d5b15a60f0dc8e91d2dd12a70858c0aa76da90206a99fa235`
- `scripts/assemble_section4_unified_global_residual.py`
  - SHA256:
    `d48834defc74119667519ecd54b6784b3980a4222087f3373ba934a4c7b39e71`
- `scripts/dml_launch_section4_placebo_shards.py`
  - SHA256:
    `cc8b9cb5dbf1f3ebd0c7e8546bed46f1f192f0bcbe57ff6b3c7dd8d718fe524f`
- `scripts/summarize_high_response_placebo_ablation.py`
  - SHA256:
    `fe1c2cd363d347bb0b77341db35216018468272de1b466d383afdcafeea17df0`
- `scripts/dml_nuisance_cv_accuracy_by_response.py`
  - Diagnostic script for cross-fitted response-score and outcome-prediction
    accuracy by response bin.
  - SHA256:
    `9d276d09ad9ab7ab2df13be838b8e2fa5791a2d87acd83e77ff7311017ace02d`
- `scripts/dml_reference_error_by_response.py`
  - Diagnostic script for expert-reference outcome accuracy by response bin,
    including the fixed-floor TMLE diagnostic arm.
  - SHA256:
    `7a8765f01805cef46788d24728a15f4d4ed97931ae52763ded8cf57193affb30`
- `scripts/verify_section4_manuscript.py`
  - SHA256:
    `37d66429eb4f70e33cd39436705cea5e1fae918f245c2a86f31d59977def4320`

The verifier confirms that the public bundle and an Overleaf paper checkout
agree on the generated Section 4 files, including the current companion
ablation table.

The appendix fixed-floor TMLE diagnostic is generated setting-by-setting on the
same 24 benchmark settings as the primary table.  C-TMLE remains the primary
TMLE comparator in the manuscript.

The diagnostic bundle
`support_csv/dml_nuisance_cv_accuracy_by_response_20260901/` checks whether
the cross-fitted response, outcome, and expert-reference nuisance fits behave
differently across low- and high-response bins.  It is retained as
theory/motivation evidence and does not feed the manuscript MSE tables.

## Manifold Source Objects

The raw shard tarballs are too large for the public GitHub repository.  The
compact bundles above were reconstructed from these Manifold objects:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/dml_ks_alignment_v3/cartesian_dml_ks_alignment_v3.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/dml2_real_anchor_v3/cartesian_dml2_real_anchor_v3.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/source/unified_cartesian_bundle_20260814_v3.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/real_benchmark_expansion_20260831/dml_real_benchmark_v1.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/real_benchmark_expansion_20260831/dml_real_benchmark_support_data_v1.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/real_benchmark_expansion_20260831/dml_real_benchmark_acic2017_v1.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/real_benchmark_expansion_20260831/dml_real_benchmark_support_data_v2.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/real_benchmark_expansion_20260831/dml_real_benchmark_twins_v1.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/real_benchmark_expansion_20260831/dml_real_benchmark_support_data_v3.tar.zst
```

Hashes:

```text
dml KS/alignment tarball: 1a6db65195d507ac2c4e1a21c62010b876f015ca0333bb6467c8dcf6d22ab6aa
dml2 real/anchor tarball: 102bb358543ec1adadd78cab863cbe328e069f6512d12b9d5c015e2ebce49fa6
source bundle: 5ab6b5927e6a7634d4e6ed3d5658a5f4362ee0b13b02e51a1260601e70ac7c1a
full scientific manifest: 65720b1fce2a24d55872ab9e008cf7ef62945b30b791272f1fdfe65280e2287f
raw reps manifest: 08d0e7f95d71773fe54eb137107e73c9f0346955247432a8ebb0e0dd1d195e92
estimator source snapshot: 98987b31cf7c883d4776996ae7b28f7f1b9fe134d6da323e95250f00232842ce
experiment wrapper: b1b08b9fc32b03e969f2f24ba7816a850de12336bbf6a335f092238715ccb332
```

## Archived and Companion Sources

The following sources are retained for historical replication and companion
checks, but they are not the current EJS Section 4 source.

### CUI Published Source

- `scripts/build_frozen_cui_published_entry.py`
- `scripts/section4_cui_published_experiments.py`

Current public SHAs:

- CUI entry builder:
  `99327ea3aaa222ef8fe2b031c3613a064db551cb116059cfe73c6fd9116211f9`
- CUI experiment adapter:
  `c4fa4e83dd6112ecdd2c4dfce1fc6cd3d3e0c4a97d4f48b093c78f4f6ba31d70`

### Canonical Region-Gated Repair Driver

The maintained high-response-capable region-gated driver is:

- `scripts/validated_reference_transfer_canonical_region_if_library.py`

SHA:

`7d1478ce3529d8ab8391e8445ac121b08a269913679458e9ce91b4c32c8b9f50`

This source gates the deployed `if_library` residual correction by the
selected analysis region:

```text
candidate_outcome = base_outcome + gamma * correction * analysis_region
```

Older runs where the deployed correction ignored the region are not valid
high-response ablation results.

### High-Response Placebo Ablation

The current 2026-08-31 region-placement diagnostic is:

- Compact public bundle:
  `support_csv/dml_high_response_placebo_ablation_20260831/`
- Full raw Manifold bundle:
  `manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/high_response_placebo_ablation_20260831/dml_high_response_placebo_ablation_20260831_raw.tar.zst`
- Raw archive SHA256:
  `6b3b12b38797375846b99b7042de80e99402f798902e241f9a2c9f76f7a96e7b`

This diagnostic uses the same 24 benchmark settings and four primary expert
families as the manuscript table.  It compares a support-restricted regional
residual correction in the selected low-response support with the same
correction in a matched high-response support.

```text
AIPW:         low +5.862% [4.448%, 7.033%], high +0.120% [-0.372%, 0.527%], low-high +5.742% [4.289%, 6.963%]
selective ML: low +0.675% [-0.364%, 1.618%], high +0.535% [0.095%, 0.944%], low-high +0.140% [-0.952%, 1.130%]
Ma DR-BC:     low +1.473% [0.582%, 2.267%], high +1.178% [0.743%, 1.591%], low-high +0.294% [-0.649%, 1.179%]
C-TMLE:       low +0.117% [-0.116%, 0.392%], high +0.035% [-0.057%, 0.126%], low-high +0.082% [-0.170%, 0.372%]
```

The 2026-08-30 C-TMLE-only low/high-response run remains in
`support_csv/dml_low_high_response_ablation_20260830/` as archived provenance.

### Archived Scalar-Damping Diagnostics

These derived bundles are retained for audit, but they are not current
manuscript tables or appendix targets:

- Compact public bundle:
  `support_csv/dml_no_shrinkage_ablation_20260830/`
- Compact Manifold bundle:
  `manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/no_shrinkage_ablation_20260830/unified_cartesian_c2_vs_unshrunk/`
- Summary script:
  `scripts/summarize_no_shrinkage_ablation.py`
  - SHA256:
    `3c07f867bca0b8f62c93ff62e5d2058ba829f37c6703be9cf00a8a79b9a5645e`

- Compact public bundle:
  `support_csv/dml_shrinkage_c_grid_20260901/`
- Summary script:
  `scripts/summarize_shrinkage_c_grid.py`
  - SHA256:
    `585cd5a93316d1a27e213c848a45e1595336ac59128f6ddb285198554f0b8f30`

### Old Mixed Section 4 Release

`support_csv/dml_section4_release_20260812_v1/` and the c-atlas bundles remain
available as archived provenance.  They used the earlier mixed evidence path
and are not the current manuscript tables.
