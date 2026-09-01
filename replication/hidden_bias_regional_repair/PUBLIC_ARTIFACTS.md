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

- `scripts/recreate_unified_cartesian_global_residual.py`
  - SHA256:
    `a0676b1bcf3d854d5b15a60f0dc8e91d2dd12a70858c0aa76da90206a99fa235`
- `scripts/assemble_section4_unified_global_residual.py`
  - SHA256:
    `e36fb3b508ddf2de72b00f42935fd445e2f8ca2737e5931cc545df0aac9a1c55`
- `scripts/dml_launch_section4_placebo_shards.py`
  - SHA256:
    `cc8b9cb5dbf1f3ebd0c7e8546bed46f1f192f0bcbe57ff6b3c7dd8d718fe524f`
- `scripts/summarize_high_response_placebo_ablation.py`
  - SHA256:
    `8f2c20a2f139c1d63646d03c70c0645c1824bd860dd418af765d7e6683940ac6`
- `scripts/summarize_no_shrinkage_ablation.py`
  - SHA256:
    `ac994817b46089fe8812d42c73a0941b905168ffe64454689cfb5b78f8756dda`
- `scripts/verify_section4_manuscript.py`
  - SHA256:
    `5f348dccc26cc0329eaf08b434199ed8b4d72d978c0e384c68ed4ceff61fbaa1`

The verifier confirms that the public bundle and an Overleaf paper checkout
agree on the generated Section 4 files, including the companion ablation
tables.

The appendix fixed-floor TMLE diagnostic is generated setting-by-setting on the
same 24 benchmark settings as the primary table.  C-TMLE remains the primary
TMLE comparator in the manuscript.

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

`8db53d4b39d4874884cd810337030d6825f70c19fd891d21b16de2302228d119`

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
AIPW:         low +1.945% [1.262%, 2.602%], high -0.036% [-0.285%, 0.121%], low-high +1.981% [1.263%, 2.664%]
selective ML: low +0.372% [-0.076%, 0.790%], high +0.195% [0.013%, 0.408%], low-high +0.178% [-0.294%, 0.596%]
Ma DR-BC:     low +1.052% [0.373%, 1.687%], high +0.959% [0.596%, 1.306%], low-high +0.093% [-0.653%, 0.825%]
C-TMLE:       low +0.124% [-0.046%, 0.357%], high +0.009% [-0.029%, 0.053%], low-high +0.115% [-0.064%, 0.351%]
```

The 2026-08-30 C-TMLE-only low/high-response run remains in
`support_csv/dml_low_high_response_ablation_20260830/` as archived provenance.

### No-Shrinkage Ablation

The accepted 2026-08-31 derived ablation is:

- Compact public bundle:
  `support_csv/dml_no_shrinkage_ablation_20260830/`
- Compact Manifold bundle:
  `manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/no_shrinkage_ablation_20260830/unified_cartesian_c2_vs_unshrunk/`
- Summary script:
  `scripts/summarize_no_shrinkage_ablation.py`
  - SHA256:
    `3c07f867bca0b8f62c93ff62e5d2058ba829f37c6703be9cf00a8a79b9a5645e`

This ablation reuses the Aug. 14 unified Cartesian raw rows and compares the
selected unshrunk candidate with the final \(c=2\) plug-in
contrast-shrinkage rule.  No new estimator fitting is performed.

Headline 24-setting benchmark gains:

```text
AIPW:         no shrinkage +9.049% [7.042%, 10.635%], c=2 +4.158% [2.943%, 5.241%]
selective ML: no shrinkage +4.948% [3.497%, 6.258%],  c=2 +2.081% [1.259%, 2.915%]
Ma DR-BC:     no shrinkage +5.793% [4.528%, 6.925%],  c=2 +4.667% [3.663%, 5.654%]
C-TMLE:       no shrinkage +0.037% [-0.084%, 0.187%], c=2 +0.048% [-0.048%, 0.176%]
```

### Old Mixed Section 4 Release

`support_csv/dml_section4_release_20260812_v1/` and the c-atlas bundles remain
available as archived provenance.  They used the earlier mixed evidence path
and are not the current manuscript tables.
