# Public Artifact Index

This file records the public-facing code and data locations for the
hidden-bias regional-repair project.

## Current EJS Section 4 Source

The current manuscript Section 4 is generated from the unified global-residual
bundle:

- `support_csv/dml_unified_cartesian_global_residual_20260814/`

This is a compact public reconstruction of the Aug. 14 v3 Cartesian run.  It
contains 170 expert-by-cell summaries, 25 family summaries, generated TeX for
the manuscript tables/macros, reconstruction provenance, and public checksums.

The paper-facing scripts are:

- `scripts/recreate_unified_cartesian_global_residual.py`
  - SHA256:
    `28e87aecded07604adf4de4187eecbb0c1212f2e142192f71c5b689c17aa87c7`
- `scripts/assemble_section4_unified_global_residual.py`
  - SHA256:
    `bc3b5edfdcb043bc0ee1afb875d707a142455f497645aed39401ad1baf4e3904`
- `scripts/verify_section4_manuscript.py`
  - SHA256:
    `8903d328781a651edf8857da43a626319c0fa1ad84532535c111f06ac23180b4`

The verifier confirms that the public bundle and an Overleaf paper checkout
agree on the generated Section 4 files.

## Manifold Source Objects

The raw shard tarballs are too large for the public GitHub repository.  The
compact bundle above was reconstructed from these Manifold objects:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/dml_ks_alignment_v3/cartesian_dml_ks_alignment_v3.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/dml2_real_anchor_v3/cartesian_dml2_real_anchor_v3.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/source/unified_cartesian_bundle_20260814_v3.tar.zst
```

Hashes:

```text
dml KS/alignment tarball: 1a6db65195d507ac2c4e1a21c62010b876f015ca0333bb6467c8dcf6d22ab6aa
dml2 real/anchor tarball: 102bb358543ec1adadd78cab863cbe328e069f6512d12b9d5c015e2ebce49fa6
source bundle: 5ab6b5927e6a7634d4e6ed3d5658a5f4362ee0b13b02e51a1260601e70ac7c1a
full scientific manifest: 65720b1fce2a24d55872ab9e008cf7ef62945b30b791272f1fdfe65280e2287f
raw reps manifest: 08d0e7f95d71773fe54eb137107e73c9f0346955247432a8ebb0e0dd1d195e92
frozen estimator source: 98987b31cf7c883d4776996ae7b28f7f1b9fe134d6da323e95250f00232842ce
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

### Low/High-Response Repair-Region Ablation

The accepted 2026-08-30 rerun is:

- Compact public bundle:
  `support_csv/dml_low_high_response_ablation_20260830/`
- Full raw Manifold bundle:
  `manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/low_high_response_ablation_20260830/ctmle_region_targeting_b50/`
- Summary script:
  `scripts/summarize_low_high_response_ablation.py`
  - SHA256:
    `9098358aa198669e0070a1b033ad57cef4948a43f7df639a86da47cf3357cd55`

The rerun reconstructs the pre-unified region-targeted C-TMLE driver at public
commit `3e131a9` and changes only the repair region: the true low-response
box versus a disjoint high-response placebo box.  It uses strengths
`0,3,5,8`, 96 paired replications per strength and region, `B=50`,
three-fold cross-fitting, XGBoost nuisances, damping grid
`0,0.25,0.5,1`, one-SE selection, and `c=2` shrinkage.

Headline over signal strengths 3, 5, and 8:

```text
low-response repair:       +14.569% [11.688%, 17.693%]
high-response placebo:      +0.001% [-0.013%, 0.014%]
paired low-minus-high gain: +14.568% [11.697%, 17.774%]
```

This is a mechanism check for region placement.  It is not the paper-facing
unified global-residual Section 4 source.

### Old Mixed Section 4 Release

`support_csv/dml_section4_release_20260812_v1/` and the c-atlas bundles remain
available as archived provenance.  They used the earlier mixed evidence path
and are not the current manuscript tables.
