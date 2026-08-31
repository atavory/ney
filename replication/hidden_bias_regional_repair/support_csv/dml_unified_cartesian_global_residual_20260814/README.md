# Unified Cartesian Global-Residual Reconstruction

This directory records a 2026-08-30 reconstruction of the Aug. 14 v3 unified
Cartesian baseline.  As of the 2026-08-30 manuscript cleanup, this is the
paper-facing Section 4 result source for the single global-residual repair.

Source Manifold objects:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/dml_ks_alignment_v3/cartesian_dml_ks_alignment_v3.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/dml2_real_anchor_v3/cartesian_dml2_real_anchor_v3.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/source/unified_cartesian_bundle_20260814_v3.tar.zst
```

Hashes verified before extraction:

```text
dml KS/alignment tarball: 1a6db65195d507ac2c4e1a21c62010b876f015ca0333bb6467c8dcf6d22ab6aa
dml2 real/anchor tarball: 102bb358543ec1adadd78cab863cbe328e069f6512d12b9d5c015e2ebce49fa6
source bundle: 5ab6b5927e6a7634d4e6ed3d5658a5f4362ee0b13b02e51a1260601e70ac7c1a
full scientific manifest: 65720b1fce2a24d55872ab9e008cf7ef62945b30b791272f1fdfe65280e2287f
```

Recreation command:

```bash
python3 scripts/recreate_unified_cartesian_global_residual.py \
  --run-dir /tmp/dml_unified_cartesian_20260814_extract_run/dml/cartesian_dml_ks_alignment_v3 \
  --run-dir /tmp/dml_unified_cartesian_20260814_extract_run/dml2/dml2_real_anchor_v3 \
  --full-manifest /tmp/dml_unified_cartesian_20260814_extract_run/source/full/manifest.tsv \
  --out-dir support_csv/dml_unified_cartesian_global_residual_20260814 \
  --draws 20000 \
  --seed 20260814
```

The reconstruction passed fail-closed checks for 4,080 shard files, 16,320
replication rows, and 170 expert-by-cell combinations.  Every row uses
`repair_mode=if_residual`, `validation_loss_se=1.0`, `shrink_c=2.0`,
`region_damp_grid=0.0|0.25|0.5|1.0`, and the frozen source SHA
`98987b31cf7c883d4776996ae7b28f7f1b9fe134d6da323e95250f00232842ce`.

Primary paper-facing readout at `c=2`:

```text
AIPW all: +2.791% [1.957%, 3.528%]
C-TMLE all: +0.028% [-0.040%, 0.119%]
selective ML all: +1.723% [1.184%, 2.252%]
Ma DR-BC all: +2.999% [2.337%, 3.656%]
```

The nonadaptive plain-TMLE arm is retained as a diagnostic fixed-floor
baseline:

```text
plain TMLE all: -0.410% [-1.203%, 0.235%]
plain TMLE Kang-Schafer: -1.886% [-5.265%, 0.862%]
```

The manuscript tables, including the fixed-floor diagnostic appendix table,
are generated from `family_summary.csv` by
`scripts/assemble_section4_unified_global_residual.py`.

Companion generated ablation tables are stored outside this primary bundle:

- `support_csv/dml_low_high_response_ablation_20260830/` checks true
  low-response repair-region placement against a disjoint high-response
  placebo.
- `support_csv/dml_no_shrinkage_ablation_20260830/` compares the selected
  unshrunk candidate with the final `c=2` shrinkage rule on the same unified
  raw rows.
