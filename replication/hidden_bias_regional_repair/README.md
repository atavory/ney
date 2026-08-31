# Hidden Bias and Residual Repair for Low-Response AIPW

This directory is the public replication package for the EJS manuscript.  As
of the 2026-08-30 manuscript cleanup, the paper-facing Section 4 source is the
single global-residual repair run on the unified Cartesian matrix.

## Current Paper Source

The authoritative compact bundle is:

- `support_csv/dml_unified_cartesian_global_residual_20260814/`

That bundle contains:

- `cell_summary.csv`: 170 expert-by-cell summaries.
- `family_summary.csv`: the sole source for the Section 4 tables and numbers.
- `summary.json`: nested method/family readout.
- `verification.json`: reconstruction provenance and fail-closed checks.
- `section4_values.tex`: generated manuscript macros.
- `section4_unified_overview_table.tex`: generated all-family table.
- `section4_unified_family_table.tex`: generated family table.
- `SHA256SUMS`: checksums for the compact public bundle.

The corresponding public scripts are:

- `scripts/recreate_unified_cartesian_global_residual.py`
- `scripts/assemble_section4_unified_global_residual.py`
- `scripts/verify_section4_manuscript.py`

Every row in the current Section 4 source uses the same repair mode and gate:
`repair_mode=if_residual`, `validation_loss_se=1.0`, `shrink_c=2.0`, and
`region_damp_grid=0.0|0.25|0.5|1.0`.  The run checks 4,080 shard files,
16,320 paired replication rows, and 170 expert-by-cell combinations.

Primary all-family readout at the frozen `c=2`:

```text
AIPW all: +2.791% [1.957%, 3.528%]
plain TMLE all: -0.410% [-1.203%, 0.235%]
C-TMLE all: +0.028% [-0.040%, 0.119%]
selective ML all: +1.723% [1.184%, 2.252%]
Ma DR-BC all: +2.999% [2.337%, 3.656%]
```

## Low/High-Response Region-Placement Check

A companion bundle records the region-placement ablation:

- `support_csv/dml_low_high_response_ablation_20260830/`

This run holds the C-TMLE reference, data law, region-targeted repair,
damping grid, one-SE gate, and `c=2` shrinkage fixed, then changes only the
repair region.  The low-response arm uses the true low-response box from the
MAR law; the high-response placebo uses a disjoint box with mean response rate
about 0.81 and zero overlap with the true low-response region.

Across non-null signal strengths 3, 5, and 8, the low-response repair gains
14.569% MSE [11.688%, 17.693%], while the high-response placebo gains 0.001%
[-0.013%, 0.014%].  The paired low-minus-high gain is 14.568%
[11.697%, 17.774%].

This is a mechanism check for region placement.  It uses the pre-unified
region-targeted C-TMLE driver at public commit `3e131a9` and is not pooled
into the unified global-residual Section 4 tables.

The bundle includes `section4_low_high_response_ablation_table.tex`, generated
by `scripts/summarize_low_high_response_ablation.py` from the committed
replication rows.

## No-Shrinkage Check

A second companion bundle records the final-shrinkage ablation:

- `support_csv/dml_no_shrinkage_ablation_20260830/`

This summary reuses the same 16,320 raw replication rows as the current
paper-facing unified global-residual run and changes only the final contrast:
the selected unshrunk candidate versus the \(c=2\) plug-in shrinkage rule.

All-family equal-cell percent MSE gains:

```text
AIPW:             no shrinkage +5.965% [4.586%, 7.041%], c=2 +2.791% [1.957%, 3.528%]
C-TMLE:           no shrinkage +0.021% [-0.066%, 0.126%], c=2 +0.028% [-0.040%, 0.119%]
selective ML:     no shrinkage +4.399% [3.414%, 5.305%], c=2 +1.723% [1.184%, 2.252%]
Ma DR-BC:         no shrinkage +3.797% [2.977%, 4.542%], c=2 +2.999% [2.337%, 3.656%]
fixed-floor TMLE: no shrinkage -6.837% [-9.784%, -4.532%], c=2 -0.410% [-1.203%, 0.235%]
```

The no-shrinkage path is more aggressive.  It increases point gains for the
responsive experts but exposes the fixed-floor TMLE over-activation failure.
The \(c=2\) rule is therefore a stability guard, not a gain-maximizing
postprocessing step.

## Rebuild the Manuscript Tables

From this directory:

```bash
python scripts/assemble_section4_unified_global_residual.py \
  --summary support_csv/dml_unified_cartesian_global_residual_20260814/family_summary.csv \
  --out-dir /tmp/section4_unified_rebuilt
```

The rebuilt files should match the primary generated TeX files in
`support_csv/dml_unified_cartesian_global_residual_20260814/` byte for byte.

To verify the public bundle against an Overleaf paper clone:

```bash
python scripts/verify_section4_manuscript.py \
  --data-root . \
  --paper-root /path/to/overleaf-paper
```

The verifier checks the bundle checksums, reconstruction invariants, summary
shape, headline values, generated TeX bytes, and Section 4 manuscript inputs.

## Recreate the Compact Bundle from Manifold

The raw shard tarballs are not committed to GitHub.  They are recoverable from
these Manifold objects:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/dml_ks_alignment_v3/cartesian_dml_ks_alignment_v3.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/dml2_real_anchor_v3/cartesian_dml2_real_anchor_v3.tar.zst
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/source/unified_cartesian_bundle_20260814_v3.tar.zst
```

Verify the tarballs before extraction:

```text
dml KS/alignment tarball: 1a6db65195d507ac2c4e1a21c62010b876f015ca0333bb6467c8dcf6d22ab6aa
dml2 real/anchor tarball: 102bb358543ec1adadd78cab863cbe328e069f6512d12b9d5c015e2ebce49fa6
source bundle: 5ab6b5927e6a7634d4e6ed3d5658a5f4362ee0b13b02e51a1260601e70ac7c1a
full scientific manifest: 65720b1fce2a24d55872ab9e008cf7ef62945b30b791272f1fdfe65280e2287f
raw reps manifest: 08d0e7f95d71773fe54eb137107e73c9f0346955247432a8ebb0e0dd1d195e92
frozen source: 98987b31cf7c883d4776996ae7b28f7f1b9fe134d6da323e95250f00232842ce
wrapper: b1b08b9fc32b03e969f2f24ba7816a850de12336bbf6a335f092238715ccb332
```

After extracting the three tarballs, recreate the compact public bundle with:

```bash
python scripts/recreate_unified_cartesian_global_residual.py \
  --run-dir /tmp/dml_unified_cartesian_20260814_extract_run/dml/cartesian_dml_ks_alignment_v3 \
  --run-dir /tmp/dml_unified_cartesian_20260814_extract_run/dml2/dml2_real_anchor_v3 \
  --full-manifest /tmp/dml_unified_cartesian_20260814_extract_run/source/full/manifest.tsv \
  --out-dir support_csv/dml_unified_cartesian_global_residual_20260814 \
  --draws 20000 \
  --seed 20260814
```

Then regenerate the manuscript tables with
`scripts/assemble_section4_unified_global_residual.py` as shown above and
refresh `SHA256SUMS`.

## Archived Section 4 Bundles

The older `dml_section4_release_20260812_v1` and c-atlas bundles remain in
`support_csv/` as archived provenance.  They are not the current EJS Section 4
source and should not be pooled into the unified global-residual tables.

The standalone Ma DiD experiment is also not part of the unified Cartesian
matrix because it has a different estimand.  It remains a separate historical
diagnostic.

## Protocol and Environment

`unified_cartesian_protocol_20260814.md` records the frozen common-estimand
matrix and single-repair rule.  The experiment drivers use the full pinned
stack, including scikit-learn and XGBoost.  The compact assemblers and
verifier use only the Python standard library.
