# No-Shrinkage Ablation

This bundle summarizes the final-shrinkage ablation for the unified
global-residual Section 4 run.  It reuses the same 16,320 raw replication rows
as the paper-facing run and changes only the last reporting contrast:

- `no_shrinkage`: the selected candidate before the final scalar
  \(c=2\) attenuation.
- `c2_shrinkage`: the final estimator reported in the primary Section 4
  tables.

No new estimator fitting is performed for this ablation.  The raw rows are the
Aug. 14 unified Cartesian global-residual shards stored under:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/
```

## Headline

The unshrunk selected path is more aggressive: it increases all-family gains
for AIPW, selective ML, and Ma DR-BC, but it also creates the fixed-floor TMLE
failure.  The \(c=2\) shrinkage rule gives up some positive movement in the
responsive arms and sharply reduces the fixed-floor TMLE loss.

All-family equal-cell percent MSE gains:

| expert | no shrinkage | \(c=2\) shrinkage |
|---|---:|---:|
| AIPW | 5.965 [4.586, 7.041] | 2.791 [1.957, 3.528] |
| C-TMLE | 0.021 [-0.066, 0.126] | 0.028 [-0.040, 0.119] |
| selective ML | 4.399 [3.414, 5.305] | 1.723 [1.184, 2.252] |
| Ma DR-BC | 3.797 [2.977, 4.542] | 2.999 [2.337, 3.656] |
| fixed-floor TMLE | -6.837 [-9.784, -4.532] | -0.410 [-1.203, 0.235] |

## Reproduce

Extract the three Aug. 14 unified Cartesian tarballs listed in the public
artifact index, then run:

```bash
python scripts/summarize_no_shrinkage_ablation.py \
  --run-dir /tmp/dml_unified_cartesian_20260814_extract/cartesian_dml_ks_alignment_v3 \
  --run-dir /tmp/dml_unified_cartesian_20260814_extract/dml2_real_anchor_v3 \
  --full-manifest /tmp/dml_unified_cartesian_20260814_extract/full/manifest.tsv \
  --out-dir support_csv/dml_no_shrinkage_ablation_20260830 \
  --draws 20000 \
  --seed 20260814
```

## Files

- `no_shrinkage_cell_summary.csv`: cell-level no-shrinkage and \(c=2\)
  summaries.
- `no_shrinkage_family_summary.csv`: family and all-family summaries.
- `section4_no_shrinkage_ablation_table.tex`: generated manuscript table.
- `verification.json`: row counts, source manifests, frozen settings, and
  provenance.
