# No-Shrinkage Ablation

This bundle summarizes the final-shrinkage ablation for the unified
global-residual Section 4 run.  The manuscript table uses the same 24 benchmark
settings as the primary table: eight Kang--Schafer settings from the Aug. 14
unified Cartesian run and four settings each for IHDP, ACIC 2016, ACIC 2017,
and Twins from the 2026-08-31 benchmark expansion runs.  The ablation changes
only the last reporting contrast:

- `no_shrinkage`: the selected candidate before the final scalar
  \(c=2\) attenuation.
- `c2_shrinkage`: the final estimator reported in the primary Section 4
  tables.

No new estimator fitting is performed for this ablation.  The Kang--Schafer raw
rows are the Aug. 14 unified Cartesian global-residual shards stored under:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/
```

## Headline

The unshrunk selected path is more aggressive: it increases benchmark gains for
AIPW, selective ML, and Ma DR-BC.  The \(c=2\) shrinkage rule gives up some
positive movement in the responsive arms while keeping their benchmark
intervals above zero.

Primary 24-setting benchmark percent MSE gains:

| expert | no shrinkage | \(c=2\) shrinkage |
|---|---:|---:|
| AIPW | 9.049 [7.042, 10.635] | 4.158 [2.943, 5.241] |
| selective ML | 4.948 [3.497, 6.258] | 2.081 [1.259, 2.915] |
| Ma DR-BC | 5.793 [4.528, 6.925] | 4.667 [3.663, 5.654] |
| C-TMLE | 0.037 [-0.084, 0.187] | 0.048 [-0.048, 0.176] |

The old full-matrix rows, including fixed-floor TMLE, remain in
`no_shrinkage_family_summary.csv` for audit and diagnostic use.  The manuscript
table reads `no_shrinkage_benchmark_summary.csv`, which is restricted to the
same 24 benchmark settings and four primary experts as the main table.

## Reproduce

Extract the three Aug. 14 unified Cartesian tarballs listed in the public
artifact index.  The IHDP, ACIC 2016, ACIC 2017, and Twins compact raw rows
must also be present in their `support_csv/dml_real_benchmark_*` bundles.  Then
run:

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
- `no_shrinkage_family_summary.csv`: family, primary, and full-matrix summaries.
- `no_shrinkage_benchmark_summary.csv`: the 24-setting benchmark summary used
  by the manuscript table.
- `section4_no_shrinkage_ablation_table.tex`: generated manuscript table.
- `verification.json`: row counts, source manifests, frozen settings, and
  provenance.
