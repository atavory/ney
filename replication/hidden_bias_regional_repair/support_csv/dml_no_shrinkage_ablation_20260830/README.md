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

The unshrunk selected path is more aggressive: it increases primary gains for
AIPW, selective ML, and Ma DR-BC, but it also creates the fixed-floor TMLE
failure.  The \(c=2\) shrinkage rule gives up some positive movement in the
responsive arms and sharply reduces the fixed-floor TMLE loss.

Primary equal-setting percent MSE gains:

| expert | no shrinkage | \(c=2\) shrinkage |
|---|---:|---:|
| AIPW | 9.754 [7.437, 11.533] | 4.584 [3.186, 5.832] |
| C-TMLE | 0.035 [-0.112, 0.215] | 0.048 [-0.068, 0.201] |
| selective ML | 6.277 [4.710, 7.713] | 2.485 [1.612, 3.342] |
| Ma DR-BC | 6.412 [5.014, 7.682] | 5.077 [3.953, 6.186] |
| fixed-floor TMLE | -11.674 [-16.682, -7.766] | -0.709 [-2.056, 0.391] |

The full-matrix rows remain in `no_shrinkage_family_summary.csv` for audit
and diagnostic use.

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
- `no_shrinkage_family_summary.csv`: family, primary, and full-matrix summaries.
- `section4_no_shrinkage_ablation_table.tex`: generated manuscript table.
- `verification.json`: row counts, source manifests, frozen settings, and
  provenance.
