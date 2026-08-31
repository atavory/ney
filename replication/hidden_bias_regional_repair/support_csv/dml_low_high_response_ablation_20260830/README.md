# Low-Response vs High-Response Repair-Region Ablation

This bundle reconstructs the regional-control ablation requested for Section 4: hold the C-TMLE reference, data law, region-targeted repair, damping grid, one-SE gate, and `c=2` shrinkage fixed, then change only the repair region.

- `low_response`: `analysis_region=true`, the true low-response box used by the MAR law.
- `high_response_placebo`: `analysis_region=wrong`, a disjoint high-response box. Its mean response rate is about 0.81 and its overlap with the true low-response region is zero.

The driver is the pre-unified region-targeted repair source at public git commit `3e131a9`. This is a region-placement ablation; it is not the current unified global-residual Section 4 release.

## Headline

Across signal strengths 3, 5, and 8, the low-response repair gains 14.569% MSE [11.688, 17.693], while the high-response placebo gains 0.001% [-0.013, 0.014]. The paired low-minus-high gain is 14.568% [11.697, 17.774].

## Per-Strength Results

| strength | low-response gain | high-response placebo gain | low final activation | high final activation | low response rate | high response rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000% [0.000, 0.000] | 0.070% [0.000, 0.177] | 0.0% | 3.1% | 0.050 | 0.809 |
| 3 | 14.111% [9.144, 19.810] | 0.003% [-0.028, 0.039] | 34.4% | 10.4% | 0.050 | 0.810 |
| 5 | 15.200% [9.772, 21.339] | 0.009% [-0.006, 0.026] | 39.6% | 8.3% | 0.050 | 0.807 |
| 8 | 14.404% [10.446, 18.881] | -0.003% [-0.024, 0.015] | 46.9% | 6.2% | 0.049 | 0.809 |

## Reproduce summaries

```bash
python scripts/summarize_low_high_response_ablation.py \
  --replications support_csv/dml_low_high_response_ablation_20260830/replications.csv \
  --out-dir /tmp/dml_low_high_response_ablation_rebuilt \
  --draws 20000 \
  --seed 20260830
```

## Manifold Raw Bundle

The full raw run, including chunk logs and per-chunk CSV files, is stored at:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/low_high_response_ablation_20260830/ctmle_region_targeting_b50/
```

Use bucket/path syntax to recover it:

```bash
ROSETTA_DISABLE_AOT=1 manifold getr \
  aai_research_tlv/tree/atavory/dml_reference_transfer/low_high_response_ablation_20260830/ctmle_region_targeting_b50 \
  /tmp/dml_low_high_response_ablation_20260830_b50
```

## Files

- `replications.csv`: consolidated 768 replication rows with paired ids.
- `summary_by_strength.csv`: per-strength summaries and paired low-minus-high intervals.
- `summary_overall.csv`: all-strength and signal-strength pooled summaries.
- `section4_low_high_response_ablation_table.tex`: generated manuscript table.
- `LIVE_STATUS.md`: monitor snapshots recorded while the parallel run was active.
- `manifest.tsv`: exact chunk commands and seeds.
- `raw/`: chunk-level summary, replication, and log files in the Manifold raw bundle.
- `source/validated_reference_transfer_3e131a9.py`: source snapshot used for the rerun.
- `verification.json`: frozen settings and environment metadata.
