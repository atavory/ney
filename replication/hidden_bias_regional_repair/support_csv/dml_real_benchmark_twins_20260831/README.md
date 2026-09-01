# Twins Benchmark Expansion

This directory records the 2026-08-31 DML Twins expansion for the unified
global-residual repair.  It is companion breadth evidence and does not replace
the Aug. 14 primary Section 4 source unless the manuscript table policy changes.
For paper-facing benchmark organization, Twins is one dataset source.  The
`twins_semisynth` and `twins_misaligned` source-design names record hidden
known-truth construction settings and should not be presented as observable
dataset categories.

## Scope

```text
dataset: Twins same-sex 3-year linked birth/infant-death benchmark
source settings: twins_semisynth, twins_misaligned
strengths: 0, 3
experts: AIPW, fixed-floor TMLE, C-TMLE, selective ML, Ma DR-BC
jobs: 480
replication rows: 1,920
expert-setting combinations: 20
```

The input files are the CEVAE Twins covariate, birthweight, and mortality CSVs.
The run uses the Twins covariates plus birthweight/mortality columns to define
a real-covariate baseline response surface, then adds known synthetic defects
so MSE truth is available.  In the `twins_semisynth` design, the added defect is
placed in the low-response region.  In the `twins_misaligned` design, the
defect is placed in a disjoint signal region.  These labels are used for
known-truth audit only.

Every accepted row uses `repair_mode=if_residual`,
`validation_risk=balanced_mse`, `validation_loss_se=1.0`, `shrink_c=2.0`,
`bootstraps=0`, `region_selector_ablation=legacy`, and damping grid
`0.0|0.25|0.5|1.0`.

## Generated Outputs

The public generator commit is `fe7e8f2`; the archived algorithm source SHA-256
is `98987b31cf7c883d4776996ae7b28f7f1b9fe134d6da323e95250f00232842ce`.
The launch command and wrapper hashes are recorded in `provenance.json`.

The generated files are:

- `cell_summary.csv`: setting-level gains and harm rates.
- `family_summary.csv`: equal-setting summaries by source setting.
- `raw_rows.csv`: compact per-replication rows behind the summaries.
- `release_manifest.csv`: portable manifest with one row per shard.
- `verification.json`: materializer completeness report.
- `manifest.tsv`, `provenance.json`, and `status.json`: launcher provenance.
- `support_inputs.json`: hashes and remote recovery details for Twins input data.

## Readout

Auxiliary dataset-level average MSE gains over the four Twins settings are:

| dataset | AIPW | selective ML | Ma DR-BC | fixed-floor TMLE | C-TMLE |
|---|---:|---:|---:|---:|---:|
| Twins | +0.09% [+0.00, +0.26] | +3.73% [+1.01, +6.72] | +0.07% [-0.53, +0.87] | -0.18% [-0.44, +0.00] | +0.00% [+0.00, +0.00] |

Source-setting readout for audit:

| design | AIPW | selective ML | Ma DR-BC | fixed-floor TMLE | C-TMLE |
|---|---:|---:|---:|---:|---:|
| Twins semisynth | +0.000% [+0.000, +0.000] | +7.052% [+3.391, +11.304] | +0.356% [-0.685, +1.914] | +0.000% [+0.000, +0.000] | +0.000% [+0.000, +0.000] |
| Twins misaligned | +0.190% [+0.000, +0.518] | +0.416% [-3.762, +4.572] | -0.209% [-0.796, +0.183] | -0.353% [-0.898, +0.000] | +0.000% [+0.000, +0.000] |

For the manuscript's primary benchmark table, report the four Twins settings
separately.  Twins supports a selective-ML gain at the dataset level.  The
source-setting split is a known-truth diagnostic, not an observable dataset
classification.

## Remote Recovery

The full shard/log archive is stored in Manifold:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/real_benchmark_expansion_20260831/dml_real_benchmark_twins_v1.tar.zst
```

Archive SHA-256:

```text
7b9ae28e4b6fe4385baf77053f3ed246dceb519a9f2420b383b65ef3fc6a84a2
```

The support-data archive is:

```text
manifold://aai_research_tlv/tree/atavory/dml_reference_transfer/real_benchmark_expansion_20260831/dml_real_benchmark_support_data_v3.tar.zst
```

Support-data archive SHA-256:

```text
72cbeac61f82630afcaea1c6f388b574e1b39ac8f2516e97a4f4d092ad584a86
```
