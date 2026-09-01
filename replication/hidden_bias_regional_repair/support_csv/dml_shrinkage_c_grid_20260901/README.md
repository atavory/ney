# Shrinkage-c Grid

This bundle re-evaluates the final scalar shrinkage constant using the
saved Section 4 replication rows.  It does not refit nuisance models or
reselect the damping candidate; it changes only the final weight
`max(0, 1 - c * vd / delta^2)` applied to the selected movement.

The benchmark target is the same 24-setting table used in the current
manuscript: eight Kang--Schafer settings and four settings each for
IHDP, ACIC 2016, ACIC 2017, and Twins, over AIPW, selective ML,
Ma DR-BC, and C-TMLE.

Grid: 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 6, 8, 10.

Best equal-setting benchmark gains on this grid:

| expert | best c | gain | interval | harm | activation |
|---|---:|---:|---:|---:|---:|
| AIPW | 0 | 9.049% | [7.061, 10.678] | 10.634% | 27.691% |
| selective ML | 0 | 4.948% | [3.517, 6.288] | 12.587% | 32.682% |
| Ma DR-BC | 0 | 5.793% | [4.536, 6.946] | 8.116% | 22.309% |
| C-TMLE | 1 | 0.051% | [-0.051, 0.188] | 0.130% | 0.217% |

Best gains subject to harm no greater than 5%:

| expert | c | gain | harm | activation |
|---|---:|---:|---:|---:|
| AIPW | 0.5 | 7.734% | 4.731% | 15.799% |
| selective ML | 1 | 2.908% | 4.905% | 16.016% |
| Ma DR-BC | 0.75 | 5.387% | 4.514% | 15.755% |
| C-TMLE | 1 | 0.051% | 0.130% | 0.217% |

Files:

- `c_grid_setting_summary.csv`: setting-level summaries.
- `c_grid_dataset_summary.csv`: dataset-level summaries.
- `c_grid_benchmark_summary.csv`: 24-setting summaries by expert and c.
- `c_grid_best_by_expert.csv`: best c by equal-setting benchmark gain.
- `c_grid_best_under_harm.csv`: best c by expert under harm caps.
- `verification.json`: row counts, grid values, and source hashes.
