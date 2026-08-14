# Hidden Bias and Regionally Targeted Repair for Low-Response AIPW

## 2026-08-14 research status

The previously released paper artifacts below remain byte-frozen and
reproducible, but current research has identified that they mix repair
construction and gate calibration across upstream experts. They do not by
themselves validate one universal expert-repair function.

New work is governed by:

- `PROJECT_HANDOFF_20260814.md`: primary cross-driver project map, current
  execution state, public/exploratory boundary, artifact locations, and
  migration/resumption order;
- `UNIFIED_REPAIR_RESEARCH_LOG_20260814.md`: findings, corrections,
  hypotheses, reporting requirements, and experiment sequence;
- `unified_cartesian_protocol_20260814.md`: the operative common-estimand
  5-expert x 34-cell baseline protocol;
- `scripts/unified_expert_repair.py`: the single public repair entry point.

Until the full 170-cell Cartesian matrix is complete, no new pilot or partial
aggregate is a paper-wide result. In particular, disconnected Ma-DiD and
Kang--Schafer pilots are execution/mechanism diagnostics only.

The compact operational inventory is
`support_csv/dml_expert_bank_handoff_20260814_v1/artifact_inventory.csv`.
Large fitted-value banks are stored on Manifold and are not committed here.

This directory is the public replication package for the EJS manuscript of
the same name. It contains the paper-facing values, the paired replication
rows from which the Section 4 sensitivity atlas is assembled, and the frozen
scripts used to generate and verify those artifacts.

## Authoritative paper artifacts

- `support_csv/dml_section4_release_20260812_v1/` is the submission-facing
  release. `paper_values.csv` maps every reported value to its source file and
  selector; the generated TeX tables and macros are included verbatim.
- `support_csv/dml_section4_c_atlas_20260812_v2/` contains the complete
  `c = 0,1,2,3,4,5,6,8` cell and panel curves used by Figures 4--7.
- The seven source directories under `support_csv/` contain the paired raw
  rows or Ma shards, certified summaries, and available verification records
  used by the release and atlas assemblers.
- `scripts/assemble_section4_release.py` builds the manuscript value ledger
  and generated tables from the certified summaries.
- `scripts/assemble_section4_c_atlas.py` rebuilds all 21 estimator/design
  curves from the paired rows.
- `scripts/plot_section4_c_atlas.py` renders the compact efficacy, safety,
  emphasized, and diagnostic figures without fitting an estimator or choosing
  `c` from observed MSE.
- `scripts/verify_section4_manuscript.py` checks source hashes, generated
  tables, figure bytes, panel coverage, and manuscript includes when run with
  the paper repository.

The primary missing-outcome shrinkage constant is `c = 2`; the candidate
damping grid is `0, 0.25, 0.5, 1`. Every reported repaired/reference pair uses
the same observations, folds, and seed. Family summaries weight native cells
equally, and intervals use 20,000 paired percentile-bootstrap draws stratified
by cell.

## Rebuild the paper values

From this directory, create an empty destination and run:

```bash
python scripts/assemble_section4_release.py \
  --data-root . \
  --out-dir /tmp/section4_release_rebuilt
```

The rebuilt files should match
`support_csv/dml_section4_release_20260812_v1/` byte for byte.

The Ma weak-overlap DiD summary can also be regenerated directly from its 768
paired rows:

```bash
python scripts/dml_ma_did_aggregate.py \
  --run-dir support_csv/dml_section4_confirmatory_20260810_v1/ma_xgboost_shards \
  --out-json /tmp/ma_xgboost_summary.json
```

## Rebuild the sensitivity atlas

```bash
python scripts/assemble_section4_c_atlas.py \
  --data-root . \
  --out-dir /tmp/section4_atlas_rebuilt

python scripts/plot_section4_c_atlas.py \
  --data-dir /tmp/section4_atlas_rebuilt \
  --out-dir /tmp/section4_figures_rebuilt
```

The assembler enforces the fixed source policy, 21-panel coverage, 108 native
cells, 864 cell-curve rows, 168 panel-curve rows, and the full prespecified
`c` grid. The plotting script consumes only those generated CSVs.

## Experiment drivers

The package also includes the frozen experiment drivers and launchers for the
Kang--Schafer, Cui--Tchetgen Tchetgen, aligned-anchor, placement,
real-covariate, and Ma DiD comparisons. The final paper compares each repaired
estimator with its own upstream reference and reports C-TMLE, faithful
selective ML, AIPW, Ma DiD, and plain-TMLE negative-control results separately.

That paragraph describes the archived release. The new universal-function
evaluation instead places AIPW, TMLE, C-TMLE, Cui selective ML, and Ma DR-BC
on every common MAR benchmark cell. The native Ma DiD experiment remains a
separate-estimand external study and is not pooled into that Cartesian matrix.

## Optional companion simulation

`scripts/regional_repair_companion.py` is a small standalone mechanism check.
It is useful for a quick smoke test but is not the source of the manuscript's
Section 4 results:

```bash
python scripts/regional_repair_companion.py --quick
```

Its historical outputs are retained under `data/` and are labeled as legacy
companion artifacts to avoid confusing them with the authoritative release.

## Environment

Install the Python dependencies in `requirements.txt`. The assemblers require
NumPy; figure rendering additionally requires Matplotlib. The experiment
drivers use the full pinned stack, including scikit-learn and XGBoost.
