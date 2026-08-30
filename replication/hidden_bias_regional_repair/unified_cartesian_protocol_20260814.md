# Unified Cartesian repair protocol (2026-08-14)

## 2026-08-30 paper-facing amendment

The Aug. 14 v3 unified Cartesian run is now the current EJS Section 4
empirical source.  The compact public reconstruction is
`support_csv/dml_unified_cartesian_global_residual_20260814/`, and the
manuscript tables are generated from that directory by
`scripts/assemble_section4_unified_global_residual.py`.

The historical cautions below remain provenance: they record what was known at
execution time and prevent the empirical run from being overread as a theorem
validation experiment.  They no longer mean the v3 matrix is an unfinished
pilot or a non-paper source.  Current manuscript claims should be read as
empirical performance of the implemented single global-residual repair, with
the theorem stated under its own assumptions.

The binding registry of datasets, experts, and callable interfaces is
`EXPERT_CV_API_AND_DATASETS_20260814.md`. This protocol is subordinate to that
source of truth and must fail closed on any disagreement.

## Estimand and matrix

- Common estimand: population mean with outcomes missing at random.
- Experts: `aipw`, `tmle`, `ctmle`, `cui_selective_ml`, `ma_dr_bc`.
- Cells: all 34 frozen breadth cells: 8 Kang--Schafer, 10 alignment,
  12 real-covariate, and 4 nonlinear-MAR anchor cells.
- Required coverage: 170 expert/cell pairs, 96 paired replications each.

The standalone Ma DiD experiment is not part of this matrix because it has a
different estimand.

## One repair rule

- Public entry point: `unified_expert_repair.repair_expert`.
- Scope: global.
- Construction: outcome residual.
- Selection threshold: 1 standard error.
- Positive-part shrinkage constant: `c=2`.
- Candidate grid: `0, 0.25, 0.5, 1`.
- Three honest folds; XGBoost 3.4.0; no whole-procedure bootstrap because the
  common shrinkage function uses the influence-score contrast variance.

No expert-specific repair selection or shrinkage is permitted. Each expert
may provide its own reference score and repaired endpoint through the common
expert protocol.

The baseline is deliberately a measurement point, not a final paper choice.
It tests the current simplest shared rule before the exact paired SE/c grid and
uniform safety variants. No parameter may be changed in response to an
individual expert's result.

## Known theory/interface defect (recorded during execution)

The public `repair_expert()` selector does not inspect a propensity model, but
the current missing-outcome residual adapter is not fully black-box: it fits a
cross-fitted propensity, uses it to construct the reference expert, and uses
the same propensity again to construct/certify the residual proposal. In the
historical fixed-floor TMLE failure, the `aipw_variance` loss used
`(1-p_hat)/p_hat^2` after flooring `p_hat` at `0.05`.

The floor bounds weights; it does not establish a normal approximation. The
paired improvements can remain strongly skewed and high-kurtosis, and no
finite-sample or uniform-after-search justification currently licenses the
ordinary 1-SE gate as a confidence statement. Moreover, where the true
propensity falls below the floor, the clipped nuisance is not the propensity
in the MAR population-risk identity.

This is a required paper fix, not an estimator-specific exception. The final
procedure must either use only the supplied expert score or fit its own common
nuisance identically for every expert, and its theorem must cover the actual
cross-fitted, clipped, data-dependent selector with explicit tail, nuisance,
multiplicity, and remainder assumptions.

The frozen v3 Cartesian baseline also uses `balanced_mse`, not the
theorem-aligned `aipw_variance` loss. Its rows may finish as exploratory
measurement, but they cannot be used to claim validation of the population
risk theorem. Any corrected theorem-aligned experiment must receive a new
manifest and fresh output identity.

## Frozen fitted-value bank amendment

Future gate and safety-rule comparisons must not refit the expert separately
for each rule. A versioned JSON+NPZ bank freezes the full candidate path and
whole-pipeline repeated-crossfit/delete-block fitted values. Pickled models are
prohibited as scientific artifacts because they execute code, depend on local
library versions, and are not canonically hashable. The bank records source,
seed, fold, configuration, and content hashes and retains every candidate in
the frozen grid rather than only the selected winner.

### Full-bank analysis policy (user decision, 2026-08-14)

The full 16,320-entry bank is an **exploratory observable atlas**, not a
confirmatory development/validation experiment. Complete every one of the 170
expert-by-cell combinations before computing or comparing features. Every
entry contains its full fit plus 20 whole-pipeline repeated-crossfit versions.
The pre-existing `development` and `validation` labels remain immutable
provenance fields, but they do not restrict the first atlas analysis: that
analysis deliberately studies observable behavior across all five experts and
all 34 cells. Simulation truth remains sealed during the observable atlas.

Do not impose an expert holdout while constructing this atlas. The immediate
scientific question is descriptive: which observable behaviors distinguish
stable, unstable, active, and standing-down repair paths across the complete
Cartesian product? No final rule or paper claim is validated by that
description.

After the observable atlas produces a proposed single repair rule, freeze and
hash its feature code, parameters, and decision rule. Confirmatory performance
must then use a newly generated bank with fresh, independent seeds across the
full expert-by-dataset Cartesian product. The current bank cannot be relabeled
as confirmatory after it has been explored.

The earlier locked AIPW/TMLE pilot remains a recorded historical diagnostic in
`EXPERT_BANK_DIRECTION_PILOT_PROTOCOL_20260814.md`; it does not gate completion
or analysis of the full atlas.

## Frozen execution

- 24 shards per cell, 4 replications per shard.
- Seed base: `1800000000`.
- Seeds depend on the frozen full-cell index and shard only, never on expert,
  host partition, or method order.
- dml owns `kang_schafer alignment` (2,160 jobs).
- dml2 owns `real anchor` (1,920 jobs).
- The two scientific identity sets must be disjoint and their union must
  exactly equal the 4,080-job full set. Identity is the ordered tuple
  `(group,design,method,n,strength,chunk,seed)`; output/log paths are
  host-specific execution fields and are excluded from the portable hash.

## Fail-closed completion

Do not aggregate a cell unless it has exactly 96 replication rows and the
expected method, design, n, strength, repair mode, SE threshold, and source
hashes. Do not publish pooled results until all 170 cells pass coverage.

Required outputs include absolute MSE, paired relative gain, path activation,
final activation, gain and harm conditional on final activation, unconditional
harm, shrink-weight/move distributions, and weight/score tail diagnostics.

## Prespecified follow-up hypotheses

The exact paired follow-up crosses SE `{1, 2.83}` with
`c={0,1,2,3,4,5,6,8}` while holding source, rows, folds, and seeds fixed.
Further candidates add only uniform move bounds, trust preconditions,
tail-robust gates, or symmetric fitting/validation weight winsorization. The
primary hypotheses and interpretation constraints are frozen in
`UNIFIED_REPAIR_RESEARCH_LOG_20260814.md`.

## Frozen bundle

- Bundle: `unified_cartesian_bundle_20260814_v3.tar.zst`; verify against the
  adjacent `.sha256` object.

V2 was invalidated before completion because its missing-outcome expert closure
discarded the `RepairParameters` object and therefore unified only shrinkage.
V3 makes `repair_expert` invoke parameterized reference/proposal evaluation and
routes every held-out gate through the shared selector. Only v3 rows are
eligible for this protocol.

The portable v3 scientific hashes are full
`65720b1fce2a24d55872ab9e008cf7ef62945b30b791272f1fdfe65280e2287f`,
dml local `d831b4b29afb66fe910faecec5e3f4ec28ac066fcef02dae229e08480788eda8`,
and dml2 remote
`583033b998053db3f20eb4d08a019208a12a9b915240b95ef72e296e99ccf132`.
