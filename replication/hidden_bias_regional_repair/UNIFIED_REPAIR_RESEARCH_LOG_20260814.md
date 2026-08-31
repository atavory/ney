# Unified expert-repair research log

## 2026-08-30 paper-facing status

This file is a historical research and audit log.  It records the concerns and
decision rules that governed the Aug. 14 v3 unified Cartesian run.  The current
EJS Section 4 empirical source is the compact reconstruction in
`support_csv/dml_unified_cartesian_global_residual_20260814/`, as indexed by
`README.md` and `PUBLIC_ARTIFACTS.md`.

2026-08-31 update: Section 4 now also includes two generated companion
ablations.  The low/high-response region-placement check lives in
`support_csv/dml_low_high_response_ablation_20260830/`; the no-shrinkage versus
`c=2` check lives in `support_csv/dml_no_shrinkage_ablation_20260830/`.  The
cross-repository verifier checks both generated ablation tables against those
bundles.

Historical cautions below still constrain interpretation: the empirical run is
not a theorem-validation experiment, and alternatives or later diagnostics
should not be silently pooled into the current Section 4 tables.  They do not
mean the v3 matrix is still incomplete or that the old mixed release is the
current manuscript source.

The sole binding registry for datasets, experts, and repair interfaces is
`EXPERT_CV_API_AND_DATASETS_20260814.md`. This log records hypotheses and
history; it does not define experimental scope.

Updated 2026-08-14. This is the operative scientific record for new work. It
does not alter the bytes or provenance of previously certified artifacts.

## Non-negotiable design requirement

The paper may expose exactly one repair function. It takes an expert through a
common estimate/influence-score interface plus explicit repair parameters. It
must not inspect the expert's name or silently dispatch C-TMLE, Cui, AIPW,
TMLE, or Ma to different repair rules.

The candidate construction (`residual` or `projection`), support (`global` or
`regional`), selection threshold, shrinkage constant, candidate grid, robust
gate, trust precondition, and move bound are parameters of that one function.
The paper must choose one frozen setting and apply it to every expert.

## Correction to the experimental design

Earlier reports combined disconnected benchmark slices: Ma only on its native
DiD DGPs and AIPW/TMLE only on Kang--Schafer. That was not a Cartesian product
and cannot support a cross-expert conclusion. The associated pilot headlines
(Ma about +7.02%, AIPW about +8.08%, and TMLE about +0.59%) are withdrawn as
paper-wide evidence. The raw rows remain useful only as execution tests and
mechanism diagnostics.

The new common matrix uses the MAR population mean, for which all five
repository experts are implemented: AIPW, plain TMLE, adaptive C-TMLE,
faithful Cui selective ML, and Ma DR-BC. The native Ma DiD simulation has a
different estimand and is retained as a separate external mechanism study; it
does not enter the common Cartesian aggregate.

## What prior results actually establish

The original certified paper artifacts are preserved. They establish results
for their stated estimator-specific adapters and calibrations, not for a
single universal black-box repair:

- C-TMLE, global residual, 1-SE, `c=2`, Kang--Schafer: about +3.41%.
- Cui, global residual, 1-SE, `c=2`, Kang--Schafer: about +5.76%.
- AIPW, projection, Kang--Schafer: about +1.54%.
- plain TMLE, residual, Kang--Schafer: about -4.90%; projection stood down in
  the tested KS/Cui cells.
- Ma DiD, projection with a 2.83-SE gate: DGP3 about +17.73%, equal-DGP family
  about +8.86%.

These are valuable historical findings but they mix construction and gate by
expert. They therefore cannot be presented as validation of the new single
function.

The later complete global-residual matrix used a different exploratory source
lineage and a 2.83-SE gate. Its near-null C-TMLE result and negative TMLE
result are informative diagnostics, not replications of the original 1-SE
results.

## Activation accounting correction

Two activation concepts must never be conflated:

- **Path activation:** held-out selection chooses a nonzero candidate.
- **Final activation:** positive-part shrinkage assigns positive weight.

For the certified C-TMLE `c=2` run, path activation was 597/768 (77.73%) and
final activation was 95/768 (12.37%). Both numbers are from the same `c=2`
run. The earlier claim that 597/768 represented `c=0` was wrong.

Every future table reports both rates explicitly.

## Current gate diagnosis

The audit produced the following conditional diagnostics:

| run | gate | final activation | gain conditional on activation | harm conditional on activation |
|---|---:|---:|---:|---:|
| TMLE residual, certified | 1-SE | 10.0% | -62.3% | 49.4% |
| TMLE residual, exploratory | 2.83-SE | 7.7% | -25.0% | 50.8% |
| C-TMLE residual, certified | 1-SE | 12.4% | +31.3% | 22.1% |

Interpretation: for plain TMLE, activation is approximately a coin flip by
frequency and the harmful tail is much larger than the beneficial tail. For
C-TMLE, activation is substantially more informative. Mean gain alone hid
this difference.

The leading mechanism hypothesis is that the missing-outcome adapter's
estimated propensity, fixed at a small floor for the plain-TMLE run, produces
large inverse-propensity weights and a heavy-tailed score. This is not an
internal propensity object obtained from a black-box TMLE. The experimental
pipeline fits one cross-fitted propensity, uses it to construct the TMLE
reference, and reuses it in residual fitting and the `aipw_variance`
validation risk. Thus a pathological nuisance estimate is partly certifying
the repair proposal built from that same nuisance estimate. Adaptive C-TMLE
produces a better-behaved score, making the same nominal gate more meaningful.

Clipping does **not**, by itself, logically imply non-normality. It bounds the
largest weight. A small floor can nevertheless leave a highly skewed,
high-kurtosis finite-sample distribution, with a small number of responders
near the floor dominating the paired improvement. We have not established
that the studentized mean improvement is approximately normal at the sample
sizes used. Therefore `mean improvement > k * estimated SE` is only a nominal
gate here, not a demonstrated confidence statement.

## Critical theory/implementation gap to fix

The population risk identity and oracle projection result do not require a
normal gate. They identify a population/oracle improvement. The implemented
selection step adds two assumptions that the paper has not proved:

1. replacing the true propensity in the MAR-identifiable risk by the fitted,
   clipped propensity preserves the required risk ordering; and
2. the paired empirical improvement and its ordinary standard error provide a
   valid uniform certificate after candidate search.

Both are open. In particular, if the true propensity is below the fixed
`0.05` floor, the clipped propensity is not the true propensity appearing in
the observable-risk identity. The clipping may be a deliberate stabilization
policy, but then the theorem must cover the resulting misspecification and
remainder rather than silently substitute the clipped value for the truth.

There is also an abstraction defect. `repair_expert()` itself consumes only an
estimate, scores, and a proposal, but the current residual proposal adapter
uses the same fitted propensity that was used to build the reference expert.
Consequently the end-to-end residual procedure is not yet a strictly
black-box repair of an already-fitted expert. Before the paper can claim that
interface, we must do one of the following uniformly for every expert:

- use a score-only construction such as the generic projection repair; or
- make the repair own and cross-fit an explicitly common nuisance model,
  independent of how the expert was fitted, and prove the guarantee for that
  full observable procedure.

Required theorem repair: state a finite-sample or high-probability guarantee
for the actual data-dependent candidate search using observable held-out
quantities, with explicit MAR, overlap, moment/tail, nuisance-rate,
cross-fitting, multiplicity, and remainder assumptions. The ordinary 1-SE or
2.83-SE rule is not licensed as a confidence guarantee until those conditions
are proved and empirically diagnosed. Historical TMLE residual results remain
failure diagnostics; they are not evidence that the current theory protects
TMLE.

## Direction decomposition and frozen-fit redesign

The completed KS decomposition uses
`repaired_error^2-reference_error^2 = move^2 + 2*move*reference_error`.
The observable move-size diagnostics were not the missing axis. AIPW paid a
larger movement cost than TMLE but its move opposed reference error; TMLE's
aiming correlation was approximately zero and its move direction was
effectively random. This changes the safety target from attenuation to
abstention: a smaller randomly directed move remains a bad bet.

The next API therefore consumes repeated whole-pipeline fitted values, not a
single realized expert score. For every algorithm-by-dataset realization we
freeze the observed data, fold IDs, reference score, initial/reference outcome
fits, propensity fit, every candidate score and endpoint, selected candidate,
shrink weight, seeds, exact source hash, and repeated-crossfit/delete-block
refits. We do **not** pickle executable estimator objects. The scientific
format is inspectable JSON plus compressed NPZ with a model-independent
content digest; simulation truth is separately named and hash-covered so the
observable repair diagnostic cannot consume it.

The initial direction pilot used an immutable dataset-level
development/validation split and one primary statistic: signed direction
reproducibility across 20 whole-pipeline repeated cross-fits, counting
stand-downs as zero. It failed its frozen 0.70 threshold and remains a recorded
historical diagnostic in `EXPERT_BANK_DIRECTION_PILOT_PROTOCOL_20260814.md`.

The subsequent user decision is to finish the complete 16,320-entry fitted-
value bank before inspecting full-bank features and to use it as an
**exploratory observable atlas**. The atlas includes every expert and every
dataset cell; no expert is held out at this descriptive stage. We inspect
observable full-fit and 20-refit behavior across the complete Cartesian
product while keeping simulation truth sealed. Because this intentionally
uses the entire bank for scientific discovery, no result from it is called
confirmatory. A proposed uniform repair rule must be frozen and hash-committed
before a separate fresh-seed Cartesian bank is generated for confirmation.

The full bank has `delete_blocks=0`. It supports repeated-crossfit direction
reproducibility, not genuine leave-block-out sensitivity. The latter is
available only in the earlier pilot and must not be claimed for the full atlas.

The table does **not** identify the causal effect of changing 1-SE to 2.83-SE:
the certified and exploratory rows differ in source lineage, seeds, grids,
and other execution details. Statements such as "2.83 suppressed C-TMLE by
95x" remain hypotheses until an exact paired ablation changes only the SE
threshold.

## Hypotheses to test

1. **Reliability hypothesis.** Conditional repair gain is predicted by
   influence-score/weight reliability, not by expert identity.
2. **Heavy-tail hypothesis.** Low effective sample size, high maximum-weight
   share, or high improvement kurtosis explains TMLE's adverse activated tail.
3. **Adaptive-floor hypothesis.** C-TMLE passes the same reliability
   precondition more often because its selected propensity floor stabilizes
   the score.
4. **Threshold hypothesis.** Raising a normal-theory SE threshold reduces the
   number of bets but does not calibrate heavy-tailed false positives.
5. **Bounded-move hypothesis.** A uniform cap
   `|theta_repaired-theta_ref| <= kappa * SE(theta_ref)` materially limits
   catastrophic harm without eliminating C-TMLE/Cui gains.
6. **Robust-gate hypothesis.** Paired sign/rank or bootstrap calibration has a
   better conditional-gain/harm profile than mean divided by a normal SE.
7. **Weight-stabilization hypothesis.** Uniform winsorization of
   `(1-p)/p^2` in both residual fitting and validation improves calibration;
   any bias/efficacy cost must be measured, not assumed away.

## Experiment sequence

### A. Baseline Cartesian matrix

- One rule: global outcome residual, 1-SE, `c=2`, grid
  `{0, .25, .5, 1}`.
- Five experts by all 34 common MAR cells: 170 cells.
- 96 paired replications per cell: 16,320 scientific rows.
- Three honest folds, XGBoost 3.4.0, common influence-contrast shrinkage.
- dml runs Kang--Schafer and alignment (2,160 four-replication jobs).
- dml2 runs real-covariate and nonlinear-MAR anchor cells (1,920 jobs).
- The scientific manifest identities are disjoint; their union exactly equals
  the 4,080-job full identity set; seeds are invariant to host partition and
  expert. Raw TSV bytes differ by run directory because output/log paths are
  execution fields and are deliberately excluded from the scientific hash.

Execution note: the frozen v3 baseline uses the historical `balanced_mse`
default rather than the theorem-aligned `aipw_variance` loss. It remains useful
as a complete Cartesian measurement, but it cannot validate the population
risk theorem or be described as its confirmatory experiment. Do not reinterpret
or relabel these rows after completion; a corrected theorem-aligned procedure
requires a separately frozen run.

### B. Exact paired knob grid

After the baseline plumbing and coverage audit, rerun the same rows/seeds over
the SE grid `{1, 2.83}` and the existing shrinkage grid
`{0,1,2,3,4,5,6,8}`. Only SE and `c` may change. This is the experiment that
can identify their effects; cross-lineage comparisons cannot.

### C. Uniform safety variants

Evaluate, without expert-specific exceptions:

- move-bound multipliers `kappa` on a prespecified grid;
- effective-sample-size, maximum-weight-share, and kurtosis trust
  preconditions;
- sign/rank and paired-bootstrap gates;
- symmetric weight winsorization in fitting and validation.

A safety variant is eligible only if its parameters and diagnostics are
computed identically for every expert.

## Required reporting

For every expert/cell and every family aggregate report:

- reference and repaired absolute MSE;
- paired relative MSE gain with stratified paired interval;
- path activation and final activation;
- gain conditional on final activation;
- harm rate conditional on final activation and unconditional harm rate;
- mean/quantiles of shrink weight and normalized move size;
- effective sample size, maximum-weight share, improvement kurtosis, and any
  trust-precondition stand-down rate;
- exact source, manifest, seed, and configuration hashes.

No paper-wide headline is permitted before all 170 baseline cells pass exact
coverage. A matrix used to choose parameters is exploratory; the selected
single rule requires fresh-seed confirmation.

## Implementation audit and frozen baseline bundle

The first v2 launch was stopped and invalidated after 263/2,160 local jobs when
audit showed that the missing-outcome `proposal_factory` ignored the parameter
object: the new entry point owned shrinkage but did not yet own proposal
construction/selection. No v2 row may enter an aggregate.

In v3, `repair_expert(expert, parameters)` calls `expert.evaluate(parameters)`;
that evaluation returns both the reference and proposal, and every held-out
gate calls the shared `select_candidate_from_improvements`. The four explicit
scope/construction combinations are global/regional by residual/projection;
the baseline freezes global residual.

- Bundle: `unified_cartesian_bundle_20260814_v3.tar.zst`.
- The authoritative archive SHA-256 is stored in the adjacent `.sha256`
  object to avoid a self-referential hash inside the archive.
- Shared seed base: `1800000000`.

The bundle freezes the universal repair entry point, scientific runner,
wrapper, launcher, structural tests, research record, protocol, and
full/remote manifests.

V3 scientific-manifest SHA-256 values are computed from ordered
`group,design,method,n,strength,chunk,seed` records:

- full: `65720b1fce2a24d55872ab9e008cf7ef62945b30b791272f1fdfe65280e2287f`;
- dml local: `d831b4b29afb66fe910faecec5e3f4ec28ac066fcef02dae229e08480788eda8`;
- dml2 remote: `583033b998053db3f20eb4d08a019208a12a9b915240b95ef72e296e99ccf132`.

The earlier raw-TSV hashes were template-location hashes, not portable
scientific identifiers, and must not be used to validate a run on another
host.

### Execution status

- dml v3 Kang--Schafer/alignment partition: complete, 2,160/2,160 jobs,
  8,640 rows, 90/90 cells with 96 rows each, zero failures.
- Archive SHA-256:
  `1a6db65195d507ac2c4e1a21c62010b876f015ca0333bb6467c8dcf6d22ab6aa`.
- dml2 real/anchor partition: assigned; it is not counted as live or complete
  until PID, portable manifest hash, and counts are acknowledged.

## Public protocol note

Operational migration details and private storage locations are not part of the
public replication package. This log records scientific design decisions and
does not certify exploratory fitted-bank coverage.
