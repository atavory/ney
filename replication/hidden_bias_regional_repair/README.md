# Hidden Bias and Regional Repair: Replication Package

This directory is the current public replication surface for the EJS manuscript
"Hidden Bias and Regional Repair in Low-Response AIPW."

## What Is Included

- `data/current_manuscript_tables.csv`: exact numeric entries for the public
  companion tables cited in the manuscript.
- `data/regional_repair_summary.csv` and
  `data/observed_outcome_budget.csv`: expected output from the default
  companion run.
- `scripts/regional_repair_companion.py`: a self-contained Python simulation for
  the visible-region repair mechanism and the observed-outcome budget check.
- `scripts/validated_reference_transfer.py`: the exact experiment driver used
  for the frozen influence-residual reference-transfer run.
- `scripts/launch_validated_reference_transfer.py`: the chunked, paired-seed
  launcher used for C-TMLE, AIPW, and the global DR-risk proxy.
- `scripts/section4_breadth_experiments.py`: frozen adapter for Kang--Schafer,
  alignment, and real-covariate Section 4 designs.
- `scripts/launch_section4_breadth_shards.py`: deterministic, resumable,
  manifest-producing breadth launcher.
- `scripts/ma_published_did_projection.py`: exact published Ma DiD/ATT design
  with the target-preserving bounded influence-score projection.
- `requirements.txt`: Python dependencies for the companion simulation.

The package contains only the public replication files listed above.

## Frozen Section 4 implementation

The current repair learns an influence-residual correction by cross-fitting,
selects from a path containing the reference action using held-out
influence-relevant loss, and applies the positive-part contrast rule.  The
Section 4 primary constant is `c=2`; the fixed damping grid is
`0, .25, .5, 1`.  The breadth adapter changes only the generated design and,
for Kang--Schafer, the covariates exposed to each nuisance fit.  It delegates
estimation, selection, shrinkage, and output construction to
`validated_reference_transfer.py`.

The public file SHA-256 values used for the 2026-08-10 confirmation are:

```text
validated_reference_transfer.py  7f193f39159a554738b5a1fdd9ccbd9abd3fccfb62e639b04462caf5ce925991
section4_breadth_experiments.py    8681e6cbfdd41a7d5046de7113202283a2772e32af38c3f6bc3a61598fc56f35
launch_section4_breadth_shards.py  13594bbf3d7491b234dc0e26663b730e594f68738a8a68a4d2eef108cf84dfac
ma_published_did_projection.py     5ce093a091dbb37e1a58a807160d5b3efa42ada80acaa7ce20c63b4231ec5101
```

The data release contains raw paired rows, manifests, provenance, and
checked-in aggregators.  Generator output is not manuscript evidence until
that data-side chain verifies full coverage.

## Earlier reference-transfer interface

The validated driver uses the learned residual-score detector, sequential
global-then-regional targeting for C-TMLE and the global DR-risk proxy, and a
regional-only addition to the untargeted outcome regression for AIPW. Thus each
candidate is compared with its own reference while holding its fitted response
model fixed. The internal command name `cui_tchetgen` denotes the global
DR-risk proxy; it is not presented as a faithful implementation of the full
Cui--Tchetgen procedure.

The full run uses strengths `0, 3, 5, 8`, 96 Monte Carlo replications per cell,
and 50 whole-procedure bootstraps per replication. The launcher records every
command and seed, snapshots the driver, and stores SHA-256 hashes for the source
and executable. Point it to an executable copy of the driver:

```bash
export USHMOO_EXPERIMENT_BINARY="$PWD/scripts/validated_reference_transfer.py"
export USHMOO_EXPERIMENT_SOURCE="$PWD/scripts/validated_reference_transfer.py"
python scripts/launch_validated_reference_transfer.py
```

The corresponding data release contains the raw per-replication CSVs and the
aggregation script that regenerates every reported result. Summary CSVs are
generated artifacts, not hand-maintained inputs.

The manuscript accepts a numerical result only when this public generator is
pushed, the data release contains complete raw rows and provenance, and the
checked-in data aggregation script regenerates and verifies the cited summary.
Interactive calculations and manually edited summaries are not paper evidence.

## External Baseline Roadmap

The next public implementations are: plain cross-fitted TMLE;
Ma--Sant'Anna--Sasaki--Ura bias-corrected trimmed DR; and faithful
Cui--Tchetgen Tchetgen selective ML. Dorn's thresholded DR is queued after its
missing-at-random mean specialization is verified. Crump trimming and overlap
weights target different populations and will be kept out of the main
same-target MSE comparison.

The existing `global_dr_risk` code is an internal proxy, not a faithful
Cui--Tchetgen Tchetgen implementation. Historical GL-risk code selects a point
on the C-TMLE regional path and is a diagnostic rather than an external
reference family.

## Quick Check

```bash
python scripts/regional_repair_companion.py --quick
```

This writes:

```text
results/regional_repair_summary.csv
results/observed_outcome_budget.csv
```

The quick mode uses fewer Monte Carlo replications so that reviewers can check
the code path immediately.

## Full Companion Run

```bash
python scripts/regional_repair_companion.py
```

The default run uses 200 replications for the regional-repair benchmark and 200
replications for each observed-outcome budget cell. The command is synthetic
only and needs no external data.

The public companion tables were generated with the default seed `1729`. The
main regional-repair benchmark uses `numpy.random.default_rng(1729)`. The
observed-outcome budget sweep uses `numpy.random.default_rng(101729)`, so it is
independent of the first benchmark. The target value `theta` is computed on a
deterministic grid of 20,001 equally spaced points in `[0, 1]`.

The script was checked with Python 3.9 and 3.12. Runtime dependencies are only
NumPy and pandas, as listed in `requirements.txt`; on a laptop the default run
finishes in well under a minute.

## Protocol

The companion target is the missing-outcome mean

```text
theta = E[Y]
```

under MAR nonresponse. For each observation,

```text
X ~ Uniform(0, 1)
G = 1{X <= q}
R | X ~ Bernoulli(pi(X))
Y = mu_design(X) + noise
```

with lower response probability inside the visible region `G`. The AIPW score
uses a deliberately misspecified globally smoothed score propensity
`p0 = q * 0.25 + (1 - q) * 0.85 = 0.76`, so the experiment isolates the
paper's hidden-bias mechanism. Estimation uses the AIPW score

```text
m(X) + R / p0 * (Y - m(X)).
```

The reference candidate fits a global polynomial outcome regression among
observed outcomes. The regional candidate adds interactions between the
polynomial basis and the visible low-response region. The guarded candidate
uses an honest split and adopts the regional candidate only when the observed
regional validation loss improves by a margin and the validation split contains
enough observed outcomes in `G`.

The observed-outcome budget check repeats the same comparison across sample
sizes. It reports the average number of observed outcomes in `G`, RMSE for the
reference, raw regional repair, screened repair, and the fraction of
replications in which each repair has larger squared error than the reference.

## Scope of This Companion

This companion reproduces the visible-regional-repair and observed-outcome
budget tables bundled here. It is a mechanism check for the outcome-axis repair
under a fixed misspecified score propensity `p0`. It does not implement the full
theoretical grid from the paper: there is no propensity-axis search, no
same-propensity Goldenshluger--Lepski AIPW contrast, and no bootstrap
simultaneous band. Those components are part of the theoretical protected-score
rule, not the simplified empirical screen used in this public companion.

## Status Relative to the Manuscript Tables

The CSV in `data/current_manuscript_tables.csv` records the exact values for
the public companion tables. The default script run regenerates the unrounded
expected-output CSVs in `data/regional_repair_summary.csv` and
`data/observed_outcome_budget.csv`.
