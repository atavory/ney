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
- `requirements.txt`: Python dependencies for the companion simulation.

The package contains only the public replication files listed above.

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
