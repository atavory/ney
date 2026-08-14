# Hidden Bias and Regional Repair in Orthogonal Estimator Selection

This repository contains replication material for the current EJS manuscript,
"Hidden Bias and Regional Repair in Orthogonal Estimator Selection."

The current manuscript package is in
`replication/hidden_bias_regional_repair/`.

For the August 14 universal-repair experiments, fitted-value banks, public
versus exploratory boundary, and devvm migration state, start with
`replication/hidden_bias_regional_repair/PROJECT_HANDOFF_20260814.md`.

The older top-level `algs/`, `experiments/`, and `demo_*.py` files are legacy
material from an earlier score-aligned-selection draft. They remain available
for continuity, but they are not the primary replication surface for the current
EJS manuscript.

## Current Manuscript Package

```bash
cd replication/hidden_bias_regional_repair
python scripts/regional_repair_companion.py --quick
```

The package includes the public reproducibility protocol cited by the current
manuscript, a CSV snapshot of the bundled companion tables, and a self-contained
Python simulation for the regional-repair mechanism and observed-outcome budget
check.

The journal-facing frozen supplement should use this package as the starting
point. GitHub is a convenience mirror, not the only archival artifact.

## Legacy Material

The material below documents the earlier public release for "Score-Aligned
Model Selection for Orthogonal Estimation."

## Setup

```bash
pip install -r requirements.txt
```

Requires `numpy`, `scipy`, `scikit-learn`, and `pandas`. No GPU needed.

## Quick start

```bash
python demo_kang_schafer.py   # AIPW on Kang-Schafer DGP (200 seeds)
python demo_plm.py            # PLM with aggregation-weighted Robinson
python demo_ate.py            # ATE with per-arm fitting-loss alignment
python demo_iv.py             # IV with optimal instrument alignment
python demo_sieve_sweep.py    # Sieve phase transition (AIPW)
```

## What this does

Every orthogonal score induces a sensitivity factor `a(X)^2` that identifies where nuisance-model error is costly. The *diagnostic* is universal across estimators. The *intervention* depends on how the estimator consumes nuisance error:

| Estimator | Sensitivity factor | Aligned stage |
|-----------|-------------------|---------------|
| **AIPW** | `1/pi(X)^2` | Fitting loss |
| **ATE** | `1/pi(X)^2` per arm | Fitting loss (per arm) |
| **PLM** | `Var(D\|X)` | Theta aggregation |
| **IV** | `gamma(X)^2 Var(Z\|X)` | Fitting loss + optimal instrument |

## Files

- `algs/weighted_dr.py` -- Score-aligned DR-AIPW estimator with configurable exponent (power=0 unweighted, power=1 stabilized, power=2 score-aligned)
- `algs/dr_aipw.py` -- Standard DR-AIPW baseline (unweighted)
- `demo_kang_schafer.py` -- AIPW: compares unweighted, stabilized (1/pi), and aligned (1/pi^2) on Kang-Schafer
- `demo_plm.py` -- PLM: theta-aggregation alignment with Var(D|X)
- `demo_ate.py` -- ATE: per-arm fitting-loss alignment
- `demo_iv.py` -- IV: outcome-fit weighting + optimal instrument alignment
- `demo_sieve_sweep.py` -- Sieve phase transition showing three regimes

## Reproducing paper results

All experiment scripts are in `experiments/`. Each script is self-contained and can be run from the `experiments/` directory.

```bash
cd experiments
python <script_name>.py [--seeds 200] [--workers 8] [--output results/output.csv]
```

### Table/figure mapping

| Paper table/figure | Script | Description |
|-------------------|--------|-------------|
| Table 2 | `experiments/nk_sweep.py` | Joint sample-size and capacity sweep (KS DGP) |
| Figure 2 | `experiments/capacity_sweep.py` | Capacity sweep: selective-overload prediction |
| Figure 2a / Tables A3-A4 | `experiments/sieve_sweep.py` | Sieve (polynomial) capacity sweep |
| Table 4 (coverage) | `experiments/coverage.py` | Coverage simulations for AIPW confidence intervals |
| Table 4 (frontier) | `experiments/aipw_frontier.py` | Fixed-regime AIPW capacity frontier |
| Table 5 (capacity curve) | `experiments/capacity_curve_real.py` | Practitioner's capacity curve on real survey data |
| Table 5 (Card IV) | `experiments/iv_card.py` | IV stage ablation on Card and 401(k) data |
| Table 7 | `experiments/plm_ablation.py` | PLM bottleneck ablation (Var(D\|X) stages) |
| Table 9 | `experiments/iv_ablation.py` | IV bottleneck ablation (optimal instrument) |
| Table A5 | `experiments/mediation_ablation.py` | Mediation (NIE) ablation |
| Table A6 | `experiments/qte_ablation.py` | QTE ablation with density-sensitive weights |
| Table A7 | `experiments/dtr_ablation.py` | Two-stage DTR ablation |
| ATE benchmarks | `experiments/ate_cattaneo.py` | ATE on Cattaneo smoking/birthweight data |
| RHC benchmark | `experiments/ate_rhc.py` | ATE on Right Heart Catheterization data |
| KS supplement | `experiments/capacity_curve_ks.py` | Capacity curve on KS synthetic DGP |
| Real-covariate AIPW | `experiments/real_tail_aipw.py` | AIPW with real covariates and tail structure |

### Shared utilities

`experiments/common.py` contains shared functions used across experiment scripts (DGP helpers, weight normalization, cross-fitting utilities, CSV I/O).

## Downloading datasets

Several experiments use real-data covariates. To download the datasets:

```bash
python data/download_datasets.py
```

Some datasets (Card, RHC, Cattaneo, LaLonde, IHDP, 401k) are downloaded automatically. Others (ACS PUMS, CPS ASEC, BRFSS, CES, GSS) require manual download due to their size or access restrictions -- the script prints instructions for each.

Downloaded files go into the `data/` directory.

### Synthetic-only experiments

The following experiments are fully synthetic and need no data downloads:
- `nk_sweep.py`, `capacity_sweep.py`, `sieve_sweep.py`, `coverage.py`
- `plm_ablation.py`, `iv_ablation.py`
- `mediation_ablation.py`, `qte_ablation.py`, `dtr_ablation.py`
- `capacity_curve_ks.py`

## Hyperparameters

Default hyperparameters match the paper:
- Clipping threshold: `c = 10`
- Number of seeds: `200`
- Propensity clipping: `[0.025, 0.975]`
- Outcome model: `RidgeCV` (sieve demos) or `HistGradientBoostingRegressor`
- Cross-fitting: 5 folds, seed derived from experiment seed
- OMP_NUM_THREADS: set to 1 for sklearn multiprocessing compatibility

## Citation

Paper under review.
