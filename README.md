# Score-Aligned Model Selection for Orthogonal Estimation

Code for "Score-Aligned Model Selection for Orthogonal Estimation" by Ami Tavory and Tal Sarig.

## Setup

```bash
pip install -r requirements.txt
```

Requires only `numpy`, `scipy`, and `scikit-learn`. No GPU needed.

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

## Hyperparameters

Default hyperparameters match the paper:
- Clipping threshold: `c = 10`
- Number of seeds: `200`
- Propensity clipping: `[0.025, 0.975]`
- Outcome model: `RidgeCV` (sieve demos) or `HistGradientBoostingRegressor`
- Cross-fitting: 5 folds, seed derived from experiment seed

## Citation

```bibtex
@article{tavory2026scorealigned,
  title={Score-Aligned Model Selection for Orthogonal Estimation},
  author={Tavory, Ami and Sarig, Tal},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```
