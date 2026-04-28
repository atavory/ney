# Score-Aligned Outcome Model Selection for Finite-Capacity DML

Code for "Finite-Capacity Double Machine Learning: Score-Aligned Model Selection for Orthogonal Estimation."

## Setup

```bash
pip install -r requirements.txt
```

Requires only `numpy`, `scipy`, and `scikit-learn`. No GPU needed.

## Quick start

```bash
python demo_kang_schafer.py
```

Runs the Kang-Schafer (2007) benchmark with 100 seeds, comparing standard DR-AIPW against score-aligned DR. Prints paired RMSE comparison.

## What this does

Standard DML tunes the outcome model by ordinary cross-validation on respondent prediction error.
But the orthogonal correction uses that model through inverse-propensity-weighted residuals, amplifying errors in low-overlap regions.

Score-aligned model selection weights the validation loss by `1/π̂(x)²`, matching the score's local sensitivity.
This is a one-line change to the tuning objective. The estimator, cross-fitting, and propensity model are unchanged.

## Files

- `weighted_dr.py` — Score-aligned DR-AIPW estimator (the method)
- `dr_aipw.py` — Standard DR-AIPW baseline
- `demo_kang_schafer.py` — Reproduces the main linear-model results
- `demo_sieve_sweep.py` — Reproduces the sieve phase transition
- `demo_plm.py` — Partially linear model extension (Var(D|X) alignment)

## Method in one equation

Standard CV selects θ by minimizing: `Σ (Yᵢ - m̂_θ(Xᵢ))²`

Score-aligned CV selects θ by minimizing: `Σ wᵢ (Yᵢ - m̂_θ(Xᵢ))²`

where `wᵢ = min(1/π̂(Xᵢ)², c)` and `c` is a clipping threshold (default 20).

## Citation

```bibtex
@article{tavory2026finite,
  title={Finite-Capacity Double Machine Learning: Score-Aligned Model Selection for Orthogonal Estimation},
  author={Tavory, Ami and Sarig, Tal},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```
