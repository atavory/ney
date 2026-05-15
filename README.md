# Score-Aligned Model Selection for Orthogonal Estimation

Code for "Score-Aligned Model Selection for Orthogonal Estimation."

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

Every orthogonal score induces a sensitivity factor `a(X)` that determines how local outcome-model error translates into estimand variance. Standard model selection ignores this geometry. Score-aligned model selection weights the fitting loss by `a(X)²`, matching the score's deployment geometry.

This is a one-line change to model selection. The estimator, cross-fitting scheme, and all other nuisance models are unchanged.

The mechanism is identical across four estimators:
- **AIPW**: `a(X) = 1/π(X)`, weight `= 1/π(X)²`
- **ATE**: per-arm inverse propensity
- **PLM** (Robinson score): `a(X)² = Var(D|X)`
- **IV** (Robinson score): same structure, compliance-weighted

## Files

- `weighted_dr.py` — Score-aligned DR-AIPW estimator (the method)
- `dr_aipw.py` — Standard DR-AIPW baseline
- `demo_kang_schafer.py` — Reproduces the main linear-model results (AIPW)
- `demo_sieve_sweep.py` — Reproduces the sieve phase transition (AIPW)
- `demo_plm.py` — Partially linear model demo (Var(D|X) alignment)

## Method in one equation

Standard CV selects θ by minimizing: `Σ (Yᵢ - m̂_θ(Xᵢ))²`

Score-aligned CV selects θ by minimizing: `Σ wᵢ (Yᵢ - m̂_θ(Xᵢ))²`

where `wᵢ = min(a(Xᵢ)², c)` and `c` is a clipping threshold. For AIPW, `a(X) = 1/π̂(X)`; for PLM, `a(X)² = Var(D|X)`.

## Citation

```bibtex
@article{tavory2026scorealigned,
  title={Score-Aligned Model Selection for Orthogonal Estimation},
  author={Tavory, Ami and Sarig, Tal},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```
