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

Every orthogonal score induces a sensitivity factor `a(X)²` that identifies where nuisance-model error is costly. The *diagnostic* is universal across estimators. The *intervention* depends on how the estimator consumes nuisance error:

- **AIPW** (training mismatch): `a(X)² = 1/π(X)²`. Error enters as pointwise weighted prediction error → align the **fitting loss**.
- **ATE**: same as AIPW, per arm (`1/π²` for treated, `1/(1-π)²` for control).
- **PLM** (aggregation mismatch): `a(X)² = Var(D|X)`. Error enters through the Robinson regression ratio → align the **θ aggregation step**.
- **IV** (joint alignment): `a(X)² = γ(X)²Var(Z|X)`. Error enters through fitting and the instrument-weighted moment → align **fitting loss + optimal instrument** jointly.

## Files

- `weighted_dr.py` — Score-aligned DR-AIPW estimator (fitting-loss alignment)
- `dr_aipw.py` — Standard DR-AIPW baseline
- `demo_kang_schafer.py` — Reproduces the main linear-model results (AIPW)
- `demo_sieve_sweep.py` — Reproduces the sieve phase transition (AIPW)
- `demo_plm.py` — PLM demo: θ-aggregation alignment with `Var(D|X)`

## Stage alignment in one sentence

Derive the score-sensitive geometry first, then align the stage of the estimator that actually uses it.

## Citation

```bibtex
@article{tavory2026scorealigned,
  title={Score-Aligned Model Selection for Orthogonal Estimation},
  author={Tavory, Ami and Sarig, Tal},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```
