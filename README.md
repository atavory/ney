# ney — Score-Aligned Tuning for Doubly Robust Estimation

Code for "Finite-Capacity Double Machine Learning: Score-Aligned Nuisance Tuning for Robust Causal Inference."

## Quick start

```bash
pip install numpy scipy scikit-learn
python demo_kang_schafer.py
```

## What this does

Standard doubly robust estimators (AIPW/DML) tune the outcome nuisance model by cross-validation on respondent prediction error. But the estimator uses that model through a weighted correction where errors in low-overlap regions are amplified by inverse propensity.

Score-aligned tuning weights the validation loss by 1/e(x)^2 to match the correction's sensitivity. This is a one-line change to the CV objective that reallocates model capacity toward regions where the score is most fragile.

## Files

- `algs/weighted_dr.py` — Score-aligned DR-AIPW (the method)
- `algs/dr_aipw.py` — Standard DR-AIPW (baseline)
- `demo_kang_schafer.py` — Reproduces the main result on Kang-Schafer (2007)

## Requirements

- Python 3.10+
- numpy, scipy, scikit-learn
