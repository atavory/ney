#!/usr/bin/env python3
"""
Demo: Score-aligned ATE estimation via AIPW.

The ATE sensitivity factor is per-arm (Table 1):
  treated arm:  a(X)^2 = 1/pi(X)^2
  control arm:  a(X)^2 = 1/(1 - pi(X))^2

The aligned stage is the fitting loss for each arm's outcome model,
the same mechanism as AIPW but applied separately per arm.

AIPW-ATE estimator:
  mu_1 = (1/n) sum [ m_1(X_i) + T_i (Y_i - m_1(X_i)) / e(X_i) ]
  mu_0 = (1/n) sum [ m_0(X_i) + (1 - T_i)(Y_i - m_0(X_i)) / (1 - e(X_i)) ]
  ATE  = mu_1 - mu_0

Requirements: numpy, scipy, scikit-learn
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import KFold


def make_ate_dgp(n: int, seed: int):
    """Kang-Schafer-style ATE DGP with confounded treatment."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 4))

    # Observed = nonlinear transforms of latent
    X = np.column_stack(
        [
            np.exp(Z[:, 0] / 2),
            Z[:, 1] / (1 + np.exp(Z[:, 0])) + 10,
            (Z[:, 0] * Z[:, 2] / 25 + 0.6) ** 3,
            (Z[:, 1] + Z[:, 3] + 20) ** 2,
        ]
    )

    # Confounded treatment assignment
    e_true = 1 / (1 + np.exp(-(Z[:, 0] - 0.5 * Z[:, 1])))
    T = rng.binomial(1, e_true)

    # Potential outcomes: ATE = 2
    Y0 = Z[:, 0] + Z[:, 1] + rng.standard_normal(n)
    Y1 = Y0 + 2 + Z[:, 0]
    Y = T * Y1 + (1 - T) * Y0
    ate_true = 2.0

    return X, T, Y, ate_true


def estimate_ate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    power: int = 0,
    clip: float = 10.0,
    seed: int = 0,
) -> float:
    """AIPW ATE with optional per-arm score-aligned outcome fitting.

    Parameters
    ----------
    X : covariates
    T : binary treatment indicator
    Y : outcome
    power : exponent for fitting weights. 0 = unweighted, 2 = aligned.
    clip : max weight for outcome training.
    seed : random seed for cross-fitting.
    """
    n = len(Y)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Cross-fitted propensity scores
    e_hat = np.full(n, 0.5)
    for train_idx, val_idx in kf.split(X):
        clf = HistGradientBoostingClassifier(
            max_depth=4, max_iter=200, random_state=seed
        )
        clf.fit(X[train_idx].astype(float), T[train_idx])
        e_hat[val_idx] = np.clip(
            clf.predict_proba(X[val_idx].astype(float))[:, 1], 0.025, 0.975
        )

    # Cross-fitted per-arm outcome models
    m1_hat = np.zeros(n)
    m0_hat = np.zeros(n)

    for train_idx, val_idx in kf.split(X):
        t1 = train_idx[T[train_idx] == 1]
        t0 = train_idx[T[train_idx] == 0]

        # Per-arm weights
        if power > 0:
            w1_raw = (1.0 / e_hat[t1]) ** power
            w1 = np.minimum(w1_raw, clip)
            w1 = w1 / w1.mean()
            w0_raw = (1.0 / (1 - e_hat[t0])) ** power
            w0 = np.minimum(w0_raw, clip)
            w0 = w0 / w0.mean()
        else:
            w1, w0 = None, None

        reg1 = HistGradientBoostingRegressor(
            max_depth=4, max_iter=200, random_state=seed
        )
        reg0 = HistGradientBoostingRegressor(
            max_depth=4, max_iter=200, random_state=seed
        )
        reg1.fit(X[t1].astype(float), Y[t1], sample_weight=w1)
        reg0.fit(X[t0].astype(float), Y[t0], sample_weight=w0)
        m1_hat[val_idx] = reg1.predict(X[val_idx].astype(float))
        m0_hat[val_idx] = reg0.predict(X[val_idx].astype(float))

    # AIPW ATE
    mu1 = np.mean(m1_hat + T * (Y - m1_hat) / e_hat)
    mu0 = np.mean(m0_hat + (1 - T) * (Y - m0_hat) / (1 - e_hat))
    return float(mu1 - mu0)


def main():
    n = 2000
    n_seeds = 200
    clip = 10.0

    print(f"ATE score-aligned estimation (N={n}, {n_seeds} seeds, clip={clip})")
    print(f"Sensitivity: 1/pi^2 (treated), 1/(1-pi)^2 (control)")
    print()

    biases_unwt = []
    biases_w2 = []

    for seed in range(n_seeds):
        X, T, Y, ate_true = make_ate_dgp(n, seed)

        ate_unwt = estimate_ate(X, T, Y, power=0, seed=seed)
        ate_w2 = estimate_ate(X, T, Y, power=2, clip=clip, seed=seed)

        biases_unwt.append(ate_unwt - ate_true)
        biases_w2.append(ate_w2 - ate_true)

    biases_unwt = np.array(biases_unwt)
    biases_w2 = np.array(biases_w2)

    rmse_unwt = np.sqrt(np.mean(biases_unwt**2))
    rmse_w2 = np.sqrt(np.mean(biases_w2**2))
    imp = (rmse_unwt - rmse_w2) / rmse_unwt * 100
    _, p = stats.ttest_rel(biases_unwt**2, biases_w2**2)

    print(f"  Standard ATE:            RMSE = {rmse_unwt:.4f}")
    print(
        f"  Score-aligned (1/pi^2):  RMSE = {rmse_w2:.4f}  "
        f"({imp:+.1f}%, p={p:.4f})"
    )


if __name__ == "__main__":
    main()
