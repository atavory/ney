#!/usr/bin/env python3
"""
Demo: Score-aligned estimation for the Partially Linear Model (Robinson 1988).

The sensitivity factor is a(X)^2 = Var(D|X), the same as in the paper's
PLM instance (Table 1). The aligned intervention is the theta aggregation
step, NOT the fitting loss.

Standard Robinson:
    theta = sum D_tilde_i * Y_tilde_i / sum D_tilde_i^2

Score-aligned Robinson:
    theta_SA = sum Var(D|X_i) * D_tilde_i * Y_tilde_i
             / sum Var(D|X_i) * D_tilde_i^2

where D_tilde = D - E[D|X] and Y_tilde = Y - E[Y|X] are cross-fitted
residuals.

Fitting-loss reweighting HURTS PLM (paper reports -1% to -136%) because
the nuisance error enters through the Robinson ratio, not the pointwise
prediction. The correct aligned stage is the aggregation step.

Runs a heteroskedastic PLM with KS-style nonlinear covariate transforms.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


def make_plm_data(
    n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Heteroskedastic PLM DGP.

    Returns (X, Y, D, Var(D|X)) where Var(D|X) varies with X.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    x1 = np.exp(z[:, 0] / 2)
    x2 = (z[:, 1] + 1.5) ** 2
    x = np.column_stack([x1, x2])

    # Heteroskedastic treatment noise
    sigma = 0.3 + 2.0 / (1.0 + np.exp(2.0 * z[:, 0]))
    d = z[:, 0] + sigma * rng.standard_normal(n)
    g0 = 5 * z[:, 0] + 3 * z[:, 1]
    y = 2.0 * d + g0 + rng.standard_normal(n) * 0.5

    var_d = sigma**2
    return x, y, d, var_d


def estimate_theta(
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    theta_weights: np.ndarray | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> float:
    """Cross-fitted Robinson estimator with optional theta aggregation weights.

    Parameters
    ----------
    x : covariates
    y : outcome
    d : treatment
    theta_weights : per-observation weights for the theta ratio.
        None = standard Robinson. Var(D|X) = score-aligned.
    n_folds : number of cross-fitting folds
    seed : random seed for splits and nuisance model
    """
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    y_resid = np.zeros(len(x))
    d_resid = np.zeros(len(x))

    for tr, te in kf.split(x):
        reg_y = RidgeCV(alphas=(0.1, 1.0, 10.0))
        reg_d = RidgeCV(alphas=(0.1, 1.0, 10.0))
        reg_y.fit(x[tr], y[tr])
        reg_d.fit(x[tr], d[tr])
        y_resid[te] = y[te] - reg_y.predict(x[te])
        d_resid[te] = d[te] - reg_d.predict(x[te])

    if theta_weights is not None:
        return float(
            np.sum(theta_weights * d_resid * y_resid)
            / np.sum(theta_weights * d_resid**2)
        )
    return float(np.sum(d_resid * y_resid) / np.sum(d_resid**2))


def main():
    n = 2000
    n_seeds = 200
    theta_true = 2.0

    biases_unwt = []
    biases_aligned = []

    for seed in range(n_seeds):
        x, y, d, var_d = make_plm_data(n, seed)

        theta_unwt = estimate_theta(x, y, d, theta_weights=None, seed=seed)
        theta_aligned = estimate_theta(x, y, d, theta_weights=var_d, seed=seed)

        biases_unwt.append(theta_unwt - theta_true)
        biases_aligned.append(theta_aligned - theta_true)

    rmse_unwt = np.sqrt(np.mean(np.array(biases_unwt) ** 2))
    rmse_aligned = np.sqrt(np.mean(np.array(biases_aligned) ** 2))
    imp = (rmse_unwt - rmse_aligned) / rmse_unwt * 100

    print(f"PLM score-aligned estimation (N={n}, {n_seeds} seeds)")
    print(f"  Sensitivity factor: a(X)^2 = Var(D|X)")
    print(f"  Aligned stage:      theta aggregation (NOT fitting loss)")
    print()
    print(f"  Standard Robinson:         RMSE = {rmse_unwt:.4f}")
    print(f"  Aligned theta aggregation: RMSE = {rmse_aligned:.4f}")
    print(f"  Improvement:               {imp:+.1f}%")


if __name__ == "__main__":
    main()
