#!/usr/bin/env python3
"""
Demo: Score-aligned estimation for the Partially Linear Model (Robinson 1988).

The sensitivity factor is Var(D|X), the same as in the paper's PLM instance.
Unlike AIPW, the aligned intervention is theta aggregation weighting,
not fitting-loss reweighting. The outcome fitting loss is unchanged.

Runs a heteroskedastic PLM with KS-style nonlinear covariate transforms.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def make_plm_data(n, seed):
    rng = np.random.RandomState(seed)
    z = rng.randn(n, 2)
    x1 = np.exp(z[:, 0] / 2)
    x2 = (z[:, 1] + 1.5) ** 2
    x = np.column_stack([x1, x2])

    sigma = 0.3 + 2.0 / (1.0 + np.exp(2.0 * z[:, 0]))
    d = z[:, 0] + sigma * rng.randn(n)
    g0 = 5 * z[:, 0] + 3 * z[:, 1]
    y = 2.0 * d + g0 + rng.randn(n) * 0.5

    var_d = sigma ** 2
    return x, y, d, var_d


def estimate_theta(x, y, d, theta_weights=None, n_folds=5, seed=0):
    """Cross-fitted Robinson estimator with optional theta aggregation weights."""
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    y_resid = np.zeros(len(x))
    d_resid = np.zeros(len(x))

    for tr, te in kf.split(x):
        reg_y = LinearRegression()
        reg_d = LinearRegression()
        reg_y.fit(x[tr], y[tr])
        reg_d.fit(x[tr], d[tr])
        y_resid[te] = y[te] - reg_y.predict(x[te])
        d_resid[te] = d[te] - reg_d.predict(x[te])

    if theta_weights is not None:
        return np.sum(theta_weights * d_resid * y_resid) / np.sum(
            theta_weights * d_resid**2
        )
    return np.sum(d_resid * y_resid) / np.sum(d_resid**2)


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
    print(f"  Standard Robinson:         RMSE = {rmse_unwt:.4f}")
    print(f"  Aligned theta aggregation: RMSE = {rmse_aligned:.4f}")
    print(f"  Improvement:               {imp:+.1f}%")
    print()
    print("The sensitivity factor is Var(D|X), but the aligned stage is")
    print("the theta aggregation, not the fitting loss.")
    print("Same diagnostic as AIPW; different aligned stage.")


if __name__ == "__main__":
    main()
