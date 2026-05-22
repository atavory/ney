#!/usr/bin/env python3
"""
Demo: Score-aligned IV estimation with optimal instrument alignment.

The IV sensitivity factor is (Table 1):
  a(X)^2 = gamma(X)^2 * Var(Z|X)

where gamma(X) = E[D|X,Z=1] - E[D|X,Z=0] is the compliance function.

The aligned stages are:
  1. Outcome-fit weighting by gamma(X)^2  (fitting loss alignment)
  2. Optimal instrument h*(X,Z) = gamma(X) * (Z - E[Z|X])
     instead of h(X,Z) = Z - E[Z|X]

The estimator is:
  theta = sum h(X_i, Z_i) * Y_tilde_i / sum h(X_i, Z_i) * D_tilde_i

where Y_tilde and D_tilde are cross-fitted residuals.

Runs a synthetic IV DGP with heterogeneous compliance.

Requirements: numpy, scipy, scikit-learn
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


def make_iv_data(n: int, seed: int):
    """IV DGP with heterogeneous compliance gamma(X).

    Z is Bernoulli(0.5) instrument, D = r(X) + gamma(X)*Z + noise.
    """
    rng = np.random.default_rng(seed)
    z_lat = rng.standard_normal((n, 4))

    # Observed covariates (KS-style transforms)
    X = np.column_stack(
        [
            np.exp(z_lat[:, 0] / 2),
            z_lat[:, 1] / (1 + np.exp(z_lat[:, 0])) + 10,
            (z_lat[:, 0] * z_lat[:, 2] / 25 + 0.6) ** 3,
            (z_lat[:, 1] + z_lat[:, 3] + 20) ** 2,
        ]
    )

    # Heterogeneous compliance
    gamma_x = 0.3 + 0.7 * expit(2.0 * z_lat[:, 0])

    # Structural equations
    g0 = (
        27.4 * z_lat[:, 0]
        + 13.7 * z_lat[:, 1]
        + 13.7 * z_lat[:, 2]
        + 13.7 * z_lat[:, 3]
    )
    r0 = 0.5 * z_lat[:, 0] - 0.3 * z_lat[:, 1]
    Z_inst = rng.binomial(1, 0.5, size=n).astype(float)
    D = r0 + gamma_x * Z_inst + 0.3 * rng.standard_normal(n)
    Y = 2.0 * D + g0 + 0.5 * rng.standard_normal(n)

    theta_true = 2.0
    return X, Y, D, Z_inst, gamma_x, theta_true


def estimate_iv(
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    gamma_x: np.ndarray,
    weight_outcome: bool = False,
    use_optimal_instrument: bool = False,
    seed: int = 0,
) -> float:
    """IV estimator with optional score-aligned fitting and instrument.

    Parameters
    ----------
    X : covariates
    Y : outcome
    D : endogenous treatment
    Z : binary instrument
    gamma_x : compliance function (oracle, for demo purposes)
    weight_outcome : weight outcome fit by gamma(X)^2
    use_optimal_instrument : use h*(X,Z) = gamma(X)*(Z-0.5) instead of Z-0.5
    seed : random seed for cross-fitting
    """
    n = len(Y)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    y_resid = np.zeros(n)
    d_resid = np.zeros(n)
    sensitivity = gamma_x**2

    for tr, te in kf.split(X):
        # Outcome nuisance l(X)
        y_model = RidgeCV(alphas=(0.1, 1.0, 10.0))
        y_w = sensitivity[tr] if weight_outcome else None
        y_model.fit(X[tr], Y[tr], sample_weight=y_w)
        y_resid[te] = Y[te] - y_model.predict(X[te])

        # Treatment nuisance r(X)
        d_model = RidgeCV(alphas=(0.1, 1.0, 10.0))
        d_model.fit(X[tr], D[tr])
        d_resid[te] = D[te] - d_model.predict(X[te])

    # Instrument
    z_resid = Z - 0.5
    if use_optimal_instrument:
        h = gamma_x * z_resid
    else:
        h = z_resid

    numerator = np.sum(h * y_resid)
    denominator = np.sum(h * d_resid)
    return float(numerator / denominator)


def main():
    n = 2000
    n_seeds = 200

    print(f"IV score-aligned estimation (N={n}, {n_seeds} seeds)")
    print(f"Sensitivity: gamma(X)^2 * Var(Z|X)")
    print(f"Aligned stages: outcome-fit weighting + optimal instrument")
    print()

    biases = {
        "unwt": [],
        "y_fit_only": [],
        "opt_inst_only": [],
        "y_fit_opt_inst": [],
    }

    for seed in range(n_seeds):
        X, Y, D, Z, gamma_x, theta_true = make_iv_data(n, seed)

        # Standard IV
        t0 = estimate_iv(X, Y, D, Z, gamma_x, False, False, seed)
        biases["unwt"].append(t0 - theta_true)

        # Outcome-fit weighting only
        t1 = estimate_iv(X, Y, D, Z, gamma_x, True, False, seed)
        biases["y_fit_only"].append(t1 - theta_true)

        # Optimal instrument only
        t2 = estimate_iv(X, Y, D, Z, gamma_x, False, True, seed)
        biases["opt_inst_only"].append(t2 - theta_true)

        # Both
        t3 = estimate_iv(X, Y, D, Z, gamma_x, True, True, seed)
        biases["y_fit_opt_inst"].append(t3 - theta_true)

    rmse_unwt = np.sqrt(np.mean(np.array(biases["unwt"]) ** 2))

    labels = {
        "unwt": "Standard IV",
        "y_fit_only": "Outcome-fit aligned",
        "opt_inst_only": "Optimal instrument",
        "y_fit_opt_inst": "Both aligned",
    }

    for key, label in labels.items():
        b = np.array(biases[key])
        rmse = np.sqrt(np.mean(b**2))
        if key == "unwt":
            print(f"  {label:26s} RMSE = {rmse:.4f}")
        else:
            imp = (rmse_unwt - rmse) / rmse_unwt * 100
            _, p = stats.ttest_rel(
                np.array(biases["unwt"]) ** 2, b**2
            )
            print(
                f"  {label:26s} RMSE = {rmse:.4f}  "
                f"({imp:+.1f}%, p={p:.4f})"
            )


if __name__ == "__main__":
    main()
