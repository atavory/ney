#!/usr/bin/env python3
"""
Demo: score-aligned tuning on the Kang-Schafer (2007) DGP.

Compares standard DR-AIPW with score-aligned DR (1/pi^2 weighted
outcome fitting) and stabilized DR (1/pi weighted outcome fitting).
Runs 200 seeds, prints paired RMSE comparison.

This reproduces the AIPW instance from Table 1 of the paper:
  sensitivity factor a(X)^2 = 1/pi(X)^2
  aligned stage: fitting loss
  clip c = 10

Requirements: numpy, scipy, scikit-learn
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import expit

from algs.dr_aipw import DRAIPW
from algs.weighted_dr import WeightedDR


def kang_schafer(N: int, seed: int, sel_coeff: float = 1.0):
    """Kang-Schafer (2007) DGP with nonlinear covariate transforms."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, 4))
    Y = (
        210
        + 27.4 * Z[:, 0]
        + 13.7 * Z[:, 1]
        + 13.7 * Z[:, 2]
        + 13.7 * Z[:, 3]
        + rng.standard_normal(N)
    )
    e0 = expit(
        sel_coeff
        * (-Z[:, 0] + 0.5 * Z[:, 1] - 0.25 * Z[:, 2] - 0.1 * Z[:, 3])
    )
    R = rng.random(N) < e0
    X = np.column_stack(
        [
            np.exp(Z[:, 0] / 2),
            Z[:, 1] / (1 + np.exp(Z[:, 0])) + 10,
            (Z[:, 0] * Z[:, 2] / 25 + 0.6) ** 3,
            (Z[:, 1] + Z[:, 3] + 20) ** 2,
        ]
    )
    return X, Y, R


def main():
    N = 2000
    TRUE_MEAN = 210.0
    N_SEEDS = 200
    CLIP = 10.0

    print(f"Kang-Schafer DGP: N={N}, {N_SEEDS} seeds, clip={CLIP}")
    print(f"True population mean: {TRUE_MEAN}")
    print()

    biases_std = []
    biases_w1 = []
    biases_w2 = []

    for seed in range(N_SEEDS):
        X, Y, R = kang_schafer(N, seed)
        X_s, Y_s = X[R], Y[R]
        if len(Y_s) < 30:
            continue

        # Standard DR-AIPW (unweighted)
        std = DRAIPW(pop_X=X)
        std.fit(X_s, Y_s, seed=seed)
        biases_std.append(std.predict_population_mean() - TRUE_MEAN)

        # Stabilized DR: power=1, weight = 1/pi
        w1 = WeightedDR(pop_X=X, clip=CLIP, power=1, stabilize=False)
        w1.fit(X_s, Y_s, seed=seed)
        biases_w1.append(w1.predict_population_mean() - TRUE_MEAN)

        # Score-aligned DR: power=2, weight = 1/pi^2
        w2 = WeightedDR(pop_X=X, clip=CLIP, power=2, stabilize=False)
        w2.fit(X_s, Y_s, seed=seed)
        biases_w2.append(w2.predict_population_mean() - TRUE_MEAN)

    biases_std = np.array(biases_std)
    biases_w1 = np.array(biases_w1)
    biases_w2 = np.array(biases_w2)

    rmse_std = np.sqrt(np.mean(biases_std**2))
    rmse_w1 = np.sqrt(np.mean(biases_w1**2))
    rmse_w2 = np.sqrt(np.mean(biases_w2**2))
    improvement_w1 = (rmse_std - rmse_w1) / rmse_std * 100
    improvement_w2 = (rmse_std - rmse_w2) / rmse_std * 100

    _, p_w1 = stats.ttest_rel(biases_std**2, biases_w1**2)
    _, p_w2 = stats.ttest_rel(biases_std**2, biases_w2**2)

    print(f"Standard DR (unwt):    RMSE = {rmse_std:.4f}")
    print(
        f"Stabilized (1/pi):     RMSE = {rmse_w1:.4f}  "
        f"({improvement_w1:+.1f}%, p={p_w1:.4f})"
    )
    print(
        f"Score-aligned (1/pi²): RMSE = {rmse_w2:.4f}  "
        f"({improvement_w2:+.1f}%, p={p_w2:.4f})"
    )


if __name__ == "__main__":
    main()
