#!/usr/bin/env python3
"""
Demo: score-aligned tuning on the Kang-Schafer (2007) DGP.

Compares standard DR-AIPW with correction-aware DR (1/e^2 weighted outcome training).
Runs 100 seeds, prints paired RMSE comparison.

Requirements: numpy, scipy, scikit-learn
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import expit

from algs.dr_aipw import DRAIPW
from algs.weighted_dr import WeightedDR


def kang_schafer(N: int, seed: int, sel_coeff: float = 1.0):
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
        sel_coeff * (-Z[:, 0] + 0.5 * Z[:, 1] - 0.25 * Z[:, 2] - 0.1 * Z[:, 3])
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
    N_SEEDS = 100

    print(f"Kang-Schafer DGP: N={N}, {N_SEEDS} seeds")
    print(f"True population mean: {TRUE_MEAN}")
    print()

    biases_std = []
    biases_w2 = []

    for seed in range(N_SEEDS):
        X, Y, R = kang_schafer(N, seed)
        X_s, Y_s = X[R], Y[R]
        if len(Y_s) < 30:
            continue

        std = DRAIPW(pop_X=X)
        std.fit(X_s, Y_s)
        biases_std.append(std.predict_population_mean() - TRUE_MEAN)

        w2 = WeightedDR(pop_X=X, clip=20, stabilize=False)
        w2.fit(X_s, Y_s)
        biases_w2.append(w2.predict_population_mean() - TRUE_MEAN)

    biases_std = np.array(biases_std)
    biases_w2 = np.array(biases_w2)

    rmse_std = np.sqrt(np.mean(biases_std**2))
    rmse_w2 = np.sqrt(np.mean(biases_w2**2))
    improvement = (rmse_std - rmse_w2) / rmse_std * 100

    _, p = stats.ttest_rel(biases_std**2, biases_w2**2)

    print(f"Standard DR:        RMSE = {rmse_std:.4f}")
    print(f"Score-aligned DR:   RMSE = {rmse_w2:.4f}")
    print(f"Improvement:        {improvement:+.1f}%")
    print(f"Paired t-test:      p = {p:.6f}")


if __name__ == "__main__":
    main()
