#!/usr/bin/env python3
"""
Demo: sieve phase transition on Kang-Schafer (2007).

Reproduces the three-regime phase transition: sweeps polynomial basis
dimension K and shows underfitting, selective overload, and weighted
instability regimes.

Uses score-aligned weights (1/pi^2, clipped at c=10) for the outcome
fitting loss on sieve (polynomial) basis regression.

Requirements: numpy, scipy, scikit-learn
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def make_kang_schafer(n: int, sel: float, seed: int):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 4))
    X = np.column_stack(
        [
            np.exp(Z[:, 0] / 2),
            Z[:, 1] / (1 + np.exp(Z[:, 0])) + 10,
            (Z[:, 0] * Z[:, 2] / 25 + 0.6) ** 3,
            (Z[:, 1] + Z[:, 3] + 20) ** 2,
        ]
    )
    Y = (
        210
        + 27.4 * Z[:, 0]
        + 13.7 * (Z[:, 1] + Z[:, 2] + Z[:, 3])
        + rng.standard_normal(n)
    )
    logit = sel * (
        -Z[:, 0] + 0.5 * Z[:, 1] - 0.25 * Z[:, 2] - 0.1 * Z[:, 3]
    )
    pi = 1 / (1 + np.exp(-logit))
    R = rng.binomial(1, pi)
    return X, Y, R, pi


def run_one(X, Y, R, degree, weighting, seed):
    mu_true = 210.0
    clip = 10.0
    resp = np.where(R == 1)[0]
    if len(resp) < 30:
        return float("nan")

    X_r, Y_r = X[resp], Y[resp]

    # Sieve features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    scaler = StandardScaler()
    X_poly = scaler.fit_transform(X_poly)
    X_poly_r = X_poly[resp]

    # Propensity (cross-fitted on full data)
    nonresp = np.where(R == 0)[0]
    X_prop = np.vstack([X_r, X[nonresp]])
    R_prop = np.concatenate([np.ones(len(resp)), np.zeros(len(nonresp))])
    clf = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, random_state=seed
    )
    clf.fit(X_prop, R_prop)
    pi_hat_r = np.clip(clf.predict_proba(X_r)[:, 1], 0.025, 0.975)
    pi_hat_all = np.clip(clf.predict_proba(X)[:, 1], 0.025, 0.975)

    # Cross-fitted outcome with sieve basis
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    m_hat_r = np.zeros(len(resp))
    for tr, va in kf.split(X_poly_r):
        if weighting == "unwt":
            w = None
        elif weighting == "w2":
            w = np.minimum(1.0 / pi_hat_r[tr] ** 2, clip)
            w = w / w.mean()
        else:
            w = None
        reg = RidgeCV(alphas=(0.1, 1.0, 10.0))
        reg.fit(X_poly_r[tr], Y_r[tr], sample_weight=w)
        m_hat_r[va] = reg.predict(X_poly_r[va])

    # Full model for outcome regression term
    reg_full = RidgeCV(alphas=(0.1, 1.0, 10.0))
    if weighting == "w2":
        w_full = np.minimum(1.0 / pi_hat_r**2, clip)
        w_full = w_full / w_full.mean()
        reg_full.fit(X_poly_r, Y_r, sample_weight=w_full)
    else:
        reg_full.fit(X_poly_r, Y_r)
    m_all = reg_full.predict(X_poly)

    # DR estimate
    or_part = m_all.mean()
    m_corr = np.zeros(len(Y))
    m_corr[resp] = m_hat_r
    m_corr[R == 0] = m_all[R == 0]
    corr_part = np.mean(R * (Y - m_corr) / pi_hat_all)
    return or_part + corr_part - mu_true


def main():
    N = 2000
    sel = 1.0
    n_seeds = 200
    degrees = [1, 2, 3, 4, 5]

    print("Sieve Phase Transition on Kang-Schafer")
    print(f"N={N}, sel={sel}, {n_seeds} seeds, clip=10")
    print()
    print(
        f"{'Degree':>6s} {'K':>5s} {'RMSE_std':>10s} "
        f"{'RMSE_SA':>10s} {'Improv%':>8s}"
    )
    print("-" * 45)

    for deg in degrees:
        K = math.comb(deg + 4, 4)
        biases_std, biases_sa = [], []
        for s in range(n_seeds):
            X, Y, R, pi = make_kang_schafer(N, sel, s)
            b_std = run_one(X, Y, R, deg, "unwt", s)
            b_sa = run_one(X, Y, R, deg, "w2", s)
            if not (math.isnan(b_std) or math.isnan(b_sa)):
                biases_std.append(b_std)
                biases_sa.append(b_sa)

        rmse_std = math.sqrt(
            sum(b**2 for b in biases_std) / len(biases_std)
        )
        rmse_sa = math.sqrt(sum(b**2 for b in biases_sa) / len(biases_sa))
        improv = (rmse_std - rmse_sa) / rmse_std * 100
        print(
            f"{deg:>6d} {K:>5d} {rmse_std:>10.4f} "
            f"{rmse_sa:>10.4f} {improv:>7.1f}%"
        )

    print()
    print(
        "Expected: gain near 0 at degree 1, positive at degree 2-3, "
        "collapse at degree 4-5"
    )


if __name__ == "__main__":
    main()
