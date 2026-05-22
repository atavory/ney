#!/usr/bin/env python3
"""
Sieve capacity sweep: polynomial basis with varying K.

This is the cleanest test of the abstract H_K projection theory.
H_K = span(phi_1, ..., phi_K) where phi are polynomial features.
Capacity = K (number of basis functions). No tree/boosting abstractions.

Sweep K on KS and spike DGPs, compare standard vs score-aligned CV.

Output: results/sieve_sweep_v1.csv
"""

import numpy as np
import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

os.environ.setdefault("OMP_NUM_THREADS", "1")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_FILE = os.path.join(RESULTS_DIR, "sieve_sweep_v1.csv")
FIELDS = ["dgp", "N", "sel", "K", "weighting", "seed", "est", "bias", "abs_bias"]

N_SEEDS = 200
MU_TRUE_KS = 210.0


def make_ks(N, sel, seed):
    rng = np.random.RandomState(seed)
    Z = rng.randn(N, 4)
    X = np.column_stack([
        np.exp(Z[:, 0] / 2),
        Z[:, 1] / (1 + np.exp(Z[:, 0])) + 10,
        (Z[:, 0] * Z[:, 2] / 25 + 0.6) ** 3,
        (Z[:, 1] + Z[:, 3] + 20) ** 2
    ])
    Y = 210 + 27.4 * Z[:, 0] + 13.7 * (Z[:, 1] + Z[:, 2] + Z[:, 3]) + rng.randn(N)
    logit = sel * (-Z[:, 0] + 0.5 * Z[:, 1] - 0.25 * Z[:, 2] - 0.1 * Z[:, 3])
    e = 1 / (1 + np.exp(-logit))
    R = rng.binomial(1, e)
    return X, Y, R, MU_TRUE_KS


def make_spike(N, sel, seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(N, 4)
    logit = sel * (-X[:, 0] + 0.5 * X[:, 1])
    e = 1 / (1 + np.exp(-logit))
    spike_mask = X[:, 0] > 2.0
    Y = X[:, 0] + X[:, 1] + X[:, 2] + rng.randn(N)
    Y[spike_mask] += 20
    R = rng.binomial(1, e)
    mu_true = Y.mean()
    return X, Y, R, mu_true


def make_sieve_features(X, degree):
    """Generate polynomial features up to given degree."""
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X)
    scaler = StandardScaler()
    X_poly = scaler.fit_transform(X_poly)
    return X_poly, poly.n_output_features_


def run_one_seed(args):
    dgp, N, sel, degree, seed = args

    if dgp == "ks":
        X, Y, R, mu_true = make_ks(N, sel, seed)
    elif dgp == "spike":
        X, Y, R, mu_true = make_spike(N, sel, seed)
    else:
        return []

    resp_idx = np.where(R == 1)[0]
    if len(resp_idx) < 30:
        return []

    X_resp = X[resp_idx]
    Y_resp = Y[resp_idx]
    nonresp_idx = np.where(R == 0)[0]

    # Fit propensity (always full-capacity HGB)
    X_prop = np.vstack([X_resp, X[nonresp_idx]])
    R_prop = np.concatenate([np.ones(len(resp_idx)), np.zeros(len(nonresp_idx))])
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    clf.fit(X_prop, R_prop)
    e_hat_resp = np.clip(clf.predict_proba(X_resp)[:, 1], 0.025, 0.975)
    e_hat_all = np.clip(clf.predict_proba(X)[:, 1], 0.025, 0.975)

    # Generate sieve features
    X_poly_all, K = make_sieve_features(X, degree)
    X_poly_resp = X_poly_all[resp_idx]

    rows = []
    for wt in ["unwt", "w2", "stab2"]:
        # Cross-fitted sieve outcome model
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        m_hat_resp = np.zeros(len(resp_idx))

        for train_idx, val_idx in kf.split(X_poly_resp):
            X_tr = X_poly_resp[train_idx]
            Y_tr = Y_resp[train_idx]
            e_tr = e_hat_resp[train_idx]

            if wt == "unwt":
                w = None
            elif wt == "w2":
                w = np.minimum(1.0 / e_tr ** 2, 50)
            elif wt == "stab2":
                sw = e_tr / e_tr.mean()
                w = np.minimum(sw / e_tr ** 2, 50)
            else:
                w = None

            # Ridge regression on polynomial features
            reg = Ridge(alpha=1.0)
            reg.fit(X_tr, Y_tr, sample_weight=w)
            m_hat_resp[val_idx] = reg.predict(X_poly_resp[val_idx])

        # Full model for OR term
        if wt == "unwt":
            reg_full = Ridge(alpha=1.0)
            reg_full.fit(X_poly_resp, Y_resp)
        elif wt == "w2":
            w_full = np.minimum(1.0 / e_hat_resp ** 2, 50)
            reg_full = Ridge(alpha=1.0)
            reg_full.fit(X_poly_resp, Y_resp, sample_weight=w_full)
        elif wt == "stab2":
            sw_full = e_hat_resp / e_hat_resp.mean()
            w_full = np.minimum(sw_full / e_hat_resp ** 2, 50)
            reg_full = Ridge(alpha=1.0)
            reg_full.fit(X_poly_resp, Y_resp, sample_weight=w_full)

        m_hat_all = reg_full.predict(X_poly_all)

        # DR estimate
        or_part = m_hat_all.mean()
        m_corr = np.zeros(len(Y))
        m_corr[resp_idx] = m_hat_resp
        m_corr[R == 0] = m_hat_all[R == 0]
        corr_part = np.mean(R * (Y - m_corr) / e_hat_all)
        est = or_part + corr_part
        bias = est - mu_true

        rows.append({
            "dgp": dgp, "N": N, "sel": sel, "K": K,
            "weighting": wt, "seed": seed,
            "est": est, "bias": bias, "abs_bias": abs(bias),
        })

    return rows


def run_all():
    import sys
    part = sys.argv[1] if len(sys.argv) > 1 else "all"

    jobs = []

    # Polynomial degree 1-5 gives K = 4, 14, 34, 69, 125 features (from 4 input vars)
    degrees = [1, 2, 3, 4, 5]

    if part in ("all", "t1"):
        # KS across N values
        for N in [1000, 2000, 5000]:
            for degree in degrees:
                for seed in range(N_SEEDS):
                    jobs.append(("ks", N, 1.0, degree, seed))

    if part in ("all", "t2"):
        # KS across sel values
        for sel in [0.5, 1.0, 1.5]:
            for degree in degrees:
                for seed in range(N_SEEDS):
                    jobs.append(("ks", 2000, sel, degree, seed))

    if part in ("all", "t3"):
        # Spike across sel values
        for sel in [0.5, 1.0, 1.5]:
            for degree in degrees:
                for seed in range(N_SEEDS):
                    jobs.append(("spike", 2000, sel, degree, seed))

    if part in ("all", "t4"):
        # KS at N=5000 with all sel values (highest quality)
        for sel in [0.5, 1.0, 1.5]:
            for degree in degrees:
                for seed in range(N_SEEDS):
                    jobs.append(("ks", 5000, sel, degree, seed))

    total = len(jobs)
    print(f"Total jobs: {total}")

    n_workers = max(1, os.cpu_count() // 2)
    print(f"Workers: {n_workers}")

    t0 = time.time()
    done = 0

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(run_one_seed, job): job for job in jobs}
            for future in as_completed(futures):
                result_rows = future.result()
                for row in result_rows:
                    writer.writerow(row)
                done += 1
                if done % 500 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (total - done) / rate
                    print(f"[{done}/{total}] {rate:.1f}/s ETA {eta/60:.0f}m")
                    f.flush()

    print(f"\nDone. {OUT_FILE}")


if __name__ == "__main__":
    run_all()
