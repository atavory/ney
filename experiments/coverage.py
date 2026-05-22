#!/usr/bin/env python3
"""
Coverage simulations for score-aligned CV.
Tests whether AIPW confidence intervals achieve nominal coverage
under both standard and score-aligned tuning.

1000 simulations for tight coverage estimates (SE ~ 0.7%).
Output: results/coverage_v1.csv
"""

import numpy as np
import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold

os.environ.setdefault("OMP_NUM_THREADS", "1")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_FILE = os.path.join(RESULTS_DIR, "coverage_v1.csv")
FIELDS = ["N", "sel", "learner", "weighting", "seed",
          "est", "se", "ci_lo", "ci_hi", "covers", "bias"]

N_SEEDS = 1000
MU_TRUE = 210.0


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
    return X, Y, R, e


def fit_dr_with_ci(X, Y, R, learner, weighting, seed):
    n_pop = len(Y)
    resp_idx = np.where(R == 1)[0]
    n_resp = len(resp_idx)
    X_resp = X[resp_idx]
    Y_resp = Y[resp_idx]

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    e_hat = np.zeros(n_pop)
    for train_idx, val_idx in kf.split(X):
        resp_train = np.intersect1d(train_idx, resp_idx)
        nonresp_train = np.setdiff1d(train_idx, resp_idx)
        R_train = np.concatenate([np.ones(len(resp_train)), np.zeros(len(nonresp_train))])
        X_train = np.vstack([X[resp_train], X[nonresp_train]])
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=100, random_state=42)
        clf.fit(X_train, R_train)
        e_hat[val_idx] = np.clip(clf.predict_proba(X[val_idx])[:, 1], 0.025, 0.975)

    m_hat = np.zeros(n_pop)
    for train_idx, val_idx in kf.split(X_resp):
        X_tr, Y_tr = X_resp[train_idx], Y_resp[train_idx]
        e_tr = e_hat[resp_idx[train_idx]]

        if weighting == "unwt":
            w = None
        elif weighting == "w2":
            w = np.minimum(1.0 / e_tr ** 2, 50)
        elif weighting == "stab2":
            sw = e_tr / e_tr.mean()
            w = np.minimum(sw / e_tr ** 2, 50)
        else:
            w = None

        if learner == "linear":
            reg = LinearRegression()
            reg.fit(X_tr, Y_tr, sample_weight=w)
        elif learner == "hgb":
            reg = HistGradientBoostingRegressor(max_depth=4, max_iter=100, random_state=seed)
            reg.fit(X_tr, Y_tr, sample_weight=w)
        else:
            raise ValueError(learner)

        global_val_idx = resp_idx[val_idx]
        m_hat[global_val_idx] = reg.predict(X_resp[val_idx])

    # For non-respondents, predict with full model
    nonresp_mask = R == 0
    if learner == "linear":
        reg_full = LinearRegression()
        reg_full.fit(X_resp, Y_resp)
    else:
        reg_full = HistGradientBoostingRegressor(max_depth=4, max_iter=100, random_state=seed)
        reg_full.fit(X_resp, Y_resp)
    m_hat[nonresp_mask] = reg_full.predict(X[nonresp_mask])

    # AIPW estimate
    or_term = m_hat.mean()
    corr_term = np.mean(R * (Y - m_hat) / e_hat)
    est = or_term + corr_term

    # Influence function for SE
    phi = m_hat - est + R * (Y - m_hat) / e_hat
    se = np.std(phi) / np.sqrt(n_pop)

    ci_lo = est - 1.96 * se
    ci_hi = est + 1.96 * se
    covers = int(ci_lo <= MU_TRUE <= ci_hi)

    return est, se, ci_lo, ci_hi, covers


def run_all():
    grid = [
        (1000, 0.5), (1000, 1.0),
        (2000, 0.5), (2000, 1.0), (2000, 1.5),
        (5000, 0.5), (5000, 1.0), (5000, 1.5),
    ]
    learners = ["linear", "hgb"]
    weightings = ["unwt", "w2", "stab2"]

    total = len(grid) * len(learners) * len(weightings) * N_SEEDS
    done = 0
    t0 = time.time()

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for N, sel in grid:
            for seed in range(N_SEEDS):
                X, Y, R, e_true = make_ks(N, sel, seed)
                if R.sum() < 20:
                    continue

                for learner in learners:
                    for wt in weightings:
                        try:
                            est, se, ci_lo, ci_hi, covers = fit_dr_with_ci(
                                X, Y, R, learner, wt, seed)
                        except Exception:
                            continue

                        writer.writerow({
                            "N": N, "sel": sel, "learner": learner,
                            "weighting": wt, "seed": seed,
                            "est": est, "se": se,
                            "ci_lo": ci_lo, "ci_hi": ci_hi,
                            "covers": covers,
                            "bias": est - MU_TRUE,
                        })
                        done += 1

                if seed % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"[{done}/{total}] N={N} sel={sel} seed={seed} | "
                          f"{rate:.1f}/s ETA {eta/60:.0f}m")
                    f.flush()

    print(f"\nDone. Wrote {OUT_FILE}")


if __name__ == "__main__":
    run_all()
