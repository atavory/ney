#!/usr/bin/env python3
"""
Capacity sweep: direct test of the selective-overload prediction.

Fix DGP and sample size, sweep HGB capacity (max_depth, max_iter),
compare standard vs score-aligned CV at each capacity level.

The prediction: improvement is near zero at very low capacity,
positive in an intermediate regime, and near zero again at high capacity.

Output: results/capacity_sweep_v1.csv
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

OUT_FILE = os.path.join(RESULTS_DIR, "capacity_sweep_v1.csv")
FIELDS = ["dgp", "N", "sel", "capacity_type", "capacity_value",
          "weighting", "seed", "est", "bias", "abs_bias"]

N_SEEDS = 200
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
    return X, Y, R


def make_spike(N, sel, spike_k, seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(N, 4)
    logit = sel * (-X[:, 0] + 0.5 * X[:, 1])
    e = 1 / (1 + np.exp(-logit))
    spike_mask = X[:, 0] > 2.0
    Y = X[:, 0] + X[:, 1] + X[:, 2] + rng.randn(N)
    Y[spike_mask] += spike_k
    R = rng.binomial(1, e)
    return X, Y, R


def run_one_seed(args):
    dgp, N, sel, capacity_type, capacity_value, seed = args

    if dgp == "ks":
        X, Y, R = make_ks(N, sel, seed)
        mu_true = MU_TRUE
    elif dgp == "spike":
        X, Y, R = make_spike(N, sel, 20, seed)
        mu_true = None  # will compute from full Y
    else:
        return []

    if mu_true is None:
        mu_true = Y.mean()

    resp_idx = np.where(R == 1)[0]
    if len(resp_idx) < 30:
        return []

    X_resp = X[resp_idx]
    Y_resp = Y[resp_idx]
    nonresp_idx = np.where(R == 0)[0]

    # Fit propensity (fixed capacity — always good)
    X_prop = np.vstack([X_resp, X[nonresp_idx]])
    R_prop = np.concatenate([np.ones(len(resp_idx)), np.zeros(len(nonresp_idx))])
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, random_state=42)
    clf.fit(X_prop, R_prop)
    e_hat_resp = np.clip(clf.predict_proba(X_resp)[:, 1], 0.025, 0.975)
    e_hat_all = np.clip(clf.predict_proba(X)[:, 1], 0.025, 0.975)

    rows = []
    for wt in ["unwt", "w2", "stab2"]:
        # Cross-fitted outcome with variable capacity
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        m_hat_resp = np.zeros(len(resp_idx))

        for train_idx, val_idx in kf.split(X_resp):
            X_tr, Y_tr = X_resp[train_idx], Y_resp[train_idx]
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

            if capacity_type == "max_depth":
                reg = HistGradientBoostingRegressor(
                    max_depth=capacity_value, max_iter=200,
                    min_samples_leaf=5, random_state=seed)
            elif capacity_type == "max_iter":
                reg = HistGradientBoostingRegressor(
                    max_depth=4, max_iter=capacity_value,
                    min_samples_leaf=5, random_state=seed)
            elif capacity_type == "max_leaf_nodes":
                reg = HistGradientBoostingRegressor(
                    max_depth=None, max_leaf_nodes=capacity_value,
                    max_iter=200, min_samples_leaf=5, random_state=seed)
            else:
                raise ValueError(capacity_type)

            reg.fit(X_tr, Y_tr, sample_weight=w)
            m_hat_resp[val_idx] = reg.predict(X_resp[val_idx])

        # Full model for OR term
        if capacity_type == "max_depth":
            reg_full = HistGradientBoostingRegressor(
                max_depth=capacity_value, max_iter=200,
                min_samples_leaf=5, random_state=seed)
        elif capacity_type == "max_iter":
            reg_full = HistGradientBoostingRegressor(
                max_depth=4, max_iter=capacity_value,
                min_samples_leaf=5, random_state=seed)
        elif capacity_type == "max_leaf_nodes":
            reg_full = HistGradientBoostingRegressor(
                max_depth=None, max_leaf_nodes=capacity_value,
                max_iter=200, min_samples_leaf=5, random_state=seed)

        if wt == "unwt":
            reg_full.fit(X_resp, Y_resp)
        elif wt == "w2":
            w_full = np.minimum(1.0 / e_hat_resp ** 2, 50)
            reg_full.fit(X_resp, Y_resp, sample_weight=w_full)
        elif wt == "stab2":
            sw_full = e_hat_resp / e_hat_resp.mean()
            w_full = np.minimum(sw_full / e_hat_resp ** 2, 50)
            reg_full.fit(X_resp, Y_resp, sample_weight=w_full)

        m_hat_all = reg_full.predict(X)

        # DR estimate
        or_part = m_hat_all.mean()
        m_corr = np.zeros(len(Y))
        m_corr[resp_idx] = m_hat_resp
        m_corr[R == 0] = m_hat_all[R == 0]
        corr_part = np.mean(R * (Y - m_corr) / e_hat_all)
        est = or_part + corr_part
        bias = est - mu_true

        rows.append({
            "dgp": dgp, "N": N, "sel": sel,
            "capacity_type": capacity_type,
            "capacity_value": capacity_value,
            "weighting": wt, "seed": seed,
            "est": est, "bias": bias, "abs_bias": abs(bias),
        })

    return rows


def run_all():
    import sys
    part = sys.argv[1] if len(sys.argv) > 1 else "all"

    jobs = []

    if part in ("all", "t1"):
        # t1: Depth sweep on KS + extra N values
        for N in [1000, 2000, 5000]:
            for depth in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
                for seed in range(N_SEEDS):
                    jobs.append(("ks", N, 1.0, "max_depth", depth, seed))

    if part in ("all", "t2"):
        # t2: Depth sweep on spike + extra sel values
        for sel in [0.5, 1.0, 1.5]:
            for depth in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
                for seed in range(N_SEEDS):
                    jobs.append(("spike", 2000, sel, "max_depth", depth, seed))

    if part in ("all", "t3"):
        # t3: Iteration sweep on KS at multiple N
        for N in [1000, 2000, 5000]:
            for n_iter in [5, 10, 20, 50, 100, 200, 500]:
                for seed in range(N_SEEDS):
                    jobs.append(("ks", N, 1.0, "max_iter", n_iter, seed))

    if part in ("all", "t4"):
        # t4: Leaf node sweep on KS + depth sweep on KS with sel variation
        for n_leaves in [2, 4, 8, 16, 31, 63, 127, 255]:
            for seed in range(N_SEEDS):
                jobs.append(("ks", 2000, 1.0, "max_leaf_nodes", n_leaves, seed))
        for sel in [0.5, 1.5]:
            for depth in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
                for seed in range(N_SEEDS):
                    jobs.append(("ks", 2000, sel, "max_depth", depth, seed))

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
