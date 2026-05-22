#!/usr/bin/env python3
from __future__ import annotations

"""
Joint sample-size and capacity sweep for AIPW survey means on the KS DGP.

This runner tests the simple rebuttal to "just use more capacity":
for any fixed dataset, reducing sample size shifts the best-performing model
toward lower complexity, and score alignment can only help inside the
finite-sample capacity window that the data supports.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from common import (
    make_observed_covariates,
    make_poly_features,
    normalize_weights,
    parse_csv_ints,
    parse_csv_floats,
    write_rows,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")

MU_TRUE = 210.0
RIDGE_ALPHAS = np.logspace(-3, 6, 19)
RESULT_FIELDS = [
    "N",
    "degree",
    "seed",
    "variant",
    "estimate",
    "true_value",
    "bias",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=str, default="500,1000,2000,5000")
    parser.add_argument("--degrees", type=str, default="1,2,3,4,5")
    parser.add_argument("--sels", type=str, default="1.0")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=str,
        default="results/nk_sweep_v1.csv",
    )
    return parser.parse_args()


def make_ks_data(
    n: int,
    selection_strength: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    x = make_observed_covariates(z)
    y = 210.0 + 27.4 * z[:, 0] + 13.7 * (z[:, 1] + z[:, 2] + z[:, 3]) + rng.standard_normal(n)
    raw = selection_strength * (
        -z[:, 0] + 0.5 * z[:, 1] - 0.25 * z[:, 2] - 0.1 * z[:, 3]
    )
    pi = 1.0 / (1.0 + np.exp(-raw))
    r = rng.binomial(1, pi).astype(int)
    return x, y, r


def fit_propensity(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    model = HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=200,
        min_samples_leaf=10,
        random_state=0,
    )
    model.fit(x, r)
    return np.clip(model.predict_proba(x)[:, 1], 0.025, 0.975)


def crossfit_outcome(
    x_poly: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    pred = np.zeros(len(y))
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in splitter.split(x_poly):
        model = RidgeCV(alphas=RIDGE_ALPHAS)
        fold_weight = None if weights is None else weights[tr]
        model.fit(x_poly[tr], y[tr], sample_weight=fold_weight)
        pred[te] = model.predict(x_poly[te])
    return pred


def run_task(args: tuple[int, float, int, int]) -> list[dict[str, object]]:
    n, selection_strength, degree, seed = args
    x, y, r = make_ks_data(n=n, selection_strength=selection_strength, seed=seed)
    resp = r == 1
    if int(resp.sum()) < 30:
        return []

    pi_hat = fit_propensity(x, r)
    x_poly = make_poly_features(x, degree)

    x_resp = x_poly[resp]
    y_resp = y[resp]
    pi_resp = pi_hat[resp]

    unwt_fit = None
    stab_fit = normalize_weights(
        1.0 / np.maximum(pi_resp, 1e-6),
        clip=25.0,
    )
    aligned_fit = normalize_weights(1.0 / np.maximum(pi_resp, 1e-6) ** 2, clip=25.0)

    rows: list[dict[str, object]] = []
    for variant, weights in (
        ("unwt", unwt_fit),
        ("stab", stab_fit),
        ("w2", aligned_fit),
    ):
        m_resp_cf = crossfit_outcome(x_resp, y_resp, weights, seed + 17)
        full_model = RidgeCV(alphas=RIDGE_ALPHAS)
        full_model.fit(x_resp, y_resp, sample_weight=weights)
        m_all = full_model.predict(x_poly)

        m_blend = m_all.copy()
        m_blend[resp] = m_resp_cf
        estimate = float(np.mean(m_all) + np.mean(r * (y - m_blend) / pi_hat))
        bias = estimate - MU_TRUE
        rows.append(
            {
                "N": n,
                "degree": degree,
                "seed": seed,
                "variant": variant,
                "estimate": estimate,
                "true_value": MU_TRUE,
                "bias": bias,
            }
        )
    return rows


def build_tasks(
    ns: tuple[int, ...],
    sels: tuple[float, ...],
    degrees: tuple[int, ...],
    seeds: int,
) -> list[tuple[int, float, int, int]]:
    tasks: list[tuple[int, float, int, int]] = []
    for n in ns:
        for selection_strength in sels:
            for degree in degrees:
                for seed in range(seeds):
                    tasks.append((n, selection_strength, degree, seed))
    return tasks


def main() -> None:
    args = parse_args()
    ns = parse_csv_ints(args.ns)
    degrees = parse_csv_ints(args.degrees)
    sels = parse_csv_floats(args.sels)
    tasks = build_tasks(ns=ns, sels=sels, degrees=degrees, seeds=args.seeds)

    print(f"Running {len(tasks)} N×K tasks")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, task_rows in enumerate(executor.map(run_task, tasks), start=1):
            rows.extend(task_rows)
            if index % 250 == 0:
                print(f"  completed {index}/{len(tasks)}")

    write_rows(args.output, RESULT_FIELDS, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
