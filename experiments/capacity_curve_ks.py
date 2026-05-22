#!/usr/bin/env python3
from __future__ import annotations

"""
Practitioner's capacity curve on Kang-Schafer synthetic DGP.

Same idea as the real-data version: at each N, a practitioner cross-validates
to pick the best learner complexity, and we report the alignment gain at
that naturally-selected complexity.

Self-contained (no data files needed). Split this across machines by --ns.
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")

MU_TRUE = 210.0
RIDGE_ALPHAS = np.logspace(-3, 6, 19)

RESULT_FIELDS = [
    "N",
    "sel_strength",
    "response_rate",
    "learner",
    "seed",
    "variant",
    "propensity_mode",
    "estimate",
    "true_value",
    "bias",
    "n_resp",
    "cv_mse_unwt",
    "weight_clip",
]

LEARNER_SPECS: list[tuple[str, str, dict]] = [
    ("ridge1", "ridge", {"degree": 1}),
    ("ridge2", "ridge", {"degree": 2}),
    ("ridge3", "ridge", {"degree": 3}),
    ("ridge4", "ridge", {"degree": 4}),
    ("ridge5", "ridge", {"degree": 5}),
    ("hgb_d1_r10", "hgb", {"depth": 1, "rounds": 10}),
    ("hgb_d1_r50", "hgb", {"depth": 1, "rounds": 50}),
    ("hgb_d2_r20", "hgb", {"depth": 2, "rounds": 20}),
    ("hgb_d2_r50", "hgb", {"depth": 2, "rounds": 50}),
    ("hgb_d2_r100", "hgb", {"depth": 2, "rounds": 100}),
    ("hgb_d3_r50", "hgb", {"depth": 3, "rounds": 50}),
    ("hgb_d3_r200", "hgb", {"depth": 3, "rounds": 200}),
]


def calibrate_intercept(linear: np.ndarray, response_rate: float) -> float:
    lo, hi = -40.0, 40.0
    for _ in range(90):
        mid = (lo + hi) / 2.0
        pi = 1.0 / (1.0 + np.exp(-(mid + linear)))
        if float(np.mean(pi)) < response_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def make_ks_data(
    n: int,
    sel: float,
    response_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    x = np.column_stack((
        np.exp(z[:, 0] / 2.0),
        z[:, 1] / (1.0 + np.exp(z[:, 0])) + 10.0,
        (z[:, 0] * z[:, 2] / 25.0 + 0.6) ** 3,
        (z[:, 1] + z[:, 3] + 20.0) ** 2,
    ))
    y = 210.0 + 27.4 * z[:, 0] + 13.7 * (z[:, 1] + z[:, 2] + z[:, 3]) + rng.standard_normal(n)
    linear = sel * (-z[:, 0] + 0.5 * z[:, 1] - 0.25 * z[:, 2] - 0.1 * z[:, 3])
    intercept = calibrate_intercept(linear, response_rate)
    logit = intercept + linear
    pi = 1.0 / (1.0 + np.exp(-logit))
    r = rng.binomial(1, pi).astype(int)
    return x, y, r, pi


def fit_propensity(x: np.ndarray, r: np.ndarray, seed: int) -> np.ndarray:
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, min_samples_leaf=10, random_state=seed,
    )
    model.fit(x, r)
    return np.clip(model.predict_proba(x)[:, 1], 0.025, 0.975)


def make_design(x: np.ndarray, kind: str, params: dict) -> np.ndarray:
    if kind == "ridge":
        poly = PolynomialFeatures(degree=params["degree"], include_bias=False)
        return StandardScaler().fit_transform(poly.fit_transform(x))
    return x


def crossfit_outcome(
    x_design: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    kind: str,
    params: dict,
    seed: int,
) -> tuple[np.ndarray, float]:
    n = len(y)
    pred = np.zeros(n)
    cv_parts: list[float] = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed + 17)
    for tr, te in splitter.split(x_design):
        sw = None if weights is None else weights[tr]
        if kind == "ridge":
            model = RidgeCV(alphas=RIDGE_ALPHAS)
            model.fit(x_design[tr], y[tr], sample_weight=sw)
        else:
            model = HistGradientBoostingRegressor(
                max_depth=params["depth"],
                max_iter=params["rounds"],
                min_samples_leaf=20,
                random_state=seed,
            )
            model.fit(x_design[tr], y[tr], sample_weight=sw)
        pred[te] = model.predict(x_design[te])
        cv_parts.append(float(np.mean((y[te] - pred[te]) ** 2)))
    return pred, float(np.mean(cv_parts))


def make_weights(pi_resp: np.ndarray, exponent: float, weight_clip: float) -> np.ndarray | None:
    if exponent == 0.0:
        return None
    raw_w = 1.0 / np.maximum(pi_resp, 1e-6) ** exponent
    weights = raw_w / float(np.mean(raw_w))
    weights = np.minimum(weights, weight_clip)
    weights = weights / float(np.mean(weights))
    return weights


def run_task(args: tuple) -> list[dict[str, object]]:
    (
        n,
        sel,
        response_rate,
        learner_name,
        kind,
        params,
        seed,
        weight_clip,
        propensity_mode,
    ) = args

    x, y, r, pi = make_ks_data(n, sel, response_rate, seed)
    resp = r == 1
    n_resp = int(resp.sum())
    if n_resp < 30:
        return []

    if propensity_mode == "oracle":
        pi_hat = np.clip(pi, 0.025, 0.975)
    elif propensity_mode == "estimated":
        pi_hat = fit_propensity(x, r, seed)
    else:
        raise ValueError(f"Unknown propensity mode: {propensity_mode}")
    x_design = make_design(x, kind, params)
    x_resp = x_design[resp]
    y_resp = y[resp]
    pi_resp = pi_hat[resp]

    rows: list[dict[str, object]] = []

    for variant_name, exponent in [("unwt", 0.0), ("stab", 1.0), ("w2", 2.0)]:
        weights = make_weights(pi_resp, exponent, weight_clip)

        pred_cf, cv_mse = crossfit_outcome(x_resp, y_resp, weights, kind, params, seed)

        if kind == "ridge":
            full_model = RidgeCV(alphas=RIDGE_ALPHAS)
            full_model.fit(x_resp, y_resp, sample_weight=weights)
        else:
            full_model = HistGradientBoostingRegressor(
                max_depth=params["depth"],
                max_iter=params["rounds"],
                min_samples_leaf=20,
                random_state=seed,
            )
            full_model.fit(x_resp, y_resp, sample_weight=weights)
        m_all = full_model.predict(x_design)

        m_blend = m_all.copy()
        m_blend[resp] = pred_cf
        estimate = float(np.mean(m_all) + np.mean(r * (y - m_blend) / pi_hat))
        bias = estimate - MU_TRUE

        rows.append({
            "N": n,
            "sel_strength": sel,
            "response_rate": response_rate,
            "learner": learner_name,
            "seed": seed,
            "variant": variant_name,
            "propensity_mode": propensity_mode,
            "estimate": estimate,
            "true_value": MU_TRUE,
            "bias": bias,
            "n_resp": n_resp,
            "cv_mse_unwt": cv_mse if variant_name == "unwt" else -1.0,
            "weight_clip": weight_clip,
        })

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=str, default="200,300,500,1000,2000,5000,10000")
    parser.add_argument("--sels", type=str, default="1.0")
    parser.add_argument("--response-rate", type=float, default=0.5)
    parser.add_argument(
        "--learners",
        type=str,
        default=",".join(name for name, _, _ in LEARNER_SPECS),
    )
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--weight-clip", type=float, default=25.0)
    parser.add_argument(
        "--propensity-mode",
        choices=["estimated", "oracle"],
        default="estimated",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/practitioner_capacity_curve_ks_v1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ns = [int(s.strip()) for s in args.ns.split(",")]
    sels = [float(s.strip()) for s in args.sels.split(",")]
    learner_names = {s.strip() for s in args.learners.split(",") if s.strip()}
    learners = [spec for spec in LEARNER_SPECS if spec[0] in learner_names]
    unknown_learners = learner_names - {spec[0] for spec in LEARNER_SPECS}
    if unknown_learners:
        raise ValueError(f"Unknown learners: {sorted(unknown_learners)}")

    tasks: list[tuple] = []
    for n in ns:
        for sel in sels:
            for lname, kind, params in learners:
                for seed in range(args.seeds):
                    tasks.append(
                        (
                            n,
                            sel,
                            args.response_rate,
                            lname,
                            kind,
                            params,
                            seed,
                            args.weight_clip,
                            args.propensity_mode,
                        )
                    )

    n_tasks = len(tasks)
    print(f"Running {n_tasks} tasks ({len(ns)} sizes x {len(learners)} learners x {args.seeds} seeds)")

    directory = os.path.dirname(args.output)
    if directory:
        os.makedirs(directory, exist_ok=True)

    out_handle = open(args.output, "w", newline="")
    writer = csv.DictWriter(out_handle, fieldnames=RESULT_FIELDS)
    writer.writeheader()
    out_handle.flush()
    total_rows = 0
    done = 0
    t0 = time.time()

    if args.workers == 1:
        task_results = (run_task(t) for t in tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        task_results = (future.result() for future in as_completed(
            {executor.submit(run_task, t): t for t in tasks},
        ))

    try:
        for task_rows in task_results:
            if task_rows:
                writer.writerows(task_rows)
                total_rows += len(task_rows)
            done += 1
            if done % 50 == 0:
                out_handle.flush()
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_tasks - done) / rate if rate > 0 else 0
                print(f"  {done}/{n_tasks} ({100*done/n_tasks:.0f}%) | {total_rows} rows | {rate:.0f} tasks/s | ETA {eta:.0f}s")
    finally:
        if args.workers != 1:
            executor.shutdown()

    out_handle.close()
    elapsed_total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE. {total_rows} rows in {elapsed_total:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
