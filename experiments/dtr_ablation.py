#!/usr/bin/env python3
from __future__ import annotations

"""
Two-stage DTR ablation with sequential pseudo-outcome estimation.

The implementation is a practical research prototype: two weighted Robinson
steps with known stage-specific score features, where stage 2 is estimated
first and then blipped down to form the stage-1 pseudo-outcome.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from common import (
    make_poly_features,
    normalize_weights,
    parse_csv_ints,
    parse_csv_strs,
    fit_crossfit_propensity,
    weighted_mean,
    write_rows,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")

THETA1_TRUE = 1.2
THETA2_TRUE = 0.9

RESULT_FIELDS = [
    "dgp",
    "N",
    "learner",
    "capacity_param",
    "seed",
    "variant",
    "parameter",
    "weight_y1_fit",
    "weight_y2_fit",
    "weight_agg1",
    "weight_agg2",
    "estimate",
    "true_value",
    "bias",
]


@dataclass(frozen=True)
class Variant:
    name: str
    weight_y2_fit: bool
    weight_y1_fit: bool
    weight_agg2: bool
    weight_agg1: bool


VARIANTS = (
    Variant("unwt", False, False, False, False),
    Variant("y2_fit_only", True, False, False, False),
    Variant("y1_fit_only", False, True, False, False),
    Variant("agg2_only", False, False, True, False),
    Variant("agg1_only", False, False, False, True),
    Variant("y_both", True, True, False, False),
    Variant("agg_both", False, False, True, True),
    Variant("all", True, True, True, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--degrees", type=str, default="1,2,3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dgps",
        type=str,
        default="dtr_hetero,dtr_const",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/dtr_ablation_v2.csv",
    )
    return parser.parse_args()


def make_stage1_observed(z0: np.ndarray, z1: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.exp(z0 / 2.0),
            (z1 + 1.5) ** 2,
            z0 * z1,
            z0 - z1,
        )
    )


def make_stage2_observed(
    z0: np.ndarray,
    z1: np.ndarray,
    l2: np.ndarray,
    a1: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        (
            np.exp(z0 / 2.0),
            (z1 + 1.5) ** 2,
            np.exp(l2 / 2.0),
            (l2 + 1.0) ** 2,
            a1,
            z0 * l2,
        )
    )


def make_dtr_data(
    n: int,
    seed: int,
    dgp_name: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    z0, z1 = z[:, 0], z[:, 1]

    if dgp_name == "dtr_hetero":
        pi1 = expit(-0.8 * z0 + 0.6 * z1)
        s1 = 0.4 + 1.3 * expit(-1.2 * z0 + 0.8 * z1)
    elif dgp_name == "dtr_const":
        pi1 = np.full(n, 0.5)
        s1 = np.full(n, 1.0)
    else:
        raise ValueError(f"Unknown DGP: {dgp_name}")

    a1 = rng.binomial(1, pi1).astype(int)
    l2 = 0.7 * z0 - 0.5 * z1 + 0.6 * a1 + 0.5 * rng.standard_normal(n)

    if dgp_name == "dtr_hetero":
        pi2 = expit(0.7 * l2 - 0.4 * z1 + 0.3 * a1)
        s2 = 0.25 + 1.5 * expit(1.4 * l2 - 0.6 * z0 + 0.4 * a1)
    else:
        pi2 = np.full(n, 0.5)
        s2 = np.full(n, 1.0)

    a2 = rng.binomial(1, pi2).astype(int)

    baseline = (
        1.5 * z0
        - 1.0 * z1
        + 1.1 * l2
        + 0.35 * z0 * l2
        + 0.35 * np.sin(1.2 * l2) * (1.0 + 0.4 * z0)
    )
    y = (
        baseline
        + THETA1_TRUE * s1 * a1
        + THETA2_TRUE * s2 * a2
        + 0.5 * rng.standard_normal(n)
    )

    w2_fit = normalize_weights(1.0 + s2**2, clip=10.0)
    w2_agg = normalize_weights(
        s2**2 / np.maximum(pi2 * (1.0 - pi2), 1e-4),
        clip=25.0,
    )
    future_load = 1.0 + 0.75 * normalize_weights(
        s2**2 / np.maximum(pi2 * (1.0 - pi2), 1e-4),
        clip=None,
    )
    w1_fit = normalize_weights(1.0 + s1**2, clip=10.0)
    w1_agg = normalize_weights(
        s1**2 / np.maximum(pi1 * (1.0 - pi1), 1e-4) * future_load,
        clip=25.0,
    )

    x1 = make_stage1_observed(z0, z1)
    x2 = make_stage2_observed(z0, z1, l2, a1)
    return x1, x2, a1, a2, y, s1, s2, w1_fit, w1_agg, w2_fit, w2_agg


def fit_crossfit_outcome(
    x_poly: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    pred = np.zeros(len(target))
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in splitter.split(x_poly):
        model = Ridge(alpha=1.0)
        fold_weight = None if weights is None else weights[tr]
        model.fit(x_poly[tr], target[tr], sample_weight=fold_weight)
        pred[te] = model.predict(x_poly[te])
    return pred


def aggregate_theta(
    outcome_resid: np.ndarray,
    treatment: np.ndarray,
    propensity_hat: np.ndarray,
    score_feature: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    score = score_feature * (treatment - propensity_hat)
    numerand = score * outcome_resid
    denominator = score**2
    if weights is None:
        return float(np.sum(numerand) / np.sum(denominator))
    return float(np.sum(weights * numerand) / np.sum(weights * denominator))


def estimate_stage_parameters(
    x1_poly: np.ndarray,
    x2_poly: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    y: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    w1_fit: np.ndarray,
    w1_agg: np.ndarray,
    w2_fit: np.ndarray,
    w2_agg: np.ndarray,
    variant: Variant,
    seed: int,
) -> tuple[float, float]:
    p2_hat = fit_crossfit_propensity(x2_poly, a2, seed + 11)
    y2_pred = fit_crossfit_outcome(
        x2_poly,
        y,
        w2_fit if variant.weight_y2_fit else None,
        seed + 17,
    )
    theta2_hat = aggregate_theta(
        outcome_resid=y - y2_pred,
        treatment=a2,
        propensity_hat=p2_hat,
        score_feature=s2,
        weights=w2_agg if variant.weight_agg2 else None,
    )

    pseudo_y1 = y - theta2_hat * s2 * a2
    p1_hat = fit_crossfit_propensity(x1_poly, a1, seed + 23)
    y1_pred = fit_crossfit_outcome(
        x1_poly,
        pseudo_y1,
        w1_fit if variant.weight_y1_fit else None,
        seed + 29,
    )
    theta1_hat = aggregate_theta(
        outcome_resid=pseudo_y1 - y1_pred,
        treatment=a1,
        propensity_hat=p1_hat,
        score_feature=s1,
        weights=w1_agg if variant.weight_agg1 else None,
    )
    return theta1_hat, theta2_hat


def run_task(args: tuple[str, int, int, Variant, int]) -> list[dict[str, object]]:
    dgp_name, n, degree, variant, seed = args
    x1, x2, a1, a2, y, s1, s2, w1_fit, w1_agg, w2_fit, w2_agg = make_dtr_data(
        n=n,
        seed=seed,
        dgp_name=dgp_name,
    )
    x1_poly = make_poly_features(x1, degree)
    x2_poly = make_poly_features(x2, degree)
    theta1_hat, theta2_hat = estimate_stage_parameters(
        x1_poly=x1_poly,
        x2_poly=x2_poly,
        a1=a1,
        a2=a2,
        y=y,
        s1=s1,
        s2=s2,
        w1_fit=w1_fit,
        w1_agg=w1_agg,
        w2_fit=w2_fit,
        w2_agg=w2_agg,
        variant=variant,
        seed=seed,
    )
    return [
        {
            "dgp": dgp_name,
            "N": n,
            "learner": "poly_stagewise",
            "capacity_param": str(degree),
            "seed": seed,
            "variant": variant.name,
            "parameter": "gamma_1",
            "weight_y1_fit": int(variant.weight_y1_fit),
            "weight_y2_fit": int(variant.weight_y2_fit),
            "weight_agg1": int(variant.weight_agg1),
            "weight_agg2": int(variant.weight_agg2),
            "estimate": theta1_hat,
            "true_value": THETA1_TRUE,
            "bias": theta1_hat - THETA1_TRUE,
        },
        {
            "dgp": dgp_name,
            "N": n,
            "learner": "poly_stagewise",
            "capacity_param": str(degree),
            "seed": seed,
            "variant": variant.name,
            "parameter": "gamma_2",
            "weight_y1_fit": int(variant.weight_y1_fit),
            "weight_y2_fit": int(variant.weight_y2_fit),
            "weight_agg1": int(variant.weight_agg1),
            "weight_agg2": int(variant.weight_agg2),
            "estimate": theta2_hat,
            "true_value": THETA2_TRUE,
            "bias": theta2_hat - THETA2_TRUE,
        },
    ]


def build_tasks(
    dgps: tuple[str, ...],
    n: int,
    degrees: tuple[int, ...],
    seeds: int,
) -> list[tuple[str, int, int, Variant, int]]:
    tasks: list[tuple[str, int, int, Variant, int]] = []
    for dgp_name in dgps:
        for degree in degrees:
            for variant in VARIANTS:
                for seed in range(seeds):
                    tasks.append((dgp_name, n, degree, variant, seed))
    return tasks


def main() -> None:
    args = parse_args()
    degrees = parse_csv_ints(args.degrees)
    dgps = parse_csv_strs(args.dgps)
    tasks = build_tasks(dgps=dgps, n=args.n, degrees=degrees, seeds=args.seeds)

    print(f"Running {len(tasks)} DTR tasks")
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
