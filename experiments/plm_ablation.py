#!/usr/bin/env python3
from __future__ import annotations

"""
PLM bottleneck ablation: which stage carries the gains from Var(D|X)?

This script isolates three possible intervention points in cross-fitted PLM:
  1. Weight the outcome nuisance g(X) fit by Var(D|X)
  2. Weight the treatment nuisance m_D(X) fit by Var(D|X)
  3. Weight the final theta aggregation by Var(D|X)

It runs the full 2^3 factorial over those knobs on both:
  - a heteroskedastic DGP where Var(D|X) varies
  - a homoskedastic null where Var(D|X) is constant

The goal is to determine whether PLM behaves like AIPW
(loss-weighting mismatch) or whether the main leverage sits in the
final aggregation step.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")

THETA_TRUE = 2.0
DEFAULT_DEGREES = (1, 2, 3, 4, 5)
RESULT_FIELDS = [
    "dgp",
    "N",
    "learner",
    "capacity_param",
    "seed",
    "variant",
    "weight_y_fit",
    "weight_d_fit",
    "weight_theta",
    "theta_est",
    "theta_true",
    "bias",
]


@dataclass(frozen=True)
class Variant:
    name: str
    weight_y_fit: bool
    weight_d_fit: bool
    weight_theta: bool


VARIANTS = (
    Variant("unwt", False, False, False),
    Variant("y_fit_only", True, False, False),
    Variant("d_fit_only", False, True, False),
    Variant("theta_only", False, False, True),
    Variant("yd_fit", True, True, False),
    Variant("y_fit_theta", True, False, True),
    Variant("d_fit_theta", False, True, True),
    Variant("all_three", True, True, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--degrees", type=str, default="1,2,3,4,5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=str, default="results/plm_bottleneck_ablation_v1.csv")
    return parser.parse_args()


def parse_degrees(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def make_plm_data(n: int, seed: int, heteroskedastic: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    x = np.column_stack((np.exp(z[:, 0] / 2), (z[:, 1] + 1.5) ** 2))

    sigma = (
        0.3 + 2.0 / (1.0 + np.exp(2.0 * z[:, 0]))
        if heteroskedastic
        else np.ones(n)
    )
    d = z[:, 0] + sigma * rng.standard_normal(n)
    g0 = 5.0 * z[:, 0] + 3.0 * z[:, 1]
    y = THETA_TRUE * d + g0 + 0.5 * rng.standard_normal(n)
    return x, y, d, sigma**2


def make_features(x: np.ndarray, degree: int) -> np.ndarray:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly.fit_transform(x)
    return StandardScaler().fit_transform(x_poly)


def fit_residuals(
    x_poly: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    weights: np.ndarray,
    variant: Variant,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_resid = np.zeros(len(x_poly))
    d_resid = np.zeros(len(x_poly))
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)

    for tr, te in splitter.split(x_poly):
        y_model = Ridge(alpha=1.0)
        d_model = LinearRegression()
        y_weight = weights[tr] if variant.weight_y_fit else None
        d_weight = weights[tr] if variant.weight_d_fit else None
        y_model.fit(x_poly[tr], y[tr], sample_weight=y_weight)
        d_model.fit(x_poly[tr], d[tr], sample_weight=d_weight)
        y_resid[te] = y[te] - y_model.predict(x_poly[te])
        d_resid[te] = d[te] - d_model.predict(x_poly[te])
    return y_resid, d_resid


def aggregate_theta(y_resid: np.ndarray, d_resid: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        numerator = np.sum(d_resid * y_resid)
        denominator = np.sum(d_resid**2)
    else:
        numerator = np.sum(weights * d_resid * y_resid)
        denominator = np.sum(weights * d_resid**2)
    return numerator / denominator


def run_task(args: tuple[str, int, int, Variant, int]) -> dict[str, object]:
    dgp_name, n, degree, variant, seed = args
    heteroskedastic = dgp_name == "plm_hetero"
    x, y, d, var_d = make_plm_data(n=n, seed=seed, heteroskedastic=heteroskedastic)
    x_poly = make_features(x, degree)
    y_resid, d_resid = fit_residuals(x_poly, y, d, var_d, variant, seed)
    theta_weights = var_d if variant.weight_theta else None
    theta_hat = aggregate_theta(y_resid, d_resid, theta_weights)
    return {
        "dgp": dgp_name,
        "N": n,
        "learner": "poly",
        "capacity_param": str(degree),
        "seed": seed,
        "variant": variant.name,
        "weight_y_fit": int(variant.weight_y_fit),
        "weight_d_fit": int(variant.weight_d_fit),
        "weight_theta": int(variant.weight_theta),
        "theta_est": theta_hat,
        "theta_true": THETA_TRUE,
        "bias": theta_hat - THETA_TRUE,
    }


def write_rows(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_tasks(n: int, degrees: tuple[int, ...], seeds: int) -> list[tuple[str, int, int, Variant, int]]:
    tasks: list[tuple[str, int, int, Variant, int]] = []
    for dgp_name in ("plm_hetero", "plm_homo"):
        for degree in degrees:
            for variant in VARIANTS:
                for seed in range(seeds):
                    tasks.append((dgp_name, n, degree, variant, seed))
    return tasks


def main() -> None:
    args = parse_args()
    degrees = parse_degrees(args.degrees)
    tasks = build_tasks(args.n, degrees, args.seeds)

    print(f"Running {len(tasks)} tasks")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(run_task, tasks), start=1):
            rows.append(row)
            if index % 500 == 0:
                print(f"  completed {index}/{len(tasks)}")

    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
