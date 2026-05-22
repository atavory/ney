#!/usr/bin/env python3
from __future__ import annotations

"""
IV bottleneck ablation using the correct PLIV score.

This script verifies where the gain lives for instrumental variables when
the score is written correctly.

Estimator:
    theta = E[h(X, Z) * (Y - l(X))] / E[h(X, Z) * (D - r(X))]

with:
    h_basic(X, Z) = Z - E[Z | X]
    h_opt(X, Z)   = E[D | X, Z] - E[D | X]

For the DGPs below, Z is randomized Bernoulli(0.5), so:
    h_opt(X, Z) = gamma(X) * (Z - 0.5)

The local sensitivity for outcome-nuisance error under the optimal-instrument
score is proportional to:
    q(X) = E[h_opt(X, Z)^2 | X] = gamma(X)^2 * Var(Z | X)

This ablation isolates three intervention points:
  1. Weight the outcome nuisance l(X) fit by q(X)
  2. Weight the treatment nuisance r(X) fit by q(X)
  3. Use the oracle optimal instrument h_opt instead of h_basic

The goal is to determine whether IV behaves like AIPW
(outcome-fit weighting carries the gain) or like PLM
(the main leverage sits in the instrument / aggregation stage).
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")

THETA_TRUE = 2.0
RESULT_FIELDS = [
    "dgp",
    "N",
    "learner",
    "capacity_param",
    "seed",
    "variant",
    "weight_y_fit",
    "weight_d_fit",
    "use_opt_inst",
    "theta_est",
    "theta_true",
    "bias",
]


@dataclass(frozen=True)
class Variant:
    name: str
    weight_y_fit: bool
    weight_d_fit: bool
    use_opt_inst: bool


VARIANTS = (
    Variant("unwt", False, False, False),
    Variant("y_fit_only", True, False, False),
    Variant("d_fit_only", False, True, False),
    Variant("yd_fit", True, True, False),
    Variant("opt_inst_only", False, False, True),
    Variant("y_fit_optinst", True, False, True),
    Variant("d_fit_optinst", False, True, True),
    Variant("all_three", True, True, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--degrees", type=str, default="1,2,3,4,5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dgps",
        type=str,
        default="iv_hetero,iv_weak,iv_const",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/iv_bottleneck_ablation_v1.csv",
    )
    return parser.parse_args()


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_csv_strs(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def make_observed_covariates(z_lat: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.exp(z_lat[:, 0] / 2),
            z_lat[:, 1] / (1 + np.exp(z_lat[:, 0])) + 10,
            (z_lat[:, 0] * z_lat[:, 2] / 25 + 0.6) ** 3,
            (z_lat[:, 1] + z_lat[:, 3] + 20) ** 2,
        )
    )


def make_iv_data(n: int, seed: int, dgp_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z_lat = rng.standard_normal((n, 4))
    x = make_observed_covariates(z_lat)

    if dgp_name == "iv_hetero":
        gamma_x = 0.3 + 0.7 * expit(2.0 * z_lat[:, 0])
    elif dgp_name == "iv_weak":
        gamma_x = np.clip(0.8 * expit(3.0 * z_lat[:, 0]) - 0.05, 0.02, 0.8)
    elif dgp_name == "iv_const":
        gamma_x = np.full(n, 0.6)
    else:
        raise ValueError(f"Unknown DGP: {dgp_name}")

    g0 = 27.4 * z_lat[:, 0] + 13.7 * z_lat[:, 1] + 13.7 * z_lat[:, 2] + 13.7 * z_lat[:, 3]
    r0 = 0.5 * z_lat[:, 0] - 0.3 * z_lat[:, 1]
    z_inst = rng.binomial(1, 0.5, size=n).astype(float)
    d = r0 + gamma_x * z_inst + 0.3 * rng.standard_normal(n)
    y = THETA_TRUE * d + g0 + 0.5 * rng.standard_normal(n)
    return x, y, d, z_inst, gamma_x


def make_features(x: np.ndarray, degree: int) -> np.ndarray:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly.fit_transform(x)
    return StandardScaler().fit_transform(x_poly)


def fit_nuisances(
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
        d_model = Ridge(alpha=1.0)
        y_weight = weights[tr] if variant.weight_y_fit else None
        d_weight = weights[tr] if variant.weight_d_fit else None
        y_model.fit(x_poly[tr], y[tr], sample_weight=y_weight)
        d_model.fit(x_poly[tr], d[tr], sample_weight=d_weight)
        y_resid[te] = y[te] - y_model.predict(x_poly[te])
        d_resid[te] = d[te] - d_model.predict(x_poly[te])
    return y_resid, d_resid


def estimate_theta(
    y_resid: np.ndarray,
    d_resid: np.ndarray,
    z_inst: np.ndarray,
    gamma_x: np.ndarray,
    use_opt_inst: bool,
) -> float:
    z_resid = z_inst - 0.5
    h = gamma_x * z_resid if use_opt_inst else z_resid
    numerator = np.sum(h * y_resid)
    denominator = np.sum(h * d_resid)
    return numerator / denominator


def run_task(args: tuple[str, int, int, Variant, int]) -> dict[str, object]:
    dgp_name, n, degree, variant, seed = args
    x, y, d, z_inst, gamma_x = make_iv_data(n=n, seed=seed, dgp_name=dgp_name)
    x_poly = make_features(x, degree)
    align_weight = gamma_x**2
    y_resid, d_resid = fit_nuisances(x_poly, y, d, align_weight, variant, seed)
    theta_hat = estimate_theta(
        y_resid=y_resid,
        d_resid=d_resid,
        z_inst=z_inst,
        gamma_x=gamma_x,
        use_opt_inst=variant.use_opt_inst,
    )
    return {
        "dgp": dgp_name,
        "N": n,
        "learner": "poly",
        "capacity_param": str(degree),
        "seed": seed,
        "variant": variant.name,
        "weight_y_fit": int(variant.weight_y_fit),
        "weight_d_fit": int(variant.weight_d_fit),
        "use_opt_inst": int(variant.use_opt_inst),
        "theta_est": theta_hat,
        "theta_true": THETA_TRUE,
        "bias": theta_hat - THETA_TRUE,
    }


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


def write_rows(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    degrees = parse_csv_ints(args.degrees)
    dgps = parse_csv_strs(args.dgps)
    tasks = build_tasks(dgps, args.n, degrees, args.seeds)

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
