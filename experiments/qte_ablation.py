#!/usr/bin/env python3
from __future__ import annotations

"""
Cross-fitted QTE ablation with oracle density-sensitive weights.

The implementation uses a Gaussian conditional-outcome plug-in:
fit arm-specific mean / scale models, aggregate them into marginal mixture CDFs,
and invert the CDF to get arm-specific quantiles. This is intentionally fast
enough to run large sweeps while preserving the density-threshold geometry that
motivated the task.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from common import (
    make_observed_covariates,
    make_poly_features,
    normalize_weights,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strs,
    fit_log_variance_model,
    weighted_mean,
    write_rows,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")

RESULT_FIELDS = [
    "dgp",
    "N",
    "learner",
    "capacity_param",
    "tau",
    "seed",
    "variant",
    "weight_mean_fit",
    "weight_scale_fit",
    "weight_agg",
    "estimate",
    "true_value",
    "bias",
]

TRUTH_CACHE: dict[tuple[str, float], tuple[float, float, float]] = {}


@dataclass(frozen=True)
class Variant:
    name: str
    weight_mean_fit: bool
    weight_scale_fit: bool
    weight_agg: bool


VARIANTS = (
    Variant("unwt", False, False, False),
    Variant("y_fit_only", True, False, False),
    Variant("density_wt_only", False, True, False),
    Variant("agg_only", False, False, True),
    Variant("y_fit_agg", True, False, True),
    Variant("all", True, True, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--degrees", type=str, default="1,2,3,4")
    parser.add_argument("--taus", type=str, default="0.5,0.1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--truth-samples", type=int, default=200000)
    parser.add_argument(
        "--dgps",
        type=str,
        default="qte_hetero,qte_const",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/qte_ablation_v3.csv",
    )
    return parser.parse_args()


def make_qte_data(
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
]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    x = make_observed_covariates(z)

    z0, z1, z2 = z[:, 0], z[:, 1], z[:, 2]
    if dgp_name == "qte_hetero":
        pi = expit(0.85 * z0 - 0.35 * z1 + 0.2 * z2)
        sigma0 = 0.55 + 0.95 * expit(-1.5 * z0 + 0.4 * z2)
        sigma1 = 0.65 + 1.05 * expit(1.1 * z1 - 0.6 * z0)
        mu0 = 1.2 * z0 - 0.6 * z1 + 0.4 * np.sin(z2)
        mu1 = mu0 + 0.75 + 0.5 * expit(z0)
    elif dgp_name == "qte_const":
        pi = np.full(n, 0.5)
        sigma0 = np.full(n, 0.9)
        sigma1 = np.full(n, 0.9)
        mu0 = np.zeros(n)
        mu1 = np.full(n, 0.75)
    else:
        raise ValueError(f"Unknown DGP: {dgp_name}")

    t = rng.binomial(1, pi).astype(int)
    y0 = mu0 + sigma0 * rng.standard_normal(n)
    y1 = mu1 + sigma1 * rng.standard_normal(n)
    y = np.where(t == 1, y1, y0)
    return x, t, y, pi, mu0, mu1, sigma0, sigma1, y0, y1


def precompute_truth(
    dgp_name: str,
    tau: float,
    truth_samples: int,
) -> tuple[float, float, float]:
    _, _, _, _, _, _, _, _, y0, y1 = make_qte_data(
        n=truth_samples,
        seed=1313 + int(1000 * tau),
        dgp_name=dgp_name,
    )
    q0 = float(np.quantile(y0, tau))
    q1 = float(np.quantile(y1, tau))
    return q1 - q0, q1, q0


def local_density(quantile_value: float, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 0.05)
    z_score = (quantile_value - np.asarray(mean, dtype=float)) / sigma_safe
    return np.maximum(norm.pdf(z_score) / sigma_safe, 1e-4)


def fit_arm_models(
    x_poly: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    weights_mean: dict[int, np.ndarray] | None,
    weights_scale: dict[int, np.ndarray] | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu_hat = {0: np.zeros(len(y)), 1: np.zeros(len(y))}
    sigma_hat = {0: np.zeros(len(y)), 1: np.zeros(len(y))}
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)

    for tr, te in splitter.split(x_poly):
        for arm in (0, 1):
            tr_arm = tr[t[tr] == arm]
            mean_weight = None if weights_mean is None else weights_mean[arm][tr_arm]
            scale_weight = None if weights_scale is None else weights_scale[arm][tr_arm]

            mean_model = Ridge(alpha=2.0)
            mean_model.fit(x_poly[tr_arm], y[tr_arm], sample_weight=mean_weight)
            mu_hat[arm][te] = mean_model.predict(x_poly[te])
            resid = y[tr_arm] - mean_model.predict(x_poly[tr_arm])
            sigma_pred = fit_log_variance_model(
                x_poly[tr_arm],
                resid,
                x_poly[te],
                sample_weight=scale_weight,
                alpha=5.0,
                sigma_floor=0.05,
            )
            sigma_cap = max(3.0, 2.5 * float(np.quantile(np.abs(resid), 0.9)))
            sigma_hat[arm][te] = np.clip(sigma_pred, 0.05, sigma_cap)

    return mu_hat[0], sigma_hat[0], mu_hat[1], sigma_hat[1]


def mixture_cdf(
    value: float,
    mean: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    sigma_safe = np.maximum(sigma, 0.05)
    cdf_vals = norm.cdf((value - mean) / sigma_safe)
    return weighted_mean(cdf_vals, weights)


def solve_marginal_quantile(
    mean: np.ndarray,
    sigma: np.ndarray,
    tau: float,
    weights: np.ndarray | None,
) -> float:
    sigma_safe = np.maximum(sigma, 0.05)
    lo = float(np.min(mean - 8.0 * sigma_safe))
    hi = float(np.max(mean + 8.0 * sigma_safe))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mixture_cdf(mid, mean, sigma_safe, weights) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_task(args: tuple[str, int, int, float, Variant, int]) -> dict[str, object]:
    dgp_name, n, degree, tau, variant, seed = args
    x, t, y, pi, mu0_true, mu1_true, sigma0_true, sigma1_true, _, _ = make_qte_data(
        n=n,
        seed=seed,
        dgp_name=dgp_name,
    )
    qte_true, q1_true, q0_true = TRUTH_CACHE[(dgp_name, tau)]

    density1 = local_density(q1_true, mu1_true, sigma1_true)
    density0 = local_density(q0_true, mu0_true, sigma0_true)
    fit_weights = {
        1: normalize_weights(1.0 / np.maximum(pi**2, 1e-5), clip=12.0),
        0: normalize_weights(1.0 / np.maximum((1.0 - pi) ** 2, 1e-5), clip=12.0),
    }
    density_weights = {
        1: normalize_weights(1.0 / np.maximum(density1, 1e-4), clip=12.0),
        0: normalize_weights(1.0 / np.maximum(density0, 1e-4), clip=12.0),
    }

    x_poly = make_poly_features(x, degree)
    weights_mean = fit_weights if variant.weight_mean_fit else None
    weights_scale = density_weights if variant.weight_scale_fit else None
    mu0_hat, sigma0_hat, mu1_hat, sigma1_hat = fit_arm_models(
        x_poly=x_poly,
        t=t,
        y=y,
        weights_mean=weights_mean,
        weights_scale=weights_scale,
        seed=seed,
    )

    agg_w1 = density_weights[1] if variant.weight_agg else None
    agg_w0 = density_weights[0] if variant.weight_agg else None
    q1_hat = solve_marginal_quantile(mu1_hat, sigma1_hat, tau, agg_w1)
    q0_hat = solve_marginal_quantile(mu0_hat, sigma0_hat, tau, agg_w0)
    estimate = q1_hat - q0_hat
    return {
        "dgp": dgp_name,
        "N": n,
        "learner": "poly_gaussian_plugin",
        "capacity_param": str(degree),
        "tau": tau,
        "seed": seed,
        "variant": variant.name,
        "weight_mean_fit": int(variant.weight_mean_fit),
        "weight_scale_fit": int(variant.weight_scale_fit),
        "weight_agg": int(variant.weight_agg),
        "estimate": estimate,
        "true_value": qte_true,
        "bias": estimate - qte_true,
    }


def build_tasks(
    dgps: tuple[str, ...],
    n: int,
    degrees: tuple[int, ...],
    taus: tuple[float, ...],
    seeds: int,
) -> list[tuple[str, int, int, float, Variant, int]]:
    tasks: list[tuple[str, int, int, float, Variant, int]] = []
    for dgp_name in dgps:
        for degree in degrees:
            for tau in taus:
                for variant in VARIANTS:
                    for seed in range(seeds):
                        tasks.append((dgp_name, n, degree, tau, variant, seed))
    return tasks


def main() -> None:
    args = parse_args()
    degrees = parse_csv_ints(args.degrees)
    taus = parse_csv_floats(args.taus)
    dgps = parse_csv_strs(args.dgps)

    for dgp_name in dgps:
        for tau in taus:
            TRUTH_CACHE[(dgp_name, tau)] = precompute_truth(
                dgp_name=dgp_name,
                tau=tau,
                truth_samples=args.truth_samples,
            )

    tasks = build_tasks(
        dgps=dgps,
        n=args.n,
        degrees=degrees,
        taus=taus,
        seeds=args.seeds,
    )

    print(f"Running {len(tasks)} QTE tasks")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(run_task, tasks), start=1):
            rows.append(row)
            if index % 250 == 0:
                print(f"  completed {index}/{len(tasks)}")

    write_rows(args.output, RESULT_FIELDS, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
