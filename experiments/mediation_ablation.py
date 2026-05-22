#!/usr/bin/env python3
from __future__ import annotations

"""
Cross-fitted mediation ablation for a natural indirect effect prototype.

This is a research runner, not a polished semiparametric package. It implements
an oracle-weighted nested g-formula experiment with three intervention points:

1. weight the treated outcome nuisance fit
2. weight the mediator nuisance fits
3. weight the downstream unit-level aggregation (diagnostic reweighting)

The aggregation-weight variants are included because they were part of the
original task list. Unlike the PLM / IV ratio estimators, this normalized
downstream reweighting is a diagnostic candidate rather than a theorem-backed
target-preserving intervention.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import (
    gaussian_kernel_separation,
    make_observed_covariates,
    make_poly_features,
    normalize_weights,
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
    "seed",
    "variant",
    "weight_y_fit",
    "weight_m_fit",
    "weight_agg",
    "estimate",
    "true_value",
    "bias",
]


@dataclass(frozen=True)
class Variant:
    name: str
    weight_y_fit: bool
    weight_m_fit: bool
    weight_agg: bool


VARIANTS = (
    Variant("unwt", False, False, False),
    Variant("y_fit_only", True, False, False),
    Variant("m_fit_only", False, True, False),
    Variant("agg_only", False, False, True),
    Variant("y_m_fit", True, True, False),
    Variant("y_fit_agg", True, False, True),
    Variant("all_three", True, True, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--degrees", type=str, default="1,2,3,4")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mc-samples", type=int, default=40)
    parser.add_argument(
        "--dgps",
        type=str,
        default="mediation_hetero,mediation_const",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/mediation_ablation_v2.csv",
    )
    return parser.parse_args()


def make_mediation_data(
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
]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    x = make_observed_covariates(z)

    z0, z1, z2 = z[:, 0], z[:, 1], z[:, 2]
    pi = expit(0.8 * z0 - 0.45 * z1 + 0.2 * z2)
    d = rng.binomial(1, pi).astype(int)

    mu_m0 = 0.6 * z0 - 0.5 * z1 + 0.3 * z2
    if dgp_name == "mediation_hetero":
        delta_m = 0.2 + 1.4 * expit(-1.8 * z0 + 0.7 * z1 - 0.2 * z2)
        sigma_m = 0.35 + 0.55 * expit(1.4 * z1 - 0.8 * z0)
        beta_m = 0.6 + 0.4 * expit(z2 - 0.4 * z0)
    elif dgp_name == "mediation_const":
        delta_m = np.full(n, 0.8)
        sigma_m = np.full(n, 0.6)
        beta_m = np.full(n, 0.9)
    else:
        raise ValueError(f"Unknown DGP: {dgp_name}")

    mediator_interaction = 0.15
    m = mu_m0 + d * delta_m + sigma_m * rng.standard_normal(n)

    base = 1.0 + 1.4 * z0 - 1.0 * z1 + 0.5 * z2 + 0.3 * z0 * z1
    y = (
        base
        + 0.7 * d
        + beta_m * m
        + 0.2 * (m**2)
        + mediator_interaction * d * m
        + 0.35 * rng.standard_normal(n)
    )

    beta_eff = beta_m + mediator_interaction
    true_unit_nie = beta_eff * delta_m + 0.2 * (2.0 * mu_m0 * delta_m + delta_m**2)
    kernel_weight = gaussian_kernel_separation(delta_m, sigma_m)
    y_weight = normalize_weights(kernel_weight * (1.0 + beta_eff**2), clip=8.0)
    m_weight = normalize_weights(kernel_weight, clip=12.0)
    agg_weight = normalize_weights(kernel_weight * (1.0 + beta_eff**2), clip=12.0)
    return x, d, m, y, true_unit_nie, y_weight, m_weight, agg_weight


def make_outcome_design(x_poly: np.ndarray, mediator: np.ndarray) -> np.ndarray:
    m = np.asarray(mediator).reshape(-1)
    interaction_cols = min(2, x_poly.shape[1])
    interaction = x_poly[:, :interaction_cols] * m[:, None]
    return np.column_stack((x_poly, m, m**2, interaction))


def fit_unit_effects(
    x_poly: np.ndarray,
    d: np.ndarray,
    m: np.ndarray,
    y: np.ndarray,
    y_align_weight: np.ndarray,
    m_align_weight: np.ndarray,
    variant: Variant,
    seed: int,
    mc_samples: int,
) -> np.ndarray:
    unit_effect = np.zeros(len(y))
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)

    for fold_idx, (tr, te) in enumerate(splitter.split(x_poly)):
        tr_treated = tr[d[tr] == 1]
        y_weight = y_align_weight[tr_treated] if variant.weight_y_fit else None
        outcome_model = Ridge(alpha=10.0)
        design_scaler = StandardScaler()
        train_design = make_outcome_design(x_poly[tr_treated], m[tr_treated])
        train_design_scaled = design_scaler.fit_transform(train_design)
        outcome_model.fit(train_design_scaled, y[tr_treated], sample_weight=y_weight)

        mediator_means: dict[int, np.ndarray] = {}
        mediator_sigmas: dict[int, np.ndarray] = {}
        for arm in (0, 1):
            tr_arm = tr[d[tr] == arm]
            m_weight = m_align_weight[tr_arm] if variant.weight_m_fit else None
            mean_model = Ridge(alpha=5.0)
            mean_model.fit(x_poly[tr_arm], m[tr_arm], sample_weight=m_weight)
            mediator_means[arm] = mean_model.predict(x_poly[te])
            resid = m[tr_arm] - mean_model.predict(x_poly[tr_arm])
            mediator_sigmas[arm] = fit_log_variance_model(
                x_poly[tr_arm],
                resid,
                x_poly[te],
                sample_weight=m_weight,
                alpha=1.0,
                sigma_floor=0.05,
            )

        rng = np.random.default_rng(seed * 1000 + fold_idx)
        te_size = len(te)
        eps1 = rng.standard_normal((te_size, mc_samples))
        eps0 = rng.standard_normal((te_size, mc_samples))
        clip_lo, clip_hi = np.quantile(m[tr], [0.01, 0.99])

        mediator_1 = mediator_means[1][:, None] + mediator_sigmas[1][:, None] * eps1
        mediator_0 = mediator_means[0][:, None] + mediator_sigmas[0][:, None] * eps0
        mediator_1 = np.clip(mediator_1, clip_lo - 0.5, clip_hi + 0.5)
        mediator_0 = np.clip(mediator_0, clip_lo - 0.5, clip_hi + 0.5)

        x_rep = np.repeat(x_poly[te], mc_samples, axis=0)
        design_1 = design_scaler.transform(
            make_outcome_design(x_rep, mediator_1.reshape(-1))
        )
        design_0 = design_scaler.transform(
            make_outcome_design(x_rep, mediator_0.reshape(-1))
        )
        pred_1 = outcome_model.predict(
            design_1
        ).reshape(te_size, mc_samples)
        pred_0 = outcome_model.predict(
            design_0
        ).reshape(te_size, mc_samples)
        unit_effect[te] = pred_1.mean(axis=1) - pred_0.mean(axis=1)

    return unit_effect


def run_task(args: tuple[str, int, int, Variant, int, int]) -> dict[str, object]:
    dgp_name, n, degree, variant, seed, mc_samples = args
    x, d, m, y, true_unit_nie, y_align_weight, m_align_weight, agg_weight = make_mediation_data(
        n=n,
        seed=seed,
        dgp_name=dgp_name,
    )
    x_poly = make_poly_features(x, degree)
    unit_effect = fit_unit_effects(
        x_poly=x_poly,
        d=d,
        m=m,
        y=y,
        y_align_weight=y_align_weight,
        m_align_weight=m_align_weight,
        variant=variant,
        seed=seed,
        mc_samples=mc_samples,
    )
    outer_weight = agg_weight if variant.weight_agg else None
    estimate = weighted_mean(unit_effect, outer_weight)
    true_value = float(np.mean(true_unit_nie))
    return {
        "dgp": dgp_name,
        "N": n,
        "learner": "poly_nested_plugin",
        "capacity_param": str(degree),
        "seed": seed,
        "variant": variant.name,
        "weight_y_fit": int(variant.weight_y_fit),
        "weight_m_fit": int(variant.weight_m_fit),
        "weight_agg": int(variant.weight_agg),
        "estimate": estimate,
        "true_value": true_value,
        "bias": estimate - true_value,
    }


def build_tasks(
    dgps: tuple[str, ...],
    n: int,
    degrees: tuple[int, ...],
    seeds: int,
    mc_samples: int,
) -> list[tuple[str, int, int, Variant, int, int]]:
    tasks: list[tuple[str, int, int, Variant, int, int]] = []
    for dgp_name in dgps:
        for degree in degrees:
            for variant in VARIANTS:
                for seed in range(seeds):
                    tasks.append((dgp_name, n, degree, variant, seed, mc_samples))
    return tasks


def main() -> None:
    args = parse_args()
    degrees = parse_csv_ints(args.degrees)
    dgps = parse_csv_strs(args.dgps)
    tasks = build_tasks(
        dgps=dgps,
        n=args.n,
        degrees=degrees,
        seeds=args.seeds,
        mc_samples=args.mc_samples,
    )

    print(f"Running {len(tasks)} mediation tasks")
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
