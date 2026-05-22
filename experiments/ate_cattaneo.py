#!/usr/bin/env python3
from __future__ import annotations

"""
ATE capacity benchmark on the Cattaneo smoking/birthweight data.

The data are real covariates and real treatment assignment. Outcomes are
semi-synthetic so the ATE is known. The purpose is not to claim a real causal
effect; it is to test whether score-aligned outcome fitting helps when overlap
and finite capacity interact in a standard observational benchmark.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")

DATA = Path(__file__).resolve().parent.parent / "data"

RESULT_FIELDS = [
    "experiment",
    "n_subsample",
    "n_covariates",
    "learner",
    "capacity_param",
    "seed",
    "variant",
    "ate_est",
    "ate_true",
    "bias",
    "abs_bias",
    "ps_min",
    "ps_p05",
    "ps_mean",
    "ps_p95",
    "ps_max",
    "fit_deff",
]


@dataclass(frozen=True)
class Variant:
    name: str
    exponent: float
    clip: float | None


@dataclass(frozen=True)
class LearnerSpec:
    name: str
    kind: str
    degree: int = 1
    depth: int = 1
    rounds: int = 20
    alpha: float = 1.0


VARIANTS = (
    Variant("unwt", 0.0, None),
    Variant("w05", 0.5, 25.0),
    Variant("stab", 1.0, 25.0),
    Variant("w2_c10", 2.0, 10.0),
)

LEARNERS = {
    "ridge1": LearnerSpec(name="ridge1", kind="ridge", degree=1, alpha=10.0),
    "ridge2": LearnerSpec(name="ridge2", kind="ridge", degree=2, alpha=10.0),
    "ridge3": LearnerSpec(name="ridge3", kind="ridge", degree=3, alpha=10.0),
    "stump10": LearnerSpec(name="stump10", kind="hgb", depth=1, rounds=10),
    "stump50": LearnerSpec(name="stump50", kind="hgb", depth=1, rounds=50),
    "hgb2_50": LearnerSpec(name="hgb2_50", kind="hgb", depth=2, rounds=50),
    "hgb2_100": LearnerSpec(name="hgb2_100", kind="hgb", depth=2, rounds=100),
}


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_strs(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def load_cattaneo(n_covariates: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(DATA / "data_cattaneo.csv")
    treatment = frame["mbsmoke_num"].fillna(frame["msmoke"]).astype(float).to_numpy()
    treatment = (treatment > 0).astype(float)
    outcome = frame["bweight"].astype(float).to_numpy()
    drop = {
        "bweight",
        "lbweight",
        "msmoke",
        "mbsmoke",
        "mbsmoke_num",
    }
    columns = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in drop]
    columns = columns[: min(n_covariates, len(columns))]
    x = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    return x, outcome, treatment


def subsample(
    x: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    n_subsample: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x) <= n_subsample:
        return x, outcome, treatment
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(x), size=n_subsample, replace=False)
    return x[keep], outcome[keep], treatment[keep]


def generate_outcome(
    x: np.ndarray,
    outcome_real: np.ndarray,
    treatment: np.ndarray,
    seed: int,
    ate_true: float,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    oracle = HistGradientBoostingRegressor(
        max_depth=4,
        max_iter=200,
        min_samples_leaf=10,
        random_state=seed,
    )
    oracle.fit(np.column_stack([x, treatment]), outcome_real)
    mu0 = oracle.predict(np.column_stack([x, np.zeros(len(x))]))
    # Put some effect heterogeneity in the low-overlap direction but keep the
    # sample ATE exactly equal to ate_true.
    first = StandardScaler().fit_transform(x[:, :1]).ravel()
    hetero = 70.0 * (first - float(np.mean(first)))
    effect = ate_true + hetero
    y0 = mu0 + 120.0 * rng.standard_normal(len(x))
    y1 = y0 + effect
    y = treatment * y1 + (1.0 - treatment) * y0
    return y, float(np.mean(effect))


def estimate_propensity(x: np.ndarray, treatment: np.ndarray, seed: int) -> np.ndarray:
    ps = np.zeros(len(treatment))
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed + 1000)
    for train, test in splitter.split(x):
        model = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=100,
            min_samples_leaf=20,
            random_state=seed,
        )
        model.fit(x[train], treatment[train].astype(int))
        ps[test] = model.predict_proba(x[test])[:, 1]
    return np.clip(ps, 0.01, 0.99)


def make_features(x: np.ndarray, learner: LearnerSpec) -> np.ndarray:
    x_std = StandardScaler().fit_transform(x)
    if learner.kind == "ridge":
        poly = PolynomialFeatures(degree=learner.degree, include_bias=False)
        return StandardScaler().fit_transform(poly.fit_transform(x_std))
    return x_std


def make_fit_weights(ps: np.ndarray, treatment: np.ndarray, variant: Variant) -> np.ndarray | None:
    if variant.name == "unwt":
        return None
    sensitivity = np.where(treatment == 1.0, 1.0 / ps, 1.0 / (1.0 - ps))
    weights = sensitivity**variant.exponent
    if variant.clip is not None:
        weights = np.minimum(weights, variant.clip)
    return weights / float(np.mean(weights))


def design_effect(weights: np.ndarray | None) -> float:
    if weights is None:
        return 1.0
    normalized = weights / float(np.mean(weights))
    return float(np.mean(normalized**2))


def fit_outcomes(
    x_design: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray | None,
    learner: LearnerSpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    mu0 = np.zeros(len(y))
    mu1 = np.zeros(len(y))
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed + 2000)
    for train, test in splitter.split(x_design):
        x_train = np.column_stack([x_design[train], treatment[train]])
        x0 = np.column_stack([x_design[test], np.zeros(len(test))])
        x1 = np.column_stack([x_design[test], np.ones(len(test))])
        sample_weight = weights[train] if weights is not None else None
        if learner.kind == "ridge":
            model = Ridge(alpha=learner.alpha)
        else:
            model = HistGradientBoostingRegressor(
                max_depth=learner.depth,
                max_iter=learner.rounds,
                min_samples_leaf=20,
                random_state=seed,
            )
        model.fit(x_train, y[train], sample_weight=sample_weight)
        mu0[test] = model.predict(x0)
        mu1[test] = model.predict(x1)
    return mu0, mu1


def aipw(y: np.ndarray, treatment: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, ps: np.ndarray) -> float:
    score = mu1 - mu0 + treatment * (y - mu1) / ps - (1.0 - treatment) * (y - mu0) / (1.0 - ps)
    return float(np.mean(score))


def run_task(
    task: tuple[np.ndarray, np.ndarray, np.ndarray, int, int, LearnerSpec, Variant, float],
) -> dict[str, object]:
    x_full, outcome_full, treatment_full, n_subsample, seed, learner, variant, ate_true = task
    x, outcome_real, treatment = subsample(x_full, outcome_full, treatment_full, n_subsample, seed)
    y, true_value = generate_outcome(x, outcome_real, treatment, seed, ate_true)
    ps = estimate_propensity(x, treatment, seed)
    x_design = make_features(x, learner)
    weights = make_fit_weights(ps, treatment, variant)
    mu0, mu1 = fit_outcomes(x_design, y, treatment, weights, learner, seed)
    estimate = aipw(y, treatment, mu0, mu1, ps)
    return {
        "experiment": "cattaneo",
        "n_subsample": len(x),
        "n_covariates": x.shape[1],
        "learner": learner.name,
        "capacity_param": learner.degree if learner.kind == "ridge" else learner.rounds,
        "seed": seed,
        "variant": variant.name,
        "ate_est": estimate,
        "ate_true": true_value,
        "bias": estimate - true_value,
        "abs_bias": abs(estimate - true_value),
        "ps_min": float(np.min(ps)),
        "ps_p05": float(np.quantile(ps, 0.05)),
        "ps_mean": float(np.mean(ps)),
        "ps_p95": float(np.quantile(ps, 0.95)),
        "ps_max": float(np.max(ps)),
        "fit_deff": design_effect(weights),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", type=str, default="500,1000,2000,4000")
    parser.add_argument("--n-covariates", type=str, default="5,10,20")
    parser.add_argument(
        "--learners",
        type=str,
        default="ridge1,ridge2,ridge3,stump10,stump50,hgb2_50,hgb2_100",
    )
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--ate-true", type=float, default=-200.0)
    parser.add_argument("--output", type=str, default="results/real_ate_cattaneo_v1.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_sizes = parse_csv_ints(args.sample_sizes)
    n_covariates_values = parse_csv_ints(args.n_covariates)
    learners = tuple(LEARNERS[name] for name in parse_csv_strs(args.learners))
    tasks: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int, LearnerSpec, Variant, float]] = []
    for n_covariates in n_covariates_values:
        x, outcome, treatment = load_cattaneo(n_covariates)
        print(f"Cattaneo covariates={n_covariates} n={len(x)} treated={int(treatment.sum())}")
        for n_subsample in sample_sizes:
            for learner in learners:
                for variant in VARIANTS:
                    for seed in range(args.seeds):
                        tasks.append((x, outcome, treatment, n_subsample, seed, learner, variant, args.ate_true))
    print(f"Running {len(tasks)} tasks")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    row_count = 0
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        handle.flush()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, row in enumerate(executor.map(run_task, tasks), start=1):
                writer.writerow(row)
                row_count += 1
                if index % 500 == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
                    print(f"  completed {index}/{len(tasks)}")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"Wrote {row_count} rows to {args.output}")


if __name__ == "__main__":
    main()
