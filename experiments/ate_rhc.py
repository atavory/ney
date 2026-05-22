#!/usr/bin/env python3
from __future__ import annotations

"""
ATE on Right Heart Catheterization (RHC) data with semi-synthetic outcomes.

Connors et al. (1996). n=5735, 50+ covariates. Famous for severe
propensity overlap failure: propensity scores range from 0.005 to 0.96.

Design: use real covariates and real treatment assignment. Generate
semi-synthetic potential outcomes Y(0), Y(1) from high-capacity models
fitted on the real data, so ground-truth ATE is known. Then estimate
ATE with limited-capacity nuisance models under different score-alignment
variants.

This gives real covariate geometry, real treatment assignment (real
overlap problems), and known truth.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

RESULT_FIELDS = [
    "experiment",
    "N",
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
    "ps_p95",
    "ps_max",
    "fit_deff",
]


@dataclass(frozen=True)
class Variant:
    name: str
    exponent: float
    clip: float | None


VARIANTS = (
    Variant("unwt", 0.0, None),
    Variant("w05", 0.5, 25.0),
    Variant("stab", 1.0, 25.0),
    Variant("w2_c10", 2.0, 10.0),
    Variant("w2_c25", 2.0, 25.0),
)


@dataclass(frozen=True)
class LearnerSpec:
    name: str
    kind: str
    degree: int = 1
    depth: int = 1
    rounds: int = 20


LEARNERS = {
    "ridge1": LearnerSpec(name="ridge1", kind="ridge", degree=1),
    "ridge2": LearnerSpec(name="ridge2", kind="ridge", degree=2),
    "stump20": LearnerSpec(name="stump20", kind="hgb", depth=1, rounds=20),
    "hgb2_50": LearnerSpec(name="hgb2_50", kind="hgb", depth=2, rounds=50),
}


def load_rhc() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(DATA, "data_rhc.csv"))
    treat = (df["swang1"] == "RHC").astype(float).values
    outcome = df["death"].map({"Yes": 1.0, "No": 0.0, 1: 1.0, 0: 0.0}).fillna(0.0).values.astype(float)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = ["Unnamed: 0", "sadmdte", "dschdte", "dthdte", "lstctdte", "death"]
    x_cols = [c for c in num_cols if c not in drop]
    x = df[x_cols].fillna(0).values.astype(float)
    return x, outcome, treat


def generate_semi_synthetic(
    x: np.ndarray,
    outcome_real: np.ndarray,
    treat: np.ndarray,
    seed: int,
    ate_true: float = -0.05,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    model = HistGradientBoostingRegressor(
        max_depth=5, max_iter=200, min_samples_leaf=10, random_state=seed,
    )
    model.fit(np.column_stack([x, treat]), outcome_real)
    mu0 = model.predict(np.column_stack([x, np.zeros(len(x))]))
    y0 = mu0 + 0.1 * rng.standard_normal(len(x))
    y1 = y0 + ate_true
    y_obs = treat * y1 + (1 - treat) * y0
    return y_obs, ate_true


def make_features(x: np.ndarray, learner: LearnerSpec) -> np.ndarray:
    x_std = StandardScaler().fit_transform(x)
    if learner.kind == "ridge":
        poly = PolynomialFeatures(degree=learner.degree, include_bias=False)
        return StandardScaler().fit_transform(poly.fit_transform(x_std))
    return x_std


def estimate_propensity(
    x: np.ndarray,
    treat: np.ndarray,
    seed: int,
) -> np.ndarray:
    ps = np.zeros(len(treat))
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed + 3333)
    for tr, te in splitter.split(x):
        model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, min_samples_leaf=20, random_state=seed,
        )
        model.fit(x[tr], treat[tr].astype(int))
        ps[te] = model.predict_proba(x[te])[:, 1]
    return np.clip(ps, 0.01, 0.99)


def make_fit_weights(
    ps: np.ndarray,
    treat: np.ndarray,
    variant: Variant,
) -> np.ndarray | None:
    if variant.name == "unwt":
        return None
    sensitivity = np.where(treat == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    base = sensitivity ** variant.exponent
    if variant.clip is not None:
        base = np.minimum(base, variant.clip)
    return base / np.mean(base)


def design_effect(weights: np.ndarray | None) -> float:
    if weights is None:
        return 1.0
    normalized = weights / np.mean(weights)
    return float(np.mean(normalized ** 2))


def fit_outcome_model(
    x_design: np.ndarray,
    y: np.ndarray,
    treat: np.ndarray,
    weights: np.ndarray | None,
    learner: LearnerSpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    mu0 = np.zeros(n)
    mu1 = np.zeros(n)
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in splitter.split(x_design):
        w_tr = weights[tr] if weights is not None else None
        if learner.kind == "ridge":
            model = Ridge(alpha=1.0)
            xd = np.column_stack([x_design, treat.reshape(-1, 1)])
            model.fit(xd[tr], y[tr], sample_weight=w_tr)
            mu0[te] = model.predict(np.column_stack([x_design[te], np.zeros((len(te), 1))]))
            mu1[te] = model.predict(np.column_stack([x_design[te], np.ones((len(te), 1))]))
        else:
            model = HistGradientBoostingRegressor(
                max_depth=learner.depth,
                max_iter=learner.rounds,
                min_samples_leaf=20,
                random_state=seed,
            )
            xd = np.column_stack([x_design, treat.reshape(-1, 1)])
            model.fit(xd[tr], y[tr], sample_weight=w_tr)
            mu0[te] = model.predict(np.column_stack([x_design[te], np.zeros((len(te), 1))]))
            mu1[te] = model.predict(np.column_stack([x_design[te], np.ones((len(te), 1))]))
    return mu0, mu1


def aipw_ate(
    y: np.ndarray,
    treat: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    ps: np.ndarray,
) -> float:
    phi = (
        mu1 - mu0
        + treat * (y - mu1) / ps
        - (1 - treat) * (y - mu0) / (1 - ps)
    )
    return float(np.mean(phi))


def run_task(
    args: tuple[np.ndarray, np.ndarray, np.ndarray, LearnerSpec, Variant, int, float],
) -> dict[str, object]:
    x, outcome_real, treat, learner, variant, seed, ate_true_default = args
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.choice(n, size=n, replace=True)
    x_b = x[idx]
    outcome_b = outcome_real[idx]
    treat_b = treat[idx]

    y_obs, ate_true = generate_semi_synthetic(x_b, outcome_b, treat_b, seed, ate_true_default)
    ps = estimate_propensity(x_b, treat_b, seed)
    x_design = make_features(x_b, learner)
    weights = make_fit_weights(ps, treat_b, variant)
    mu0, mu1 = fit_outcome_model(x_design, y_obs, treat_b, weights, learner, seed)
    ate_est = aipw_ate(y_obs, treat_b, mu0, mu1, ps)

    return {
        "experiment": "rhc",
        "N": n,
        "learner": learner.name,
        "capacity_param": learner.degree if learner.kind == "ridge" else learner.rounds,
        "seed": seed,
        "variant": variant.name,
        "ate_est": ate_est,
        "ate_true": ate_true,
        "bias": ate_est - ate_true,
        "abs_bias": abs(ate_est - ate_true),
        "ps_min": float(np.min(ps)),
        "ps_p05": float(np.quantile(ps, 0.05)),
        "ps_p95": float(np.quantile(ps, 0.95)),
        "ps_max": float(np.max(ps)),
        "fit_deff": design_effect(weights),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learners", type=str, default="ridge1,ridge2,stump20,hgb2_50")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ate-true", type=float, default=-0.05)
    parser.add_argument(
        "--output",
        type=str,
        default="results/real_ate_rhc_v1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    learner_names = [s.strip() for s in args.learners.split(",")]
    learners = [LEARNERS[name] for name in learner_names]

    x, outcome_real, treat = load_rhc()
    print(f"RHC: n={len(x)}, p={x.shape[1]}, treated={treat.sum():.0f}")

    tasks = []
    for learner in learners:
        for variant in VARIANTS:
            for seed in range(args.seeds):
                tasks.append((x, outcome_real, treat, learner, variant, seed, args.ate_true))

    print(f"Running {len(tasks)} tasks")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, row in enumerate(executor.map(run_task, tasks), start=1):
            rows.append(row)
            if index % 500 == 0:
                print(f"  completed {index}/{len(tasks)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
