#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common import (
    normalize_weights,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strs,
    write_rows,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")

logger: logging.Logger = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent.parent / "data"
RESULT_FIELDS = [
    "dataset",
    "population_size",
    "response_rate",
    "tail_fraction",
    "tail_effect",
    "feature_count",
    "learner",
    "pi_source",
    "seed",
    "variant",
    "estimate",
    "true_value",
    "bias",
    "abs_bias",
    "n_resp",
    "tail_resp_rate",
    "body_resp_rate",
    "tail_share_resp",
    "fit_deff",
    "corr_risk",
    "tail_rmse",
    "body_rmse",
]


@dataclass(frozen=True)
class Dataset:
    name: str
    x: np.ndarray


@dataclass(frozen=True)
class LearnerSpec:
    name: str
    kind: str
    degree: int = 1
    depth: int = 1
    rounds: int = 20
    alpha: float = 1.0


@dataclass(frozen=True)
class Variant:
    name: str
    exponent: float
    clip: float | None


LEARNERS = {
    "ridge1": LearnerSpec(name="ridge1", kind="ridge", degree=1, alpha=10.0),
    "ridge2": LearnerSpec(name="ridge2", kind="ridge", degree=2, alpha=10.0),
    "stump20": LearnerSpec(name="stump20", kind="hgb", depth=1, rounds=20),
    "hgb2_50": LearnerSpec(name="hgb2_50", kind="hgb", depth=2, rounds=50),
}
VARIANTS = (
    Variant("unwt", 0.0, None),
    Variant("w05", 0.5, 25.0),
    Variant("stab", 1.0, 25.0),
    Variant("w15", 1.5, 25.0),
    Variant("w2_c10", 2.0, 10.0),
    Variant("w2_c25", 2.0, 25.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="acs,cps,brfss,ces,gss",
    )
    parser.add_argument("--population-size", type=int, default=30000)
    parser.add_argument("--response-rates", type=str, default="0.02,0.05,0.10")
    parser.add_argument("--tail-fractions", type=str, default="0.05,0.10")
    parser.add_argument("--tail-effects", type=str, default="2.0,4.0")
    parser.add_argument("--feature-counts", type=str, default="1,2,4")
    parser.add_argument("--learners", type=str, default="ridge1,ridge2,stump20")
    parser.add_argument("--pi-sources", type=str, default="oracle,estimated")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--pi-floor", type=float, default=0.01)
    parser.add_argument("--population-seed", type=int, default=20260518)
    parser.add_argument(
        "--output",
        type=str,
        default="results/real_tail_aipw_v1.csv",
    )
    return parser.parse_args()


def clean_frame(frame: pd.DataFrame, columns: list[str], name: str) -> Dataset:
    clean = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    return Dataset(name=name, x=clean.to_numpy(dtype=float))


def load_dataset(name: str) -> Dataset:
    if name == "acs":
        frame = pd.read_csv(DATA / "acs_pums_2022.csv")
        return clean_frame(frame, ["age_group", "sex", "race", "edu", "region"], "acs")
    if name == "cps":
        columns = [
            "PERRP",
            "PRDTHSP",
            "PRDTRACE",
            "PRCITSHP",
            "A_AGE",
            "A_SEX",
            "A_MARITL",
            "A_HGA",
            "A_CLSWKR",
        ]
        frame = pd.read_csv(DATA / "cps_asec_2022.csv", usecols=columns)
        return clean_frame(frame, columns, "cps")
    if name == "brfss":
        columns = ["_AGE_G", "_SEX", "_IMPRACE", "_EDUCAG", "_STATE"]
        frame = pd.read_csv(DATA / "brfss_2022_raw.csv", usecols=columns, low_memory=False)
        return clean_frame(frame, columns, "brfss")
    if name == "ces":
        columns = ["age_group", "sex", "race", "edu", "region"]
        frame = pd.read_csv(DATA / "ces_2022.csv", usecols=columns)
        return clean_frame(frame, columns, "ces")
    if name == "gss":
        columns = ["age", "sex", "race", "educ", "region", "income"]
        frame = pd.read_csv(DATA / "gss_cumulative.csv", usecols=columns, low_memory=False)
        return clean_frame(frame, columns, "gss")
    raise ValueError(f"Unknown dataset: {name}")


def subsample(dataset: Dataset, size: int, seed: int) -> Dataset:
    if len(dataset.x) <= size:
        return dataset
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(dataset.x), size=size, replace=False)
    return Dataset(name=dataset.name, x=dataset.x[keep])


def standardized_prefix(x: np.ndarray, feature_count: int) -> np.ndarray:
    width = min(feature_count, x.shape[1])
    return StandardScaler().fit_transform(x[:, :width])


def safe_standardize(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    if scale <= 1e-12:
        return values - float(np.mean(values))
    return (values - float(np.mean(values))) / scale


def make_scores(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = x[:, 0]
    second = x[:, 1] if x.shape[1] > 1 else np.zeros(len(x))
    third = x[:, 2] if x.shape[1] > 2 else np.zeros(len(x))
    main = safe_standardize(0.8 * first - 0.5 * second + 0.25 * third)
    tail = safe_standardize(first + 0.75 * second + 0.5 * first * second)
    return main, tail


def make_tail_indicator(tail_score: np.ndarray, tail_fraction: float) -> np.ndarray:
    threshold = float(np.quantile(tail_score, 1.0 - tail_fraction))
    return tail_score >= threshold


def make_conditional_mean(
    main_score: np.ndarray,
    tail_score: np.ndarray,
    tail: np.ndarray,
    tail_effect: float,
) -> np.ndarray:
    smooth = 0.7 * main_score + 0.35 * np.sin(1.5 * main_score)
    body = 0.25 * tail_score
    tail_jump = tail_effect * tail.astype(float) * (1.0 + 0.25 * np.maximum(tail_score, 0.0))
    return smooth + body + tail_jump


def calibrate_propensity(
    main_score: np.ndarray,
    tail: np.ndarray,
    response_rate: float,
) -> np.ndarray:
    raw = -0.35 * main_score - 3.0 * tail.astype(float)
    lo = -40.0
    hi = 40.0
    for _ in range(90):
        mid = (lo + hi) / 2.0
        if float(np.mean(expit(mid + raw))) < response_rate:
            lo = mid
        else:
            hi = mid
    return expit((lo + hi) / 2.0 + raw)


def estimate_propensity(x: np.ndarray, r: np.ndarray, seed: int, floor: float) -> np.ndarray:
    pred = np.zeros(len(r))
    splitter = KFold(n_splits=3, shuffle=True, random_state=seed)
    for train, test in splitter.split(x):
        model = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=80,
            min_samples_leaf=20,
            random_state=seed,
        )
        model.fit(x[train], r[train])
        pred[test] = model.predict_proba(x[test])[:, 1]
    return np.clip(pred, floor, 1.0 - floor)


def make_design(x: np.ndarray, learner: LearnerSpec) -> np.ndarray:
    if learner.kind != "ridge":
        return x
    poly = PolynomialFeatures(degree=learner.degree, include_bias=False)
    return StandardScaler().fit_transform(poly.fit_transform(x))


def crossfit_ridge(
    x_resp: np.ndarray,
    y_resp: np.ndarray,
    weights: np.ndarray | None,
    learner: LearnerSpec,
) -> np.ndarray:
    pred = np.zeros(len(y_resp))
    splitter = KFold(n_splits=5, shuffle=True, random_state=17)
    for train, test in splitter.split(x_resp):
        model = Ridge(alpha=learner.alpha)
        sample_weight = None if weights is None else weights[train]
        model.fit(x_resp[train], y_resp[train], sample_weight=sample_weight)
        pred[test] = model.predict(x_resp[test])
    return pred


def crossfit_hgb(
    x_resp: np.ndarray,
    y_resp: np.ndarray,
    weights: np.ndarray | None,
    learner: LearnerSpec,
    seed: int,
) -> np.ndarray:
    pred = np.zeros(len(y_resp))
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed + 17)
    for train, test in splitter.split(x_resp):
        model = HistGradientBoostingRegressor(
            max_depth=learner.depth,
            max_iter=learner.rounds,
            min_samples_leaf=20,
            random_state=seed,
        )
        sample_weight = None if weights is None else weights[train]
        model.fit(x_resp[train], y_resp[train], sample_weight=sample_weight)
        pred[test] = model.predict(x_resp[test])
    return pred


def fit_predict(
    x_all: np.ndarray,
    y: np.ndarray,
    resp: np.ndarray,
    learner: LearnerSpec,
    weights: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_resp = x_all[resp]
    y_resp = y[resp]
    if learner.kind == "ridge":
        model = Ridge(alpha=learner.alpha)
        model.fit(x_resp, y_resp, sample_weight=weights)
        return model.predict(x_all), crossfit_ridge(x_resp, y_resp, weights, learner)
    model = HistGradientBoostingRegressor(
        max_depth=learner.depth,
        max_iter=learner.rounds,
        min_samples_leaf=20,
        random_state=seed,
    )
    model.fit(x_resp, y_resp, sample_weight=weights)
    return model.predict(x_all), crossfit_hgb(x_resp, y_resp, weights, learner, seed)


def make_fit_weights(pi_resp: np.ndarray, variant: Variant) -> np.ndarray | None:
    if variant.name == "unwt":
        return None
    raw = 1.0 / np.maximum(pi_resp, 1e-6) ** variant.exponent
    return normalize_weights(raw, clip=variant.clip)


def design_effect(weights: np.ndarray | None) -> float:
    if weights is None:
        return 1.0
    norm = weights / float(np.mean(weights))
    return float(np.mean(norm**2))


def run_cell(
    dataset: Dataset,
    response_rate: float,
    tail_fraction: float,
    tail_effect: float,
    feature_count: int,
    learner: LearnerSpec,
    pi_source: str,
    seed: int,
    pi_floor: float,
) -> list[dict[str, object]]:
    x = standardized_prefix(dataset.x, feature_count)
    main_score, tail_score = make_scores(x)
    tail = make_tail_indicator(tail_score, tail_fraction)
    mu = make_conditional_mean(main_score, tail_score, tail, tail_effect)
    pi_true = calibrate_propensity(main_score, tail, response_rate)
    rng = np.random.default_rng(seed)
    y = mu + rng.normal(scale=0.25, size=len(mu))
    r = rng.binomial(1, pi_true).astype(int)
    resp = r == 1
    if int(resp.sum()) < 50 or int(np.sum(tail & resp)) < 5:
        return []
    pi_used = pi_true if pi_source == "oracle" else estimate_propensity(x, r, seed, pi_floor)
    pi_used = np.clip(pi_used, pi_floor, 1.0 - pi_floor)
    x_design = make_design(x, learner)
    return run_variants(dataset, x_design, y, mu, resp, r, pi_used, tail, response_rate, tail_fraction, tail_effect, feature_count, learner, pi_source, seed)


def run_variants(
    dataset: Dataset,
    x_design: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    resp: np.ndarray,
    r: np.ndarray,
    pi_used: np.ndarray,
    tail: np.ndarray,
    response_rate: float,
    tail_fraction: float,
    tail_effect: float,
    feature_count: int,
    learner: LearnerSpec,
    pi_source: str,
    seed: int,
) -> list[dict[str, object]]:
    true_value = float(np.mean(y))
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        fit_weights = make_fit_weights(pi_used[resp], variant)
        m_all, m_resp_cf = fit_predict(x_design, y, resp, learner, fit_weights, seed)
        m_blend = m_all.copy()
        m_blend[resp] = m_resp_cf
        estimate = float(np.mean(m_all) + np.mean(r * (y - m_blend) / pi_used))
        rows.append(
            make_row(
                dataset,
                response_rate,
                tail_fraction,
                tail_effect,
                feature_count,
                learner,
                pi_source,
                seed,
                variant,
                estimate,
                true_value,
                resp,
                tail,
                pi_used,
                fit_weights,
                m_all,
                mu,
            )
        )
    return rows


def make_row(
    dataset: Dataset,
    response_rate: float,
    tail_fraction: float,
    tail_effect: float,
    feature_count: int,
    learner: LearnerSpec,
    pi_source: str,
    seed: int,
    variant: Variant,
    estimate: float,
    true_value: float,
    resp: np.ndarray,
    tail: np.ndarray,
    pi_used: np.ndarray,
    fit_weights: np.ndarray | None,
    m_all: np.ndarray,
    mu: np.ndarray,
) -> dict[str, object]:
    bias = estimate - true_value
    resid = mu - m_all
    body = ~tail
    return {
        "dataset": dataset.name,
        "population_size": len(dataset.x),
        "response_rate": response_rate,
        "tail_fraction": tail_fraction,
        "tail_effect": tail_effect,
        "feature_count": feature_count,
        "learner": learner.name,
        "pi_source": pi_source,
        "seed": seed,
        "variant": variant.name,
        "estimate": estimate,
        "true_value": true_value,
        "bias": bias,
        "abs_bias": abs(bias),
        "n_resp": int(np.sum(resp)),
        "tail_resp_rate": float(np.mean(resp[tail])),
        "body_resp_rate": float(np.mean(resp[body])),
        "tail_share_resp": float(np.mean(tail[resp])),
        "fit_deff": design_effect(fit_weights),
        "corr_risk": float(np.mean((resid**2) / pi_used)),
        "tail_rmse": float(np.sqrt(np.mean(resid[tail] ** 2))),
        "body_rmse": float(np.sqrt(np.mean(resid[body] ** 2))),
    }


def build_tasks(
    datasets: tuple[Dataset, ...],
    response_rates: tuple[float, ...],
    tail_fractions: tuple[float, ...],
    tail_effects: tuple[float, ...],
    feature_counts: tuple[int, ...],
    learners: tuple[LearnerSpec, ...],
    pi_sources: tuple[str, ...],
    seeds: int,
    pi_floor: float,
) -> list[tuple[Dataset, float, float, float, int, LearnerSpec, str, int, float]]:
    tasks: list[tuple[Dataset, float, float, float, int, LearnerSpec, str, int, float]] = []
    for dataset in datasets:
        for response_rate in response_rates:
            for tail_fraction in tail_fractions:
                for tail_effect in tail_effects:
                    for feature_count in feature_counts:
                        for learner in learners:
                            for pi_source in pi_sources:
                                for seed in range(seeds):
                                    tasks.append(
                                        (
                                            dataset,
                                            response_rate,
                                            tail_fraction,
                                            tail_effect,
                                            feature_count,
                                            learner,
                                            pi_source,
                                            seed,
                                            pi_floor,
                                        )
                                    )
    return tasks


def run_task(task: tuple[Dataset, float, float, float, int, LearnerSpec, str, int, float]) -> list[dict[str, object]]:
    return run_cell(*task)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    datasets = tuple(
        subsample(load_dataset(name), args.population_size, args.population_seed + index)
        for index, name in enumerate(parse_csv_strs(args.datasets))
    )
    learners = tuple(LEARNERS[name] for name in parse_csv_strs(args.learners))
    for dataset in datasets:
        logger.info(f"Loaded {dataset.name}: n={len(dataset.x)}, p={dataset.x.shape[1]}")
    tasks = build_tasks(
        datasets=datasets,
        response_rates=parse_csv_floats(args.response_rates),
        tail_fractions=parse_csv_floats(args.tail_fractions),
        tail_effects=parse_csv_floats(args.tail_effects),
        feature_counts=parse_csv_ints(args.feature_counts),
        learners=learners,
        pi_sources=parse_csv_strs(args.pi_sources),
        seeds=args.seeds,
        pi_floor=args.pi_floor,
    )
    logger.info(f"Running {len(tasks)} real-tail AIPW cells")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, task_rows in enumerate(executor.map(run_task, tasks), start=1):
            rows.extend(task_rows)
            if index % 500 == 0:
                logger.info(f"  completed {index}/{len(tasks)}")
    write_rows(args.output, RESULT_FIELDS, rows)
    logger.info(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
