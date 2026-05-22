#!/usr/bin/env python3
from __future__ import annotations

"""
Practitioner's capacity curve on real survey data.

At each sample size N, a practitioner cross-validates to pick the best
learner complexity, and we report the alignment gain at that
naturally-selected complexity. Uses ProcessPoolExecutor with incremental
CSV flushing.
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")

RESULT_FIELDS = [
    "dataset",
    "N",
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

RIDGE_ALPHAS = np.logspace(-3, 6, 19)

_GLOBAL_POP: dict = {}


@dataclass(frozen=True)
class LearnerSpec:
    name: str
    kind: str
    degree: int = 1
    depth: int = 2
    rounds: int = 50


LEARNER_GRID: list[LearnerSpec] = [
    LearnerSpec("ridge1", "ridge", degree=1),
    LearnerSpec("ridge2", "ridge", degree=2),
    LearnerSpec("ridge3", "ridge", degree=3),
    LearnerSpec("ridge4", "ridge", degree=4),
    LearnerSpec("ridge5", "ridge", degree=5),
    LearnerSpec("hgb_d1_r10", "hgb", depth=1, rounds=10),
    LearnerSpec("hgb_d1_r50", "hgb", depth=1, rounds=50),
    LearnerSpec("hgb_d2_r20", "hgb", depth=2, rounds=20),
    LearnerSpec("hgb_d2_r50", "hgb", depth=2, rounds=50),
    LearnerSpec("hgb_d2_r100", "hgb", depth=2, rounds=100),
    LearnerSpec("hgb_d3_r50", "hgb", depth=3, rounds=50),
    LearnerSpec("hgb_d3_r200", "hgb", depth=3, rounds=200),
]


def load_dataset(name: str, data_dir: str) -> np.ndarray:
    import pandas as pd

    if name == "acs":
        path = os.path.join(data_dir, "acs_pums_2022.csv")
        frame = pd.read_csv(path)
        cols = ["age_group", "sex", "race", "edu", "region"]
    elif name == "cps":
        path = os.path.join(data_dir, "cps_asec_2022.csv")
        cols = ["PERRP", "PRDTHSP", "PRDTRACE", "PRCITSHP", "A_AGE", "A_SEX", "A_MARITL", "A_HGA", "A_CLSWKR"]
        frame = pd.read_csv(path, usecols=cols)
    elif name == "brfss":
        path = os.path.join(data_dir, "brfss_2022_raw.csv")
        cols = ["_AGE_G", "_SEX", "_IMPRACE", "_EDUCAG", "_STATE"]
        frame = pd.read_csv(path, usecols=cols, low_memory=False)
    elif name == "ces":
        path = os.path.join(data_dir, "ces_2022.csv")
        cols = ["age_group", "sex", "race", "edu", "region"]
        frame = pd.read_csv(path, usecols=cols)
    elif name == "gss":
        path = os.path.join(data_dir, "gss_cumulative.csv")
        cols = ["age", "sex", "race", "educ", "region", "income"]
        frame = pd.read_csv(path, usecols=cols, low_memory=False)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    clean = frame[cols].replace([np.inf, -np.inf], np.nan).dropna()
    return clean.to_numpy(dtype=float)


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


def make_population(
    x_raw: np.ndarray,
    pop_size: int,
    response_rate: float,
    tail_fraction: float,
    tail_effect: float,
    tail_logit_penalty: float,
    pop_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(pop_seed)
    if len(x_raw) > pop_size:
        idx = rng.choice(len(x_raw), size=pop_size, replace=False)
        x_raw = x_raw[idx]
    x = StandardScaler().fit_transform(x_raw[:, :4])

    main, tail_score = make_scores(x)
    threshold = float(np.quantile(tail_score, 1.0 - tail_fraction))
    in_tail = tail_score >= threshold

    m0 = 0.7 * main + 0.35 * np.sin(1.5 * main) + 0.25 * tail_score
    m0 += tail_effect * in_tail.astype(float) * (1.0 + 0.25 * np.maximum(tail_score, 0.0))
    true_mean = float(np.mean(m0))

    raw_logit = -0.35 * main - tail_logit_penalty * in_tail.astype(float)
    lo, hi = -40.0, 40.0
    for _ in range(90):
        mid = (lo + hi) / 2.0
        if float(np.mean(expit(mid + raw_logit))) < response_rate:
            lo = mid
        else:
            hi = mid
    pi0 = expit((lo + hi) / 2.0 + raw_logit)

    return x, m0, pi0, true_mean


def _init_worker(x_pop: np.ndarray, m0: np.ndarray, pi0: np.ndarray, true_mean: float) -> None:
    _GLOBAL_POP["x"] = x_pop
    _GLOBAL_POP["m0"] = m0
    _GLOBAL_POP["pi0"] = pi0
    _GLOBAL_POP["true_mean"] = true_mean


def fit_propensity_cf(x: np.ndarray, r: np.ndarray, seed: int) -> np.ndarray:
    pred = np.zeros(len(r))
    splitter = KFold(n_splits=3, shuffle=True, random_state=seed)
    for train, test in splitter.split(x):
        model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=80, min_samples_leaf=20, random_state=seed,
        )
        model.fit(x[train], r[train])
        pred[test] = model.predict_proba(x[test])[:, 1]
    return np.clip(pred, 0.01, 0.99)


def make_poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return StandardScaler().fit_transform(poly.fit_transform(x))


def crossfit_outcome(
    x_design: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    learner: LearnerSpec,
    seed: int,
) -> tuple[np.ndarray, float]:
    n = len(y)
    pred = np.zeros(n)
    cv_mse_parts: list[float] = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed + 17)
    for tr, te in splitter.split(x_design):
        if learner.kind == "ridge":
            model = RidgeCV(alphas=RIDGE_ALPHAS)
            sw = None if weights is None else weights[tr]
            model.fit(x_design[tr], y[tr], sample_weight=sw)
        else:
            model = HistGradientBoostingRegressor(
                max_depth=learner.depth,
                max_iter=learner.rounds,
                min_samples_leaf=20,
                random_state=seed,
            )
            sw = None if weights is None else weights[tr]
            model.fit(x_design[tr], y[tr], sample_weight=sw)
        pred[te] = model.predict(x_design[te])
        cv_mse_parts.append(float(np.mean((y[te] - pred[te]) ** 2)))
    cv_mse = float(np.mean(cv_mse_parts))
    return pred, cv_mse


def make_weights(pi_resp: np.ndarray, exponent: float, weight_clip: float) -> np.ndarray | None:
    if exponent == 0.0:
        return None
    raw_w = 1.0 / np.maximum(pi_resp, 1e-6) ** exponent
    weights = raw_w / float(np.mean(raw_w))
    weights = np.minimum(weights, weight_clip)
    weights = weights / float(np.mean(weights))
    return weights


def run_cell(args: tuple) -> list[dict[str, object]]:
    (ds_name, n_sub, learner, seed, weight_clip, propensity_mode) = args

    try:
        return _run_cell_inner(ds_name, n_sub, learner, seed, weight_clip, propensity_mode)
    except (np.linalg.LinAlgError, ValueError):
        return []


def _run_cell_inner(
    ds_name: str, n_sub: int, learner: LearnerSpec, seed: int,
    weight_clip: float, propensity_mode: str,
) -> list[dict[str, object]]:
    x_pop = _GLOBAL_POP["x"]
    m0 = _GLOBAL_POP["m0"]
    pi0 = _GLOBAL_POP["pi0"]
    true_mean = _GLOBAL_POP["true_mean"]

    rng = np.random.default_rng(seed)
    n_pop = len(x_pop)

    if n_sub < n_pop:
        idx = rng.choice(n_pop, size=n_sub, replace=False)
        x = x_pop[idx]
        m = m0[idx]
        pi = pi0[idx]
    else:
        x = x_pop
        m = m0
        pi = pi0

    y = m + rng.standard_normal(len(m))
    r = rng.binomial(1, pi).astype(int)
    resp = r == 1
    n_resp = int(resp.sum())
    if n_resp < 30:
        return []

    if propensity_mode == "oracle":
        pi_hat = np.clip(pi, 0.01, 0.99)
    elif propensity_mode == "estimated":
        pi_hat = fit_propensity_cf(x, r, seed)
    else:
        raise ValueError(f"Unknown propensity mode: {propensity_mode}")
    pi_resp = pi_hat[resp]

    if learner.kind == "ridge":
        x_design = make_poly_features(x, learner.degree)
    else:
        x_design = x

    x_resp = x_design[resp]
    y_resp = y[resp]

    rows: list[dict[str, object]] = []

    for variant_name, exponent in [("unwt", 0.0), ("stab", 1.0), ("w2", 2.0)]:
        weights = make_weights(pi_resp, exponent, weight_clip)

        pred_cf, cv_mse = crossfit_outcome(x_resp, y_resp, weights, learner, seed)

        if learner.kind == "ridge":
            full_model = RidgeCV(alphas=RIDGE_ALPHAS)
            full_model.fit(x_resp, y_resp, sample_weight=weights)
        else:
            full_model = HistGradientBoostingRegressor(
                max_depth=learner.depth,
                max_iter=learner.rounds,
                min_samples_leaf=20,
                random_state=seed,
            )
            full_model.fit(x_resp, y_resp, sample_weight=weights)
        m_all = full_model.predict(x_design)

        m_blend = m_all.copy()
        m_blend[resp] = pred_cf
        estimate = float(np.mean(m_all) + np.mean(r * (y - m_blend) / pi_hat))
        bias = estimate - true_mean

        rows.append({
            "dataset": ds_name,
            "N": n_sub,
            "learner": learner.name,
            "seed": seed,
            "variant": variant_name,
            "propensity_mode": propensity_mode,
            "estimate": estimate,
            "true_value": true_mean,
            "bias": bias,
            "n_resp": n_resp,
            "cv_mse_unwt": cv_mse if variant_name == "unwt" else -1.0,
            "weight_clip": weight_clip,
        })

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="acs,cps,brfss,ces,gss")
    parser.add_argument("--sample-sizes", type=str, default="300,500,1000,2000,5000,10000")
    parser.add_argument(
        "--learners",
        type=str,
        default=",".join(learner.name for learner in LEARNER_GRID),
    )
    parser.add_argument("--response-rate", type=float, default=0.08)
    parser.add_argument("--tail-fraction", type=float, default=0.08)
    parser.add_argument("--tail-effect", type=float, default=5.0)
    parser.add_argument("--tail-logit-penalty", type=float, default=3.0)
    parser.add_argument("--population-size", type=int, default=30000)
    parser.add_argument("--population-seed", type=int, default=20260522)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--weight-clip", type=float, default=25.0)
    parser.add_argument(
        "--propensity-mode",
        choices=["estimated", "oracle"],
        default="estimated",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/practitioner_capacity_curve_v1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = [s.strip() for s in args.datasets.split(",")]
    sample_sizes = [int(s.strip()) for s in args.sample_sizes.split(",")]
    learner_names = {s.strip() for s in args.learners.split(",") if s.strip()}
    learners = [learner for learner in LEARNER_GRID if learner.name in learner_names]
    unknown_learners = learner_names - {learner.name for learner in LEARNER_GRID}
    if unknown_learners:
        raise ValueError(f"Unknown learners: {sorted(unknown_learners)}")

    directory = os.path.dirname(args.output)
    if directory:
        os.makedirs(directory, exist_ok=True)

    out_handle = open(args.output, "w", newline="")
    writer = csv.DictWriter(out_handle, fieldnames=RESULT_FIELDS)
    writer.writeheader()
    out_handle.flush()
    total_rows = 0
    t0 = time.time()

    for ds_idx, ds_name in enumerate(dataset_names):
        print(f"\n{'='*60}")
        print(f"[{ds_idx+1}/{len(dataset_names)}] Loading {ds_name}...")
        x_raw = load_dataset(ds_name, args.data_dir)
        print(f"  {ds_name}: {len(x_raw)} rows, {x_raw.shape[1]} cols")

        x_pop, m0, pi0, true_mean = make_population(
            x_raw,
            pop_size=args.population_size,
            response_rate=args.response_rate,
            tail_fraction=args.tail_fraction,
            tail_effect=args.tail_effect,
            tail_logit_penalty=args.tail_logit_penalty,
            pop_seed=args.population_seed,
        )
        print(f"  true_mean={true_mean:.3f}, response_rate={float(np.mean(pi0)):.3f}")

        tasks: list[tuple] = []
        for n_sub in sample_sizes:
            for learner in learners:
                for seed in range(args.seeds):
                    tasks.append(
                        (
                            ds_name,
                            n_sub,
                            learner,
                            seed,
                            args.weight_clip,
                            args.propensity_mode,
                        )
                    )

        n_tasks = len(tasks)
        print(f"  {n_tasks} tasks ({len(sample_sizes)} sizes x {len(learners)} learners x {args.seeds} seeds)")
        done = 0
        ds_t0 = time.time()

        if args.workers == 1:
            _init_worker(x_pop, m0, pi0, true_mean)
            task_results = (run_cell(t) for t in tasks)
        else:
            executor = ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(x_pop, m0, pi0, true_mean),
            )
            task_results = (future.result() for future in as_completed(
                {executor.submit(run_cell, t): t for t in tasks},
            ))

        try:
            for task_rows in task_results:
                if task_rows:
                    writer.writerows(task_rows)
                    total_rows += len(task_rows)
                done += 1
                if done % 50 == 0:
                    out_handle.flush()
                    elapsed = time.time() - ds_t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (n_tasks - done) / rate if rate > 0 else 0
                    print(f"  [{ds_name}] {done}/{n_tasks} ({100*done/n_tasks:.0f}%) | {total_rows} rows | {rate:.0f} tasks/s | ETA {eta:.0f}s")
        finally:
            if args.workers != 1:
                executor.shutdown()

        out_handle.flush()
        elapsed_ds = time.time() - ds_t0
        print(f"  {ds_name} DONE in {elapsed_ds:.0f}s. Total rows: {total_rows}")

    out_handle.close()
    elapsed_total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE. {total_rows} rows in {elapsed_total:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
