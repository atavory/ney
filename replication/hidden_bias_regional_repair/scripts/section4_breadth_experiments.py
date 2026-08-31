#!/usr/bin/env python3
"""Run frozen reference-transfer code on preregistered breadth designs.

This adapter changes only the generated data and, for Kang--Schafer, which
published covariates are supplied to each nuisance fit.  Estimation, regional
selection, scalar shrinkage, bootstrap construction, and output schemas are
delegated to the frozen public ``validated_reference_transfer.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_wine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


KS_DESIGNS = {
    "kang_schafer_cc": ("correct", "correct"),
    "kang_schafer_ci": ("correct", "incorrect"),
    "kang_schafer_ic": ("incorrect", "correct"),
    "kang_schafer_ii": ("incorrect", "incorrect"),
}
ALIGNMENT_DESIGNS = {
    "alignment_aligned": (0.00, 0.10),
    "alignment_partial": (0.05, 0.15),
    "alignment_disjoint": (0.20, 0.30),
}
# Fixed Monte Carlo approximation to population quantiles of the *raw*,
# unclipped response score.  Generated once from 2,000,000 independent
# Uniform(0,1)^5 draws with seed 2026081001; never recomputed from a run sample.
ALIGNMENT_RAW_SCORE_CUTS = {
    0.00: -np.inf,
    0.05: 0.00621474689107956,
    0.10: 0.17041363046317684,
    0.15: 0.2287269268831464,
    0.20: 0.27443068109495083,
    0.30: 0.3807565804610278,
}
REAL_DESIGNS = {
    "real_digits_misaligned": ("digits", "wider_partial"),
    "real_breast_cancer_misaligned": ("breast_cancer", "wider_partial"),
    "real_diabetes_misaligned": ("diabetes", "wider_partial"),
    "real_wine_misaligned": ("wine", "wider_partial"),
    "real_digits_aligned": ("digits", "aligned"),
    "real_breast_cancer_aligned": ("breast_cancer", "aligned"),
    "real_diabetes_aligned": ("diabetes", "aligned"),
    "real_wine_aligned": ("wine", "aligned"),
}
SUPPORT_DATA = Path(
    os.environ.get(
        "DML_SUPPORT_DATA",
        os.environ.get(
            "USHMOO_SUPPORT_DATA",
            "/home/atavory/.overleaf_git_clone/69edff47028a983c95b7fcc2/support/data",
        ),
    )
)


def _load_frozen(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_reference_transfer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = (np.arange(len(values), dtype=float) + 0.5) / len(values)
    return ranks


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (x - center) / scale


def _col(x: np.ndarray, index: int) -> np.ndarray:
    return x[:, min(index, x.shape[1] - 1)]


def _kang_schafer(n: int, seed: int):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    z1, z2, z3, z4 = z.T
    wrong = np.column_stack(
        [
            np.exp(z1 / 2.0),
            z2 / (1.0 + np.exp(z1)) + 10.0,
            (z1 * z3 / 25.0 + 0.6) ** 3,
            (z2 + z4 + 20.0) ** 2,
        ]
    )
    mu = 210.0 + 27.4 * z1 + 13.7 * (z2 + z3 + z4)
    true_pi = _sigmoid(-z1 + 0.5 * z2 - 0.25 * z3 - 0.1 * z4)
    y = mu + rng.standard_normal(n)
    response = rng.binomial(1, true_pi)
    low_response = _rank01(true_pi) < 0.10
    return np.column_stack([z, wrong]), y, response, low_response, true_pi, 210.0, mu


def _alignment(n: int, epsilon: float, strength: float, design: str, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, (n, 5))
    raw_pi = (
        0.5
        + 0.4 * np.sin(2.0 * np.pi * x[:, 0] * x[:, 1])
        - 0.3 * (x[:, 2] > x[:, 3]).astype(float)
    )
    true_pi = np.clip(raw_pi, max(epsilon, 0.03), 0.95)
    lo, hi = ALIGNMENT_DESIGNS[design]
    defect = (raw_pi >= ALIGNMENT_RAW_SCORE_CUTS[lo]) & (
        raw_pi < ALIGNMENT_RAW_SCORE_CUTS[hi]
    )
    base = 0.3 + 0.4 * (x[:, 0] - 0.5) + 0.3 * np.sin(2.0 * np.pi * x[:, 1])
    mu = base + strength * defect.astype(float)
    y = mu + rng.standard_normal(n)
    response = rng.binomial(1, true_pi)
    return x, y, response, defect, true_pi, float(np.mean(mu)), mu


@lru_cache(maxsize=None)
def _real_pool(name: str) -> np.ndarray:
    loaded = {
        "digits": load_digits,
        "breast_cancer": load_breast_cancer,
        "diabetes": load_diabetes,
        "wine": load_wine,
    }[name]()
    return StandardScaler().fit_transform(loaded.data.astype(float))


@lru_cache(maxsize=None)
def _real_direction(name: str, width: int) -> np.ndarray:
    dataset_offsets = {
        "digits": 1,
        "breast_cancer": 2,
        "diabetes": 3,
        "wine": 4,
    }
    rng = np.random.default_rng(20260810 + dataset_offsets[name])
    direction = rng.normal(size=width)
    return direction / np.linalg.norm(direction)


def _real_semisynthetic(
    n: int,
    epsilon: float,
    strength: float,
    name: str,
    geometry: str,
    seed: int,
):
    rng = np.random.default_rng(seed)
    pool = _real_pool(name)
    rows = rng.integers(0, len(pool), n)
    x = pool[rows] + rng.normal(0.0, 0.15, (n, pool.shape[1]))
    score = x @ _real_direction(name, x.shape[1])
    score = (score - np.mean(score)) / (np.std(score) + 1e-12)
    low = score < -1.04
    base_response = _sigmoid(1.5 * score)
    true_pi = np.clip(np.where(low, epsilon, 0.15 + 0.75 * base_response), epsilon, 0.95)
    internal_index = min(2, x.shape[1] - 1)
    internal = x[:, internal_index]
    internal = (internal - np.mean(internal)) / (np.std(internal) + 1e-12)
    if geometry == "wider_partial":
        deviation = np.maximum(0.0, -0.52 - score) * (0.5 + internal)
    elif geometry == "aligned":
        deviation = low.astype(float)
    else:
        raise ValueError(f"unknown real-covariate geometry: {geometry}")
    mu = 0.3 + 0.4 * score + 0.3 * np.sin(1.5 * x[:, 1]) + strength * deviation
    y = mu + rng.standard_normal(n)
    response = rng.binomial(1, true_pi)
    defect = deviation != 0.0
    return x, y, response, defect, true_pi, float(np.mean(mu)), mu


@lru_cache(maxsize=None)
def _acic2017_population() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    path = SUPPORT_DATA / "acic" / "acic2017_x.csv"
    if not path.exists():
        raise FileNotFoundError(f"ACIC 2017 covariates not found at {path}")
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    matrix = np.column_stack([data[name] for name in data.dtype.names])
    matrix = _standardize_columns(np.nan_to_num(matrix))
    base = (
        0.35 * _col(matrix, 0)
        - 0.25 * _col(matrix, 4)
        + 0.20 * _col(matrix, 11) * _col(matrix, 12)
        + 0.25 * np.sin(_col(matrix, 20))
    )
    base = _standardize_columns(base.reshape(-1, 1))[:, 0]
    score_region = (
        0.8 * _col(matrix, 1)
        - 0.7 * _col(matrix, 5)
        + 0.6 * _col(matrix, 16)
        - 0.4 * _col(matrix, 28)
    )
    score_signal = (
        0.7 * _col(matrix, 2) + 0.6 * _col(matrix, 8) - 0.5 * _col(matrix, 24)
    )
    response_region = score_region <= np.quantile(score_region, 0.15)
    signal_region = score_signal <= np.quantile(score_signal, 0.18)
    outside_rank = _rank01(0.5 * _col(matrix, 3) + 0.4 * _col(matrix, 17))
    return matrix, base, response_region, signal_region, outside_rank


def _acic2017_semisynthetic(
    n: int, epsilon: float, strength: float, design: str, seed: int
):
    rng = np.random.default_rng(seed)
    x_pop, base, response_region, signal_region, outside_rank = _acic2017_population()
    rows = rng.integers(0, len(base), n)
    x = x_pop[rows]
    region = response_region[rows]
    outside = np.clip(0.68 + 0.22 * outside_rank[rows], 0.05, 0.95)
    true_pi = np.where(region, epsilon, outside)
    response = rng.binomial(1, true_pi)
    if design == "acic2017_semisynth":
        signal_pop = response_region.astype(float) * (
            0.70 + 0.50 * _rank01(_col(x_pop, 2))
        )
    elif design == "acic2017_misaligned":
        signal_pop = signal_region.astype(float) * (
            0.70 + 0.50 * _rank01(_col(x_pop, 6))
        )
    else:
        raise ValueError(f"unknown ACIC 2017 design: {design}")
    mu_pop = base + 0.35 * strength * signal_pop
    y = mu_pop[rows] + rng.standard_normal(n)
    theta = float(mu_pop.mean())
    return x, y, response, region, true_pi, theta, mu_pop[rows]


def _column_selector(kind: str) -> FunctionTransformer:
    slc = slice(0, 4) if kind == "correct" else slice(4, 8)
    return FunctionTransformer(lambda values: values[:, slc], validate=False)


def _install_adapter(module) -> None:
    original_make_data = module.make_data
    original_regressor = module._regressor
    original_classifier = module._classifier
    original_cui_propensity = module._cui_candidate_propensity
    original_cui_outcome = module._cui_candidate_outcome
    state = {"design": ""}

    def make_data(n, epsilon, strength, design, seed, mar_design):
        state["design"] = design
        if design in KS_DESIGNS:
            return _kang_schafer(n, seed)
        if design in ALIGNMENT_DESIGNS:
            return _alignment(n, epsilon, strength, design, seed)
        if design in REAL_DESIGNS:
            name, geometry = REAL_DESIGNS[design]
            return _real_semisynthetic(
                n, epsilon, strength, name, geometry, seed
            )
        if design in {"acic2017_semisynth", "acic2017_misaligned"}:
            return _acic2017_semisynthetic(n, epsilon, strength, design, seed)
        return original_make_data(n, epsilon, strength, design, seed, mar_design)

    def nuisance_kind(which: int) -> str | None:
        pair = KS_DESIGNS.get(state["design"])
        return pair[which] if pair is not None else None

    def regressor(seed, learner):
        estimator = original_regressor(seed, learner)
        kind = nuisance_kind(0)
        return make_pipeline(_column_selector(kind), estimator) if kind else estimator

    def classifier(seed, learner):
        estimator = original_classifier(seed, learner)
        kind = nuisance_kind(1)
        return make_pipeline(_column_selector(kind), estimator) if kind else estimator

    def cui_propensity(name, seed):
        estimator = original_cui_propensity(name, seed)
        kind = nuisance_kind(1)
        # The gradient-boosting factory delegates to module._classifier, which
        # is already column-selecting under KS.  Wrapping it again would slice
        # the 4-column result with 4:8 and produce zero features.
        if kind and name != "gradient_boosting":
            return make_pipeline(_column_selector(kind), estimator)
        return estimator

    def cui_outcome(name, seed):
        estimator = original_cui_outcome(name, seed)
        kind = nuisance_kind(0)
        if kind and name != "gradient_boosting":
            return make_pipeline(_column_selector(kind), estimator)
        return estimator

    module.make_data = make_data
    module._regressor = regressor
    module._classifier = classifier
    module._cui_candidate_propensity = cui_propensity
    module._cui_candidate_outcome = cui_outcome


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--frozen-source",
        type=Path,
        default=Path(os.environ.get("USHMOO_VALIDATED_SOURCE", "")),
    )
    known, forwarded = parser.parse_known_args()
    if not str(known.frozen_source):
        raise SystemExit("set --frozen-source or USHMOO_VALIDATED_SOURCE")
    source = known.frozen_source.resolve()
    module = _load_frozen(source)
    _install_adapter(module)
    sys.argv = [sys.argv[0], *forwarded]
    module.main()


if __name__ == "__main__":
    main()
