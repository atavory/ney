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
from sklearn.datasets import load_breast_cancer, load_digits
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
    "real_digits_misaligned": "digits",
    "real_breast_cancer_misaligned": "breast_cancer",
}


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
    loaded = {"digits": load_digits, "breast_cancer": load_breast_cancer}[name]()
    return StandardScaler().fit_transform(loaded.data.astype(float))


@lru_cache(maxsize=None)
def _real_direction(name: str, width: int) -> np.ndarray:
    rng = np.random.default_rng(20260810 + (1 if name == "digits" else 2))
    direction = rng.normal(size=width)
    return direction / np.linalg.norm(direction)


def _real_misaligned(n: int, epsilon: float, strength: float, name: str, seed: int):
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
    deviation = np.maximum(0.0, -0.52 - score) * (0.5 + internal)
    mu = 0.3 + 0.4 * score + 0.3 * np.sin(1.5 * x[:, 1]) + strength * deviation
    y = mu + rng.standard_normal(n)
    response = rng.binomial(1, true_pi)
    defect = deviation != 0.0
    return x, y, response, defect, true_pi, float(np.mean(mu)), mu


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
            return _real_misaligned(n, epsilon, strength, REAL_DESIGNS[design], seed)
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
