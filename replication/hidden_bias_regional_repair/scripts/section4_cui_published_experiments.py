#!/usr/bin/env python3
"""Adapter for preregistered Design 2: Cui--Tchetgen published DGP.

This file only supplies the data-generating process from Cui and Tchetgen
Tchetgen, Section 7.  Estimation, repair selection, damping, bootstrap logic,
and output schemas are delegated to the frozen validated_reference_transfer.py.

The ATE component E[Y(1)] is mapped to the MAR mean by treating treatment A as
the response indicator R.  The observed outcome among R=1 is Y(1), the target is
E[Y(1)], and pi(X)=Pr(A=1|X).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


D2_DESIGNS = {
    "cui_published_scenario1": "logistic_step",
    "cui_published_scenario2": "quadratic",
}


def _load_frozen(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_reference_transfer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _published_features(x: np.ndarray, scenario: str) -> np.ndarray:
    if scenario == "logistic_step":
        return _sigmoid(20.0 * (x - 0.5))
    if scenario == "quadratic":
        return x * x
    raise ValueError(f"unknown Cui scenario: {scenario}")


def _cui_published_data(n: int, design: str, seed: int):
    rng = np.random.default_rng(seed)
    scenario = D2_DESIGNS[design]
    x = rng.uniform(0.0, 1.0, (n, 5))
    fx = _published_features(x, scenario)
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    propensity = _sigmoid(fx @ signs)
    response = rng.binomial(1, propensity)

    # Cui Section 7: E(Y | A, X)=2{1+1'f(X)+1'f(X)A+A}.  For the
    # counterfactual mean E[Y(1)], the MAR outcome regression is mu1(X).
    feature_sum = np.sum(fx, axis=1)
    mu1 = 2.0 * (1.0 + feature_sum + feature_sum + 1.0)
    y1 = mu1 + rng.standard_normal(n)
    low_propensity = propensity <= float(np.quantile(propensity, 0.10))
    theta = float(np.mean(mu1))
    return x, y1, response, low_propensity, propensity, theta, mu1


def _install_adapter(module) -> None:
    original_make_data = module.make_data

    def make_data(n, epsilon, strength, design, seed, mar_design):
        if design in D2_DESIGNS:
            if mar_design != "box":
                raise ValueError("Cui published DGP adapter uses mar_design=box")
            return _cui_published_data(n, design, seed)
        return original_make_data(n, epsilon, strength, design, seed, mar_design)

    module.make_data = make_data


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
    module = _load_frozen(known.frozen_source.resolve())
    _install_adapter(module)
    sys.argv = [sys.argv[0], *forwarded]
    module.main()


if __name__ == "__main__":
    main()
