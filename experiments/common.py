#!/usr/bin/env python3
from __future__ import annotations

import csv
import os

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

EPS = 1e-6


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def parse_csv_strs(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def write_rows(path: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_observed_covariates(z_lat: np.ndarray) -> np.ndarray:
    z0 = z_lat[:, 0]
    z1 = z_lat[:, 1] if z_lat.shape[1] > 1 else np.zeros(len(z_lat))
    z2 = z_lat[:, 2] if z_lat.shape[1] > 2 else np.zeros(len(z_lat))
    z3 = z_lat[:, 3] if z_lat.shape[1] > 3 else np.zeros(len(z_lat))
    return np.column_stack(
        (
            np.exp(z0 / 2.0),
            z1 / (1.0 + np.exp(z0)) + 10.0,
            (z0 * z2 / 25.0 + 0.6) ** 3,
            (z1 + z3 + 20.0) ** 2,
        )
    )


def make_poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly.fit_transform(x)
    return StandardScaler().fit_transform(x_poly)


def normalize_weights(
    weights: np.ndarray,
    *,
    clip: float | None = 50.0,
    floor: float = EPS,
) -> np.ndarray:
    arr = np.maximum(np.asarray(weights, dtype=float), floor)
    if clip is not None:
        arr = np.minimum(arr, clip)
    return arr / np.mean(arr)


def weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(arr))
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w * arr) / np.sum(w))


def fit_crossfit_propensity(x: np.ndarray, a: np.ndarray, seed: int) -> np.ndarray:
    p_hat = np.zeros(len(a))
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in splitter.split(x):
        model = LogisticRegression(max_iter=4000, solver="lbfgs")
        model.fit(x[tr], a[tr])
        p_hat[te] = np.clip(model.predict_proba(x[te])[:, 1], 0.02, 0.98)
    return p_hat


def fit_log_variance_model(
    x_train: np.ndarray,
    resid_train: np.ndarray,
    x_test: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    alpha: float = 1.0,
    sigma_floor: float = 0.05,
) -> np.ndarray:
    target = np.log(np.maximum(resid_train**2, 1e-4))
    model = Ridge(alpha=alpha)
    model.fit(x_train, target, sample_weight=sample_weight)
    return np.sqrt(np.maximum(np.exp(model.predict(x_test)), sigma_floor**2))


def gaussian_kernel_separation(delta: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 0.05)
    ratio = np.asarray(delta, dtype=float) / sigma_safe
    return (1.0 / (np.sqrt(np.pi) * sigma_safe)) * (1.0 - np.exp(-(ratio**2) / 4.0))


def sigmoid_range(raw: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + (high - low) * expit(raw)
